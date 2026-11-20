"""Load configs/config.yaml (+ group files) with CLI overrides, using OmegaConf only.

Mimics the part of Hydra we need, without the dependency:

    model=olmo2_7b                     -> swap the config group file configs/model/olmo2_7b.yaml
    model.revision=stage1-step1000...  -> override a single value (dotlist)
    sampling.max_new_tokens=20
    backend=hf device=cpu
"""
from __future__ import annotations

import importlib.util
from pathlib import Path

from omegaconf import DictConfig, OmegaConf


def _cuda_available() -> bool:
    try:
        import torch
    except ImportError:
        return False
    return torch.cuda.is_available()


def _load_group(conf_dir: Path, group: str, name: str) -> DictConfig:
    path = conf_dir / group / f"{name}.yaml"
    if not path.exists():
        options = sorted(p.stem for p in (conf_dir / group).glob("*.yaml"))
        raise FileNotFoundError(f"No config '{group}={name}'. Available: {options}")
    sub = OmegaConf.load(path)
    # Files marked `# @package _global_` are merged at the root instead of under their group key.
    head = "\n".join(path.read_text().splitlines()[:3])
    is_global = "@package _global_" in head
    return sub if is_global else OmegaConf.create({group: sub})


def load_config(root: str | Path, overrides: list[str] | None = None) -> DictConfig:
    root = Path(root)
    conf_dir = root / "configs"
    base = OmegaConf.load(conf_dir / "config.yaml")

    defaults = OmegaConf.to_container(base.pop("defaults", []))
    groups: dict[str, str] = {}
    for item in defaults:
        if item == "_self_":
            continue
        ((group, name),) = item.items()
        groups[group] = name

    dotlist = []
    for ov in overrides or []:
        key, sep, value = ov.partition("=")
        if not sep:
            raise ValueError(f"Override must look like key=value, got '{ov}'")
        if "." not in key and (conf_dir / key).is_dir():
            groups[key] = value  # config-group switch
        else:
            dotlist.append(ov)

    parts = [_load_group(conf_dir, g, n) for g, n in groups.items()]
    cfg = OmegaConf.merge(*parts, base, OmegaConf.from_dotlist(dotlist))
    cfg = OmegaConf.create(OmegaConf.to_container(cfg, resolve=True))  # resolve ${...}

    # ---- resolve device / backend / dtype
    if cfg.device == "auto":
        cfg.device = "cuda" if _cuda_available() else "cpu"
    if cfg.backend == "auto":
        has_vllm = importlib.util.find_spec("vllm") is not None
        cfg.backend = "vllm" if (cfg.device == "cuda" and has_vllm) else "hf"
    if cfg.model.dtype == "auto":
        cfg.model.dtype = "bfloat16" if cfg.device == "cuda" else "float32"

    # ---- make relative paths absolute (relative to project root)
    for key in cfg.paths:
        cfg.paths[key] = str(root / cfg.paths[key])
    if not Path(cfg.sampling.prompt_file).is_absolute():
        cfg.sampling.prompt_file = str(root / cfg.sampling.prompt_file)

    return cfg
