"""build_generator(cfg) -> TextGenerator; picks hf or vllm from cfg.backend (auto-detect)."""
from __future__ import annotations

from .base import TextGenerator


def build_generator(cfg) -> TextGenerator:
    """Create the generator described by the (already resolved) config."""
    m = cfg.model
    if cfg.backend == "hf":
        from .hf_backend import HFGenerator

        return HFGenerator(
            hf_id=m.hf_id,
            revision=m.revision,
            dtype=m.dtype,
            device=cfg.device,
            trust_remote_code=m.trust_remote_code,
        )
    if cfg.backend == "vllm":
        from .vllm_backend import VLLMGenerator

        return VLLMGenerator(
            hf_id=m.hf_id,
            revision=m.revision,
            dtype=m.dtype,
            seed=cfg.seed,
            trust_remote_code=m.trust_remote_code,
        )
    raise ValueError(f"Unknown backend '{cfg.backend}' (expected hf or vllm)")
