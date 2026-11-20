"""Sampling loop: read prompts, call the generator, write JSONL (with model/revision/seed metadata)."""
from __future__ import annotations

from pathlib import Path

from ..io.jsonl import write_jsonl
from ..models.registry import build_generator


def read_prompts(path: str | Path) -> list[str]:
    """One prompt per line. An empty line means unconditional generation."""
    text = Path(path).read_text(encoding="utf-8")
    return [line.rstrip("\r") for line in text.split("\n")[: -1 if text.endswith("\n") else None]]


def run_sampling(cfg) -> list[dict]:
    s = cfg.sampling
    prompts = read_prompts(s.prompt_file)
    print(
        f"model={cfg.model.hf_id} revision={cfg.model.revision} "
        f"backend={cfg.backend} device={cfg.device} dtype={cfg.model.dtype}\n"
        f"{len(prompts)} prompts x {s.num_samples} samples, max_new_tokens={s.max_new_tokens}\n"
    )

    generator = build_generator(cfg)
    try:
        records = generator.generate(
            prompts,
            n=s.num_samples,
            max_new_tokens=s.max_new_tokens,
            temperature=s.temperature,
            top_p=s.top_p,
            top_k=s.top_k,
            batch_size=s.batch_size,
            seed=cfg.seed,
        )
    finally:
        generator.close()

    meta = {
        "model": cfg.model.name,
        "hf_id": cfg.model.hf_id,
        "revision": cfg.model.revision,
        "backend": cfg.backend,
        "seed": cfg.seed,
        "temperature": s.temperature,
        "top_p": s.top_p,
        "top_k": s.top_k,
        "max_new_tokens": s.max_new_tokens,
    }
    for r in records:
        r.update(meta)
        r["full_text"] = r["prompt"] + r["text"]

    n_written = write_jsonl(cfg.paths.samples, records)

    for r in records[: s.print_samples]:
        shown_prompt = r["prompt"] if r["prompt"] else "<unconditional>"
        text = r["text"].replace("\n", "\\n")
        print(f"[{r['sample_id']}] {shown_prompt!r} -> {text!r}  ({r['num_new_tokens']} tokens)")
    if len(records) > s.print_samples:
        print(f"... ({len(records) - s.print_samples} more not shown)")
    print(f"\nWrote {n_written} samples to {cfg.paths.samples}")
    return records
