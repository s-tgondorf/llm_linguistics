"""Entry point: sample strings from the model chosen in configs/ and save them as JSONL.

Examples:
    python scripts/01_sample.py
    python scripts/01_sample.py model=olmo2_7b model.revision=stage1-step100000-tokens420B
    python scripts/01_sample.py backend=hf device=cpu sampling.max_new_tokens=20 sampling.num_samples=3
"""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))  # works without `pip install -e .`

from treesample.config import load_config  # noqa: E402
from treesample.sampling.sampler import run_sampling  # noqa: E402

if __name__ == "__main__":
    cfg = load_config(ROOT, sys.argv[1:])
    run_sampling(cfg)
