"""Entry point: 02_parse (reads configs/ + CLI overrides)."""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))  # works without `pip install -e .`

from treesample.config import load_config  # noqa: E402
from treesample.parsing.benepar_parser import BeneparParser  # noqa: E402
from treesample.parsing.parse_tree import ParseTree  # noqa: E402
import treesample.io.jsonl as jsonl  # noqa: E402

if __name__ == "__main__":
    cfg = load_config(ROOT, sys.argv[1:])
    parser = BeneparParser()
    generated = jsonl.read_jsonl(cfg.paths.samples)
    generated = [rec for rec in generated if "parsed" not in rec]  # skip already parsed
    for rec in generated:
        #rec["text"] = "The quick brown foxes jump over the lazy dog."
        parsed_lines = []
        text = " ".join(rec["full_text"].split()) # only parse the first line of the generated text
        parsed = parser.parse(text)
        parsed_dict = parser.doc_to_dict(parsed)
        parsed_lines.append(parsed_dict)
        rec["parsed"] = parsed_lines
    jsonl.write_jsonl(cfg.paths.parses, generated)

