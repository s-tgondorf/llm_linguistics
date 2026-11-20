"""Entry point: 03_analyze (reads configs/ + CLI overrides)."""
import sys
from pathlib import Path

from treesample.io import jsonl

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))  # works without `pip install -e .`


from treesample.analysis.trees import ParseTree
from treesample.config import load_config

if __name__ == "__main__":
    cfg = load_config(ROOT, sys.argv[1:])
    parsed_generation = jsonl.read_jsonl(cfg.paths.parses)
    for generations in parsed_generation:
        for parsed_lines in generations["parsed"]:
            for parsed_line in parsed_lines:
                parse_tree = ParseTree(parsed_line)
                print(parse_tree.text + ":")
                print(parse_tree.count_label("NP"))
                print(parse_tree.count_spacy_tag("NNS"))
                print(parse_tree.count_spacy_pos("NOUN"))
                print()