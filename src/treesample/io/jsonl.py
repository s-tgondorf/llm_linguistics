"""read_jsonl / write_jsonl helpers (streaming, append-safe)."""
from __future__ import annotations

import json
from collections.abc import Iterable, Iterator
from pathlib import Path


def write_jsonl(path: str | Path, records: Iterable[dict], append: bool = False) -> int:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    count = 0
    with path.open("a" if append else "w", encoding="utf-8") as f:
        for rec in records:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
            count += 1
    return count


def read_jsonl(path: str | Path) -> Iterator[dict]:
    with Path(path).open(encoding="utf-8") as f:
        for line in f:
            if line.strip():
                yield json.loads(line)
