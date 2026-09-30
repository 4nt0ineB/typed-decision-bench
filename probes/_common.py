"""Shared plumbing for the probes: archives, latency, how an answer is read.

The Jev client itself is bench/models/jev.py.
"""

from __future__ import annotations

import json
import os
import statistics
import sys
import time
from pathlib import Path
from typing import Any

from typesafe_sdk import ChoiceAnswer, NoulAnswer, ScoreAnswer

ROOT = Path(__file__).resolve().parent.parent
RESULTS = ROOT / "results" / "probes"
# Probes run as scripts from probes/, and import the client, tasks and adapters from bench.
sys.path.insert(0, str(ROOT))

def fr_corpus_dir() -> Path:
    """The French filler corpus P4 and P5 bury their needle in.

    Not in the repo: the set used here is administrative prose on property and gifts that
    belongs to another sandbox. Any directory of `.md` files in comparable French register
    works, which is why this is a path rather than a download.
    """
    path = Path(os.environ.get("FR_CORPUS", ROOT / "data" / "fr-corpus"))
    if not sorted(path.glob("*.md")):
        raise SystemExit(
            f"no French corpus at {path}. Drop a few .md files of French prose there, "
            f"or point FR_CORPUS at a directory that has some. See docs/probes-bench.md."
        )
    return path


def answer_value(answer: Any) -> Any:
    if isinstance(answer, NoulAnswer):
        return answer.noul
    if isinstance(answer, ChoiceAnswer):
        return answer.choice
    return answer.score


def decided(answer: Any) -> Any:
    """What the code would branch on, not what the field holds.

    A Noul going from 0.98 to 0.99 and a Score going from 1.94 to 1.95 change no decision.
    Comparing those raw floats produced five false positives in this project (P1 twice, P8,
    P11, P16), each first read as model instability. So the thresholding lives here, not in
    the caller who will forget it.

    A Score is documented as able to land between two levels, so its decision is the nearest
    level, never equality with an integer.
    """
    if isinstance(answer, NoulAnswer):
        return answer.noul >= 0.5
    if isinstance(answer, ScoreAnswer):
        return round(answer.score)
    return answer_value(answer)


def answer_distribution(answer: Any) -> dict[str, float]:
    if isinstance(answer, NoulAnswer):
        return {"true": answer.noul}
    return dict(answer.probabilities)


def dump(name: str, records: list[dict[str, Any]]) -> Path:
    RESULTS.mkdir(parents=True, exist_ok=True)
    path = RESULTS / f"{name}-{time.strftime('%Y%m%dT%H%M%S')}.jsonl"
    with path.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False, default=str) + "\n")
    return path


def latency_report(samples: list[float]) -> str:
    ordered = sorted(samples)

    def percentile(q: float) -> float:
        return ordered[min(int(q * len(ordered)), len(ordered) - 1)]

    return (
        f"n={len(ordered)}  mean={statistics.mean(ordered):.0f}ms  "
        f"p50={percentile(0.5):.0f}ms  p95={percentile(0.95):.0f}ms  max={ordered[-1]:.0f}ms"
    )
