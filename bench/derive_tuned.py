"""<model>-tuned: a model with fitted temperatures, derived offline from its stored runs.

    .venv/bin/python bench/run.py --model laya-router --task massive_scenario \
        --split dev --sample 200 --seed 7
    .venv/bin/python bench/derive_tuned.py --model laya-router --task massive_scenario

Not a run, and no model call. The exponent s (p' ~ p^s, P14) is fitted by NLL on the stored
`dev` run, then applied to the stored test run. Dev, not train or test: criteria design and
temperature fitting both belong on the split nothing is scored on. One exponent per Laya
checkpoint when rows name one, a single exponent otherwise. Temperature is monotone, so
accuracy equals the source's by construction; confidence is Jev's formula over the tuned
distribution, which for Open-Jev is already its own.
"""

from __future__ import annotations

import argparse
import json
import math
import statistics
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from bench.run import BENCH, git_sha, row, write_manifest  # noqa: E402

# P14's fitting conditions: EN_EN and FR_FR send rows to both Laya checkpoints.
FIT_CONDITIONS = ("EN_EN", "FR_FR")


def logs(probabilities: dict[str, float]) -> dict[str, float]:
    """Log-probabilities from rounded ones. Rows that round to 1.0 lose their losers' order
    here, which is why the bench stores Laya's log-probabilities instead."""
    return {k: math.log(max(v, 1e-12)) for k, v in probabilities.items()}


def rescale(log_probabilities: dict[str, float], s: float) -> dict[str, float]:
    """p' proportional to p^s, the temperature change expressed on probabilities. Taken from
    log-probabilities so no loser underflows or clamps to a shared floor, which would tie
    every saturated row to one tuned distribution."""
    top = max(log_probabilities.values())
    powered = {k: math.exp(s * (v - top)) for k, v in log_probabilities.items()}
    total = sum(powered.values())
    return {k: v / total for k, v in powered.items()}


def nll(rows: list[tuple[dict, str]], s: float) -> float:
    return -statistics.mean(
        math.log(max(rescale(log_probabilities, s).get(gold, 1e-12), 1e-12))
        for log_probabilities, gold in rows
    )


def fit(rows: list[tuple[dict, str]], loss=None) -> float:
    """Grid search on the exponent, by NLL unless `loss` says otherwise. One scalar per
    call: a caller wanting one per checkpoint or per option count calls it once for each."""
    # Wide on purpose: the first version stopped at 0.20 and the English checkpoint pinned
    # to the floor, which reports the boundary rather than the optimum.
    grid = np.arange(0.01, 4.01, 0.01)
    loss = loss or nll
    best = float(min(grid, key=lambda s: loss(rows, s)))
    if best <= grid[1] or best >= grid[-2]:
        print(f"  WARNING: exponent {best:.2f} is at the edge of the search grid")
    return best


def read(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]


def log_probabilities(r: dict) -> dict[str, float]:
    """A probability rounded to 0.0 has lost its order against the other losers, and the
    rescaled distribution of every such row collapses onto one. Refuse rather than rank on it."""
    if "log_probabilities" in r:
        return r["log_probabilities"]
    assert min(r["probabilities"].values()) > 0, (
        f"row {r['id']}: a probability rounded to 0, store log-probabilities to tune it")
    return logs(r["probabilities"])


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--task", required=True)
    args = parser.parse_args()

    dev = BENCH / args.model / f"{args.task}.dev.jsonl"
    fitting = defaultdict(list)
    for r in read(dev):
        if r["condition"] in FIT_CONDITIONS:
            fitting[r.get("checkpoint", "all")].append((log_probabilities(r), r["gold"]))
    exponents = {checkpoint: fit(rows) for checkpoint, rows in sorted(fitting.items())}
    print(f"exponents fitted on dev: {exponents}")

    source = BENCH / args.model / f"{args.task}.jsonl"
    rows = []
    for r in read(source):
        tuned = rescale(log_probabilities(r), exponents[r.get("checkpoint", "all")])
        assert max(tuned, key=tuned.get) == r["predicted"], "temperature moved the argmax"
        rows.append(row(args.task, r["condition"], r["id"], r["state"], r["gold"],
                        r | {"probabilities": tuned, "confidence": None}, r["elapsed_ms"]))

    dev_manifest = json.loads(dev.with_suffix(".json").read_text(encoding="utf-8"))
    target = BENCH / f"{args.model}-tuned" / f"{args.task}.jsonl"
    target.parent.mkdir(exist_ok=True)
    target.write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows),
                      encoding="utf-8")
    root = BENCH.parent.parent
    write_manifest(target, {
        "model": target.parent.name, "task": args.task, "n": len(rows), "derived": True,
        "source": str(source.relative_to(root)), "temperature_exponents": exponents,
        "fit": {"source": str(dev.relative_to(root)), "split": "dev",
                "sample": dev_manifest["sample"], "seed": dev_manifest["seed"],
                "objective": "nll", "conditions": FIT_CONDITIONS,
                "rows_per_checkpoint": {c: len(v) for c, v in fitting.items()}},
        "git": git_sha(),
    })
    print(f"wrote {target}")


if __name__ == "__main__":
    main()
