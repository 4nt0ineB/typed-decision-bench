"""P2: does Jev degrade in French?

Undocumented territory. TypeSafe states no multilingual claim either way, and the token
budget is specified in "English characters", which suggests tokenization was benchmarked
on English.

The task itself (dataset, criteria, conditions) lives in bench/tasks/massive_scenario.py,
shared with every model the bench compares. This file is the historical Jev runner.
"""

from __future__ import annotations

import argparse
import statistics
from collections import defaultdict

from typesafe_sdk import Choice

from _common import client, dump, latency_report, timed
from bench.tasks.massive_scenario import (  # noqa: F401  re-exported for P12-P17
    CONDITIONS,
    DATA,
    INSTRUCTIONS_EN,
    INSTRUCTIONS_FR,
    SCENARIOS_EN,
    SCENARIOS_EN_V1,
    SCENARIOS_FR,
    ece,
    load_pairs,
    read_split,
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--sample", type=int, default=30)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    pairs = load_pairs(args.sample, args.seed)
    c = client()
    records = []

    for name, (lang, instructions, criteria) in CONDITIONS.items():
        question = {"scenario": Choice(instructions=instructions, criteria=criteria)}
        for pair in pairs:
            call = timed(c, pair[lang], question)
            answer = call.answers["scenario"]
            records.append(
                call.archive()
                | {
                    "condition": name,
                    "id": pair["id"],
                    "utterance": pair[lang],
                    "gold": pair["gold"],
                    "predicted": answer.choice,
                    "correct": answer.choice == pair["gold"],
                    "p_chosen": answer.probabilities[answer.choice],
                    "confidence": answer.confidence,
                    "input_tokens": call.input_tokens,
                }
            )

    by_condition = defaultdict(list)
    for record in records:
        by_condition[record["condition"]].append(record)

    print(f"\nMASSIVE scenario classification, 18 classes, n={len(pairs)} parallel utterances\n")
    print(f"{'condition':<8} {'accuracy':>9} {'ECE':>7} {'mean conf':>10} {'conf|right':>11} "
          f"{'conf|wrong':>11} {'tokens':>8}")
    for name in CONDITIONS:
        rows = by_condition[name]
        right = [r["confidence"] for r in rows if r["correct"]]
        wrong = [r["confidence"] for r in rows if not r["correct"]]
        print(
            f"{name:<8} "
            f"{sum(r['correct'] for r in rows) / len(rows):>8.1%} "
            f"{ece(rows):>7.3f} "
            f"{statistics.mean(r['confidence'] for r in rows):>10.3f} "
            f"{statistics.mean(right) if right else float('nan'):>11.3f} "
            f"{statistics.mean(wrong) if wrong else float('nan'):>11.3f} "
            f"{statistics.mean(r['input_tokens'] for r in rows):>8.0f}"
        )

    en_tokens = statistics.mean(r["input_tokens"] for r in by_condition["EN_EN"])
    fr_tokens = statistics.mean(r["input_tokens"] for r in by_condition["FR_FR"])
    print(f"\nFR/EN token ratio: {fr_tokens / en_tokens:.2f}x")

    en_pred = {r["id"]: r["predicted"] for r in by_condition["EN_EN"]}
    agree = sum(1 for r in by_condition["FR_EN"] if en_pred[r["id"]] == r["predicted"])
    print(f"EN_EN vs FR_EN agreement: {agree / len(pairs):.1%}")

    print(f"\nlatency: {latency_report([r['elapsed_ms'] for r in records])}")

    path = dump("p2-multilingual", records)
    print(f"wrote {path}")


if __name__ == "__main__":
    main()
