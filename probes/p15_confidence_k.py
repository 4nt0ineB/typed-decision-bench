"""P15: does the confidence baseline equal 1/k? The sweep session 13 asked for.

Session 13 identified `confidence` as a rescaling of the top probability:

    confidence = (p_max - c) / (1 - c)

and recovered c exactly at k=3 (0.3333 = 1/3) but measured c = 0.0718 at k=18 where 1/k
would be 0.0556. Two option counts cannot separate "c = 1/k with a small systematic
artifact" from "c is some other function of k", so this sweeps k = 2..12 directly.

Getting a spread of p_max at each k is the trick: a question the model finds easy always
returns p_max = 1.00, and a regression needs variation. So the criteria set is truncated to
the first k MASSIVE scenarios and fed utterances whose true scenario is often outside that
set, which produces genuinely divided distributions across the whole 1/k to 1.0 range.

Roughly 130 calls, about half a cent.
"""

from __future__ import annotations

import argparse
import statistics

import numpy as np
from typesafe_sdk import Choice

from _common import client, dump, latency_report, timed
from p2_multilingual import INSTRUCTIONS_EN, SCENARIOS_EN, load_pairs


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--utterances", type=int, default=12)
    parser.add_argument("--min-k", type=int, default=2)
    parser.add_argument("--max-k", type=int, default=12)
    parser.add_argument("--seed", type=int, default=11, help="not 42: P2's test seed")
    args = parser.parse_args()

    pairs = load_pairs(args.utterances, args.seed)
    options = list(SCENARIOS_EN.items())
    c = client()
    records = []

    for k in range(args.min_k, args.max_k + 1):
        criteria = dict(options[:k])
        question = {"scenario": Choice(instructions=INSTRUCTIONS_EN, criteria=criteria)}
        for pair in pairs:
            call = timed(c, pair["en"], question)
            answer = call.answers["scenario"]
            records.append(call.archive() | {
                "k": k,
                "utterance": pair["en"],
                "gold_in_set": pair["gold"] in criteria,
                "p_max": max(answer.probabilities.values()),
                "confidence": answer.confidence,
                "probabilities": dict(answer.probabilities),
            })

    print(f"\nconfidence = (p_max - c) / (1 - c), c recovered per option count\n")
    print(f"{'k':>3} {'n':>4} {'p_max range':>14} {'slope':>8} {'intercept':>10} "
          f"{'c fitted':>9} {'1/k':>7} {'resid sd':>9} {'MAE vs 1/k':>11}")
    for k in range(args.min_k, args.max_k + 1):
        rows = [r for r in records if r["k"] == k]
        x = np.array([r["p_max"] for r in rows])
        y = np.array([r["confidence"] for r in rows])
        span = f"{x.min():.2f}-{x.max():.2f}"
        if len(set(np.round(x, 2))) < 2:
            print(f"{k:>3} {len(rows):>4} {span:>14}   saturated, no spread to fit")
            continue
        slope, intercept = np.linalg.lstsq(
            np.vstack([x, np.ones_like(x)]).T, y, rcond=None)[0]
        fitted = 1 - 1 / slope
        predicted = (k * x - 1) / (k - 1)
        print(f"{k:>3} {len(rows):>4} {span:>14} {slope:>8.4f} {intercept:>10.4f} "
              f"{fitted:>9.4f} {1 / k:>7.4f} "
              f"{np.std(y - (slope * x + intercept)):>9.4f} "
              f"{np.mean(np.abs(predicted - y)):>11.4f}")

    everything = [((r["k"] * r["p_max"] - 1) / (r["k"] - 1), r["confidence"]) for r in records]
    errors = [abs(p - conf) for p, conf in everything]
    print(f"\n(k*p_max - 1)/(k - 1) over all {len(records)} answers: "
          f"MAE {statistics.mean(errors):.4f}  max {max(errors):.4f}  "
          f"within 0.01 {sum(e <= 0.0101 for e in errors) / len(errors):.1%}")
    print(f"latency: {latency_report([r['elapsed_ms'] for r in records])}")
    print(f"wrote {dump('p15-confidence-k', records)}")


if __name__ == "__main__":
    main()
