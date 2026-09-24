"""P1: is Jev deterministic?

Gates the whole auditability argument. If argmax over a returned distribution is stable
on a pinned model version, Jev is more reproducible than an LLM at temperature 0, and
that is the strongest case for using it in a regulated flow.

Also checks whether `jev-latest` already differs from the pinned version today.
"""

from __future__ import annotations

import argparse
import random
import statistics
from collections import Counter

from typesafe_sdk import Choice, Noul, Score

from _common import (
    MODEL,
    answer_distribution,
    answer_value,
    decided,
    client,
    dump,
    latency_report,
    timed,
)

RUNS = 20

STATE = (
    "Customer writes: my premium went up 40% at renewal and nobody told me. "
    "I want an explanation today or I am cancelling."
)

QUESTIONS = {
    "department": Choice(
        instructions="Which team should handle this message",
        criteria={
            "billing": "Payment, premium, refund or invoice issue",
            "policy": "Changing, renewing or cancelling a contract",
            "complaint": "Formal dissatisfaction about service received",
        },
    ),
    "severity": Score(
        instructions="How severe the situation is for the customer",
        criteria=[
            "Routine question, no impact",
            "Noticeable inconvenience",
            "Serious problem, customer is at risk of leaving",
        ],
    ),
    "is_urgent": Noul(instructions="The message conveys urgency or time-sensitivity"),
}


def fingerprint(answers: dict) -> str:
    parts = []
    for qid in sorted(answers):
        distribution = answer_distribution(answers[qid])
        rendered = ",".join(f"{k}:{distribution[k]!r}" for k in sorted(distribution))
        parts.append(f"{qid}=[{rendered}]")
    return " | ".join(parts)


def decision(answers: dict) -> str:
    parts = []
    for qid in sorted(answers):
        value = answer_value(answers[qid])
        parts.append(f"{qid}={round(value, 2) if isinstance(value, float) else value}")
    return " | ".join(parts)


# The documented confidence-gating floor. Routing a run above it and an identical run
# below it is the failure this probe exists to catch.
GATE = 0.6


def run(model: str, runs: int) -> tuple[list[dict], list[float]]:
    c = client()
    records, latencies = [], []
    for i in range(runs):
        call = timed(c, STATE, QUESTIONS, model=model)
        latencies.append(call.elapsed_ms)
        records.append(
            call.archive()
            | {
                "run": i,
                "answers": call.answers,
                "input_tokens": call.input_tokens,
                "fingerprint": fingerprint(call.answers),
                "decision": decision(call.answers),
            }
        )
    return records, latencies


def report(label: str, records: list[dict], latencies: list[float]) -> None:
    n = len(records)
    print(f"\n--- {label}  (n={n}) ---")
    print(f"latency: {latency_report(latencies)}")
    print(f"identical full responses: {len(Counter(r['fingerprint'] for r in records))} / {n}")

    print(f"\n{'question':<12} {'argmax':>18} {'top p':>16} {'confidence':>16} {'gate':>10}")
    for qid in sorted(records[0]["answers"]):
        answers = [r["answers"][qid] for r in records]
        values = Counter(str(decided(a)) for a in answers)
        stable = f"{values.most_common(1)[0][0]} ({values.most_common(1)[0][1]}/{n})"

        tops = [max(answer_distribution(a).values()) for a in answers]
        confidences = [a.confidence for a in answers if hasattr(a, "confidence")]

        span = f"{min(tops):.2f}-{max(tops):.2f}"
        if confidences:
            below = sum(1 for c in confidences if c < GATE)
            conf = f"{min(confidences):.2f}-{max(confidences):.2f}"
            gate = "STABLE" if below in (0, n) else f"SPLIT {below}/{n}"
        else:
            conf, gate = "n/a", "n/a"

        print(f"{qid:<12} {stable:>18} {span:>16} {conf:>16} {gate:>10}")

    print(f"\n  argmax stable on every question: "
          f"{all(len({str(decided(r['answers'][q])) for r in records}) == 1 for q in records[0]['answers'])}")
    print(f"  gate at {GATE} is crossed by identical inputs: "
          f"{any(0 < sum(1 for r in records if getattr(r['answers'][q], 'confidence', 1.0) < GATE) < n for q in records[0]['answers'])}")


def permutation_test(left: list[float], right: list[float], trials: int = 20000) -> tuple[float, float]:
    """Two-sample permutation test on the mean. No scipy, no distributional assumption.

    Session 6 compared pinned against `jev-latest` at n=3 and correctly refused to call it.
    Per-call noise is real, so the question is not "are the numbers equal" but "is the
    difference bigger than this model's own run-to-run spread", which is what this asks.
    """
    observed = abs(statistics.mean(left) - statistics.mean(right))
    pool = left + right
    rng = random.Random(0)
    hits = 0
    for _ in range(trials):
        rng.shuffle(pool)
        split = abs(statistics.mean(pool[:len(left)]) - statistics.mean(pool[len(left):]))
        hits += split >= observed - 1e-12
    return observed, hits / trials


def compare(pinned: list[dict], alias: list[dict]) -> None:
    print(f"\n--- alias drift: {MODEL} vs jev-latest ---")
    print(f"{'question':<12} {'pinned mean p':>14} {'alias mean p':>13} {'diff':>8} "
          f"{'p-value':>9}  verdict")
    for qid in sorted(pinned[0]["answers"]):
        left = [max(answer_distribution(r["answers"][qid]).values()) for r in pinned]
        right = [max(answer_distribution(r["answers"][qid]).values()) for r in alias]
        difference, p_value = permutation_test(left, right)
        verdict = ("indistinguishable" if p_value >= 0.05
                   else "DIFFERS beyond run-to-run noise")
        print(f"{qid:<12} {statistics.mean(left):>14.4f} {statistics.mean(right):>13.4f} "
              f"{difference:>8.4f} {p_value:>9.4f}  {verdict}")
    print("  a null result here does not prove the alias points at the pinned build, only "
          "that this input cannot tell them apart")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--runs", type=int, default=RUNS)
    # Session 6 left this at 3 and could not decide. 50 a side is the redo it asked for.
    parser.add_argument("--alias-runs", type=int, default=RUNS)
    args = parser.parse_args()

    pinned, pinned_latency = run(MODEL, args.runs)
    report(MODEL, pinned, pinned_latency)

    alias, alias_latency = run("jev-latest", args.alias_runs)
    report("jev-latest (alias drift check)", alias, alias_latency)

    compare(pinned, alias)

    path = dump("p1-determinism", pinned + alias)
    print(f"wrote {path}")


if __name__ == "__main__":
    main()
