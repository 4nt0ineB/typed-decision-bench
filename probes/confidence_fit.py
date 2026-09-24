"""What is `confidence`? Open since session 6, fitted here from archived responses.

The documented patterns all gate on `confidence`, and the docs never say what it is. P0
found it is not max-probability: on {billing 0.68, technical 0.32, other 0.00} it returned
0.52, where normalised entropy gives 0.43, Gini 0.35 and top-two margin 0.36.

Method, in order, because each step decides whether the next one is meaningful:

  1. Is confidence a function of the returned distribution at all? Group answers whose
     rounded probabilities are identical and look at the spread of their confidence. If
     identical distributions return different confidences, no formula over those
     probabilities can be exact and everything below is an approximation by construction.
  2. Score the closed-form candidates, per question type and per option count.
  3. Fit the one-parameter families (Renyi order, entropy exponent) by grid search.

Laya is the control. Its source computes `1 - H(p)/log(k)` and nothing else, so the same
pipeline run over P12/P13 must recover normalised entropy at ~zero error. If it does not,
the harness is wrong and the Jev verdict means nothing. Session 11's rule: three times in
one day the finding was the harness.

Zero API calls. Everything comes from results/probes/*.jsonl.
"""

from __future__ import annotations

import json
import math
from collections import defaultdict

import numpy as np

from _common import RESULTS

# Rounding is itself a finding (session 6): probabilities come back at 2 decimals, so a
# formula evaluated on them inherits that quantisation and cannot be checked below it.
QUANTUM = 0.01


def answers(prefixes: list[str], laya: bool) -> list[dict]:
    """(probabilities, confidence, type) triples out of every archived raw response."""
    out = []
    for prefix in prefixes:
        for path in sorted(RESULTS.glob(f"{prefix}-*.jsonl")):
            for line in path.read_text(encoding="utf-8").splitlines():
                raw = json.loads(line).get("raw") or {}
                for answer in (raw.get("answers") or {}).values():
                    if not isinstance(answer, dict) or "probabilities" not in answer:
                        continue
                    if answer.get("confidence") is None:
                        continue
                    probabilities = [float(v) for v in answer["probabilities"].values()]
                    out.append({
                        "p": np.array(probabilities),
                        "k": len(probabilities),
                        "type": answer.get("type", "choice"),
                        "confidence": float(answer["confidence"]),
                        "source": "laya" if laya else "jev",
                    })
    return out


def entropy(p: np.ndarray) -> float:
    q = np.clip(p, 1e-12, 1.0)
    return float(-(q * np.log(q)).sum())


def renyi(p: np.ndarray, alpha: float) -> float:
    q = np.clip(p, 1e-12, 1.0)
    if abs(alpha - 1.0) < 1e-9:
        return entropy(q)
    return float(np.log((q ** alpha).sum()) / (1.0 - alpha))


CANDIDATES = {
    "max p": lambda p, k: float(p.max()),
    "normalised max": lambda p, k: (k * float(p.max()) - 1) / (k - 1),
    "top-two margin": lambda p, k: float(np.diff(np.sort(p)[-2:])[0]),
    "1 - H/log k": lambda p, k: 1.0 - entropy(p) / math.log(k),
    "collision sum p^2": lambda p, k: float((p ** 2).sum()),
    "normalised Gini": lambda p, k: (k * float((p ** 2).sum()) - 1) / (k - 1),
}


def score(rows: list[dict], predict) -> tuple[float, float]:
    errors = [abs(predict(r["p"], r["k"]) - r["confidence"]) for r in rows]
    return float(np.mean(errors)), float(np.max(errors))


def functional_check(rows: list[dict]) -> None:
    """Same rounded distribution, same confidence? Bounds what any formula can achieve."""
    groups = defaultdict(list)
    for r in rows:
        groups[(r["type"], tuple(np.round(r["p"], 2)))].append(r["confidence"])
    repeated = {k: v for k, v in groups.items() if len(v) > 1}
    if not repeated:
        print("  no distribution repeats; cannot test functional dependence")
        return
    spreads = [max(v) - min(v) for v in repeated.values()]
    worst = max(repeated.items(), key=lambda kv: max(kv[1]) - min(kv[1]))
    print(f"  {len(repeated)} distributions seen more than once, "
          f"max confidence spread {max(spreads):.3f}, mean {np.mean(spreads):.4f}")
    # Both p and confidence are rounded to 0.01, so a formula evaluated on the unrounded
    # distribution can legitimately land 2 quanta apart on two identical rounded ones.
    if max(spreads) > 2 * QUANTUM:
        print(f"    worst: k={len(worst[0][1])} {sorted(set(worst[1]))}")
        print("    -> confidence is NOT a function of the rounded probabilities alone")
    else:
        print("    -> consistent within rounding, so a formula over p is possible")


def verdict(rows: list[dict]) -> None:
    """The winner, checked the way the number will be used rather than by mean error."""
    label, predict = min(CANDIDATES.items(), key=lambda kv: score(rows, kv[1])[0])
    errors = np.array([abs(predict(r["p"], r["k"]) - r["confidence"]) for r in rows])
    print(f"\n  best fit: {label}")
    for tolerance in (QUANTUM, 2 * QUANTUM):
        print(f"    within {tolerance:.2f}: {(errors <= tolerance + 1e-9).mean():>6.1%} "
              f"of {len(errors)} answers")
    worst = rows[int(errors.argmax())]
    print(f"    worst residual {errors.max():.4f} at k={worst['k']} "
          f"p_max={worst['p'].max():.2f} said {worst['confidence']:.2f}")

    # If confidence is (p_max - c) / (1 - c), regressing it on p_max recovers c. Reported
    # per k because the two option counts disagree about whether c is exactly 1/k.
    for k in sorted({r["k"] for r in rows}):
        sub = [r for r in rows if r["k"] == k]
        x = np.array([float(r["p"].max()) for r in sub])
        if len(set(np.round(x, 2))) < 2:
            continue
        slope, intercept = np.linalg.lstsq(
            np.vstack([x, np.ones_like(x)]).T,
            np.array([r["confidence"] for r in sub]), rcond=None)[0]
        residual = np.std([r["confidence"] for r in sub] - (slope * x + intercept))
        print(f"    k={k:<3} conf = {slope:.4f}*p_max {intercept:+.4f}  "
              f"-> baseline c={1 - 1 / slope:.4f} (1/k={1 / k:.4f})  resid sd {residual:.4f}")
    # Session 6's unexplained example, which is what sent this question to the journal.
    example = np.array([0.68, 0.32, 0.0])
    print(f"    session 6 case {{0.68, 0.32, 0.00}}: predicts "
          f"{predict(example, 3):.2f}, observed 0.52")


def report(name: str, rows: list[dict]) -> None:
    print(f"\n{name}: n={len(rows)}")
    by_shape = defaultdict(list)
    for r in rows:
        by_shape[(r["type"], r["k"])].append(r)
    print(f"  shapes: {', '.join(f'{t} k={k} (n={len(v)})' for (t, k), v in sorted(by_shape.items()))}")

    print("\n  functional dependence")
    functional_check(rows)

    print(f"\n  {'candidate':<20} {'MAE':>8} {'max err':>9}   per shape")
    for label, predict in CANDIDATES.items():
        mae, worst = score(rows, predict)
        per_shape = "  ".join(
            f"{t[0]}{k}:{score(v, predict)[0]:.3f}" for (t, k), v in sorted(by_shape.items())
        )
        print(f"  {label:<20} {mae:>8.4f} {worst:>9.4f}   {per_shape}")

    grid = np.arange(0.05, 5.01, 0.05)
    best_alpha = min(grid, key=lambda a: score(rows, lambda p, k, a=a: 1.0 - renyi(p, a) / math.log(k))[0])
    mae_alpha, max_alpha = score(rows, lambda p, k: 1.0 - renyi(p, best_alpha) / math.log(k))
    best_gamma = min(grid, key=lambda g: score(rows, lambda p, k, g=g: (1.0 - entropy(p) / math.log(k)) ** g)[0])
    mae_gamma, max_gamma = score(rows, lambda p, k: (1.0 - entropy(p) / math.log(k)) ** best_gamma)
    print(f"\n  fitted 1 - H_alpha/log k     alpha={best_alpha:.2f}  "
          f"MAE {mae_alpha:.4f}  max {max_alpha:.4f}")
    print(f"  fitted (1 - H/log k)^gamma   gamma={best_gamma:.2f}  "
          f"MAE {mae_gamma:.4f}  max {max_gamma:.4f}")
    verdict(rows)


def main() -> None:
    laya = answers(["p12-laya", "p13-laya-freetext"], laya=True)
    report("laya 0.3.4 (CONTROL: source says 1 - H/log k)", laya)
    report("jev-1.13.0", answers(["p0-smoke", "p1-determinism", "p2-multilingual"], laya=False))


if __name__ == "__main__":
    main()
