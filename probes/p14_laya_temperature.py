"""P14: Laya as its author says to run it, with temperatures fitted.

P12 measured the shipped checkpoints. Their README says not to: "Both checkpoints are
over-confident as shipped", ECE 0.466 -> 0.081 after "refitting one temperature per
(question type, option count) on held-out data", and the multilingual checkpoint ships with
no fitted temperatures at all. P12's ECE of 0.29-0.33 is that admitted 0.466 regime, so
P12's verdict on Laya's confidence gate was measuring an unfitted model. This closes that.

Method, and the shortcut that makes it cheap:

  Temperature scaling is softmax(z/T). The returned probabilities are already
  softmax(z/T0), so rescaling to T' needs no logits and no re-inference: p' proportional to
  p^(T0/T'). One scalar s = T0/T' per checkpoint is fitted on MASSIVE *train* rows, then
  applied offline to P12's archived *test* probabilities.

  Fitting on train and scoring on test is the whole point. A temperature fitted on the
  evaluation set would be reporting the best case of a free parameter, which is the trick
  this project exists to avoid.

Temperature is monotone, so argmax never moves: accuracy is identical before and after by
construction, and only calibration and the gate can change. That is the claim under test.
"""

from __future__ import annotations

import argparse
import json
import statistics
from collections import defaultdict

from _common import RESULTS, dump
from bench.derive_tuned import fit, logs, nll, rescale
from bench.models.laya import router as laya_router
from bench.report import binned
from p2_multilingual import CONDITIONS, ece, load_pairs


def train_ece(rows: list[tuple[dict, str]], s: float) -> float:
    scored = []
    for log_probabilities, gold in rows:
        tuned = rescale(log_probabilities, s)
        chosen = max(tuned, key=tuned.get)
        scored.append({"p_chosen": tuned[chosen], "correct": chosen == gold})
    return ece(scored)


def gate(points: list[tuple[float, bool]], threshold: float) -> tuple[float, float]:
    kept = [correct for p, correct in points if p >= threshold]
    return len(kept) / len(points), (statistics.mean(kept) if kept else float("nan"))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dump", default="p12-laya-20260921T093030",
                        help="archived P12 run to re-temperature; default is the "
                             "shipped-config router run, P12's best accuracy")
    parser.add_argument("--sample", type=int, default=200)
    parser.add_argument("--seed", type=int, default=7, help="not 42: the P12 test seed")
    parser.add_argument("--device", default=None)
    parser.add_argument("--head", type=int, default=192)
    parser.add_argument("--max-len", type=int, default=512)
    args = parser.parse_args()

    import time

    # Train partition, so nothing here overlaps the test rows P12 scored.
    pairs = load_pairs(args.sample, args.seed, "train")

    router = laya_router(args.device, args.head, args.max_len)

    # Fit per checkpoint, because the router picks per request and the two checkpoints are
    # separately trained. English rows exercise one, French rows the other.
    train = defaultdict(list)
    records = []
    for condition in ("EN_EN", "FR_FR"):
        lang, instructions, criteria = CONDITIONS[condition]
        question = {"scenario": {"type": "choice", "instructions": instructions,
                                 "criteria": criteria}}
        for pair in pairs:
            start = time.perf_counter()
            result = router.predict(pair[lang], question)
            answer = result["answers"]["scenario"]
            checkpoint = result["routing"]["model"]
            train[checkpoint].append((logs(answer["probabilities"]), pair["gold"]))
            records.append({
                "split": "train", "condition": condition, "checkpoint": checkpoint,
                "gold": pair["gold"], "predicted": answer["choice"],
                "correct": answer["choice"] == pair["gold"],
                "probabilities": answer["probabilities"],
                "elapsed_ms": (time.perf_counter() - start) * 1000.0,
            })

    print(f"\nfitted on {len(pairs)} MASSIVE train rows per condition, seed {args.seed}")
    print(f"{'checkpoint':<14} {'n':>5} {'s (NLL)':>9} {'s (ECE)':>9} {'NLL':>8} -> {'NLL':>7} "
          f"{'ECE':>8} -> {'ECE':>7}")
    fitted = {}
    for checkpoint, rows in sorted(train.items()):
        by_nll, by_ece = fit(rows), fit(rows, train_ece)
        fitted[checkpoint] = {"nll": by_nll, "ece": by_ece, "shipped": 1.0}
        print(f"{checkpoint:<14} {len(rows):>5} {by_nll:>9.2f} {by_ece:>9.2f} "
              f"{nll(rows, 1.0):>8.3f} -> {nll(rows, by_nll):>7.3f} "
              f"{train_ece(rows, 1.0):>8.3f} -> {train_ece(rows, by_ece):>7.3f}")
    print("  s < 1 softens the distribution (the model was over-confident), s > 1 sharpens")

    archived = [json.loads(line) for line in
                (RESULTS / f"{args.dump}.jsonl").read_text(encoding="utf-8").splitlines()]

    print(f"\napplied to {len(archived)} archived test answers from {args.dump}")
    print("accuracy is identical in every row: temperature is monotone, argmax cannot move\n")
    print(f"{'condition':<8} {'accuracy':>9} {'tuning':<9} {'ECE':>7} {'cover>=.85':>11} "
          f"{'acc|gated':>10} {'cover @ 90% acc':>16}")
    for condition in CONDITIONS:
        rows = [r for r in archived if r["condition"] == condition]
        if not rows:
            continue
        accuracy = statistics.mean(r["correct"] for r in rows)
        for tuning in ("shipped", "nll", "ece"):
            scored = []
            for record in rows:
                probabilities = record["raw"]["answers"]["scenario"]["probabilities"]
                s = fitted.get(record["checkpoint"], {}).get(tuning, 1.0)
                tuned = rescale(logs(probabilities), s)
                chosen = max(tuned, key=tuned.get)
                assert chosen == record["predicted"], "temperature moved the argmax"
                scored.append({"p_chosen": tuned[chosen], "correct": record["correct"]})
                if tuning == "nll":
                    record["p_tuned"] = tuned[chosen]
                    records.append(record | {"split": "test"})
            points = [(r["p_chosen"], r["correct"]) for r in scored]
            cover, acc = gate(points, 0.85)
            # The number a product actually needs: how much traffic can be auto-approved
            # at a quality bar, rather than at an arbitrary threshold.
            usable = max((gate(points, t / 100)[0] for t in range(1, 100)
                          if gate(points, t / 100)[1] >= 0.90), default=0.0)
            print(f"{condition if tuning == 'shipped' else '':<8} "
                  f"{accuracy if tuning == 'shipped' else float('nan'):>8.1%} "
                  f"{tuning:<9} {ece(scored):>7.3f} {cover:>10.0%} {acc:>10.0%} "
                  f"{usable:>15.0%}")

    pooled_before = [(r["p_chosen"], r["correct"]) for r in archived]
    pooled_after = [(r["p_tuned"], r["correct"]) for r in archived if "p_tuned" in r]
    print("\npooled reliability (mean p, accuracy, n)")
    for label, points in (("shipped", pooled_before), ("tuned", pooled_after)):
        bins = "  ".join(f"{mp:.2f}/{acc:.0%}/{n}" for mp, acc, n in binned(points))
        print(f"  {label:<8} {bins}")

    print(f"\nwrote {dump('p14-laya-temperature', records)}")


if __name__ == "__main__":
    main()
