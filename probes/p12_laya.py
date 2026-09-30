"""P12: the open-source challenger, on P2's exact benchmark.

Laya (convaiinnovations, Apache 2.0) copies Jev's interface: same three primitives, same
state + typed questions in one forward pass, same RLCD training story. It is an encoder
(ModernBERT-large 421M for English, mmBERT-base 322M for 100+ languages) with a router
that picks a checkpoint per request, so it is self-hostable and free to run.

Runs the same 18-scenario MASSIVE classification as P2, same 4 conditions, same sample and
seed, same criteria strings, and replays P2's archived Jev records side by side. Nothing
here is re-measured for Jev; the comparison is against the run in results/probes/.

Hardware disclaimer: Jev's numbers are a hosted API called from Europe, Laya's are local.
Accuracy and calibration are comparable; the latency columns are not measuring the same
thing and the vendor's own T4 figures (33ms) are not reproducible on a laptop.
"""

from __future__ import annotations

import argparse
import json
import statistics
import time
from collections import defaultdict

from _common import RESULTS, dump, latency_report
from bench.models.laya import router as laya_router
from p2_multilingual import CONDITIONS, ece, load_pairs


def summarise(title: str, records: list[dict], extra: str = "") -> None:
    by_condition = defaultdict(list)
    for record in records:
        by_condition[record["condition"]].append(record)

    print(f"\n{title}")
    print(f"{'condition':<8} {'accuracy':>9} {'ECE':>7} {'mean conf':>10} {'conf|right':>11} "
          f"{'conf|wrong':>11} {'tokens':>8} {'ms':>7} {extra}")
    for name in CONDITIONS:
        rows = by_condition.get(name)
        if not rows:
            continue
        right = [r["confidence"] for r in rows if r["correct"]]
        wrong = [r["confidence"] for r in rows if not r["correct"]]
        tail = ""
        if extra:
            picked = {r.get("checkpoint") for r in rows}
            tail = "/".join(sorted(p for p in picked if p))
        print(
            f"{name:<8} "
            f"{sum(r['correct'] for r in rows) / len(rows):>8.1%} "
            f"{ece(rows):>7.3f} "
            f"{statistics.mean(r['confidence'] for r in rows):>10.3f} "
            f"{statistics.mean(right) if right else float('nan'):>11.3f} "
            f"{statistics.mean(wrong) if wrong else float('nan'):>11.3f} "
            f"{statistics.mean(r['input_tokens'] for r in rows):>8.0f} "
            f"{statistics.median(r['elapsed_ms'] for r in rows):>7.0f} {tail}"
        )


def jev_baseline() -> list[dict]:
    """P2's archived Jev records, newest run, so the comparison is not retyped by hand."""
    dumps = sorted(RESULTS.glob("p2-multilingual-*.jsonl"))
    if not dumps:
        return []
    return [json.loads(line) for line in dumps[-1].read_text(encoding="utf-8").splitlines()]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--sample", type=int, default=100)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default=None, help="mps, cpu, cuda. default: laya picks")
    parser.add_argument("--model", default="router",
                        help="router, english, multilingual, typed-decisions")
    # 18 options of full-sentence criteria need ~330 tokens. At the shipped head_max_len=192
    # every option is silently truncated to 9 tokens, which is not the criteria P2 measured.
    parser.add_argument("--head", type=int, default=448)
    parser.add_argument("--max-len", type=int, default=1024)
    args = parser.parse_args()

    pairs = load_pairs(args.sample, args.seed)
    router = laya_router(args.device, args.head, args.max_len)
    forced = None if args.model == "router" else args.model

    records = []
    for name, (lang, instructions, criteria) in CONDITIONS.items():
        question = {"scenario": {"type": "choice", "instructions": instructions,
                                 "criteria": criteria}}
        for pair in pairs:
            start = time.perf_counter()
            result = router.predict(pair[lang], question, model=forced)
            elapsed_ms = (time.perf_counter() - start) * 1000.0
            answer = result["answers"]["scenario"]
            records.append({
                "model": f"laya:{result['routing']['model']}",
                "checkpoint": result["routing"]["model"],
                "elapsed_ms": elapsed_ms,
                "condition": name,
                "id": pair["id"],
                "utterance": pair[lang],
                "gold": pair["gold"],
                "predicted": answer["choice"],
                "correct": answer["choice"] == pair["gold"],
                "p_chosen": answer["probabilities"][answer["choice"]],
                "confidence": answer["confidence"],
                "input_tokens": result["usage"]["input_tokens"],
                "raw": result,
            })

    print(f"\nMASSIVE scenario classification, 18 classes, n={len(pairs)} parallel utterances")
    summarise(
        f"laya {args.model} on {router.load('english').device}, head_max_len={args.head}",
        records,
        extra="checkpoint",
    )

    baseline = jev_baseline()
    if baseline:
        summarise("jev-1.13.0, P2 archive (hosted API, network latency included)", baseline)

    print(f"\nlaya latency: {latency_report([r['elapsed_ms'] for r in records])}")

    print(f"wrote {dump('p12-laya', records)}")


if __name__ == "__main__":
    main()
