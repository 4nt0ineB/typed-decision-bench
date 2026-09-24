"""P17: a small local LLM with constrained decoding, on P2's exact benchmark.

The row the comparison table kept missing. Laya (P12) is an encoder, Jev is a hosted
service, and neither answers the question a team actually asks first: what does a 2-4B
open-weight LLM running on the laptop give me for free?

MiniCPM5-2B (OpenBMB, Apache 2.0) and Qwen3.5-4B (Alibaba, Apache 2.0), held to Jev's
output contract by constrained next-token scoring. The contract, and what it does and does
not show, is documented in bench/models/hf_logits.py, which this probe shares with the bench.
"""

from __future__ import annotations

import argparse
import statistics
import time
from collections import defaultdict

from _common import dump, latency_report
from bench.models.hf_logits import build_prompt, label_token_ids, load_model
from p12_laya import jev_baseline, summarise
from p2_multilingual import CONDITIONS, ece, load_pairs


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True, help="hugging face id")
    parser.add_argument("--sample", type=int, default=100)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="mps")
    parser.add_argument("--dtype", default="bfloat16")
    args = parser.parse_args()

    import torch

    tok, model = load_model(args.model, args.device, args.dtype)
    print(f"{args.model}: {model.config.architectures}, {sum(p.numel() for p in model.parameters()) / 1e9:.2f}B params on {args.device}")

    pairs = load_pairs(args.sample, args.seed)
    records = []

    for name, (lang, instructions, criteria) in CONDITIONS.items():
        keys = list(criteria)
        ids = label_token_ids(tok, len(keys))
        for pair in pairs:
            prompt = build_prompt(tok, instructions, criteria, pair[lang])
            inputs = tok(prompt, return_tensors="pt").to(args.device)

            start = time.perf_counter()
            with torch.inference_mode():
                logits = model(**inputs).logits[0, -1]
            probs = torch.softmax(logits[ids].float(), dim=-1).tolist()
            elapsed_ms = (time.perf_counter() - start) * 1000.0

            distribution = dict(zip(keys, probs))
            predicted = max(distribution, key=distribution.get)
            p_chosen = distribution[predicted]
            k = len(keys)
            records.append({
                "model": args.model,
                "elapsed_ms": elapsed_ms,
                "condition": name,
                "id": pair["id"],
                "utterance": pair[lang],
                "gold": pair["gold"],
                "predicted": predicted,
                "correct": predicted == pair["gold"],
                "p_chosen": p_chosen,
                # P15's formula, so the column means the same thing as Jev's.
                "confidence": (k * p_chosen - 1) / (k - 1),
                "input_tokens": int(inputs["input_ids"].shape[1]),
                "probabilities": distribution,
            })

    print(f"\nMASSIVE scenario classification, 18 classes, n={len(pairs)} parallel utterances")
    summarise(f"{args.model}, constrained to {len(keys)} label tokens, one forward pass", records)

    baseline = jev_baseline()
    if baseline:
        summarise("jev-1.13.0, P2 archive (hosted API, network latency included)", baseline)

    by_condition = defaultdict(list)
    for record in records:
        by_condition[record["condition"]].append(record)
    en_tokens = statistics.mean(r["input_tokens"] for r in by_condition["EN_EN"])
    fr_tokens = statistics.mean(r["input_tokens"] for r in by_condition["FR_FR"])
    print(f"\nFR/EN token ratio: {fr_tokens / en_tokens:.2f}x")

    print(f"latency: {latency_report([r['elapsed_ms'] for r in records])}")

    slug = args.model.split("/")[-1].lower()
    print(f"wrote {dump(f'p17-{slug}', records)}")


if __name__ == "__main__":
    main()
