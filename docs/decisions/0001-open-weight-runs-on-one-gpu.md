# 0001: Run every open-weight model on one rented GPU

Date: 2026-09-24. Status: done on branch `modal-parity`.

## Context

The open-weight rows were measured on three different setups:

| Model | Where | n per condition | Tasks |
|---|---|---|---|
| `open-jev-2b` | Modal L4 | 500 | both |
| `open-jev-9b` | Modal A100-80GB | 500 | both |
| `qwen3.5-4b`, `minicpm5-2b` | M3 laptop, bf16 | 100 | `massive_scenario` only |
| `qwen3.5-0.8b-mlx-4bit` | M3 laptop, MLX 4-bit | 500 | both |

Three problems follow. The latency column compares an M3 with an L4 and an A100. Qwen 4B
and MiniCPM 2B sit at n=100 on one task because sustained bf16 runs overheat the laptop.
And no row isolates what Open-Jev adds: Open-Jev is a LoRA adapter plus a decision head on
a frozen Qwen3.5 base, but the bench never scores that base on its own.

## Decision

1. Every open-weight model that runs through `transformers` runs on Modal, on the same GPU
   (A100-80GB, which the 9B already needs), in bf16, at n=500 per condition on both tasks.
2. Add the two frozen bases as plain zero-shot LLMs: `Qwen/Qwen3.5-2B` and
   `Qwen/Qwen3.5-9B`, pinned to the revisions Open-Jev loads. Each Open-Jev row then sits
   next to its own base on identical hardware, so the gap between them is what the Open-Jev
   fine-tune adds.
3. Move `qwen3.5-4b` and `minicpm5-2b` to Modal at full size. Re-run `open-jev-2b` on the
   A100 so its latency matches the others; its accuracy is not expected to move.
4. Keep `qwen3.5-0.8b-mlx-4bit` in the table and drop it from the charts. It answers a
   different question: can a solo developer try an idea on a laptop? Yes, but the machine
   runs hot and the model is not accurate enough to compete with 9B rows. The same
   feasibility note covers Laya: it runs well on a Mac but does not suit a long queue of
   requests with large contexts.

Hosted rows (Jev, Laya, OpenRouter) stay as they are: their hardware is not ours to choose,
and their latency includes the network from France.

## Consequences

- Accuracy, Brier score and calibration of the moved models should barely change (bf16 on
  MPS versus CUDA). If a moved row shifts beyond its confidence interval, that is a finding
  to investigate, not noise.
- The latency column becomes comparable across all open-weight rows. It remains a
  single-request forward pass without prefix caching, not a served deployment.
- Modal credit: about $25 left. The first full run measures the real cost per pass before
  the rest are launched.

## Results

All runs on Modal A100-80GB, n=500 per condition, $1.61 in total.

| model | intent acc | intent ECE | scenario acc | scenario ECE |
|---|---|---|---|---|
| qwen3.5-9b | 75.1% | 0.070 | 83.0% | 0.021 |
| open-jev-9b | 70.7% | 0.246 | 78.5% | 0.193 |
| qwen3.5-4b | 74.5% | 0.029 | 80.4% | 0.019 |
| qwen3.5-2b | 19.8% \* | 0.137 | 61.8% | 0.102 |
| open-jev-2b | 66.4% | 0.443 | 68.0% | 0.345 |
| minicpm5-2b | 30.4% \* | 0.232 | 49.5% | 0.131 |

\* Understated by the answer-label format past 26 options; see
[0002](0002-answer-labels-past-26-options.md).

- At 9B the frozen base beats Open-Jev on both tasks, on accuracy and far more on
  calibration, at a quarter of the input tokens. Open-Jev 9B only leads on coverage at the
  0.90 gate once its temperature is fitted on labeled dev data.
- At 2B Open-Jev leads on accuracy and the base leads on calibration. The 60-intent gap is
  inflated by the label format; at the base's `A-Z` rate it would be about 25 points, not 47.
- Moving hardware changed nothing beyond noise: Open-Jev 2B on the A100 matches its L4 run
  within 0.2 points, and Qwen 4B and MiniCPM stay inside their old n=100 intervals.
- Qwen 4B about matches Qwen 9B and is better calibrated.

## Alternatives considered

- **Leave everything and add a caveat.** Cheapest, but leaves Qwen 4B and MiniCPM at n=100
  and gives no base-model control for Open-Jev.
- **A second family at 9B.** Only useful if a reviewer argues that Qwen is weak at this
  task. MiniCPM's own launch chart (MiniCPM5-2B release, 2026) names `granite-4.2-3B` as a
  peer; keep it as the first candidate if that question comes up.
- **Mistral Nemo on Modal.** About 24 GB in bf16, over an L4; it stays on OpenRouter, where
  it answers the cheap-hosted-API question.

## Aside: vendor leaderboards do not replace this bench

MiniCPM5-2B's release chart scores generation (code, math, agents, tool use, long context).
This bench scores typed decisions: one label plus a probability, judged on calibration,
Brier score and how accuracy holds as the confidence gate rises, including on private French
data no model trained on. The two do not overlap.
