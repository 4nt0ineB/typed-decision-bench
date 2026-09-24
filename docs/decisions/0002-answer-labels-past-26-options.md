# 0002: Answer labels past 26 options

Date: 2026-09-24. Status: proposed, not started. Its own branch, after `modal-parity`.

## Context

LLM rows answer with one label token per option, and the probability is the softmax over
those tokens at the answer position. For 60 options the labels are `A-Z`, `a-z`, `0-7`:
60 distinct single tokens. Small models read `a` as `A`. On `massive_intent`, Qwen3.5 2B is
right 39% of the time when the gold label is in `A-Z` and 4% beyond it; the 4B and 9B are
flat across positions. Numbers in BENCH.md, "What the bench does not show".

Affected: every LLM adapter sharing `user_prompt` (`hf_logits`, `mlx_logits`,
`openai_compat`), on `massive_intent` only. Jev, Laya and Open-Jev use no labels.

## Constraint

The scheme must work for OpenRouter rows too. Their API returns only the first token's
top 20 logprobs, so it cannot score a forced multi-token label, and with 60 options the
true answer can fall outside the top 20. A scheme that only local models can follow would
split the LLM rows into two methods.

## Options to decide between

1. **Numeric labels scored as full sequences** (`1` to `60`, one or two tokens each).
   Removes case confusion for local models. Does not work on OpenRouter as is.
2. **Two-stage choice.** Group the 60 intents under their 18 scenarios, ask the scenario,
   then the intent within it; the probability is the product. At most 26 labels per step,
   works on every adapter, but doubles the calls and changes what is measured.
3. **Shuffle the option order.** One line. Spreads the damage evenly across intents instead
   of concentrating it on the lowercase ones, but recovers no accuracy.
4. **Keep the format and document it.** What `modal-parity` does.

## To measure first

Whether GPT-4o mini and Mistral Nemo lose accuracy past `Z` the way the 2B does. If they do
not, option 1 for local models plus a documented exception for OpenRouter may be enough.

## Cost

Re-running the four `hf_logits` intent rows is about $1 to $1.50 on Modal, plus the
laptop MLX row and a few cents on OpenRouter.
