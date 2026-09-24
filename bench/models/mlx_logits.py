"""hf_logits' constrained next-token contract, run through Apple's MLX runtime.

Same prompt, same label tokens, same restricted softmax; only the runtime and the weights
(quantized MLX checkpoints) change, so a gap against the hf_logits row is the cost of
deployment-realistic inference.
"""

from __future__ import annotations

from importlib.metadata import version
from pathlib import Path

from bench.models.hf_logits import build_prompt, label_token_ids


def load(hf_id: str, **_):
    import mlx.core as mx
    from huggingface_hub import hf_hub_download
    from mlx_lm import load as mlx_load

    model, wrapper, config = mlx_load(hf_id, return_config=True)
    # The wrapper's apply_chat_template injects its own enable_thinking; the HF tokenizer
    # underneath keeps the prompt byte-identical to hf_logits'.
    tok = wrapper._tokenizer

    def predict(state: str, questions: dict) -> dict:
        (question,) = questions.values()
        keys = list(question["criteria"])
        ids = label_token_ids(tok, len(keys))
        prompt = build_prompt(tok, question["instructions"], question["criteria"], state)
        tokens = tok(prompt)["input_ids"]  # as hf_logits tokenizes it, special tokens included
        logits = model(mx.array(tokens)[None])[0, -1]
        probs = mx.softmax(logits[mx.array(ids)].astype(mx.float32))
        mx.eval(probs)
        distribution = dict(zip(keys, probs.tolist()))
        return {
            "predicted": max(distribution, key=distribution.get),
            "probabilities": distribution,
            "confidence": None,
            "input_tokens": len(tokens),
            "output_tokens": 0,
            "raw": None,
        }

    quantization = config.get("quantization")
    predict.manifest = {
        "runtime": f"mlx-lm {version('mlx-lm')}",
        # The snapshot directory is named after the commit mlx_lm just loaded from the cache.
        "revision": Path(hf_hub_download(hf_id, "config.json")).parent.name,
        "quantization": quantization and {k: quantization[k] for k in ("bits", "group_size")},
        "device": "mlx",
    }
    return predict
