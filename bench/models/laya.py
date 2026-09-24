"""Laya (convaiinnovations, Apache 2.0), the open-weight encoder that copies Jev's interface.

ModernBERT-large 421M for English, mmBERT-base 322M for 100+ languages, and a router that
picks a checkpoint per request, so it is self-hostable and free to run. `model` forces a
checkpoint ("english", "multilingual"); None lets the router pick.

Defaults are the shipped head_max_len=192 and max_len=512. At 192, 18 full-sentence
options are each truncated to about 9 tokens: that is what "as shipped" measures.
"""

from __future__ import annotations

import math


def router(device: str | None = None, head: int = 192, max_len: int = 512):
    from laya import Router

    loaded = Router(device=device, max_loaded=2)
    # Without this the router rebuilds a checkpoint on every language flip (~7s each).
    loaded.preload(["english", "multilingual"])
    for checkpoint in ("english", "multilingual"):
        loaded.load(checkpoint).cfg.update(head_max_len=head, max_len=max_len)
    return loaded


def load(model: str | None = None, head: int = 192, max_len: int = 512,
         device: str | None = None):
    from laya.common import QTYPES, temp_bucket

    loaded = router(device, head, max_len)
    # Laya rounds probabilities to 4 decimals, so a confident row stores its losers as 0.0
    # and temperature rescaling cannot order it. The raw logits keep that order.
    captured = {}
    for checkpoint in ("english", "multilingual"):
        loaded.load(checkpoint).model.register_forward_hook(
            lambda module, inputs, output: captured.update(logits=output[0]))

    def log_probabilities(agent, options: list[str]) -> dict[str, float]:
        k, choice = len(options), QTYPES["choice"]
        t = agent.temperature_by_options.get(temp_bucket(choice, k), agent.temperature[choice])
        z = [float(v) / max(1e-3, float(t)) for v in captured["logits"][0, :k]]
        top = max(z)
        norm = top + math.log(sum(math.exp(v - top) for v in z))
        return {o: v - norm for o, v in zip(options, z)}

    def predict(state: str, questions: dict) -> dict:
        result = loaded.predict(state, questions, model=model)
        (answer,) = result["answers"].values()
        agent = loaded.load(result["routing"]["model"])
        return {
            "predicted": answer["choice"],
            "probabilities": answer["probabilities"],
            "log_probabilities": log_probabilities(agent, list(answer["probabilities"])),
            "confidence": answer["confidence"],
            "input_tokens": result["usage"]["input_tokens"],
            "output_tokens": result["usage"]["output_tokens"],
            "raw": result,
            "checkpoint": result["routing"]["model"],
        }

    predict.manifest = {"device": str(loaded.load("english").device)}
    return predict
