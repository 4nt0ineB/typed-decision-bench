"""Kev (Jared Palmer, Apache-2.0): a LoRA adapter and a pointer head on a frozen Qwen.

Loaded the way its author's own Modal endpoint loads it (skills/kev-deploy/scripts/
kev_serve.py): bf16, adapter fused into the weights, CUDA graphs, and the temperature the
checkpoint ships with. Answers go through kev.serve.Server in process, so the
probabilities are the served ones without an HTTP hop. The pointer head reads every
option of a question in one sequence, so options can move one another; Open-Jev scores
each option on its own.

Linux and CUDA only, in its own image (bench/modal_run.py): Kev pins torch<2.9, the bench
2.14.

At this commit the server refuses (HTTP 422) a state plus question over 8,192 tokens
(kev/model.py, SERVE_MAX_BRANCH), which records_* HAYSTACK exceeds. A refusal is a row
with no answer, counted wrong, as Kev's own README counts its over-long documents.
"""

from __future__ import annotations

import json
from importlib.metadata import distribution
from importlib.util import find_spec

from bench.models.openjev import answer


def package_commit() -> str | None:
    direct = distribution("kev").read_text("direct_url.json")
    return json.loads(direct).get("vcs_info", {}).get("commit_id") if direct else None


def load(model: str):
    import torch
    from fastapi import HTTPException
    from kev.api import SystemOneRequest
    from kev.checkpoint import Checkpoint, LoadOptions
    from kev.serve import Server

    checkpoint = Checkpoint(model)
    tok, net = checkpoint.load("cuda", LoadOptions(dtype=torch.bfloat16, cuda_graphs=True,
                                                   fused=True))
    server = Server(checkpoint, tok, net, "cuda")

    def predict(state: str, questions: dict) -> dict:
        try:
            result = server.answer(SystemOneRequest.model_validate(
                {"state": state, "model": "kev-latest", "questions": questions}))
        except HTTPException as refused:
            if refused.status_code != 422:
                raise
            return {"predicted": None, "probabilities": None, "confidence": None,
                    "input_tokens": len(tok.encode(state)), "output_tokens": 0,
                    "raw": {"refused": refused.detail}, "refused": True, "model_ms": None}
        # Model time alone, the figure Kev's serving table reports; elapsed_ms adds the rest.
        return answer(result) | {"model_ms": result["latency_ms"]}

    predict.manifest = {
        "kev_commit": package_commit(), "checkpoint": model,
        "temperature": net.head.temperature, "dtype": "bfloat16", "fused": True,
        "cuda_graphs": True, "kernels": {"fla": find_spec("fla") is not None},
    }
    return predict
