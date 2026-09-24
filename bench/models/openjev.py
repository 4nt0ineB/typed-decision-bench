"""Open-Jev (Zefan Cai, MIT): a LoRA adapter and a scalar decision head on frozen Qwen3.5-2B.

It speaks Jev's wire format, so `questions` goes through unchanged. Every option is scored
as an independent "Is this proposed answer correct? Yes/No" pass, softmaxed across options
at the checkpoint's calibration temperature. Probabilities and confidence are Open-Jev's
own; the confidence is the same rescaled chosen probability the bench applies otherwise.

Linux and CUDA only (bench/modal_run.py). Prefix caching stays off, as upstream ships it:
it exceeded their probability tolerance on 9 of 11 workloads.
"""

from __future__ import annotations

import json
from importlib.metadata import distribution
from importlib.util import find_spec


def answer(result: dict) -> dict:
    (reply,) = result["answers"].values()
    return {
        "predicted": reply["choice"],
        "probabilities": reply["probabilities"],
        "confidence": reply["confidence"],
        "input_tokens": result["usage"]["input_tokens"],
        "output_tokens": result["usage"]["output_tokens"],
        "raw": result,
    }


def package_commit() -> str | None:
    """The Open-Jev commit pip installed; their own provenance reads git and gets None."""
    direct = distribution("open-jev").read_text("direct_url.json")
    return json.loads(direct).get("vcs_info", {}).get("commit_id") if direct else None


def load(checkpoint_repo: str, revision: str, device: str = "cuda", batch_size: int = 32):
    from huggingface_hub import snapshot_download
    from jev.serving import load_predictor

    root = snapshot_download(checkpoint_repo, revision=revision,
                             allow_patterns=["package/checkpoint/*"])
    predictor = load_predictor(checkpoint=f"{root}/package/checkpoint", device=device,
                               batch_size=batch_size, prefix_cache=False)

    def predict(state: str, questions: dict) -> dict:
        return answer(predictor.predict({"state": state, "questions": questions}))

    # Multi-question probes (P18) need the whole response, which answer() narrows to one.
    predict.predictor = predictor
    predict.manifest = {
        "open_jev_commit": package_commit(), "checkpoint_revision": revision,
        "base_model": predictor.model_name,
        "base_revision": predictor.provenance["base_revision"],
        "checkpoint_sha256": predictor.provenance["checkpoint_sha256"],
        "temperature": predictor.temperature, "max_length": predictor.provenance["max_length"],
        "device": device, "dtype": "bfloat16", "batch_size": batch_size, "prefix_cache": False,
        # Without them transformers runs Qwen3.5's linear-attention layers in reference PyTorch.
        "kernels": {"causal_conv1d": find_spec("causal_conv1d") is not None,
                    "fla": find_spec("fla") is not None},
    }
    return predict


if __name__ == "__main__":
    fake = {
        "answers": {"scenario": {"type": "choice", "choice": "audio",
                                 "probabilities": {"alarm": 0.2, "audio": 0.7, "calendar": 0.1},
                                 "confidence": 0.55}},
        "model": "Qwen/Qwen3.5-2B", "usage": {"input_tokens": 312, "output_tokens": 0},
        "metadata": {"method": "lora_decision_head", "temperature": 1.5188},
    }
    mapped = answer(fake)
    assert mapped["predicted"] == "audio" and mapped["confidence"] == 0.55
    assert mapped["probabilities"] == {"alarm": 0.2, "audio": 0.7, "calendar": 0.1}
    assert (mapped["input_tokens"], mapped["output_tokens"]) == (312, 0)
    assert mapped["raw"] is fake
    try:
        answer({**fake, "answers": {"a": {}, "b": {}}})
        raise AssertionError("two answers were accepted")
    except ValueError:
        pass
    print("ok")
