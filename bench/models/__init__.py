"""Every model the bench can run: name -> (adapter module in this package, its kwargs).

An adapter module exposes `load(**kwargs)`, returning
`predict(state: str, questions: {key: wire question}) -> answer dict` with keys
predicted, probabilities (or None), confidence (or None), input_tokens, output_tokens,
raw, plus any extras worth archiving. `predict.manifest`, when set, is merged into the
run manifest. `CONCURRENT = True` in the module allows `--workers > 1`.

The name is the results directory, so it pins what it runs: never an alias that moves.
"""

REGISTRY = {
    "jev-1.13.0": ("jev", {"model": "jev-1.13.0"}),
    "laya-router": ("laya", {}),
    "laya-multilingual": ("laya", {"model": "multilingual"}),
    "qwen3.5-4b": ("hf_logits", {"hf_id": "Qwen/Qwen3.5-4B"}),
    "minicpm5-2b": ("hf_logits", {"hf_id": "openbmb/MiniCPM5-2B"}),
    # The frozen bases Open-Jev loads, at the revisions its checkpoints' model.json pin.
    "qwen3.5-2b": ("hf_logits", {"hf_id": "Qwen/Qwen3.5-2B",
                                 "revision": "15852e8c16360a2fea060d615a32b45270f8a8fc"}),
    "qwen3.5-9b": ("hf_logits", {"hf_id": "Qwen/Qwen3.5-9B",
                                 "revision": "c202236235762e1c871ad0ccb60c8ee5ba337b9a"}),
    "qwen3.5-0.8b-mlx-4bit": ("mlx_logits", {"hf_id": "mlx-community/Qwen3.5-0.8B-4bit"}),
    # CUDA only: bench/modal_run.py.
    "open-jev-2b": ("openjev", {"checkpoint_repo": "ZefanCai/Open-Jev-2B",
                                "revision": "0c7aa498b1627be8da4acf34c863ff0ee0a92785"}),
    # Base Qwen/Qwen3.5-9B@c202236235762e1c871ad0ccb60c8ee5ba337b9a, pinned by its model.json.
    # batch_size bounds the sequences per forward pass: 64 scores massive_intent's 60 in one.
    "open-jev-9b": ("openjev", {"checkpoint_repo": "ZefanCai/Open-Jev-9B",
                                "revision": "47e966881e489511c0c7f5633a9e1960a676a551",
                                "batch_size": 64}),
    # CUDA only, in their own image: bench/modal_run.py. Weights pinned to a Hub revision;
    # the kev code is pinned by the image (KEV_REF).
    "kev-9b": ("kev", {"model": "jaredpalmer/kev-9b@2629c06a5aeb0feb3b9783bafed17ed8f39ecf5c"}),
    "kev-27b": ("kev", {"model": "jaredpalmer/kev-27b@01b81998019be550f0ae858727df49bac9511195"}),
    "mistral-nemo-openrouter": ("openai_compat", {
        # Novita served logprobs=None despite require_parameters (19 of 500 massive_intent rows).
        "model": "mistralai/mistral-nemo", "ignore_providers": ["Novita"]}),
    "gpt-4o-mini-openrouter": ("openai_compat", {"model": "openai/gpt-4o-mini"}),
    # Not zero-shot: XLM-R base fine-tuned on MASSIVE en-US train, massive_intent only.
    # Its model card reports 87.75% on the full en-US test split.
    "xlm-r-base-massive-intent": ("hf_classifier", {
        "hf_id": "cartesinus/xlm-r-base-amazon-massive-intent",
        "revision": "52f2f02e1ba81f0ef8bb9d0d25ec9ccbb98128b6"}),
}

# USD per million input and output tokens. Local models are absent: they cost hardware time.
PRICES = {
    "jev-1.13.0": (0.042, 0.0),
    # OpenRouter listing, 2026-09-22.
    "mistral-nemo-openrouter": (0.019, 0.03),
    "gpt-4o-mini-openrouter": (0.15, 0.60),
}

# The models the charts draw, in legend order, as (label, colour, marker); every other model
# is in the tables only. Colours: slots 1-6 of a categorical order validated for line charts
# (worst neighbouring pair: CVD ΔE 9.1, normal vision 19.6). Crossing curves can still put a
# pair below the CVD floor (green and orange, for protanopes), hence a marker per model too.
CHARTED = {
    "jev-1.13.0": ("Jev 1.13.0", "#2a78d6", "o"),
    "kev-27b": ("Kev-27B", "#eb6834", "X"),
    "gpt-4o-mini-openrouter": ("GPT-4o mini", "#1baf7a", "s"),
    "qwen3.5-9b": ("Qwen3.5 9B", "#eda100", "^"),
    "open-jev-9b": ("Open-Jev 9B", "#e87ba4", "D"),
    "laya-multilingual": ("Laya multilingual", "#008300", "v"),
}
