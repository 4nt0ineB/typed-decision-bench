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
    "mistral-nemo-openrouter": ("openai_compat", {
        # Novita served logprobs=None despite require_parameters (19 of 500 massive_intent rows).
        "model": "mistralai/mistral-nemo", "ignore_providers": ["Novita"]}),
    "gpt-4o-mini-openrouter": ("openai_compat", {"model": "openai/gpt-4o-mini"}),
}

# USD per million input and output tokens. Local models are absent: they cost hardware time.
PRICES = {
    "jev-1.13.0": (0.042, 0.0),
    # OpenRouter listing, 2026-09-22.
    "mistral-nemo-openrouter": (0.019, 0.03),
    "gpt-4o-mini-openrouter": (0.15, 0.60),
}

# Fixed per model so a curve keeps its colour when other models come and go.
# Hue = family (validated as a categorical set), lightness = size within a family,
# dotted line = a `-tuned` row sharing its parent's colour (see report.py).
COLOURS = {
    "jev-1.13.0": "#2a78d6",
    "open-jev-2b": "#f39a66",
    "open-jev-9b": "#b8420f",
    "qwen3.5-0.8b-mlx-4bit": "#5ccb95",
    "qwen3.5-2b": "#5ccb95",
    "qwen3.5-4b": "#1baf7a",
    "qwen3.5-9b": "#00774d",
    "mistral-nemo-openrouter": "#eda100",
    "laya-router": "#e87ba4",
    "laya-multilingual": "#a8356a",
    "minicpm5-2b": "#4a3aa7",
    "gpt-4o-mini-openrouter": "#e34948",
}
# Qwen's three sizes cannot clear the normal-vision floor on lightness alone.
MARKERS = {"qwen3.5-2b": "^", "qwen3.5-4b": "s"}
# For result directories not listed above.
FALLBACK = ["#5a5a5a", "#17a2b8", "#d62728", "#7f7f7f"]
