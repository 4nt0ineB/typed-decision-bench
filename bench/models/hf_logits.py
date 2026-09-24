"""A Hugging Face causal LM held to Jev's output contract, without generating anything.

MiniCPM5-2B (OpenBMB, Apache 2.0) and Qwen3.5-4B (Alibaba, Apache 2.0) are both
autoregressive causal language models. Left alone they write text, so the comparison is
only fair if they are held to the same output contract as Jev: one key out of a closed
set, plus a probability for every key.

That contract is obtained here without generating anything. The prompt ends on an answer
prefix, one forward pass produces the next-token logits, and the softmax is taken over
the k option-label tokens alone. Three consequences, and they are the point:

  - schema conformity is absolute, exactly as absolute as Jev's, because the answer is
    picked from a fixed set of token ids rather than parsed out of generated text;
  - the run is deterministic by construction. No sampling happens, so no seed is needed.
    A seed only buys determinism back for models left free to generate;
  - the probabilities are real logits, so ECE measures the same thing it measures for
    Jev. The adapter returns no confidence, so the bench applies P15's formula.

What this does NOT show: how these models behave under ordinary generation with a JSON
schema. There the guarantee comes from the serving stack's grammar, and the probabilities
are spread over a token sequence rather than a decision, which is a different measurement.
"""

from __future__ import annotations

# One token per option, and distinct by construction. Option keys ("alarm", "calendar")
# would collide on their first token and make the restricted softmax meaningless. 62 labels
# because massive_intent has 60 options.
LABELS = "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789"

# Ends on a newline, not a space. Both vocabularies merge "Answer: " + "A" into one
# token, which leaves no position whose logits are the answer.
ANSWER_PREFIX = "Answer:\n"


def user_prompt(instructions: str, criteria: dict[str, str], state: str) -> str:
    options = "\n".join(
        f"{LABELS[i]}. {text}" for i, text in enumerate(criteria.values())
    )
    return (
        f"{instructions}?\n\n"
        f"Options:\n{options}\n\n"
        f"Input:\n{state}\n\n"
        f"Reply with the letter of the single best option."
    )


def build_prompt(tok, instructions: str, criteria: dict[str, str], state: str) -> str:
    user = user_prompt(instructions, criteria, state)
    try:
        text = tok.apply_chat_template(
            [{"role": "user", "content": user}],
            tokenize=False, add_generation_prompt=True, enable_thinking=False,
        )
    except TypeError:
        text = tok.apply_chat_template(
            [{"role": "user", "content": user}],
            tokenize=False, add_generation_prompt=True,
        )
    return text + ANSWER_PREFIX


def label_token_ids(tok, count: int) -> list[int]:
    """The id each option's letter takes when it follows the answer prefix.

    Encoded in context and sliced off the end: a leading-space variant is a different id
    in most vocabularies, and picking the wrong one silently scores noise.
    """
    base = tok.encode(ANSWER_PREFIX, add_special_tokens=False)
    ids = []
    for letter in LABELS[:count]:
        full = tok.encode(ANSWER_PREFIX + letter, add_special_tokens=False)
        assert full[: len(base)] == base, f"prefix not stable for {letter!r}"
        ids.append(full[len(base)])
    assert len(set(ids)) == count, "two options share a first token"
    return ids


def load_model(hf_id: str, device: str, dtype: str, revision: str | None = None):
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tok = AutoTokenizer.from_pretrained(hf_id, revision=revision)
    try:
        model = AutoModelForCausalLM.from_pretrained(hf_id, revision=revision,
                                                     dtype=getattr(torch, dtype))
    except ValueError:
        # Vision-capable checkpoints (Qwen3.5) are not registered for plain causal LM.
        from transformers import AutoModelForImageTextToText

        model = AutoModelForImageTextToText.from_pretrained(hf_id, revision=revision,
                                                            dtype=getattr(torch, dtype))
    return tok, model.to(device).eval()


def load(hf_id: str, revision: str | None = None, device: str | None = None,
         dtype: str = "bfloat16"):
    import torch

    device = device or ("cuda" if torch.cuda.is_available() else "mps")
    tok, model = load_model(hf_id, device, dtype, revision)

    def predict(state: str, questions: dict) -> dict:
        (question,) = questions.values()
        keys = list(question["criteria"])
        ids = label_token_ids(tok, len(keys))
        prompt = build_prompt(tok, question["instructions"], question["criteria"], state)
        inputs = tok(prompt, return_tensors="pt").to(device)
        with torch.inference_mode():
            logits = model(**inputs).logits[0, -1]
        probs = torch.softmax(logits[ids].float(), dim=-1).tolist()
        distribution = dict(zip(keys, probs))
        return {
            "predicted": max(distribution, key=distribution.get),
            "probabilities": distribution,
            "confidence": None,
            "input_tokens": int(inputs["input_ids"].shape[1]),
            "output_tokens": 0,
            "raw": None,
        }

    from transformers import AutoConfig

    # The loaded model's own config comes back without _commit_hash in transformers 5.
    revision = AutoConfig.from_pretrained(hf_id, revision=revision)._commit_hash
    predict.manifest = {"revision": revision, "device": device, "dtype": dtype}
    return predict
