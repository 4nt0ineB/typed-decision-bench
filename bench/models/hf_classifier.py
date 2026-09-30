"""A Hugging Face encoder fine-tuned on a fixed label set: the "labelled data" reference row.

Not zero-shot. Its classification head has one output per label it was trained on, so it
never reads the instructions or the criteria, only the state. It still runs under every
condition to keep the rows paired, which makes FR_EN and FR_FR identical by construction.
"""

from __future__ import annotations


def load(hf_id: str, revision: str, device: str | None = None):
    import torch
    from transformers import AutoModelForSequenceClassification, AutoTokenizer

    device = device or ("cuda" if torch.cuda.is_available() else "mps")
    tok = AutoTokenizer.from_pretrained(hf_id, revision=revision)
    model = AutoModelForSequenceClassification.from_pretrained(hf_id, revision=revision)
    model = model.to(device).eval()
    labels = [model.config.id2label[i] for i in range(model.config.num_labels)]

    def predict(state: str, questions: dict) -> dict:
        (question,) = questions.values()
        if set(question["criteria"]) != set(labels):
            raise ValueError(f"{hf_id} was trained on other labels than this task's options")
        inputs = tok(state, return_tensors="pt", truncation=True).to(device)
        with torch.inference_mode():
            probs = torch.softmax(model(**inputs).logits[0].float(), dim=-1).tolist()
        distribution = dict(zip(labels, probs))
        return {
            "predicted": max(distribution, key=distribution.get),
            "probabilities": distribution,
            "confidence": None,
            "input_tokens": int(inputs["input_ids"].shape[1]),
            "output_tokens": 0,
            "raw": None,
        }

    predict.manifest = {"revision": revision, "device": device}
    return predict
