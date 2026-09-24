"""Jev (TypeSafe), the hosted typed decision model, through typesafe_sdk."""

from __future__ import annotations

from typesafe_sdk import Choice

from _common import MODEL, client, timed

# One shared TypeSafeClient: its httpx2 connection pool is thread-safe.
CONCURRENT = True

TYPES = {"choice": Choice}


def load(model: str = MODEL):
    c = client()

    def predict(state: str, questions: dict) -> dict:
        typed = {key: TYPES[q["type"]](instructions=q["instructions"], criteria=q["criteria"])
                 for key, q in questions.items()}
        call = timed(c, state, typed, model)
        (answer,) = call.answers.values()
        return {
            "predicted": answer.choice,
            "probabilities": dict(answer.probabilities),
            "confidence": answer.confidence,
            "input_tokens": call.input_tokens,
            "output_tokens": call.response.usage.output_tokens,
            "raw": call.response.raw_http_response.json(),
            "request_id": call.response.request_id,
        }

    return predict
