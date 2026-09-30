"""Jev (TypeSafe), the hosted zero-shot classification model, through typesafe_sdk.

Also the client the probes call Jev with, so every Jev number goes through one place.
"""

from __future__ import annotations

import os
import time
from dataclasses import dataclass
from typing import Any

from typesafe_sdk import Choice, SystemOneResponse, TypeSafeClient

from bench import load_env

# Pinned deliberately. `jev-latest` is an alias that can move under us and silently
# invalidate every measurement taken before it moved.
MODEL = os.environ.get("TYPESAFE_MODEL", "jev-1.13.0")

# One shared TypeSafeClient: its httpx2 connection pool is thread-safe.
CONCURRENT = True

TYPES = {"choice": Choice}


def client() -> TypeSafeClient:
    load_env()
    if not os.environ.get("TYPESAFE_API_KEY"):
        raise SystemExit("TYPESAFE_API_KEY is not set. Create .env from .env.example.")
    return TypeSafeClient()


@dataclass(frozen=True)
class Call:
    elapsed_ms: float
    model: str
    response: SystemOneResponse

    @property
    def answers(self) -> dict[str, Any]:
        return self.response.answers

    @property
    def input_tokens(self) -> int:
        return self.response.usage.input_tokens

    def archive(self) -> dict[str, Any]:
        return {
            "model": self.model,
            "request_id": self.response.request_id,
            "elapsed_ms": self.elapsed_ms,
            "raw": self.response.raw_http_response.json(),
        }


def timed(c: TypeSafeClient, state: Any, questions: dict[str, Any], model: str = MODEL) -> Call:
    start = time.perf_counter()
    response = c.system_one(state, questions, model=model)
    return Call((time.perf_counter() - start) * 1000.0, model, response)


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
