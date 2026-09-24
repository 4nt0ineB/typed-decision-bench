"""Any OpenAI-compatible chat endpoint (OpenRouter by default), held to hf_logits' prompt.

Same user text as hf_logits, sent as one user message. When the endpoint returns logprobs
for the first generated token, the distribution is the softmax restricted to the option
labels found among the top 20, which is hf_logits' measurement through a keyhole: options
outside the top 20 get 0. Without logprobs the first label in the text is the answer,
probabilities are None, and a reply with no valid label is a schema failure.
"""

from __future__ import annotations

import json
import math
import os
import threading
import time
import urllib.error
import urllib.request

from _common import load_env
from bench.models.hf_logits import LABELS, user_prompt

CONCURRENT = True
ATTEMPTS = 5


def post(request: urllib.request.Request) -> tuple[dict, int]:
    """The decoded reply and how many retries it took, backing off on 429 and 5xx."""
    for retries in range(ATTEMPTS):
        try:
            with urllib.request.urlopen(request, timeout=60) as response:
                return json.load(response), retries
        except urllib.error.HTTPError as error:
            if retries == ATTEMPTS - 1 or not (error.code == 429 or error.code >= 500):
                raise
            after = (error.headers or {}).get("Retry-After", "")
            time.sleep(float(after) if after.isdigit() else 2 ** retries)


def distribution(content: list[dict] | None, keys: list[str]) -> dict[str, float] | None:
    """Option key -> probability from the first token's top logprobs, or None."""
    if not content:
        return None
    labels = dict(zip(LABELS, keys))
    mass = dict.fromkeys(keys, 0.0)
    # " B" and "B" are different tokens that mean the same answer.
    for candidate in content[0].get("top_logprobs") or []:
        key = labels.get(candidate["token"].strip())
        if key:
            mass[key] += math.exp(candidate["logprob"])
    total = sum(mass.values())
    return {k: v / total for k, v in mass.items()} if total else None


def parse(text: str, keys: list[str]) -> str | None:
    """The option whose label is the first non-blank character of the reply."""
    first = text.lstrip(" \n*`\"'([")[:1]
    return dict(zip(LABELS, keys)).get(first)


def load(model: str, base_url: str = "https://openrouter.ai/api/v1",
         key_env: str = "OPENROUTER_API_KEY", ignore_providers: list[str] | None = None):
    load_env()
    key = os.environ.get(key_env)
    if not key:
        raise SystemExit(f"{key_env} is not set. Add it to .env.")
    providers_lock = threading.Lock()
    # Otherwise OpenRouter may route to a provider that silently drops logprobs.
    # require_parameters alone is not enough: some providers accept logprobs and ignore it.
    routing = {"require_parameters": True}
    if ignore_providers:
        routing["ignore"] = ignore_providers

    def predict(state: str, questions: dict) -> dict:
        (question,) = questions.values()
        keys = list(question["criteria"])
        body = {
            "model": model,
            "messages": [{"role": "user", "content": user_prompt(
                question["instructions"], question["criteria"], state)}],
            "max_tokens": 4, "temperature": 0, "logprobs": True, "top_logprobs": 20,
            "provider": routing,
        }
        request = urllib.request.Request(
            f"{base_url}/chat/completions", data=json.dumps(body).encode(),
            headers={"Authorization": f"Bearer {key}", "Content-Type": "application/json"})
        # run.py's elapsed_ms includes the backoff sleeps: the caller really waited that long.
        # Filter rows with retries > 0 when reading latency.
        raw, retries = post(request)

        choice = raw["choices"][0]
        probabilities = distribution((choice.get("logprobs") or {}).get("content"), keys)
        if probabilities:
            predicted = max(probabilities, key=probabilities.get)
        else:
            predicted = parse(choice["message"].get("content") or "", keys)
        with providers_lock:
            if raw.get("provider") and raw["provider"] not in predict.manifest["providers"]:
                predict.manifest["providers"].append(raw["provider"])
        return {
            "predicted": predicted,
            "probabilities": probabilities,
            "confidence": None,
            "input_tokens": raw["usage"]["prompt_tokens"],
            "output_tokens": raw["usage"]["completion_tokens"],
            "raw": raw,
            "schema_failure": predicted is None,
            "retries": retries,
        }

    predict.manifest = {"base_url": base_url, "model_id": model, "providers": [],
                        "ignore_providers": ignore_providers}
    return predict


if __name__ == "__main__":
    import io
    from unittest import mock

    def fake(content: str, logprobs: dict | None) -> io.BytesIO:
        return io.BytesIO(json.dumps({
            "provider": "Fake", "usage": {"prompt_tokens": 120, "completion_tokens": 1},
            "choices": [{"message": {"content": content}, "logprobs": logprobs}]}).encode())

    top = [{"token": " B", "logprob": math.log(0.5)}, {"token": "B", "logprob": math.log(0.2)},
           {"token": "A", "logprob": math.log(0.1)}, {"token": "The", "logprob": math.log(0.2)}]
    questions = {"scenario": {"instructions": "Which", "criteria": {
        "alarm": "an alarm", "audio": "the volume", "calendar": "a meeting"}}}
    os.environ.setdefault("OPENROUTER_API_KEY", "test")
    predict = load("any/model")
    replies = [fake("B", {"content": [{"token": "B", "top_logprobs": top}]}),
               fake(" **C**", None), fake("I think", {"content": []})]
    with mock.patch("urllib.request.urlopen", side_effect=replies) as urlopen:
        logits, text, failure = [predict("wake me up", questions) for _ in replies]
    assert json.loads(urlopen.call_args.args[0].data)["provider"] == {"require_parameters": True}

    ignoring = load("any/model", ignore_providers=["Novita"])
    with mock.patch("urllib.request.urlopen", side_effect=[fake("B", None)]) as urlopen:
        ignoring("wake me up", questions)
    assert json.loads(urlopen.call_args.args[0].data)["provider"] == {
        "require_parameters": True, "ignore": ["Novita"]}
    assert ignoring.manifest["ignore_providers"] == ["Novita"]

    assert logits["predicted"] == "audio" and not logits["schema_failure"]
    assert math.isclose(logits["probabilities"]["alarm"], 0.125)
    assert logits["probabilities"]["calendar"] == 0.0 and logits["input_tokens"] == 120
    assert text["predicted"] == "calendar" and text["probabilities"] is None
    assert failure["predicted"] is None and failure["schema_failure"]
    assert predict.manifest["providers"] == ["Fake"]
    assert logits["retries"] == 0

    def http_error(code: int, headers: dict | None = None) -> urllib.error.HTTPError:
        return urllib.error.HTTPError("https://fake", code, "fake", headers or {}, None)

    flaky = [http_error(429, {"Retry-After": "3"}), http_error(503),
             fake("B", {"content": [{"token": "B", "top_logprobs": top}]})]
    with mock.patch("urllib.request.urlopen", side_effect=flaky), \
            mock.patch("time.sleep") as sleep:
        retried = predict("wake me up", questions)
    assert retried["retries"] == 2 and retried["predicted"] == "audio"
    assert [c.args[0] for c in sleep.call_args_list] == [3.0, 2]

    with mock.patch("urllib.request.urlopen", side_effect=[http_error(401)]) as urlopen, \
            mock.patch("time.sleep") as sleep:
        try:
            predict("wake me up", questions)
            raise AssertionError("401 did not raise")
        except urllib.error.HTTPError as error:
            assert error.code == 401
    assert urlopen.call_count == 1 and not sleep.called

    with mock.patch("urllib.request.urlopen", side_effect=[http_error(429)] * ATTEMPTS) as urlopen, \
            mock.patch("time.sleep") as sleep:
        try:
            predict("wake me up", questions)
            raise AssertionError("exhausted retries did not raise")
        except urllib.error.HTTPError as error:
            assert error.code == 429
    assert urlopen.call_count == ATTEMPTS
    assert [c.args[0] for c in sleep.call_args_list] == [1, 2, 4, 8]
    print("ok")
