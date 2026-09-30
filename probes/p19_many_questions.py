"""P19: the loan offer's 17 questions, asked of GPT-4o mini the two ways an LLM allows.

P8 sends the document and its 17 questions to Jev in one call. An LLM cannot do both
things that call does at once:

  A. per_question  one call per question, the bench's own contract (bench/models/
                   openai_compat.py): the answer and a probability read from the first
                   token's logprobs. 17 calls per document, each carrying the whole
                   document, sent in parallel as an application would.
  B. one_call      all 17 questions in one call, answered as JSON constrained to each
                   question's option keys. One document, one call, and no probability:
                   nothing left to set a threshold on.

Same document, questions and gold as P8 (a yes/no question becomes two options, the score
its four levels). What it measures per document: accuracy, wall time, tokens and dollars,
and, for A, how many wrong answers P8's policy would have prefilled silently (0.9 and
above). Jev's figures come from P8's newest archive, so this costs no Jev call.
"""

from __future__ import annotations

import argparse
import json
import os
import statistics
import sys
import time
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from typesafe_sdk import Choice, Noul, Score

from _common import RESULTS, dump
from bench.models import PRICES
from bench.models.openai_compat import load, post
from p8_prefill import AUTO, DOCUMENT, GOLD, QUESTIONS

MODEL = "gpt-4o-mini-openrouter"
MODEL_ID = "openai/gpt-4o-mini"
PER_IN, PER_OUT = (price / 1e6 for price in PRICES[MODEL])


def wire(question) -> dict:
    """P8's typed question as a Choice the LLM adapter accepts."""
    if isinstance(question, Noul):
        return {"instructions": f"Is this true of the document: {question.instructions}",
                "criteria": {"true": "Yes, it is true", "false": "No, it is not"}}
    if isinstance(question, Score):
        return {"instructions": question.instructions,
                "criteria": {str(level): text for level, text in enumerate(question.criteria)}}
    assert isinstance(question, Choice)
    return {"instructions": question.instructions, "criteria": dict(question.criteria)}


def key(gold) -> str:
    if isinstance(gold, bool):
        return str(gold).lower()
    return str(int(gold)) if isinstance(gold, float) else gold


WIRE = {qid: wire(q) for qid, q in QUESTIONS.items()}
GOLD_KEYS = {qid: key(g) for qid, g in GOLD.items()}


def cost(input_tokens: int, output_tokens: int) -> float:
    return input_tokens * PER_IN + output_tokens * PER_OUT


def per_question(predict, state: str, run: int) -> tuple[list[dict], float]:
    def ask(item):
        qid, question = item
        start = time.perf_counter()
        answer = predict(state, {qid: question})
        return qid, answer, (time.perf_counter() - start) * 1000.0

    start = time.perf_counter()
    with ThreadPoolExecutor(len(WIRE)) as pool:
        answers = list(pool.map(ask, WIRE.items()))
    wall = (time.perf_counter() - start) * 1000.0
    records = []
    for qid, answer, elapsed in answers:
        probabilities = answer["probabilities"]
        p = probabilities[answer["predicted"]] if probabilities and answer["predicted"] else None
        k = len(WIRE[qid]["criteria"])
        records.append({
            "mode": "per_question", "run": run, "question": qid, "gold": GOLD_KEYS[qid],
            "predicted": answer["predicted"], "correct": answer["predicted"] == GOLD_KEYS[qid],
            "p_chosen": p, "confidence": None if p is None else (k * p - 1) / (k - 1),
            "elapsed_ms": elapsed, "input_tokens": answer["input_tokens"],
            "output_tokens": answer["output_tokens"], "retries": answer["retries"],
            "provider": answer["raw"].get("provider"),
        })
    return records, wall


def one_call_body(state: str) -> dict:
    blocks = []
    for qid, question in WIRE.items():
        options = "\n".join(f"  - {k}: {text}" for k, text in question["criteria"].items())
        blocks.append(f"{qid}: {question['instructions']}?\n{options}")
    prompt = ("Answer every question below about the document. For each question, give the "
              "key of the single best option.\n\nQuestions:\n\n" + "\n\n".join(blocks)
              + f"\n\nDocument:\n{state}")
    schema = {"type": "object", "additionalProperties": False, "required": list(WIRE),
              "properties": {qid: {"type": "string", "enum": list(q["criteria"])}
                             for qid, q in WIRE.items()}}
    return {"model": MODEL_ID, "messages": [{"role": "user", "content": prompt}],
            "temperature": 0, "provider": {"require_parameters": True},
            "response_format": {"type": "json_schema", "json_schema": {
                "name": "answers", "strict": True, "schema": schema}}}


def one_call(key_header: str, state: str, run: int) -> tuple[list[dict], float]:
    request = urllib.request.Request(
        "https://openrouter.ai/api/v1/chat/completions",
        data=json.dumps(one_call_body(state)).encode(),
        headers={"Authorization": key_header, "Content-Type": "application/json"})
    start = time.perf_counter()
    raw, retries = post(request)
    wall = (time.perf_counter() - start) * 1000.0
    answers = json.loads(raw["choices"][0]["message"]["content"])
    usage = raw["usage"]
    return [{
        "mode": "one_call", "run": run, "question": qid, "gold": GOLD_KEYS[qid],
        "predicted": answers.get(qid), "correct": answers.get(qid) == GOLD_KEYS[qid],
        "p_chosen": None, "confidence": None, "elapsed_ms": wall,
        # Tokens and retries belong to the call, stored once on its first row.
        "input_tokens": usage["prompt_tokens"] if i == 0 else 0,
        "output_tokens": usage["completion_tokens"] if i == 0 else 0,
        "retries": retries, "provider": raw.get("provider"),
    } for i, qid in enumerate(WIRE)], wall


def jev_baseline() -> dict:
    archive = sorted(RESULTS.glob("p8-prefill-*.jsonl"))[-1]
    rows = [json.loads(line) for line in archive.read_text(encoding="utf-8").splitlines()]
    calls = {r["run"]: r for r in rows}
    tokens = statistics.median(c["input_tokens"] for c in calls.values())
    return {"right": sum(r["correct"] for r in rows), "n": len(rows),
            "wall_ms": statistics.median(c["elapsed_ms"] for c in calls.values()),
            "calls": 1, "tokens": tokens, "usd": tokens * PRICES["jev-1.13.0"][0] / 1e6,
            "silent_wrong": sum(not r["correct"] and r["certainty"] >= AUTO for r in rows),
            "source": archive.name}


def summarise(mode: str, records: list[dict], walls: list[float]) -> dict:
    rows = [r for r in records if r["mode"] == mode]
    runs = len(walls)
    tokens_in = sum(r["input_tokens"] for r in rows) / runs
    tokens_out = sum(r["output_tokens"] for r in rows) / runs
    return {"right": sum(r["correct"] for r in rows), "n": len(rows),
            "wall_ms": statistics.median(walls), "calls": 17 if mode == "per_question" else 1,
            "tokens": tokens_in, "usd": cost(tokens_in, tokens_out),
            "silent_wrong": sum(not r["correct"] and (r["confidence"] or 0) >= AUTO for r in rows)
            if mode == "per_question" else None}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repeats", type=int, default=20)
    args = parser.parse_args()

    state = DOCUMENT.read_text(encoding="utf-8")
    predict = load(MODEL_ID)
    key_header = f"Bearer {os.environ['OPENROUTER_API_KEY']}"

    records, walls = [], {"per_question": [], "one_call": []}
    for run in range(args.repeats):
        for mode, fn in (("per_question", lambda: per_question(predict, state, run)),
                         ("one_call", lambda: one_call(key_header, state, run))):
            rows, wall = fn()
            records += rows
            walls[mode].append(wall)
        print(f"\rrun {run + 1}/{args.repeats}", end="", file=sys.stderr, flush=True)
    print(file=sys.stderr)

    table = {"Jev, one call (P8)": jev_baseline(),
             "GPT-4o mini, one call per question": summarise("per_question", records,
                                                             walls["per_question"]),
             "GPT-4o mini, one JSON call": summarise("one_call", records, walls["one_call"])}
    print(f"\n{args.repeats} runs of the {len(WIRE)} questions on {Path(DOCUMENT).name}\n")
    print(f"{'':<38} {'right':>9} {'calls':>6} {'wall p50':>9} {'tokens in':>10} "
          f"{'$ / document':>13} {'wrong at 0.90+':>15}")
    for name, s in table.items():
        silent = "no probability" if s["silent_wrong"] is None else str(s["silent_wrong"])
        print(f"{name:<38} {s['right']:>4}/{s['n']:<4} {s['calls']:>6} {s['wall_ms']:>7.0f}ms "
              f"{s['tokens']:>10.0f} {s['usd']:>13.5f} {silent:>15}")
    wrong = sorted({(r["mode"], r["question"], r["predicted"]) for r in records if not r["correct"]})
    if wrong:
        print("\nwrong answers (mode, question, answered):")
        for mode, qid, got in wrong:
            count = sum(1 for r in records if (r["mode"], r["question"], r["predicted"]) == (mode, qid, got))
            print(f"  {mode:<13} {qid:<28} {got}  x{count}  (gold {GOLD_KEYS[qid]})")
    print(f"\nproviders: {sorted({r['provider'] for r in records if r['provider']})}")
    print(f"wrote {dump('p19-many-questions', records)}")


if __name__ == "__main__":
    main()
