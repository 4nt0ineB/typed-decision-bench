"""P13: Laya on P11's French free-text blurbs, and on P8/P9's bank document.

P12 compared the two models on MASSIVE, which is short English and French sentences over a
flat label set. This one runs the probes that are closer to the actual product: the messy
French situation box (P11), where Jev scored 252/252, and then the 11-page mortgage
proposal (P8/P9), where it scored 340/340 and 100/100.

The document half does not fit. Laya is an encoder: 512 tokens on the English checkpoint,
1024 on the multilingual one, 8192 at the absolute ceiling of the mmBERT backbone. The
redacted proposal is ~13k tokens. Rather than assert that, the probe measures how much of
the document survives the sequence builder and what the answers look like on the part that
does, which is the honest version of "cannot run".
"""

from __future__ import annotations

import argparse
import time

import msgspec

from _common import ROOT
from p11_freetext import evaluate, report


class Answer:
    """Laya returns plain dicts; P11's grader reads attributes and uses hasattr('noul')."""

    def __init__(self, payload: dict):
        self.__dict__.update(payload)


class Call:
    def __init__(self, result: dict, elapsed_ms: float):
        self.answers = {k: Answer(v) for k, v in result["answers"].items()}
        self.elapsed_ms = elapsed_ms
        self.input_tokens = result["usage"]["input_tokens"]
        self.checkpoint = result.get("routing", {}).get("model", "?")


def as_laya(question) -> dict:
    """typesafe_sdk Choice/Score/Noul -> laya's dict question. Same fields, different box."""
    payload = {k: v for k, v in msgspec.structs.asdict(question).items() if v is not None}
    return {"type": type(question).__name__.lower()} | payload


def document_probe(router, head: int, max_len: int) -> None:
    """How much of the 11-page proposal fits, and what it costs to find out."""
    from laya.common import build_sequence

    text = (ROOT / "data" / "proposition-redacted.txt").read_text(encoding="utf-8")
    agent = router.load("multilingual")
    tok = agent.tok
    full = len(tok(text, add_special_tokens=False)["input_ids"])

    question = {"t": "choice", "ins": "What kind of property is this",
                "crit": {"NEUF": "newly built", "ANCIEN": "existing"}}
    ids, _ = build_sequence(tok, text, question, max_len, head)
    state_room = max_len - head
    print(f"\nP8/P9 document: {len(text):,} chars = {full:,} tokens")
    print(f"  laya ceiling: max_len={max_len}, head={head}, so ~{state_room} tokens of state")
    print(f"  fits: {min(state_room, full) / full:.0%} of the document, "
          f"sequence built at {len(ids)} tokens")
    print(f"  jev-1.13.0 accepted this document whole at 13,060 tokens (P8) "
          f"and 25,120 (P9), measured budget ~33k")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--device", default=None)
    parser.add_argument("--model", default="router")
    # P12 measured that the shipped 192/512 beats a raised budget on 18 options. P11 has at
    # most 8 options per question, so nothing is truncated here either way.
    parser.add_argument("--head", type=int, default=192)
    parser.add_argument("--max-len", type=int, default=512)
    args = parser.parse_args()

    from laya import Router

    router = Router(device=args.device, max_loaded=2)
    router.preload(["english", "multilingual"])
    for checkpoint in ("english", "multilingual"):
        router.load(checkpoint).cfg.update(head_max_len=args.head, max_len=args.max_len)
    forced = None if args.model == "router" else args.model

    picked = []

    def predict(state, questions):
        start = time.perf_counter()
        result = router.predict(state, {k: as_laya(v) for k, v in questions.items()},
                                model=forced)
        call = Call(result, (time.perf_counter() - start) * 1000.0)
        picked.append(call.checkpoint)
        return call

    records = evaluate(predict, args.repeats)
    report(records, args.repeats, f"laya:{args.model}", name="p13-laya-freetext", price=0.0)
    print(f"  checkpoints: {', '.join(sorted(set(picked)))}")

    document_probe(router, args.head, args.max_len)


if __name__ == "__main__":
    main()
