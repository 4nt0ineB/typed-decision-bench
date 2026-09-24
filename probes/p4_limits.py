"""P4a: what is the actual maximum input size?

The docs do not state one. `models.md` covers pricing and rate limits, `api.md` covers
errors, neither gives a token or character budget. So measure it: send a growing state
until the API refuses, then bisect the boundary.

Filler is real French prose from the immobilier/donation corpus rather than repeated
lorem ipsum, so the tokenizer sees realistic text.
"""

from __future__ import annotations

import json

from typesafe_sdk import Noul

from _common import MODEL, ROOT, client, dump, fr_corpus_dir, timed

QUESTION = {"mentions_donation": Noul(instructions="The text mentions a gift or donation")}


def filler() -> str:
    parts = [p.read_text(encoding="utf-8") for p in sorted(fr_corpus_dir().glob("*.md"))]
    return "\n\n".join(parts)


def probe(c, text: str) -> dict:
    """One call. Returns tokens on success, or the error class and message on failure."""
    try:
        call = timed(c, text, QUESTION)
        return {"ok": True, "chars": len(text), "tokens": call.input_tokens,
                "elapsed_ms": call.elapsed_ms}
    except Exception as exc:  # the SDK's exception hierarchy is what we are mapping
        return {"ok": False, "chars": len(text), "error": type(exc).__name__,
                "message": str(exc)[:300]}


def main() -> None:
    c = client()
    text = filler()
    print(f"corpus: {len(text):,} chars from {fr_corpus_dir()}\n")
    records = []

    print(f"{'chars':>10} {'tokens':>9} {'chars/tok':>10} {'ms':>7}  result")

    # Climb until refusal, so the failure mode is observed rather than assumed.
    size, last_ok, first_bad = 2_000, 0, None
    while size <= len(text):
        result = probe(c, text[:size])
        records.append(result)
        if result["ok"]:
            print(f"{result['chars']:>10,} {result['tokens']:>9,} "
                  f"{result['chars'] / result['tokens']:>10.2f} {result['elapsed_ms']:>7.0f}  ok")
            last_ok = size
        else:
            print(f"{result['chars']:>10,} {'':>9} {'':>10} {'':>7}  "
                  f"{result['error']}: {result['message'][:100]}")
            first_bad = size
            break
        size *= 2

    if first_bad is None:
        print(f"\nno refusal up to the whole corpus ({len(text):,} chars)")
    else:
        # Bisect to find where the boundary actually sits.
        lo, hi = last_ok, first_bad
        while hi - lo > max(1_000, lo // 100):
            mid = (lo + hi) // 2
            result = probe(c, text[:mid])
            records.append(result)
            print(f"{mid:>10,} {'ok' if result['ok'] else result['error']:>9}")
            lo, hi = (mid, hi) if result["ok"] else (lo, mid)
        print(f"\nlimit between {lo:,} and {hi:,} chars")

    ok = [r for r in records if r["ok"]]
    if ok:
        best = max(ok, key=lambda r: r["tokens"])
        print(f"largest accepted: {best['chars']:,} chars = {best['tokens']:,} tokens "
              f"({best['chars'] / best['tokens']:.2f} French chars/token) in {best['elapsed_ms']:.0f}ms")

    path = dump("p4-limits", records)
    print(f"wrote {path}")


if __name__ == "__main__":
    main()
