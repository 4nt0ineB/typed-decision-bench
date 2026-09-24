"""P6: the same question, with competing records in the state.

P5 found no accuracy loss up to 31k tokens of unrelated filler. That result stands, but
the task was too easy to locate the failure the jaggedness page describes: the filler was
encyclopaedia prose with nothing in it that could be mistaken for the answer.

A real bank dossier is not like that. It contains many applications, many amounts, many
employment situations, and the question is about one of them. So here the state holds K
sibling loan files scattered through the same filler, each with different values and its
own reference number, and the questions name the file to read.

This is the realistic version of "large irrelevant state": irrelevant text that is
structurally identical to the relevant text. Sweep K and see where it breaks.
"""

from __future__ import annotations

import argparse
import random
import statistics
from collections import defaultdict

from typesafe_sdk import Choice, Noul

from _common import client, dump, latency_report, timed
from p5_haystack import QUESTIONS, fillers, make_needles

DISTRACTORS = [0, 1, 3, 7, 15, 31]
FILLER = 32_000

REFERENCE = {
    "fr": "Le dossier numéro {ref}",
    "en": "The application with file number {ref}",
}


def targeted(ref: str, lang: str) -> dict:
    """The P5 questions, re-pointed at one specific file among several."""
    subject = REFERENCE[lang].format(ref=ref)
    questions = {}
    for qid, question in QUESTIONS.items():
        if isinstance(question, Noul):
            questions[qid] = Noul(
                instructions=f"{subject}: a guarantor or co-borrower is backing it"
            )
        else:
            text = {
                "purpose": f"{subject}: what is that applicant asking to finance",
                "employment": f"{subject}: what is that applicant's employment situation",
                "amount": f"{subject}: how much financing is that applicant requesting",
            }[qid]
            questions[qid] = Choice(instructions=text, criteria=question.criteria)
    return questions


def build(target: dict, siblings: list[dict], filler: str, lang: str, rng: random.Random) -> str:
    """Scatter the target and its siblings through the filler at random paragraph breaks."""
    body = filler[:FILLER]
    files = [target[lang]] + [s[lang] for s in siblings]
    rng.shuffle(files)

    cuts = sorted(rng.randrange(len(body)) for _ in files)
    out, previous = [], 0
    for cut, text in zip(cuts, files):
        out.append(body[previous:cut])
        out.append(f"\n\n{text}\n\n")
        previous = cut
    out.append(body[previous:])
    return "".join(out)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--needles", type=int, default=24)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    # A wide pool so siblings are drawn from genuinely different files, never the target.
    pool = make_needles(args.needles + max(DISTRACTORS) + 8, args.seed)
    targets, rest = pool[: args.needles], pool[args.needles :]
    filler = fillers()
    c = client()
    records = []

    for lang in ("en", "fr"):
        for count in DISTRACTORS:
            rng = random.Random(args.seed + count)
            for target in targets:
                siblings = rng.sample(rest, count)
                state = build(target, siblings, filler[lang], lang, rng)
                call = timed(c, state, targeted(str(target["ref"]), lang))
                common = {"lang": lang, "distractors": count, "needle": target["index"],
                          "input_tokens": call.input_tokens, "elapsed_ms": call.elapsed_ms}
                for qid, answer in call.answers.items():
                    if qid == "cosigner":
                        predicted = answer.noul >= 0.5
                        p_chosen = answer.noul if predicted else 1 - answer.noul
                        gold = target["gold_cosigner"]
                    else:
                        predicted = answer.choice
                        p_chosen = answer.probabilities[answer.choice]
                        gold = target["gold"][qid]
                    records.append(common | {"question": qid, "gold": gold,
                                             "predicted": predicted,
                                             "correct": predicted == gold,
                                             "p_chosen": p_chosen})

    questions = ["purpose", "employment", "amount", "cosigner"]
    for lang in ("en", "fr"):
        by = defaultdict(list)
        for record in records:
            if record["lang"] == lang:
                by[record["distractors"]].append(record)

        print(f"\n=== {lang.upper()}: accuracy vs number of competing files "
              f"(in {FILLER:,} chars of filler) ===")
        print(f"{'others':>7} {'tokens':>8} {'ms':>6}  " + "".join(f"{q:>14}" for q in questions))
        for count in DISTRACTORS:
            rows = by[count]
            cells = ""
            for qid in questions:
                subset = [r for r in rows if r["question"] == qid]
                cells += (f"{sum(r['correct'] for r in subset) / len(subset):>8.0%}"
                          f"{statistics.mean(r['p_chosen'] for r in subset):>6.2f}")
            print(f"{count:>7} {statistics.mean(r['input_tokens'] for r in rows):>8,.0f} "
                  f"{statistics.median(r['elapsed_ms'] for r in rows):>6.0f}  {cells}")
        print(f"{'':>23}  " + "".join(f"{'acc    p':>14}" for _ in questions))

    print(f"\nlatency: {latency_report([r['elapsed_ms'] for r in records])}")
    path = dump("p6-distractors", records)
    print(f"wrote {path}")


if __name__ == "__main__":
    main()
