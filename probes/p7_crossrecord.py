"""P7: questions that require reading every record, not finding one.

P5 and P6 both scored 100%. In both, the answer sat verbatim in one identifiable place:
P6 even named the file number, so the model could anchor on an exact string and ignore the
rest. That is retrieval, and Jev is clearly good at it.

This removes the anchor and asks three things that cannot be answered from one record:

  retiree_purpose   identify the file by a described property, then answer about it
  largest           compare amounts across every file in the state
  count_over_300k   count the files above a threshold

The last two aim straight at two documented jagged edges, "does not count reliably" and
"the error grows with the size of the thing being counted", and at the interaction with
large state. If Jev holds up here it is doing something considerably more interesting
than lookup. If it breaks, that is the boundary of where this tool belongs.
"""

from __future__ import annotations

import argparse
import random
import statistics
from collections import defaultdict

from typesafe_sdk import Choice

from _common import client, dump, latency_report, timed
from p5_haystack import BANDS, fillers, make_needles

COMPETING = [3, 7, 15, 31]
FILLER = 32_000

COUNT_BUCKETS = {
    "0": "None of them",
    "1": "Exactly one",
    "2-3": "Two or three",
    "4-7": "Between four and seven",
    "8+": "Eight or more",
}

BAND_CRITERIA = {
    "moins_150k": "Less than 150,000 euros",
    "150k_300k": "Between 150,000 and 300,000 euros",
    "300k_500k": "Between 300,000 and 500,000 euros",
    "plus_500k": "More than 500,000 euros",
}

BAND_ORDER = [name for name, _, _ in BANDS]


def bucket(n: int) -> str:
    if n == 0:
        return "0"
    if n == 1:
        return "1"
    if n <= 3:
        return "2-3"
    if n <= 7:
        return "4-7"
    return "8+"


QUESTIONS = {
    "retiree_purpose": Choice(
        instructions="Exactly one of the loan applications in this document was submitted "
                     "by an applicant who has retired. What is that applicant asking to "
                     "finance",
        criteria={
            "residence_principale": "Buying a home the applicant will live in themselves",
            "investissement_locatif": "Buying a property in order to rent it out",
            "travaux": "Renovation or repair work on a property already owned",
            "rachat_credit": "Consolidating or refinancing existing loans",
        },
    ),
    "largest": Choice(
        instructions="Across every loan application in this document, which band contains "
                     "the single largest amount requested",
        criteria=BAND_CRITERIA,
    ),
    "count_over_300k": Choice(
        instructions="How many of the loan applications in this document request more than "
                     "300,000 euros",
        criteria=COUNT_BUCKETS,
    ),
}


def scatter(files: list[str], filler: str, rng: random.Random) -> str:
    body = filler[:FILLER]
    order = list(files)
    rng.shuffle(order)
    cuts = sorted(rng.randrange(len(body)) for _ in order)
    out, previous = [], 0
    for cut, text in zip(cuts, order):
        out.append(body[previous:cut])
        out.append(f"\n\n{text}\n\n")
        previous = cut
    out.append(body[previous:])
    return "".join(out)


def amount_of(needle: dict) -> int:
    return BAND_ORDER.index(needle["gold"]["amount"])


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--reps", type=int, default=20)
    parser.add_argument("--seed", type=int, default=7)
    args = parser.parse_args()

    pool = make_needles(400, args.seed)
    retirees = [n for n in pool if n["gold"]["employment"] == "retraite"]
    others = [n for n in pool if n["gold"]["employment"] != "retraite"]
    filler = fillers()
    c = client()
    records = []

    for lang in ("en", "fr"):
        for count in COMPETING:
            rng = random.Random(args.seed + count)
            for rep in range(args.reps):
                target = rng.choice(retirees)
                siblings = rng.sample(others, count - 1)
                group = [target] + siblings

                gold = {
                    "retiree_purpose": target["gold"]["purpose"],
                    "largest": max(group, key=amount_of)["gold"]["amount"],
                    "count_over_300k": bucket(
                        sum(1 for n in group if amount_of(n) >= BAND_ORDER.index("300k_500k"))
                    ),
                }

                state = scatter([n[lang] for n in group], filler[lang], rng)
                call = timed(c, state, QUESTIONS)
                common = {"lang": lang, "competing": count, "rep": rep,
                          "input_tokens": call.input_tokens, "elapsed_ms": call.elapsed_ms}
                for qid, answer in call.answers.items():
                    records.append(common | {
                        "question": qid, "gold": gold[qid], "predicted": answer.choice,
                        "correct": answer.choice == gold[qid],
                        "p_chosen": answer.probabilities[answer.choice],
                        "confidence": answer.confidence,
                    })

    questions = list(QUESTIONS)
    for lang in ("en", "fr"):
        by = defaultdict(list)
        for record in records:
            if record["lang"] == lang:
                by[record["competing"]].append(record)

        print(f"\n=== {lang.upper()}: cross-record questions vs number of files "
              f"(in {FILLER:,} chars of filler) ===")
        print(f"{'files':>6} {'tokens':>8} {'ms':>6}  " + "".join(f"{q:>18}" for q in questions))
        for count in COMPETING:
            rows = by[count]
            cells = ""
            for qid in questions:
                subset = [r for r in rows if r["question"] == qid]
                cells += (f"{sum(r['correct'] for r in subset) / len(subset):>11.0%}"
                          f"{statistics.mean(r['p_chosen'] for r in subset):>7.2f}")
            print(f"{count:>6} {statistics.mean(r['input_tokens'] for r in rows):>8,.0f} "
                  f"{statistics.median(r['elapsed_ms'] for r in rows):>6.0f}  {cells}")
        print(f"{'':>22}  " + "".join(f"{'acc     p':>18}" for _ in questions))

    print("\n--- what it says when it is wrong (EN, 31 files) ---")
    for record in records:
        if (record["lang"] == "en" and record["competing"] == 31
                and not record["correct"] and record["rep"] < 6):
            print(f"  {record['question']:<16} gold={str(record['gold']):<18} "
                  f"said={str(record['predicted']):<18} p={record['p_chosen']:.2f}")

    print(f"\nlatency: {latency_report([r['elapsed_ms'] for r in records])}")
    path = dump("p7-crossrecord", records)
    print(f"wrote {path}")


if __name__ == "__main__":
    main()
