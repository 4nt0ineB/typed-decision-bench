"""P11: free-text situation box instead of document upload.

The design: no file, no bank statement, no PDF. One textarea, the user describes their
situation in plain French however they like, and the form prefills from it. A funnel
shortcut rather than a data-entry convenience.

Why this is a better shape than the document route, beyond the obvious privacy one:

  The injection surface is small in a way an LLM's never is. Jev returns no text. It
  cannot be made to say something, leak its instructions, or call a tool. The documented
  worst case is that hostile input shifts a classification, and here that means a user
  mis-filling their own form. Self-inflicted only, no third party to harm.

  The abuse ceiling is arithmetic. Input caps at ~33k tokens and costs $0.042/M, so a
  determined abuser burns fractions of a cent per call. There is no output to inflate.
  An LLM box has to be defended against someone extracting 100k output tokens.

What this probe measures, because those are the two things that decide whether the
shortcut converts:

  1. Does it correctly tag a short, messy, incomplete blurb, and correctly say "not
     stated" for everything the user did not mention? Most fields are absent in two
     sentences, so ABSTENTION IS THE DOMINANT BEHAVIOUR, not the edge case.
  2. Can it bind a bare number to the right role? "210k" and "32000" and "2580" appear
     with no labels at all, and the only thing distinguishing them is the surrounding
     prose.

Inputs are handwritten rather than drawn from a dataset, so gold is true by construction:
the text was written to say a specific thing. The risk is writing blurbs that are too
tidy, so they are deliberately abbreviated, mistyped, unpunctuated, and in several cases
incomplete or self-contradictory.
"""

from __future__ import annotations

import argparse
import re
from collections import Counter, defaultdict

from typesafe_sdk import Choice, Noul, Score

from _common import MODEL, client, dump, latency_report, timed

NONE = "non_precise"

# Permissive on purpose: free text writes amounts every way a person might. Code
# normalises whatever comes back, so over-capturing costs nothing and under-capturing is
# the only real failure (P10: a missing candidate degrades to abstention, not a wrong pick).
#
# The first version used `\d[\d\s.,]*` and matched straight across "T3, 265k" to produce
# the candidate "3, 265k". The model then picked it, correctly, and my harness scored that
# as a model failure. Separators only bridge when a digit follows them.
NUMBER = re.compile(
    r"\d+(?:[ .]\d{3})*(?:,\d+)?\s*[kK]?\s*(?:€|euros?)?",
)

BLURBS = [
    {
        "text": "bjr je suis célibataire je gagne 2580 net par mois je veux acheter une "
                "maison ancienne vers l'est de paris 210k j'ai 32000 d'apport est ce que "
                "j'ai droit au ptz ?",
        "gold": {"projet": "residence_principale", "nature": "ANCIEN",
                 "famille": "celibataire", "ptz_evoque": True, "prix": "210k",
                 "apport": "32000", "revenus": "2580"},
    },
    {
        "text": "On est en couple avec 2 enfants, on cherche du neuf, budget 350 000 €, "
                "apport 60k. On nous a parlé du PTZ mais on ne sait pas si on y a droit.",
        "gold": {"projet": "residence_principale", "nature": "NEUF",
                 "famille": "famille_avec_enfants", "ptz_evoque": True, "prix": "350 000 €",
                 "apport": "60k", "revenus": NONE},
    },
    {
        "text": "investissement locatif, studio 95k, je paye cash pas de prêt",
        "gold": {"projet": "investissement_locatif", "nature": NONE,
                 "famille": NONE, "ptz_evoque": False, "prix": "95k",
                 # paying cash with no loan means the whole price is the contribution
                 "apport": "95k", "revenus": NONE},
    },
    {
        "text": "je voudrais faire des travaux dans ma maison, isolation et toiture, "
                "40 000 euros de budget, je gagne 3200 par mois",
        # renovating a house you already own means the property is an existing one
        "gold": {"projet": "travaux", "nature": "ANCIEN", "famille": NONE,
                 "ptz_evoque": False, "prix": NONE, "apport": NONE, "revenus": "3200"},
    },
    {
        "text": "j'ai trois crédits conso en cours et ça devient lourd, je voudrais tout "
                "regrouper en une seule mensualité",
        "gold": {"projet": "rachat_credit", "nature": NONE, "famille": NONE,
                 "ptz_evoque": False, "prix": NONE, "apport": NONE, "revenus": NONE},
    },
    {
        "text": "achat maison 200k",
        "gold": {"projet": "residence_principale", "nature": NONE, "famille": NONE,
                 "ptz_evoque": False, "prix": "200k", "apport": NONE, "revenus": NONE},
    },
    {
        "text": "j'hésite encore entre continuer à louer ou acheter, je voulais juste "
                "simuler pour voir ce que ça donne",
        "gold": {"projet": "indecis", "nature": NONE, "famille": NONE,
                 "ptz_evoque": False, "prix": NONE, "apport": NONE, "revenus": NONE},
    },
    {
        "text": "mariés 1 enfant, appart ancien à rénover 180000, on a 0 d'apport, "
                "salaires cumulés 4100 net",
        "gold": {"projet": "residence_principale", "nature": "ANCIEN",
                 "famille": "famille_avec_enfants", "ptz_evoque": False,
                 "prix": "180000", "apport": "0", "revenus": "4100"},
    },
    {
        "text": "j ai 30 ans je veux emprunter sur 25 ans pour un appart neuf a 240k",
        "gold": {"projet": "residence_principale", "nature": "NEUF", "famille": NONE,
                 "ptz_evoque": False, "prix": "240k", "apport": NONE, "revenus": NONE},
    },
    {
        "text": "en concubinage, on achète une maison dans l'ancien, prix 295 000, "
                "on met 45 000 d'apport, nos revenus c'est 5 200 à deux",
        "gold": {"projet": "residence_principale", "nature": "ANCIEN",
                 "famille": "couple_sans_enfants", "ptz_evoque": False,
                 "prix": "295 000", "apport": "45 000", "revenus": "5 200"},
    },
    {
        "text": "VEFA livraison 2028, T3, 265k, primo accédant, apport 20k, célibataire",
        "gold": {"projet": "residence_principale", "nature": "NEUF",
                 "famille": "celibataire", "ptz_evoque": False, "prix": "265k",
                 "apport": "20k", "revenus": NONE},
    },
    {
        "text": "slt je gagne 1900 est ce que je peux acheter",
        # "est-ce que je peux acheter" names no kind of purchase; both readings defensible
        "gold": {"projet": {"residence_principale", NONE}, "nature": NONE, "famille": NONE,
                 "ptz_evoque": False, "prix": NONE, "apport": NONE, "revenus": "1900"},
    },
]

QUESTIONS_STATIC = {
    "projet": Choice(
        instructions="What is this person trying to do",
        criteria={
            "residence_principale": "Buy a home they will live in themselves",
            "investissement_locatif": "Buy a property in order to rent it out",
            "travaux": "Fund renovation or repair work on a property they already own",
            "rachat_credit": "Consolidate or refinance existing loans",
            "indecis": "Still deciding, or only exploring, with no settled project",
            NONE: "They do not say what they want to do",
        },
    ),
    "nature": Choice(
        instructions="Is the property they describe newly built or existing",
        criteria={
            "NEUF": "Newly built, off-plan, or a VEFA purchase",
            "ANCIEN": "An existing, previously owned property, including one needing work",
            NONE: "They do not say which, or there is no property",
        },
    ),
    "famille": Choice(
        instructions="What is this person's household situation",
        criteria={
            "celibataire": "A single person, buying alone",
            "couple_sans_enfants": "A couple with no children mentioned",
            "famille_avec_enfants": "A household with at least one child",
            NONE: "They do not say",
        },
    ),
    "ptz_evoque": Noul(
        instructions="This person explicitly mentions the zero-interest state loan (PTZ), "
                     "whether asking about it, claiming it, or saying they were told about it"
    ),
    "utile": Score(
        instructions="How much of a mortgage simulation form could be filled in from what "
                     "this person wrote",
        criteria=[
            "Almost nothing: no project, no property, no figures",
            "A start: the kind of project is clear but the figures are missing",
            "Most of it: the project plus at least a price or an income",
            "Nearly all of it: project, property type, price, contribution and income",
        ],
    ),
}

ROLES = {
    "prix": "the price of the property being bought",
    "apport": "the personal contribution this person is putting in themselves",
    "revenus": "this person's income per month",
}


def candidates(text: str) -> list[str]:
    found = {m.group(0).strip().rstrip(".,") for m in NUMBER.finditer(text)}
    return sorted(v for v in found if any(ch.isdigit() for ch in v))


def evaluate(predict, repeats: int) -> list[dict]:
    """Grade every blurb through `predict(state, questions)`, which returns a Call.

    Split out from main so P12 can score an entirely different model on the same blurbs,
    the same gold and the same comparison operators. Three times in session 11 the harness
    was the thing that was wrong, so a second model gets the identical harness or the
    numbers are not comparable.
    """
    records = []

    for index, blurb in enumerate(BLURBS):
        options = candidates(blurb["text"])
        criteria = {v: f"The figure written as {v}" for v in options}
        criteria[NONE] = "This person does not state such a figure"

        questions = dict(QUESTIONS_STATIC)
        for role, description in ROLES.items():
            questions[role] = Choice(
                instructions=f"Which figure from this text, if any, is {description}",
                criteria=criteria,
            )

        for run in range(repeats):
            call = predict(blurb["text"], questions)
            for qid, answer in call.answers.items():
                if qid == "utile":
                    continue
                gold = blurb["gold"][qid]
                got = answer.noul >= 0.5 if hasattr(answer, "noul") else answer.choice
                certainty = (max(answer.noul, 1 - answer.noul)
                             if hasattr(answer, "noul") else answer.confidence)
                records.append({
                    "blurb": index, "run": run, "question": qid,
                    "gold": sorted(gold) if isinstance(gold, set) else gold,
                    "predicted": got, "correct": got in gold if isinstance(gold, set) else got == gold, "certainty": certainty,
                    "elapsed_ms": call.elapsed_ms, "input_tokens": call.input_tokens,
                    "candidates": len(options),
                })
            score = call.answers["utile"]
            records.append({
                "blurb": index, "run": run, "question": "utile", "gold": None,
                "predicted": round(score.score, 2), "correct": None,
                "certainty": score.confidence, "elapsed_ms": call.elapsed_ms,
                "input_tokens": call.input_tokens, "candidates": len(options),
            })

    return records


def report(records: list[dict], repeats: int, model: str,
           name: str = "p11-freetext", price: float = 0.042) -> None:
    graded = [r for r in records if r["correct"] is not None]
    by_blurb = defaultdict(list)
    by_question = defaultdict(list)
    for record in graded:
        by_blurb[record["blurb"]].append(record)
        by_question[record["question"]].append(record)

    print(f"{len(BLURBS)} free-text blurbs, {len(QUESTIONS_STATIC) + len(ROLES)} questions "
          f"per call, {repeats} repeats, model {model}\n")

    print(f"{'#':<3} {'blurb':<58} {'score':>6} {'usable':>7}  wrong")
    for index, blurb in enumerate(BLURBS):
        rows = by_blurb[index]
        right = sum(r["correct"] for r in rows)
        utility = [r["predicted"] for r in records
                   if r["blurb"] == index and r["question"] == "utile"]
        wrong = sorted({r["question"] for r in rows if not r["correct"]})
        snippet = blurb["text"][:55].replace("\n", " ")
        print(f"{index:<3} {snippet:<58} {right:>2}/{len(rows):<3} "
              f"{sum(utility) / len(utility):>7.2f}  {', '.join(wrong) or '-'}")

    print(f"\n{'question':<14} {'accuracy':>10} {'certainty':>22}")
    for qid in list(QUESTIONS_STATIC) + list(ROLES):
        if qid == "utile":
            continue
        rows = by_question[qid]
        cs = [r["certainty"] for r in rows]
        print(f"{qid:<14} {sum(r['correct'] for r in rows) / len(rows):>9.0%} "
              f"{min(cs):>13.2f}-{max(cs):.2f}")

    total = len(graded)
    right = sum(r["correct"] for r in graded)
    absent = [r for r in graded if r["gold"] in (NONE, False) or r["gold"] == [NONE]]
    print(f"\n  overall accuracy:        {right}/{total} = {right / total:.1%}")
    print(f"  on fields not stated:    "
          f"{sum(r['correct'] for r in absent)}/{len(absent)} = "
          f"{sum(r['correct'] for r in absent) / len(absent):.1%}")

    wrong_rows = [r for r in graded if not r["correct"]]
    high = [r for r in wrong_rows if r["certainty"] >= 0.9]
    print(f"  wrong and above 0.90:    {len(high)}/{total} = {len(high) / total:.1%}")

    if wrong_rows:
        print("\n--- disagreements ---")
        for record in sorted(wrong_rows, key=lambda r: (r["blurb"], r["question"])):
            print(f"  blurb {record['blurb']} {record['question']}: "
                  f"gold={record['gold']} said={record['predicted']} "
                  f"certainty={record['certainty']:.2f}")

    tokens = [r["input_tokens"] for r in records]
    print(f"\n  input tokens: {min(tokens)}-{max(tokens)} per call")
    print(f"  cost:         {sum(tokens) / len(tokens) * price / 1e6:.8f} USD per submission")
    print(f"  latency:      {latency_report([r['elapsed_ms'] for r in records])}")

    path = dump(name, records)
    print(f"\nwrote {path}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repeats", type=int, default=3)
    args = parser.parse_args()

    c = client()
    report(evaluate(lambda s, q: timed(c, s, q), args.repeats), args.repeats, MODEL)


if __name__ == "__main__":
    main()
