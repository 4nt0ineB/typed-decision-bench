"""P5: how far does accuracy fall as irrelevant state grows?

The jaggedness page states the weakness and gives no number:

    "Large Irrelevant State: Accuracy falls as the state grows with content unrelated to
     the decision."

So measure the curve. A short loan-application passage with labels known by construction
is buried in a growing amount of real prose, and the same four questions are asked at every
size. Gold cannot drift, the task cannot get harder, the only variable is how much
unrelated text surrounds the answer.

Filler is deliberately *topically adjacent*, not random noise: French administrative text
on property and gifts, English encyclopaedia text on mortgages and property. Irrelevant
text that looks relevant is the realistic case and the harder one.

Two sweeps:
  length    needle in the middle, filler 0 -> the measured 127k-char ceiling
  position  needle at start / middle / end at one large filler size

Position bias is not mentioned anywhere in the jaggedness list, so either it was not
looked for or it is not there.
"""

from __future__ import annotations

import argparse
import random
import statistics
from collections import defaultdict

from typesafe_sdk import Choice, Noul

from _common import ROOT, client, dump, fr_corpus_dir, latency_report, timed

EN_CORPUS = ROOT / "data" / "en-corpus.txt"

FILLER_SIZES = [0, 2_000, 8_000, 32_000, 64_000, 120_000]
POSITION_SIZE = 64_000

PURPOSES = {
    "residence_principale": (
        ["acquérir le logement qu'il occupera lui-même",
         "acheter la maison dans laquelle il compte habiter"],
        ["buy the home they will live in themselves",
         "purchase the house they intend to occupy"],
    ),
    "investissement_locatif": (
        ["acquérir un appartement qu'il compte mettre en location",
         "acheter un studio destiné à être loué à des étudiants"],
        ["buy a flat they intend to rent out",
         "purchase a studio to let to students"],
    ),
    "travaux": (
        ["refaire la toiture et l'isolation de sa maison",
         "financer la rénovation complète de la cuisine et de la salle de bains"],
        ["redo the roof and insulation of their house",
         "fund a full renovation of the kitchen and bathroom"],
    ),
    "rachat_credit": (
        ["regrouper ses crédits en cours en une seule mensualité",
         "racheter trois prêts existants pour alléger sa charge mensuelle"],
        ["consolidate their outstanding loans into a single monthly payment",
         "refinance three existing loans to reduce the monthly burden"],
    ),
}

EMPLOYMENTS = {
    "cdi": (["est salarié en contrat à durée indéterminée depuis six ans"],
            ["has been on a permanent employment contract for six years"]),
    "cdd": (["enchaîne les contrats à durée déterminée depuis deux ans"],
            ["has been working on successive fixed-term contracts for two years"]),
    "independant": (["exerce en libéral et déclare ses revenus en BNC"],
                    ["is self-employed and files as a sole trader"]),
    "retraite": (["a fait valoir ses droits à la retraite l'an dernier"],
                 ["retired last year and lives on a pension"]),
}

COSIGNER = (
    (["Sa sœur se porte caution solidaire sur l'intégralité du prêt."],
     ["Their sister stands as joint guarantor for the whole loan."]),
    (["Le dossier est présenté sans caution ni co-emprunteur."],
     ["The file is submitted with no guarantor and no co-borrower."]),
)

BANDS = [("moins_150k", 80_000, 149_000), ("150k_300k", 150_000, 299_000),
         ("300k_500k", 300_000, 499_000), ("plus_500k", 500_000, 900_000)]

TEMPLATE_FR = (
    "Dossier {ref}. Le demandeur sollicite un financement de {amount} euros pour "
    "{purpose}. Il {employment}. {cosigner} L'agence de {city} a transmis le dossier "
    "au comité d'engagement."
)
TEMPLATE_EN = (
    "File {ref}. The applicant is seeking financing of EUR {amount} in order to "
    "{purpose}. The borrower {employment}. {cosigner} The {city} branch has passed the "
    "file to the credit committee."
)

CITIES = ["Nantes", "Rennes", "Bordeaux", "Lille", "Toulouse", "Strasbourg"]

QUESTIONS = {
    "purpose": Choice(
        instructions="What is the applicant asking to finance",
        criteria={
            "residence_principale": "Buying a home the applicant will live in themselves",
            "investissement_locatif": "Buying a property in order to rent it out",
            "travaux": "Renovation or repair work on a property already owned",
            "rachat_credit": "Consolidating or refinancing existing loans",
        },
    ),
    "employment": Choice(
        instructions="What is the applicant's employment situation",
        criteria={
            "cdi": "Permanent employment contract",
            "cdd": "Fixed-term or temporary employment contract",
            "independant": "Self-employed, freelance or running their own business",
            "retraite": "Retired, living on a pension",
        },
    ),
    "amount": Choice(
        instructions="How much financing is being requested",
        criteria={
            "moins_150k": "Less than 150,000 euros",
            "150k_300k": "Between 150,000 and 300,000 euros",
            "300k_500k": "Between 300,000 and 500,000 euros",
            "plus_500k": "More than 500,000 euros",
        },
    ),
    "cosigner": Noul(instructions="A guarantor or co-borrower is backing this application"),
}


def make_needles(count: int, seed: int) -> list[dict]:
    rng = random.Random(seed)
    needles = []
    for index in range(count):
        purpose = rng.choice(list(PURPOSES))
        employment = rng.choice(list(EMPLOYMENTS))
        band, low, high = rng.choice(BANDS)
        amount = rng.randrange(low, high, 1_000)
        has_cosigner = rng.random() < 0.5
        which = 0 if has_cosigner else 1
        slot = rng.randrange(2)
        # Unique by construction: P6 asks questions that name one file among many, so two
        # files sharing a reference would silently make a question unanswerable.
        fields = {
            "ref": f"{4100 + index * 7}",
            "city": rng.choice(CITIES),
            "amount": f"{amount:,}".replace(",", " "),
        }
        needles.append(
            {
                "index": index,
                "ref": fields["ref"],
                "fr": TEMPLATE_FR.format(
                    purpose=PURPOSES[purpose][0][slot % len(PURPOSES[purpose][0])],
                    employment=EMPLOYMENTS[employment][0][0],
                    cosigner=COSIGNER[which][0][0],
                    **fields,
                ),
                "en": TEMPLATE_EN.format(
                    purpose=PURPOSES[purpose][1][slot % len(PURPOSES[purpose][1])],
                    employment=EMPLOYMENTS[employment][1][0],
                    cosigner=COSIGNER[which][1][0],
                    **fields | {"amount": f"{amount:,}"},
                ),
                "gold": {"purpose": purpose, "employment": employment, "amount": band},
                "gold_cosigner": has_cosigner,
            }
        )
    return needles


def fillers() -> dict[str, str]:
    french = "\n\n".join(
        p.read_text(encoding="utf-8") for p in sorted(fr_corpus_dir().glob("*.md"))
    )
    if not EN_CORPUS.exists():
        raise SystemExit(f"missing {EN_CORPUS}. Run fetch_en_corpus.py first.")
    return {"fr": french, "en": EN_CORPUS.read_text(encoding="utf-8")}


def build(needle: str, filler: str, size: int, position: str) -> str:
    if size == 0:
        return needle
    body = filler[:size]
    if position == "start":
        return f"{needle}\n\n{body}"
    if position == "end":
        return f"{body}\n\n{needle}"
    cut = len(body) // 2
    return f"{body[:cut]}\n\n{needle}\n\n{body[cut:]}"


def score(records: list[dict], qid: str) -> tuple[float, float]:
    rows = [r for r in records if r["question"] == qid]
    return (
        sum(r["correct"] for r in rows) / len(rows),
        statistics.mean(r["p_chosen"] for r in rows),
    )


def run(c, needles: list[dict], filler: str, lang: str, size: int, position: str) -> list[dict]:
    records = []
    for needle in needles:
        state = build(needle[lang], filler, size, position)
        call = timed(c, state, QUESTIONS)
        common = {"lang": lang, "size": size, "position": position,
                  "needle": needle["index"], "input_tokens": call.input_tokens,
                  "elapsed_ms": call.elapsed_ms}
        for qid, answer in call.answers.items():
            if qid == "cosigner":
                predicted = answer.noul >= 0.5
                p_chosen = answer.noul if predicted else 1 - answer.noul
                gold = needle["gold_cosigner"]
            else:
                predicted = answer.choice
                p_chosen = answer.probabilities[answer.choice]
                gold = needle["gold"][qid]
            records.append(common | {"question": qid, "gold": gold, "predicted": predicted,
                                     "correct": predicted == gold, "p_chosen": p_chosen})
    return records


def table(title: str, records: list[dict], key: str, order: list) -> None:
    by = defaultdict(list)
    for record in records:
        by[record[key]].append(record)

    questions = ["purpose", "employment", "amount", "cosigner"]
    print(f"\n{title}")
    header = f"{key:>10} {'tokens':>8} {'ms':>6}  " + "".join(f"{q:>14}" for q in questions)
    print(header)
    for value in order:
        rows = by.get(value)
        if not rows:
            continue
        cells = "".join(
            f"{score(rows, q)[0]:>8.0%}{score(rows, q)[1]:>6.2f}" for q in questions
        )
        label = f"{value:,}" if isinstance(value, int) else str(value)
        print(f"{label:>10} {statistics.mean(r['input_tokens'] for r in rows):>8,.0f} "
              f"{statistics.median(r['elapsed_ms'] for r in rows):>6.0f}  {cells}")
    print(f"{'':>26}  " + "".join(f"{'acc    p':>14}" for _ in questions))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--needles", type=int, default=24)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    needles = make_needles(args.needles, args.seed)
    filler = fillers()
    c = client()
    records = []

    print(f"{args.needles} needles, filler sizes {FILLER_SIZES}, two languages")
    print(f"example needle:\n  {needles[0]['fr']}\n  gold={needles[0]['gold']} "
          f"cosigner={needles[0]['gold_cosigner']}\n")

    for lang in ("en", "fr"):
        for size in FILLER_SIZES:
            records += run(c, needles, filler[lang], lang, size, "middle")
        for position in ("start", "end"):
            records += run(c, needles, filler[lang], lang, POSITION_SIZE, position)

    for lang in ("en", "fr"):
        rows = [r for r in records if r["lang"] == lang and r["position"] == "middle"]
        table(f"=== {lang.upper()}: accuracy vs filler size (needle in middle) ===",
              rows, "size", FILLER_SIZES)

    for lang in ("en", "fr"):
        rows = [r for r in records
                if r["lang"] == lang and r["size"] == POSITION_SIZE]
        table(f"=== {lang.upper()}: accuracy vs needle position at {POSITION_SIZE:,} chars ===",
              rows, "position", ["start", "middle", "end"])

    print(f"\nlatency: {latency_report([r['elapsed_ms'] for r in records])}")
    path = dump("p5-haystack", records)
    print(f"wrote {path}")


if __name__ == "__main__":
    main()
