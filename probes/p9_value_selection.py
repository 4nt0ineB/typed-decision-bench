"""P9: can it put the right number in the right field?

The open doubt after P8: Jev tags well, but a prefill feature needs amounts, and Jev does
not generate text. If it cannot produce `192 596,00` then tagging alone is not worth a
form.

It does not have to produce it. The documented pattern is selection, not generation: code
regexes every numeric candidate out of the document, the model picks which candidate fills
which field, and code copies the verbatim string. Two consequences worth stating.

  A selected value cannot be misread. An LLM generating "192 596,00" can drop a digit and
  nothing downstream can tell. Selection returns an index into a list code already holds,
  so transcription error is impossible by construction, not merely unlikely.

  The model never sees the arithmetic. It answers "which of these 22 figures is the
  guarantee fee", which is exactly the judgment P7 showed it is good at, and never "add
  these up", which P7 showed it is not.

The candidate set here is the real one: 22 distinct amounts regexed from the document,
several deliberately confusable (1 014,07 against 1 025,27, the same instalment with and
without insurance; 102 252,61 against 105 478,21, the same credit cost with and without
insurance; 2 796,00 against 2 796,45, the document contradicting itself; 3 225,60 against
1 075,20, the insurance over the full term against the insurance over its first eight years).

One field is absent from the document on purpose, to check the escape hatch is used rather
than the least-wrong candidate picked.
"""

from __future__ import annotations

import argparse
import re
from collections import Counter, defaultdict
from pathlib import Path

from typesafe_sdk import Choice

from _common import MODEL, ROOT, client, dump, latency_report, timed

DOCUMENT = ROOT / "data" / "proposition-redacted.txt"
AMOUNT = re.compile(r"\d{1,3}(?:\s\d{3})*,\d{2}")

ABSENT = "aucune"

# What each schema field means, in the terms the document would state it. Gold is a set:
# where the document genuinely contradicts itself, both readings are defensible and
# scoring one of them wrong would be scoring my opinion rather than the model.
FIELDS: dict[str, tuple[str, set[str]]] = {
    "montant_bien": (
        "the seller's price of the property alone, before any notary, guarantee, "
        "agency or application fees are added",
        {"214 000,00"},
    ),
    "cout_total_projet": (
        "the total cost of the whole operation: the property plus every ancillary fee",
        {"233 596,00"},
    ),
    "montant_apport": (
        "the borrower's own personal contribution, the money not borrowed",
        {"41 000,00"},
    ),
    "montant_pret": (
        "the capital borrowed under the main loan",
        {"192 596,00"},
    ),
    "frais_garantie": (
        "the estimated fee for the loan guarantee",
        {"2 796,00", "2 796,45"},
    ),
    "frais_dossier": (
        "the bank's application handling fee for setting up this loan",
        {"0,00"},
    ),
    "frais_notaire_et_divers": (
        "the combined line covering notary costs, negotiation and miscellaneous items",
        {"16 800,00"},
    ),
    "revenus_mensuels": (
        "the borrower's declared income per month",
        {"3 140,00"},
    ),
    "mensualite_hors_assurance": (
        "the first monthly instalment on the loan, excluding the insurance premium",
        {"1 014,07"},
    ),
    "mensualite_avec_assurance": (
        "the first monthly instalment on the loan, including the insurance premium",
        {"1 025,27"},
    ),
    "montant_interets": (
        "the total interest paid over the life of the loan, insurance excluded",
        {"99 456,16"},
    ),
    "cout_credit_hors_assurance": (
        "the total cost of the credit with the insurance excluded",
        {"102 252,61"},
    ),
    "cout_credit_avec_assurance": (
        "the total cost of the credit with the compulsory insurance included",
        {"105 478,21"},
    ),
    "cout_assurance_duree_totale": (
        "the cost of the loan insurance over the entire term of the loan",
        {"3 225,60"},
    ),
    "cout_assurance_huit_ans": (
        "the cost of the loan insurance over only the first eight years of the loan",
        {"1 075,20"},
    ),
    "prime_assurance_mensuelle": (
        "the insurance premium charged for one single period",
        {"11,20"},
    ),
    "taux_debiteur": (
        "the annual nominal borrowing rate at which interest accrues on the capital",
        {"3,74"},
    ),
    "taeg": (
        "the global effective annual rate, folding in insurance and fees",
        {"4,08"},
    ),
    "taux_assurance": (
        "the effective annual rate of the loan insurance considered on its own",
        {"0,13"},
    ),
    # Deliberately absent: there is no renovation budget anywhere in this document.
    "enveloppe_travaux": (
        "the budget set aside for renovation or building works on the property",
        {ABSENT},
    ),
}


def candidates(text: str) -> list[str]:
    """Every distinct amount in the document, largest first so the list reads sensibly."""
    found = set(AMOUNT.findall(text))
    return sorted(found, key=lambda s: float(s.replace(" ", "").replace(",", ".")), reverse=True)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repeats", type=int, default=5)
    args = parser.parse_args()

    text = DOCUMENT.read_text(encoding="utf-8")
    options = candidates(text)
    criteria = {value: f"The figure {value}" for value in options}
    criteria[ABSENT] = "This figure does not appear anywhere in the document"

    questions = {
        field: Choice(
            instructions=f"Which of these figures, exactly as written in the document, is "
                         f"{description}",
            criteria=criteria,
        )
        for field, (description, _) in FIELDS.items()
    }

    print(f"{len(options)} numeric candidates regexed from the document, "
          f"plus an explicit '{ABSENT}' escape hatch")
    print(f"{len(questions)} fields asked in one call, {args.repeats} repeats, model {MODEL}\n")

    c = client()
    runs, records = [], []
    for run in range(args.repeats):
        call = timed(c, text, questions)
        runs.append(call)
        for field, answer in call.answers.items():
            records.append({
                "run": run, "field": field, "gold": sorted(FIELDS[field][1]),
                "predicted": answer.choice, "correct": answer.choice in FIELDS[field][1],
                "p_chosen": answer.probabilities[answer.choice],
                "confidence": answer.confidence, "elapsed_ms": call.elapsed_ms,
                "input_tokens": call.input_tokens,
            })

    by_field = defaultdict(list)
    for record in records:
        by_field[record["field"]].append(record)

    print(f"{'field':<30} {'expected':<22} {'selected':<14} {'':<3} {'p':>10} {'conf':>11}")
    for field in FIELDS:
        rows = by_field[field]
        said = Counter(r["predicted"] for r in rows)
        correct = all(r["correct"] for r in rows)
        mark = "ok " if correct else ("~  " if any(r["correct"] for r in rows) else "X  ")
        ps = [r["p_chosen"] for r in rows]
        cs = [r["confidence"] for r in rows]
        expected = "/".join(sorted(FIELDS[field][1]))
        rendered = " ".join(f"{v}({n})" if len(said) > 1 else v for v, n in said.most_common())
        print(f"{field:<30} {expected:<22} {rendered:<14} {mark} "
              f"{min(ps):.2f}-{max(ps):.2f} {min(cs):.2f}-{max(cs):.2f}")

    total = len(records)
    right = sum(r["correct"] for r in records)
    print(f"\n  accuracy: {right}/{total} = {right / total:.1%}")
    print(f"  latency:  {latency_report([r.elapsed_ms for r in runs])}")
    print(f"  cost:     {runs[0].input_tokens * 0.042 / 1e6:.6f} USD per document, "
          f"{len(questions)} fields")

    wrong = [r for r in records if not r["correct"]]
    if wrong:
        print("\n--- wrong selections ---")
        for record in wrong:
            print(f"  {record['field']}: expected {record['gold']} "
                  f"got {record['predicted']} at p={record['p_chosen']:.2f} "
                  f"conf={record['confidence']:.2f}")

    path = dump("p9-value-selection", records)
    print(f"\nwrote {path}")


if __name__ == "__main__":
    main()
