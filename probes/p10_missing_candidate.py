"""P10: what happens when the regex missed the right answer?

P9 showed that a field genuinely absent from the document gets `aucune`, correctly and at
a flagged confidence. That is the easy case.

The dangerous case is different and untested: the value IS in the document, the model can
read it, but it is not in the candidate list because the regex did not match its format.
`169.000 EUR`, `169000`, a figure split across a line break. The model cannot select what
it was not offered.

Two possible behaviours, and the whole risk profile of a selection-based prefill turns on
which one happens:

  GOOD    probability spreads, or `aucune` wins. Confidence gating catches the gap for
          free, the field is left empty, the user is told. Regex coverage becomes a
          quality problem.

  BAD     it confidently picks the least-wrong remaining candidate. The field is silently
          prefilled with a plausible wrong number from the same document. Regex coverage
          becomes a correctness requirement, and every format the regex misses is a silent
          data-corruption bug.

Method: take P9 exactly as it stands, and for each field remove that field's correct
answer from the candidate list. Everything else is unchanged, including the document,
which still contains the right value in plain sight.
"""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict

from typesafe_sdk import Choice

from _common import MODEL, client, dump, latency_report, timed
from p9_value_selection import ABSENT, DOCUMENT, FIELDS, candidates

# The ones worth the calls: a field whose correct value has a near-twin still on the menu
# is where a confident wrong answer would hurt most.
SUBSET = [
    "montant_pret", "montant_bien", "revenus_mensuels", "mensualite_avec_assurance",
    "cout_credit_avec_assurance", "taux_debiteur", "frais_dossier", "cout_total_projet",
]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repeats", type=int, default=3)
    args = parser.parse_args()

    text = DOCUMENT.read_text(encoding="utf-8")
    options = candidates(text)
    c = client()
    records = []

    print(f"{len(options)} candidates, correct answer removed one field at a time, "
          f"{args.repeats} repeats, model {MODEL}\n")
    print(f"{'field':<28} {'removed':<13} {'now picks':<26} {'p':>10} {'conf':>11}  verdict")

    for field in SUBSET:
        description, gold = FIELDS[field]
        # Everything the regex would have produced, minus the answer it "missed".
        remaining = {v: f"The figure {v}" for v in options if v not in gold}
        remaining[ABSENT] = "This figure does not appear anywhere in the document"

        question = {
            field: Choice(
                instructions=f"Which of these figures, exactly as written in the document, "
                             f"is {description}",
                criteria=remaining,
            )
        }

        picks, ps, cs = Counter(), [], []
        for run in range(args.repeats):
            call = timed(c, text, question)
            answer = call.answers[field]
            picks[answer.choice] += 1
            ps.append(answer.probabilities[answer.choice])
            cs.append(answer.confidence)
            records.append({
                "field": field, "run": run, "removed": sorted(gold),
                "predicted": answer.choice, "abstained": answer.choice == ABSENT,
                "p_chosen": answer.probabilities[answer.choice],
                "confidence": answer.confidence, "elapsed_ms": call.elapsed_ms,
            })

        abstained = picks[ABSENT] == args.repeats
        # The schema's policy: >= 0.9 prefills silently, which is where a wrong answer hurts.
        silent = not abstained and min(cs) >= 0.9
        verdict = "SAFE, abstains" if abstained else ("SILENT WRONG" if silent else "flagged")
        rendered = " ".join(f"{v}({n})" for v, n in picks.most_common())
        print(f"{field:<28} {'/'.join(sorted(gold)):<13} {rendered:<26} "
              f"{min(ps):.2f}-{max(ps):.2f} {min(cs):.2f}-{max(cs):.2f}  {verdict}")

    by_field = defaultdict(list)
    for record in records:
        by_field[record["field"]].append(record)

    abstaining = [f for f in SUBSET if all(r["abstained"] for r in by_field[f])]
    silent_wrong = [
        f for f in SUBSET
        if not all(r["abstained"] for r in by_field[f])
        and min(r["confidence"] for r in by_field[f]) >= 0.9
    ]

    print(f"\n  abstained ('{ABSENT}') on every call: {len(abstaining)}/{len(SUBSET)}  {abstaining}")
    print(f"  confidently wrong above 0.90:         {len(silent_wrong)}/{len(SUBSET)}  {silent_wrong}")
    print(f"\n  latency: {latency_report([r['elapsed_ms'] for r in records])}")

    if not silent_wrong:
        print("\n  => a regex gap degrades to an empty flagged field, not a wrong number.")
        print("     Candidate coverage is a quality problem, not a correctness requirement.")
    else:
        print("\n  => a regex gap can silently prefill a plausible wrong number.")
        print("     Candidate coverage becomes a correctness requirement for every format.")

    path = dump("p10-missing-candidate", records)
    print(f"\nwrote {path}")


if __name__ == "__main__":
    main()
