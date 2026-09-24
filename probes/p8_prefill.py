"""P8: prefilling a real form from a real French bank document.

The target fields come from an independently authored domain schema: written before this
document existed, by someone other than the person writing these questions, for an
unrelated purpose. That matters more than it sounds. Self-authored gold labels are
circular, because a misreading gets encoded into the question and the answer at once and
nothing in the experiment can detect it. P2 only caught a backwards criterion because the
public dataset disagreed. Here the field list and the known-hard cases (insurance on
initial vs outstanding capital, total vs partial deferral, monthly vs annual figure, one
subsidised loan line or two) were fixed in advance and independently.

Division of labour, which the schema's own doctrine and P7 arrived at separately:
numbers are parsed in code, the model answers only bounded interpretation questions. So
nothing here asks Jev to read 192 596,00 off the page. A regex does that. These are the
seventeen questions a regex cannot answer.

Instructions and criteria are in English against a French document: P2 measured that
condition at English token cost and no accuracy loss, so it is the cheap configuration.

The headline metric is not accuracy. It is SILENT-WRONG RATE under the schema's own
threshold policy: >= 0.9 prefill silently, 0.6-0.9 prefill flagged, < 0.6 leave empty.
A flagged error costs a glance. A silently prefilled error on a 25-year commitment costs
a decision the user never knows to question.

Second metric, straight out of P1: repeated identical calls, counting how many fields
cross a policy boundary. A field that prefills silently for one user and flags for the
next, with no code change, is a design defect in the policy rather than in the model.
"""

from __future__ import annotations

import argparse
import statistics
from collections import Counter, defaultdict
from pathlib import Path

from typesafe_sdk import Choice, Noul, Score

from _common import MODEL, ROOT, client, dump, latency_report, timed

DOCUMENT = ROOT / "data" / "proposition-redacted.txt"

# The schema's policy, verbatim.
AUTO, FLAG = 0.9, 0.6

QUESTIONS = {
    # --- controls: unambiguous in the document, expect these to be perfect
    "nature_bien": Choice(
        instructions="Is the property being purchased new or pre-owned",
        criteria={"NEUF": "Newly built property", "ANCIEN": "Existing, previously owned property"},
    ),
    "type_taux": Choice(
        instructions="Is the main loan's interest rate fixed, variable or mixed",
        criteria={"FIXE": "Fixed for the whole term", "VARIABLE": "Revisable or indexed",
                  "MIXTE": "Fixed for a first period then variable"},
    ),

    # --- interpretation: the document states something, but which schema field it fills
    #     is a judgment
    "montant_bien_scope": Choice(
        instructions="The document states a figure labelled as the amount of the project. "
                     "Does that figure cover the seller's price for the property alone, or "
                     "the price together with the ancillary costs such as notary, guarantee "
                     "and application fees",
        criteria={
            "prix_seul": "The seller's price for the property alone, ancillary costs listed "
                         "separately and added on top",
            "prix_et_frais": "The price and the ancillary costs together, as one total",
        },
    ),
    "revenus_periodicite": Choice(
        instructions="The borrower's declared income appears in this document. Over what "
                     "period is that income figure expressed",
        criteria={"mensuel": "Per month", "annuel": "Per year", "hebdomadaire": "Per week"},
    ),
    "taux_nominal_figure": Choice(
        instructions="Several different percentage rates appear for the main loan. Which one "
                     "is the annual nominal borrowing rate, the rate at which interest "
                     "actually accrues on the capital",
        criteria={
            "taux_debiteur": "The annual debtor rate, stated as fixed, excluding insurance "
                             "and fees",
            "taeg": "The global effective annual rate, which folds in insurance and fees",
            "taea": "The effective annual rate of the insurance alone",
        },
    ),
    "assurance_mode": Choice(
        instructions="The loan insurance premium is described in this document. Is the "
                     "premium computed on the capital originally borrowed, so that it stays "
                     "the same every month for the whole term, or on the capital still "
                     "outstanding, so that it falls as the loan is repaid",
        criteria={
            "CAPITAL_INITIAL": "On the initial borrowed capital: a constant premium every "
                               "month, total cost equal to the monthly premium times the "
                               "number of months",
            "CAPITAL_RESTANT_DU": "On the outstanding balance: a premium that decreases over "
                                  "the life of the loan",
        },
    ),
    "quotite_globale": Choice(
        instructions="Summing the insured share across every insured person on this "
                     "application, what is the total insured share",
        criteria={"100": "100 % in total, typically a single borrower covered in full",
                  "200": "200 % in total, typically two borrowers each covered in full",
                  "autre": "Some other total"},
    ),

    # --- absence and traps: the answer is no, and saying yes is the expensive failure
    "frais_garantie_coherent": Noul(
        instructions="The estimated guarantee fee is stated with the same value everywhere "
                     "it appears in this document"
    ),
    "frais_notaire_separable": Noul(
        instructions="The notary fees are given as their own separate figure, not merged "
                     "into a combined line with other costs such as agency or miscellaneous "
                     "fees"
    ),
    "frais_dossier_zero_explicite": Noul(
        instructions="The application fee is explicitly stated as an amount of zero, as "
                     "opposed to not being mentioned at all"
    ),
    "a_enveloppe_travaux": Noul(
        instructions="A renovation or works budget is part of this financing plan"
    ),
    "a_differe": Noul(
        instructions="The main loan includes a deferral or grace period during which no "
                     "capital is amortised"
    ),
    "a_pret_complementaire": Noul(
        instructions="The financing includes a second loan alongside the main one"
    ),
    "a_ptz": Noul(
        instructions="The financing includes a zero-interest state-subsidised loan (PTZ)"
    ),
    "a_coemprunteur": Noul(
        instructions="There is a co-borrower or a guarantor besides the applicant on this "
                     "application"
    ),
    "lissage": Noul(
        instructions="The arrangement provides for smoothing several loans together so that "
                     "the borrower pays one constant total instalment. A clause allowing the "
                     "instalment to be adjusted up or down later is a different thing and "
                     "does not count"
    ),

    # --- derived: two figures two hundred lines apart, near a regulatory threshold
    "effort": Score(
        instructions="Comparing the monthly instalment including insurance against the "
                     "borrower's declared monthly income, how heavy is the debt burden",
        criteria=[
            "Light: the instalment is under a fifth of declared monthly income",
            "Moderate: the instalment is between a fifth and 30 % of declared monthly income",
            "Heavy: the instalment is between 30 % and the 35 % regulatory ceiling",
            "Over the ceiling: the instalment exceeds 35 % of declared monthly income",
        ],
    ),
}

# Read by hand from the document, against the schema's field list. Stated here so the
# disagreements are auditable rather than buried in a percentage.
GOLD = {
    "nature_bien": "ANCIEN",
    "type_taux": "FIXE",
    "montant_bien_scope": "prix_seul",
    "revenus_periodicite": "mensuel",
    "taux_nominal_figure": "taux_debiteur",
    "assurance_mode": "CAPITAL_INITIAL",
    "quotite_globale": "100",
    "frais_garantie_coherent": False,
    "frais_notaire_separable": False,
    "frais_dossier_zero_explicite": True,
    "a_enveloppe_travaux": False,
    "a_differe": False,
    "a_pret_complementaire": False,
    "a_ptz": False,
    "a_coemprunteur": False,
    "lissage": False,
    "effort": 2.0,
}

# Why each gold value is what it is, for anything a reader could reasonably dispute.
NOTES = {
    "montant_bien_scope": "214 000 + 2 796 + 0 + 16 800 = 233 596, stated as TOTAL",
    "assurance_mode": "11,20/month x 288 months = 3 225,60 exactly, so the premium is constant",
    "frais_garantie_coherent": "2 796,00 in the financing plan, 2 796,45 in the cost section",
    "frais_notaire_separable": "'Autres (notaire, negociation, divers) 16 800,00' is one line",
    "lissage": "the document offers a modulation clause, which is not lissage",
    "effort": "1 025,27 / 3 140 = 32.7 %, inside the 30-35 % band",
}


def certainty(answer) -> float:
    """One number per answer on the policy's scale.

    A Noul returns no confidence field, so the distance of its probability from 0.5 is
    the only thing the policy can read. Stated rather than hidden: it is not the same
    quantity as a Choice's confidence and should not be compared across primitives.
    """
    if hasattr(answer, "noul"):
        return max(answer.noul, 1 - answer.noul)
    return answer.confidence


def value(answer):
    if hasattr(answer, "noul"):
        return answer.noul >= 0.5
    if hasattr(answer, "choice"):
        return answer.choice
    return answer.score


def matches(answer, gold) -> bool:
    """A Score is a probability-weighted position and is documented to land between
    levels, so 1.98 is level 2. Comparing it for equality scores the harness, not the
    model: the same mistake P1's first verdict made with a Noul float."""
    got = value(answer)
    if isinstance(gold, float) and isinstance(got, float):
        return round(got) == round(gold)
    return got == gold


def band(c: float) -> str:
    return "auto" if c >= AUTO else ("flag" if c >= FLAG else "empty")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repeats", type=int, default=5)
    args = parser.parse_args()

    if not DOCUMENT.exists():
        raise SystemExit(f"missing {DOCUMENT}. Run redact_proposal.py first.")
    state = DOCUMENT.read_text(encoding="utf-8")

    c = client()
    runs, records = [], []
    for run in range(args.repeats):
        call = timed(c, state, QUESTIONS)
        runs.append(call)
        for qid, answer in call.answers.items():
            records.append({
                "run": run, "question": qid, "gold": GOLD[qid], "predicted": value(answer),
                "correct": matches(answer, GOLD[qid]), "certainty": certainty(answer),
                "band": band(certainty(answer)), "elapsed_ms": call.elapsed_ms,
                "input_tokens": call.input_tokens,
            })

    by_question = defaultdict(list)
    for record in records:
        by_question[record["question"]].append(record)

    print(f"document: {len(state):,} chars, {runs[0].input_tokens:,} tokens, "
          f"{len(QUESTIONS)} questions in one call, {args.repeats} repeats, model {MODEL}\n")

    print(f"{'field':<28} {'gold':<16} {'said':<16} {'':<3} {'certainty':>10} {'band':>6} {'flips':>6}")
    for qid in QUESTIONS:
        rows = by_question[qid]
        said = Counter(str(r["predicted"]) for r in rows).most_common(1)[0][0]
        certainties = [r["certainty"] for r in rows]
        bands = {r["band"] for r in rows}
        correct = all(r["correct"] for r in rows)
        mixed = len({r["correct"] for r in rows}) > 1
        mark = "ok " if correct else ("~  " if mixed else "X  ")
        print(f"{qid:<28} {str(GOLD[qid]):<16} {said:<16} {mark} "
              f"{min(certainties):.2f}-{max(certainties):.2f}".ljust(85)
              + f"{'/'.join(sorted(bands)):>8} {'YES' if len(bands) > 1 else '.':>6}")

    print("\n--- under the schema's policy (>=0.90 silent, 0.60-0.90 flagged, <0.60 empty) ---")
    tally = Counter()
    for record in records:
        tally[(record["band"], record["correct"])] += 1
    total = len(records)
    for band_name in ("auto", "flag", "empty"):
        right, wrong = tally[(band_name, True)], tally[(band_name, False)]
        if right or wrong:
            print(f"  {band_name:<6} {right + wrong:>3} answers   {right:>3} right   {wrong:>3} wrong")

    silent_wrong = tally[("auto", False)]
    print(f"\n  SILENT-WRONG RATE: {silent_wrong}/{total} answers "
          f"({silent_wrong / total:.1%}) prefilled above {AUTO} with the wrong value")

    flipping = [q for q in QUESTIONS if len({r["band"] for r in by_question[q]}) > 1]
    print(f"  fields crossing a policy boundary across identical calls: "
          f"{len(flipping)}/{len(QUESTIONS)}" + (f"  {flipping}" if flipping else ""))

    accuracy = sum(r["correct"] for r in records) / total
    print(f"\n  accuracy: {accuracy:.1%} over {total} answers")
    print(f"  latency:  {latency_report([r.elapsed_ms for r in runs])}")
    print(f"  cost:     {runs[0].input_tokens * 0.042 / 1e6:.6f} USD per document")

    print("\n--- disagreements ---")
    any_wrong = False
    for qid in QUESTIONS:
        rows = by_question[qid]
        if all(r["correct"] for r in rows):
            continue
        any_wrong = True
        said = Counter(str(r["predicted"]) for r in rows).most_common()
        print(f"  {qid}: gold={GOLD[qid]} said={said}")
        if qid in NOTES:
            print(f"      gold rationale: {NOTES[qid]}")
    if not any_wrong:
        print("  none")

    path = dump("p8-prefill", records)
    print(f"\nwrote {path}")


if __name__ == "__main__":
    main()
