"""P16: does asking one question shift another?

The docs say the questions of a call are evaluated in parallel, and that question ids are not
even passed to the model. The shorthand "they are evaluated together" suggests the opposite:
that adding a question could change the answers to the others. That had never been checked
here, and the article's sentence depends on it.

Protocol: same state, same eight questions, two conditions.

  seule      each question in its own call
  groupee    all eight questions in a single call

P1 showed that an identical input returns a top probability that moves by 0.1 from one call
to the next, so comparing two calls would prove nothing. Hence repetitions on both sides and
a per-question permutation test: the question is not "are the numbers equal" but "does the
gap exceed the model's own noise".
"""
from __future__ import annotations

import argparse
import statistics
from collections import Counter

from typesafe_sdk import Choice, Noul, Score

from _common import answer_distribution, client, decided, dump, latency_report, timed
from p1_determinism import permutation_test

STATE = (
    "Message client : suite au dégât des eaux du 3 mars dans la cuisine, j'ai envoyé le "
    "constat et deux devis de plombier il y a trois semaines. Personne ne m'a rappelé, et "
    "le prélèvement de la prime est passé normalement le 5. Je veux savoir où en est le "
    "dossier avant de saisir le médiateur."
)

QUESTIONS = {
    "service": Choice(
        instructions="Quel service doit traiter ce message",
        criteria={
            "sinistre": "Déclaration ou suivi d'un dommage",
            "facturation": "Prime, prélèvement, remboursement",
            "contrat": "Souscription, modification, résiliation",
            "reclamation": "Mécontentement formel sur le service rendu",
        },
    ),
    "urgence": Score(
        instructions="Degré d'urgence pour la personne qui écrit",
        criteria=["Aucune urgence", "Gêne réelle", "Situation bloquée depuis longtemps"],
    ),
    "menace_mediateur": Noul(instructions="La personne évoque un recours extérieur"),
    "piece_fournie": Noul(instructions="La personne dit avoir déjà envoyé des documents"),
    "sinistre_declare": Noul(instructions="Un sinistre est mentionné"),
    "delai_evoque": Noul(instructions="La personne fait état d'un délai d'attente"),
    "paiement_a_jour": Noul(instructions="La personne indique être à jour de ses paiements"),
    "demande_rappel": Noul(instructions="La personne demande un retour de la part de l'assureur"),
}


def collect(c, questions: dict, runs: int) -> tuple[list[dict], list[dict]]:
    rows, records = [], []
    for run in range(runs):
        call = timed(c, STATE, questions)
        for qid, answer in call.answers.items():
            distribution = answer_distribution(answer)
            rows.append({
                "question": qid,
                "run": run,
                "p_max": max(distribution.values()),
                "decision": str(decided(answer)),
                "posees": len(questions),
            })
        records.append(call.archive() | {"posees": len(questions), "run": run})
    return rows, records


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--runs", type=int, default=15)
    args = parser.parse_args()

    c = client()
    archive = []

    grouped, records = collect(c, QUESTIONS, args.runs)
    archive += records

    alone = []
    for qid, question in QUESTIONS.items():
        rows, records = collect(c, {qid: question}, args.runs)
        alone += rows
        archive += records

    print(f"\nsame state, {len(QUESTIONS)} questions, {args.runs} runs per condition\n")
    print(f"{'question':<18} {'alone':>18} {'grouped':>18} {'gap':>7} {'p-value':>9}  verdict")
    verdicts = []
    for qid in QUESTIONS:
        solo = [r["p_max"] for r in alone if r["question"] == qid]
        group = [r["p_max"] for r in grouped if r["question"] == qid]
        difference, p_value = permutation_test(solo, group)
        # A flipped decision is worse than a moving probability. Check both: the probability
        # can stay within noise while the decision flips.
        solo_choice = Counter(r["decision"] for r in alone if r["question"] == qid).most_common(1)[0]
        group_choice = Counter(r["decision"] for r in grouped if r["question"] == qid).most_common(1)[0]
        stable = solo_choice[0] == group_choice[0]
        verdict = ("identical" if p_value >= 0.05 and stable
                   else "DECISION CHANGED" if not stable
                   else "gap beyond noise")
        verdicts.append(verdict)
        print(f"{qid:<18} "
              f"{statistics.mean(solo):>8.3f} ({solo_choice[0][:7]:>7}) "
              f"{statistics.mean(group):>8.3f} ({group_choice[0][:7]:>7}) "
              f"{difference:>7.3f} {p_value:>9.4f}  {verdict}")

    print(f"\n  questions whose decision changes: "
          f"{sum(v == 'DECISION CHANGED' for v in verdicts)}/{len(QUESTIONS)}")
    print(f"  questions whose probability moves beyond noise: "
          f"{sum(v == 'gap beyond noise' for v in verdicts)}/{len(QUESTIONS)}")

    solo_tokens = [r["raw"]["usage"]["input_tokens"] for r in archive if r["posees"] == 1]
    group_tokens = [r["raw"]["usage"]["input_tokens"] for r in archive if r["posees"] > 1]
    print(f"\n  tokens: {statistics.mean(group_tokens):.0f} for a grouped call, "
          f"{statistics.mean(solo_tokens):.0f} on average per isolated call, "
          f"{len(QUESTIONS) * statistics.mean(solo_tokens):.0f} for the eight separately")
    print(f"  grouped latency:  {latency_report([r['elapsed_ms'] for r in archive if r['posees'] > 1])}")
    print(f"  isolated latency: {latency_report([r['elapsed_ms'] for r in archive if r['posees'] == 1])}")

    print(f"\nwrote {dump('p16-independance', archive)}")


if __name__ == "__main__":
    main()
