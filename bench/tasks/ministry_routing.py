"""Routing a député's written question to the ministry that must answer it: 36-way Choice, French.

Data: data/questions-ecrites/ (see its README), written questions published from
2026-03-01 and answered under the Lecornu II government, after the training data of most
models here. Gold is the government's own final routing: the ministry the question was
attributed to when its reply was published. Every question has the same 36 options, the
ministries in office, described by their official titles.

About a quarter of the questions were transferred: the député addressed one ministry and
the government moved the question to another, within the same government. On those, the
député's own choice is wrong by definition, which gives a human baseline, and a model that
just follows the addressee fails too.

Two conditions, same items and gold:
  RAW       the text as published, opening on the minister the député addressed
  REDACTED  that addressee replaced by "le Gouvernement": routing on content alone

The gap between them measures how much a model leans on the député's choice. On RAW, an
answer that differs from the addressee is a flagged misroute; report() scores it as one.

report() also scores the ministry family: a delegated minister counts with the ministry
its official title attaches it to ("délégué auprès de ..."). Which of the two handles a
question is set by decrees no text here states, so the family score separates reading the
question from knowing the org chart. Only the titles decide a family: two full ministries
never share one, however close.
"""

from __future__ import annotations

import json
import random
from pathlib import Path

NAME = "ministry_routing"

DATA = Path(__file__).resolve().parents[2] / "data" / "questions-ecrites"

INSTRUCTIONS = ("Quel ministère doit répondre à cette question écrite, posée par un député "
                "au Gouvernement")
CRITERIA = json.loads((DATA / "ministries.json").read_text(encoding="utf-8"))

CONDITIONS = {"RAW": "question", "REDACTED": "question_redacted"}

ECONOMY = "economie_finances_souverainete_industrielle_energetique_et_numerique"
FOREIGN = "europe_et_affaires_etrangeres"
PARENTS = {
    "relations_avec_le_parlement": {"premier_ministre"},
    "egalite_entre_les_femmes_et_les_hommes_et_lutte_contre_les_discriminations": {"premier_ministre"},
    "porte_parole_du_gouvernement_et_energie": {"premier_ministre", ECONOMY},
    "industrie": {ECONOMY},
    "intelligence_artificielle_et_numerique": {ECONOMY},
    "interieur_md": {"interieur"},
    "citoyennete": {"interieur"},
    "armees_et_anciens_combattants_md": {"armees_et_anciens_combattants"},
    "enseignement_et_formation_professionnels_et_apprentissage": {"travail_et_solidarites",
                                                                  "education_nationale"},
    "mer_et_peche": {"transition_ecologique_biodiversite_et_negociations_internationales"},
    "transition_ecologique": {"transition_ecologique_biodiversite_et_negociations_internationales"},
    "europe": {FOREIGN},
    "commerce_exterieur_et_attractivite": {FOREIGN},
    "francophonie_partenariats_internationaux_et_francais_de_letranger": {FOREIGN},
    "autonomie_et_personnes_handicapees": {"sante_familles_autonomie_et_personnes_handicapees"},
    "ruralite": {"amenagement_du_territoire_et_decentralisation"},
}
assert set(PARENTS) == {k for k, title in CRITERIA.items() if title.startswith("Ministère délégué")}


def same_family(a: str, b: str) -> bool:
    return bool(({a} | PARENTS.get(a, set())) & ({b} | PARENTS.get(b, set())))


def read(stem: str, split: str) -> list[dict]:
    path = DATA / f"{stem}.{split}.jsonl"
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]


def sampled(rows: list[dict], sample: int, seed: int) -> list[dict]:
    rows = sorted(rows, key=lambda r: r["id"])
    random.Random(seed).shuffle(rows)
    return rows[:sample]


def load_pairs(sample: int, seed: int, split: str = "test") -> list[dict]:
    return sampled(read("routing", split), sample, seed)


def question(condition: str) -> tuple[str, dict]:
    return CONDITIONS[condition], {"ministry": {"type": "choice", "instructions": INSTRUCTIONS,
                                                "criteria": CRITERIA}}


def share(hits: int, n: int) -> str:
    return f"{hits / n:.0%} ({hits}/{n})" if n else "-"


def report(models: dict[str, list[dict]]) -> str:
    items = {r["id"]: r for r in read("routing", "test")}
    header = ["model", "condition", "acc family", "acc kept", "acc transferred",
              "follows the député", "misroutes flagged", "flags right"]
    lines = ["### ministry_routing: families and transferred questions", "",
             "*Acc family*: the answer is the gold ministry, its parent, or a sibling under "
             "the same parent. *Transferred*: the government moved the question away from the ministry the "
             "député addressed, so the député's own choice scores 100% on kept questions and "
             "0% on transferred ones. *Follows the député*: share of transferred questions "
             "answered with the ministry the député addressed. *Misroutes flagged*: share of "
             "transferred questions answered with another ministry than the addressed one; "
             "*flags right*: share of all such answers that are on a transferred question.", "",
             "| " + " | ".join(header) + " |", "|" + "---|" * len(header)]
    for model, rows in models.items():
        for condition in CONDITIONS:
            subset = [(r, items[r["id"]]) for r in rows if r["condition"] == condition]
            if not subset:
                continue
            kept = [r for r, item in subset if not item["transferred"]]
            moved = [(r, item) for r, item in subset if item["transferred"]]
            flags = [item for r, item in subset if r["predicted"] != item["asked"]]
            lines.append("| " + " | ".join([
                model, condition,
                share(sum(same_family(r["predicted"], r["gold"]) for r, _ in subset), len(subset)),
                share(sum(r["correct"] for r in kept), len(kept)),
                share(sum(r["correct"] for r, _ in moved), len(moved)),
                share(sum(r["predicted"] == item["asked"] for r, item in moved), len(moved)),
                share(sum(r["predicted"] != item["asked"] for r, item in moved), len(moved)),
                share(sum(item["transferred"] for item in flags), len(flags)),
            ]) + " |")
    return "\n".join(lines) + "\n"
