"""Which question does this ministry reply answer: 6-way Choice, French.

Data: data/questions-ecrites/ (see its README). One published reply and five written
questions to the same ministry, under the same rubrique: the one it answers and four
distractors. Gold comes from the data, no labeling. Distractors on the same crisis or the
same scheme make it retrieval plus verification, not keyword matching; questions answered
by the same or a near-identical reply were excluded as distractors, so gold is never
arbitrary.

Two ways out, and "aucune" is right on both:
  absent       the answered question was replaced by a fifth distractor
  empty_reply  the reply was published without text (the question's own data), so
               nothing can be matched

States run to about 14,000 characters at the median, 24,000 at most. report() splits
accuracy by kind.
"""

from __future__ import annotations

from bench.tasks.ministry_routing import read, sampled, share

NAME = "reply_matching"

INSTRUCTIONS = "À laquelle de ces questions écrites cette réponse du ministère répond-elle"
NONE = "aucune"
KINDS = ("match", "absent", "empty_reply")
CANDIDATES = 5
CRITERIA = {str(k): f"La question {k}" for k in range(1, CANDIDATES + 1)} | {
    NONE: "Aucune de ces questions : la réponse est vide, ou elle ne répond à aucune d'elles"}

CONDITIONS = {"FR": "state"}


def state(row: dict) -> str:
    questions = "\n\n".join(f"Question {k} :\n{c['question']}"
                            for k, c in enumerate(row["candidates"], start=1))
    return f"Réponse du ministère :\n{row['reply']}\n\n{questions}"


def gold(row: dict) -> str:
    ids = [c["id"] for c in row["candidates"]]
    return str(ids.index(row["gold"]) + 1) if row["gold"] in ids else NONE


def load_pairs(sample: int, seed: int, split: str = "test") -> list[dict]:
    return [{"id": row["id"], "state": state(row), "gold": gold(row)}
            for row in sampled(read("matching", split), sample, seed)]


def question(condition: str) -> tuple[str, dict]:
    return CONDITIONS[condition], {"question": {"type": "choice", "instructions": INSTRUCTIONS,
                                                "criteria": CRITERIA}}


def report(models: dict[str, list[dict]]) -> str:
    kind = {r["id"]: r["kind"] for r in read("matching", "test")}
    header = ["model"] + [f"acc {k}" for k in KINDS] + ["says aucune on match"]
    lines = ["### reply_matching: by kind", "",
             "*match*: the answered question is among the five. *absent* and *empty_reply*: "
             "gold is aucune. *Says aucune on match*: the way out taken when it was wrong to.",
             "", "| " + " | ".join(header) + " |", "|" + "---|" * len(header)]
    for model, rows in models.items():
        by_kind = {k: [r for r in rows if kind[r["id"]] == k] for k in KINDS}
        cells = [model] + [share(sum(r["correct"] for r in by_kind[k]), len(by_kind[k]))
                           for k in KINDS]
        cells.append(share(sum(r["predicted"] == NONE for r in by_kind["match"]),
                           len(by_kind["match"])))
        lines.append("| " + " | ".join(cells) + " |")
    return "\n".join(lines) + "\n"
