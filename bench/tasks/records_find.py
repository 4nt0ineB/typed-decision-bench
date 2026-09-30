"""Finding one record among many: 4-way Choice, English only, on records_count's texts.

The retrieval control for records_count. Every one of its texts holds exactly one retired
applicant; asking what that loan finances measures finding on the very inputs the count is
measured on. The purposes are balanced, so a constant answer scores 25%. It was the one P7
question a constant guess could not answer.
"""

from __future__ import annotations

from bench.tasks import records_count

NAME = "records_find"

INSTRUCTIONS = ("Exactly one of the loan applications in this document was submitted by an "
                "applicant who has retired. What is that applicant asking to finance")
CRITERIA = {
    "own_home": "Buying a home the applicant will live in themselves",
    "rental": "Buying a property in order to rent it out",
    "renovation": "Renovation or repair work on a property already owned",
    "refinancing": "Consolidating or refinancing existing loans",
}
CONDITIONS = {"PLAIN": "plain", "HAYSTACK": "haystack"}


def load_pairs(sample: int, seed: int, split: str = "test") -> list[dict]:
    return [pair | {"gold": pair["retiree_purpose"]}
            for pair in records_count.load_pairs(sample, seed, split)]


def question(condition: str) -> tuple[str, dict]:
    return CONDITIONS[condition], {"purpose": {"type": "choice", "instructions": INSTRUCTIONS,
                                              "criteria": CRITERIA}}


def report(models: dict[str, list[dict]]) -> str:
    header = ["model", "condition", "acc", "readable", "acc if readable"]
    lines = ["### records_find: readable answers", "",
             f"A constant answer scores 25%. {records_count.READABLE_NOTE}", "",
             "| " + " | ".join(header) + " |", "|" + "---|" * len(header)]
    for model, rows in models.items():
        for condition in CONDITIONS:
            subset = [r for r in rows if r["condition"] == condition]
            legible = records_count.readable(subset)
            cells = [model, condition, records_count.share(subset)]
            cells += ["-", "-"] if legible is None else [
                f"{len(legible) / len(subset):.0%}", records_count.share(legible)]
            lines.append("| " + " | ".join(cells) + " |")
    return "\n".join(lines) + "\n"
