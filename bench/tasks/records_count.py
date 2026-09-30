"""Counting records in a text: 13-way Choice, English only.

P7 asked a count with five ranges and let the number of files decide the answer: with 31
files it was always "eight or more", and on the one case that needed a count a constant
guess did as well as Jev. This task removes both shortcuts:

  - the true count K runs 0 to 12, independently of the number of records N (12 or 24),
    with as many items in every (N, K) cell, so no constant answer beats 1 in 13;
  - the options are exact counts, so an off-by-one is wrong, and report() measures how far
    and in which direction;
  - no amount sits near the 300,000 threshold (80k to 250k, or 350k to 900k), so an error
    comes from counting, not from misreading one amount.

Three conditions, same items and gold, to locate a failure:
  TAGGED    matching records carry a marker and the question counts it: counting alone
  RULE      plain records: counting, plus comparing each amount to the threshold
  HAYSTACK  the same records scattered in 32,000 characters of prose: plus finding them

Every text holds exactly one retired applicant, so records_find asks a retrieval question
on the very same texts.
"""

from __future__ import annotations

import random
import statistics
from pathlib import Path

NAME = "records_count"

DATA = Path(__file__).resolve().parents[2] / "data"
FILLER_CHARS = 32_000
SIZES = (12, 24)
COUNTS = range(13)
CELLS = [(n, k) for n in SIZES for k in COUNTS]
THRESHOLD = 300_000
LOW, HIGH = (80_000, 250_000), (350_000, 900_000)
TAG = "[OVER 300K] "

# Wording from P5's loan files, so both probes and the bench read the same kind of record.
PURPOSES = {
    "own_home": ["buy the home they will live in themselves",
                 "purchase the house they intend to occupy"],
    "rental": ["buy a flat they intend to rent out", "purchase a studio to let to students"],
    "renovation": ["redo the roof and insulation of their house",
                   "fund a full renovation of the kitchen and bathroom"],
    "refinancing": ["consolidate their outstanding loans into a single monthly payment",
                    "refinance three existing loans to reduce the monthly burden"],
}
EMPLOYMENTS = ["has been on a permanent employment contract for six years",
               "has been working on successive fixed-term contracts for two years",
               "is self-employed and files as a sole trader"]
RETIRED = "retired last year and lives on a pension"
COSIGNERS = ["Their sister stands as joint guarantor for the whole loan.",
             "The file is submitted with no guarantor and no co-borrower."]
CITIES = ["Nantes", "Rennes", "Bordeaux", "Lille", "Toulouse", "Strasbourg"]
TEMPLATE = ("File {ref}. The applicant is seeking financing of EUR {amount:,} in order to "
            "{purpose}. The borrower {employment}. {cosigner} The {city} branch has passed "
            "the file to the credit committee.")

INSTRUCTIONS = "How many of the loan applications in this document request more than 300,000 euros"
INSTRUCTIONS_TAGGED = "How many of the loan applications in this document are marked [OVER 300K]"
CRITERIA = {"0": "None of them"} | {str(k): f"Exactly {k}" for k in COUNTS if k}

CONDITIONS = {
    "TAGGED": ("tagged", INSTRUCTIONS_TAGGED),
    "RULE": ("plain", INSTRUCTIONS),
    "HAYSTACK": ("haystack", INSTRUCTIONS),
}


def records(rng: random.Random, n: int, k: int, purpose: str) -> list[tuple[bool, str]]:
    """n loan files, k of them above the threshold, one from a retiree financing `purpose`."""
    above = set(rng.sample(range(n), k))
    retiree = rng.randrange(n)
    refs = rng.sample(range(4100, 10_000), n)
    files = []
    for i in range(n):
        low, high = HIGH if i in above else LOW
        own = purpose if i == retiree else rng.choice(list(PURPOSES))
        files.append((i in above, TEMPLATE.format(
            ref=refs[i], amount=rng.randrange(low, high + 1, 1_000),
            purpose=rng.choice(PURPOSES[own]),
            employment=RETIRED if i == retiree else rng.choice(EMPLOYMENTS),
            cosigner=rng.choice(COSIGNERS), city=rng.choice(CITIES))))
    return files


def scatter(texts: list[str], filler: str, rng: random.Random) -> str:
    """The texts, in order, inserted at random word boundaries of the filler."""
    cuts = []
    for _ in texts:
        space = filler.find(" ", rng.randrange(len(filler)))
        cuts.append(len(filler) if space < 0 else space)
    cuts.sort()
    out, previous = [], 0
    for cut, text in zip(cuts, texts):
        out += [filler[previous:cut], f"\n\n{text}\n\n"]
        previous = cut
    return "".join(out + [filler[previous:]])


def load_pairs(sample: int, seed: int, split: str = "test") -> list[dict]:
    """`sample` items, rep-major over the 26 (N, K) cells: every multiple of 26 is balanced,
    and 156 is six items per cell. Retiree purposes cycle, so each gets a quarter."""
    filler = (DATA / "en-corpus.txt").read_text(encoding="utf-8")[:FILLER_CHARS]
    pairs = []
    for index in range(sample):
        rep, (n, k) = index // len(CELLS), CELLS[index % len(CELLS)]
        rng = random.Random(f"{seed}-{split}-{n}-{k}-{rep}")
        purpose = list(PURPOSES)[index % len(PURPOSES)]
        files = records(rng, n, k, purpose)
        texts = [text for _, text in files]
        pairs.append({
            "id": f"n{n}-k{k}-r{rep}", "gold": str(k), "retiree_purpose": purpose,
            "tagged": "\n\n".join(TAG + text if hit else text for hit, text in files),
            "plain": "\n\n".join(texts),
            "haystack": scatter(texts, filler, rng),
        })
    return pairs


def question(condition: str) -> tuple[str, dict]:
    variant, instructions = CONDITIONS[condition]
    return variant, {"count": {"type": "choice", "instructions": instructions,
                               "criteria": CRITERIA}}


def size(row: dict) -> int:
    return int(row["id"].split("-")[0].removeprefix("n"))


def share(rows: list[dict]) -> str:
    return f"{statistics.mean(r['correct'] for r in rows):.0%}" if rows else "-"


def readable(rows: list[dict]) -> list[dict] | None:
    """Rows where the option letters got at least half of the model's first-token
    probability. None when the adapter records no label_mass: Jev answers by construction."""
    if not rows or rows[0].get("label_mass") is None:
        return None
    return [r for r in rows if r["label_mass"] >= 0.5]


READABLE_NOTE = ("Readable: share of answers where the option letters got at least half of the "
                 "model's first-token probability. Below that, an LLM wanted to write a word "
                 "and its answer is read out of leftovers.")


def report(models: dict[str, list[dict]]) -> str:
    """How far off, in which direction, and whether it worsens with the count or the size."""
    header = ["model", "condition", "exact", "within 1", "mean error", "K 0-3", "K 4-7",
              "K 8-12", "N 12", "N 24", "cover@0.90", "acc@0.90", "readable",
              "exact if readable"]
    lines = ["### records_count: how far off", "",
             f"A constant answer is exact on 1 item in {len(CRITERIA)} "
             f"({1 / len(CRITERIA):.1%}). Mean error is predicted minus true count: "
             f"negative means it undercounts. A row with no answer (an input the model "
             f"refused) counts wrong in exact and is left out of the error sizes. "
             f"{READABLE_NOTE}", "",
             "| " + " | ".join(header) + " |", "|" + "---|" * len(header)]
    for model, rows in models.items():
        for condition in CONDITIONS:
            subset = [r for r in rows if r["condition"] == condition]
            if not subset:
                continue
            errors = [int(r["predicted"]) - int(r["gold"]) for r in subset
                      if r["predicted"] is not None]
            gated = [r for r in subset if r["confidence"] is not None and r["confidence"] >= 0.90]
            cells = [model, condition, share(subset),
                     f"{statistics.mean(abs(e) <= 1 for e in errors):.0%}" if errors else "-",
                     f"{statistics.mean(errors):+.2f}" if errors else "-"]
            cells += [share([r for r in subset if low <= int(r["gold"]) <= high])
                      for low, high in ((0, 3), (4, 7), (8, 12))]
            cells += [share([r for r in subset if size(r) == n]) for n in SIZES]
            cells += [f"{len(gated) / len(subset):.0%}", share(gated)]
            legible = readable(subset)
            cells += ["-", "-"] if legible is None else [
                f"{len(legible) / len(subset):.0%}", share(legible)]
            lines.append("| " + " | ".join(cells) + " |")
    return "\n".join(lines) + "\n"


if __name__ == "__main__":
    import re
    from collections import Counter

    pairs = load_pairs(156, 42)
    assert Counter((size(p), int(p["gold"])) for p in pairs) == {cell: 6 for cell in CELLS}
    assert Counter(p["retiree_purpose"] for p in pairs) == {p: 39 for p in PURPOSES}
    for p in pairs:
        amounts = [int(a.replace(",", "")) for a in re.findall(r"EUR ([\d,]+)", p["plain"])]
        assert len(amounts) == size(p) and sum(a > THRESHOLD for a in amounts) == int(p["gold"])
        assert not any(LOW[1] < a < HIGH[0] for a in amounts)
        assert p["tagged"].count(TAG) == int(p["gold"])
        assert p["plain"].count(RETIRED) == p["haystack"].count(RETIRED) == 1
        assert all(text in p["haystack"] for text in p["plain"].split("\n\n"))
    assert load_pairs(26, 42) == pairs[:26] and load_pairs(26, 42, "dev") != pairs[:26]
    print(f"ok: {len(pairs)} items, longest state {max(len(p['haystack']) for p in pairs):,} chars")
