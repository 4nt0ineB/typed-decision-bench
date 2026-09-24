"""P3: behaviour at the edges of the answer space.

Three cases the docs do not cover:
  A. genuinely ambiguous input, where two options are equally defensible
  B. input that matches no option, with no escape hatch offered
  C. the same input, with an explicit "other" option available

Case B is the one that matters. If the model spreads probability mass, confidence gating
catches the no-match case for free. If it confidently picks the least-wrong option, then
every closed criteria set needs an explicit escape hatch and that is a design rule worth
publishing.

Result on jev-1.13.0 (one call each): A. billing 0.75 / policy 0.25, confidence 0.62.
B. policy 0.74, confidence 0.60: not spread out, a clear pick of the least-wrong option,
though below a 0.85 gate. C. other 1.00. So an escape hatch is worth offering: with it the
no-match case is certain, without it only a strict gate catches it. Open-Jev on the same
three cases is in P18.
"""

from __future__ import annotations

from typesafe_sdk import Choice

from _common import client, dump, timed

BASE_CRITERIA = {
    "billing": "Payment, premium, refund or invoice issue",
    "claim": "Reporting a new incident or damage",
    "policy": "Changing, renewing or cancelling a contract",
}

INSTRUCTIONS = "Which team should handle this"

WITH_ESCAPE = BASE_CRITERIA | {"other": "None of the above applies"}

CASES = [
    (
        "A. ambiguous",
        "I want to cancel my contract because you still have not refunded the "
        "overcharge from March.",
        BASE_CRITERIA,
    ),
    (
        "B. no match, no escape hatch",
        "Do you have any openings for a summer internship in your data team?",
        BASE_CRITERIA,
    ),
    (
        "C. no match, escape hatch offered",
        "Do you have any openings for a summer internship in your data team?",
        WITH_ESCAPE,
    ),
]


def show(label: str, state: str, answer, elapsed_ms: float) -> None:
    print(f"\n--- {label} ---")
    print(f"  {state}")
    print(f"  choice={answer.choice}  confidence={answer.confidence:.3f}  "
          f"latency={elapsed_ms:.0f}ms")
    for option, probability in sorted(answer.probabilities.items(), key=lambda kv: -kv[1]):
        print(f"    {option:<10} {probability:.4f}")


def main() -> None:
    c = client()
    records = []

    for label, state, criteria in CASES:
        call = timed(
            c,
            state,
            {"department": Choice(instructions=INSTRUCTIONS, criteria=criteria)},
        )
        show(label, state, call.answers["department"], call.elapsed_ms)
        records.append(call.archive() | {"case": label, "state": state})

    path = dump("p3-edges", records)
    print(f"\nwrote {path}")


if __name__ == "__main__":
    main()
