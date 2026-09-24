"""P0: reproduce the quickstart and dump the raw response shape.

Run this first. Everything else assumes the response fields documented at
docs.typesafe.ai/api.md; this probe is what confirms them against the live service.
"""

from __future__ import annotations

import json

from typesafe_sdk import Choice, Noul, Score

from _common import MODEL, client, dump, timed

TICKET = (
    "Hi, I've been trying to connect my Stripe account for 3 days and it keeps failing. "
    "I'm losing sales. Please help ASAP."
)

QUESTIONS = {
    "department": Choice(
        instructions="Which team should handle this",
        criteria={
            "billing": "Payment or subscription issues",
            "technical": "Bugs or integration problems",
            "sales": "Pricing or account questions",
        },
    ),
    "frustration": Score(
        instructions="How frustrated the customer appears",
        criteria=[
            "Calm, just stating facts",
            "Frustrated but civil",
            "Very angry, strong language",
        ],
    ),
    "is_urgent": Noul(instructions="The message conveys urgency or time-sensitivity"),
}


def main() -> None:
    call = timed(client(), TICKET, QUESTIONS)
    archive = call.archive()

    print(f"model={MODEL}  latency={call.elapsed_ms:.0f}ms  input_tokens={call.input_tokens}")
    print(json.dumps(archive["raw"], indent=2, ensure_ascii=False))

    path = dump("p0-smoke", [archive])
    print(f"\nwrote {path}")


if __name__ == "__main__":
    main()
