"""P18: Open-Jev on P11's French blurbs and P3's edge cases.

The bench only compared Open-Jev on MASSIVE, one Choice over short commands. P11 is the
shape of the product: messy French free text, Choice and Noul together, and "not stated"
as the right answer most of the time. Jev scored 252/252 there over 3 repetitions; Laya 123/288
(P13). The dangerous outcome is confident invention on a field the user never mentioned.

P3's question comes along because it is three calls: when no option fits and no escape
hatch is offered, does the probability spread out, or does one option win confidently?

Same blurbs, gold, questions and grader as P11 (its evaluate()/report(), through P13's
wrapper). Open-Jev scores every option as an independent Yes/No pass with no sampling, so
one repetition is the run. Zero-shot, at the checkpoint's shipped temperature. Linux and
CUDA only:

    modal run bench/modal_run.py --probe p18_openjev.py --model open-jev-2b
    modal run bench/modal_run.py --probe p18_openjev.py --model open-jev-9b --gpu A100-80GB

Result (2026-09-23, one run each, results/probes/p18-open-jev-*):

  P11, 84 graded answers per model (Jev: 84/84 per repetition).
    2B: 64/84. Only 25/39 on fields not stated: it fills famille on 7/7 blurbs that name
        no household, and invents a nature, an apport or a price on 7 more. Every one of
        those inventions is at confidence 0.09-0.43 except a price at 0.67, so under P8's
        policy (empty below 0.6) they stay empty. The one confident error is the other way:
        blurb 0 asks about the PTZ and the Noul says no at 0.98. Two more stated fields
        are wrong in the flag band: projet travaux at 0.74, PTZ no at 0.77.
    9B: 80/84, and 39/39 on fields not stated. The four misses all answer "not stated"
        (or "not mentioned") to something the blurb does say, at 0.25-0.59. Nothing wrong
        above 0.6. It is less sure than Jev when right: 50 of 80 above 0.9.

  P3, no option fits and no escape hatch: the probability spreads out on both, the good
  outcome. 2B billing 0.52 / claim 0.32 / policy 0.16, confidence 0.28; 9B claim 0.43 /
  policy 0.32 / billing 0.26, confidence 0.14. Jev picked policy at 0.74 (confidence
  0.60). With "other" offered, both pick it but only at 0.46 and 0.45 (confidence 0.28,
  0.27), where Jev said 1.00. On the ambiguous case 9B matches Jev (billing 0.74); 2B
  splits policy 0.55 / billing 0.45.
"""

from __future__ import annotations

import argparse
import time

from typesafe_sdk import Choice

from _common import dump
from bench.models import REGISTRY
from bench.models.openjev import load
from p3_edges import CASES, INSTRUCTIONS, show
from p11_freetext import evaluate, report
from p13_laya_freetext import Call, as_laya


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True, choices=["open-jev-2b", "open-jev-9b"])
    args = parser.parse_args()

    predictor = load(**REGISTRY[args.model][1]).predictor

    def predict(state, questions):
        start = time.perf_counter()
        result = predictor.predict({"state": state,
                                    "questions": {k: as_laya(v) for k, v in questions.items()}})
        return Call(result, (time.perf_counter() - start) * 1000.0)

    print("1 repetition: Open-Jev is deterministic (no sampling); Jev's 3 were for stability")
    report(evaluate(predict, 1), 1, args.model, name=f"p18-{args.model}-freetext", price=0.0)

    records = []
    for label, state, criteria in CASES:
        call = predict(state, {"department": Choice(instructions=INSTRUCTIONS, criteria=criteria)})
        show(label, state, call.answers["department"], call.elapsed_ms)
        records.append({"case": label, "state": state, "elapsed_ms": call.elapsed_ms,
                        "answer": vars(call.answers["department"])})
    print(f"\nwrote {dump(f'p18-{args.model}-edges', records)}")


if __name__ == "__main__":
    main()
