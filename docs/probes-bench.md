# Probes: Jev on its own, and one real document

The loan-document results are in the
[README](../README.md#4-a-real-french-mortgage-offer-and-messy-messages). Here: what a probe
is, those probes in detail, and one line per other probe with what it found.

## What a probe is

A script in `probes/` that answers one question, mostly on Jev alone, written before the bench
existed. Each writes every raw response to `results/probes/<name>-<timestamp>.jsonl` and prints
its own summary. Later scripts replay those archives rather than call the API again: that is
how `confidence_fit.py` checked the confidence formula against 800 stored responses without a
single call. Where a probe ran more than once, the newest archive is the one quoted; the
earlier ones are pilots at a smaller n.

```bash
cd probes
../.venv/bin/python p2_multilingual.py --sample 100
```

## The loan document and the messages

P8 to P11, P13, P18 and P19 are not a benchmark: one document and twelve messages say "the task
is within reach", not "this generalises". They are here because no model could have trained
on them, and because they have the shape of the actual product: prefilling a mortgage form.

The document is `data/proposition-redacted.txt`, a real eleven-page French loan offer (about
13,000 tokens). Every direct identifier is a bracketed placeholder, and its 26 amounts were
replaced by an invented profile that keeps the document's arithmetic: its eight identities,
its two deliberate traps, and the few cents between the theoretical and the displayed
instalment. That was done once, offline, and the tools were deleted: their substitution table
listed one person's identifiers.

- **P8, prefill from the document.** 17 questions in one call, instructions in English: is the
  property new or pre-owned, is the rate fixed or variable, is the insurance on the initial or
  the outstanding capital, is there a co-borrower, a deferral, a works budget, a state
  subsidised loan... Seven choices, nine yes/no, one score, repeated 20 times: 340 of 340
  right. Under the form's own policy (prefill silently above 0.9, prefill with a flag between
  0.6 and 0.9, leave empty below), 290 answers were prefilled silently, 50 flagged, none
  silently wrong.
- **P19, the same 17 questions asked of GPT-4o mini**, the two ways an LLM allows, 20 runs
  each. One call per question, with a probability read from the first token: 17 calls per
  document, each billing the whole document (173,029 tokens, $0.026, 47 times Jev's price),
  2.3 s in parallel, and 208 of 340 right. 100 of its wrong answers come at 0.90 or more: it
  says yes to a deferral, a works budget and a second loan the offer does not contain, and
  still does with the options reversed. All 17 in one call, as JSON constrained to the option
  keys: 11,577 tokens, $0.0018, 2.1 s, 260 of 340 right, and no probability to set a
  threshold on. Jev: one call, 324 ms, $0.0006, 340 of 340.
- **P9, pick the number rather than write it.** Code extracts 22 distinct amounts and rates
  from the document with regexes. For each of 20 fields (loan amount, monthly instalment with
  and without insurance, notary fees, APR...), Jev picks the right candidate, with "none" as
  an option. Several candidates differ by one digit on purpose. Five repetitions: 100 of 100,
  and the one field absent from the document got "none".
- **P10, the right number is missing from the list.** Same as P9, with the correct amount
  removed from the candidates but still in the document. The dangerous outcome would be a
  confident pick of the nearest wrong number. Observed: 24 of 24 answers were "none", at
  confidences between 0.53 and 0.97. A regex that misses is a coverage problem, not silent
  corruption.
- **P11, a free-text box instead of a document.** Twelve handwritten French messages, kept
  messy: abbreviations, typos, no punctuation, missing or contradictory information. Seven
  questions: project type, new or pre-owned, household, whether a subsidised loan is
  mentioned, and which bare number is the price, the income and the deposit. Most fields are
  absent from most messages, so the right answer is often "not stated". Three repetitions: 252
  of 252. A usefulness score is recorded for stability, but has no gold.
- **P13, Laya on the same inputs.** 123 of 252 on the messages. The document does not fit: Laya
  ships with a 512 to 1,024 token window, 8,192 at most when raised, and the document is about
  13,000 tokens. The probe measures how much of it survives truncation rather than pretending
  to run it.
- **P18, Open-Jev on the messages**, one run each, since nothing is sampled. 2B: 64 of 84, and
  25 of 39 on fields not stated, mostly a household invented at low confidence; one confident
  error, a subsidised-loan question answered no at 0.98. 9B: 80 of 84, 39 of 39 on fields not
  stated, and its four misses are all "not stated" answers below 0.6.

## The other probes

| probe | question | what it found |
|---|---|---|
| `p0_smoke` | Does a live response match the docs? | Yes. Probabilities come back rounded to two decimals. |
| `p1_determinism` | The same input 50 times on a pinned version: does the answer move? | The chosen answer never moved. Its confidence ranged from 0.50 to 0.66, so a 0.6 threshold split the identical calls 28 to 22. `jev-latest` was indistinguishable from the pinned version on this input. |
| `p3_edges` | What happens when no option fits? | With no way out, Jev picks the least-wrong option at 0.74 (confidence 0.60). With an "other" option, "other" at 1.00. Offer a way out. Open-Jev (P18) spreads its probability instead: top option at 0.52 (2B) and 0.43 (9B). |
| `p4_limits` | What is the real maximum input size? | About 33,000 tokens (127,000 French characters). Beyond, a clean `400 max_tokens_exceeded`, no silent truncation. 64 times the input costs under twice the latency. |
| `p5_haystack` | How far does accuracy fall as irrelevant text grows? | Not at all: 100% up to 120,000 characters of filler, needle at the start, middle or end, in both languages. The filler held nothing that looked like an answer, so it was too easy. |
| `p6_distractors` | Up to 31 competing loan files, the question naming one? | 100% at every count. Retrieval by a reference number, not judgment. |
| `p7_crossrecord` | Questions that need every record? | Finding a record by a described property, and comparing amounts across all: 100%. Counting: 60% to 95%. Superseded by the bench's [records tasks](records-bench.md). |
| `p15_confidence_k` | Does the confidence formula hold at every option count? | Yes, from 2 to 12 options: `(k * p_max - 1) / (k - 1)`. At two options it is `p_max` doubled minus one, so binary questions have the twitchiest threshold. |
| `p16_independance` | Does adding a question to a call move the others? | No. Eight questions alone or together, 15 repetitions each: no answer moved beyond the model's own noise. Together, they cost a fifth of the tokens (607 against 2,987), and one call of eight takes as long as a call of one (about 250 ms in each of three runs). |

Two scripts are not probes. `calibration.py` draws the reliability diagram over every graded
Jev answer archived ([`figures/calibration.png`](../figures/calibration.png)): it checks the
vendor's calibration claim, and ranks nothing. `confidence_fit.py` fits the confidence
formula on archived responses at every option count, with Laya as the control: its source
states its formula, so the fit must recover it exactly.

**History.** P2 (Jev on MASSIVE scenarios), P12 and P14 (Laya on the same items, untuned
and with fitted temperatures) and P17 (small local LLMs under Jev's contract) became the
[MASSIVE bench](massive-bench.md) rows. Their archives stay: `calibration.py` and P12 replay
them.

## Data

- **P2, P12, P14, P15, P17**: MASSIVE, fetched as in [massive-bench.md](massive-bench.md#run-it).
- **P8, P9, P10, P13**: `data/proposition-redacted.txt`, in the repo.
- **P11, P13, P18**: `data/freetext-blurbs.json`, the twelve messages with their gold, in the
  repo.
- **P4 to P7**: French filler that is not in the repo. Point `FR_CORPUS` at a directory of
  French `.md` prose of a comparable register, or put files in `data/fr-corpus/`. The English
  filler is built once by `fetch_en_corpus.py`, from Wikipedia.

No client data is in this repo, anonymised or otherwise. The one real document is the
author's own.
