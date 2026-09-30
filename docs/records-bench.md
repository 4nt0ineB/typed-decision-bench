# Counting records: synthetic loan files, in English

The headline table is in the [README](../README.md#3-counting-records-in-a-long-text). Here:
how the texts are built so no shortcut works, the bars set before the first run, how far off
the counts are, and why some LLM answers are read out of leftovers.

## The rows the README leaves out

| model | count, `TAGGED` | `RULE` | `HAYSTACK` | find, plain | find, `HAYSTACK` |
|---|---|---|---|---|---|
| Kev-9B | 49% | 19% | 8%\* | 95% | 44%\* |
| Qwen3.5 4B | 15% | 15% | 11% | 45% | 47% |

\* Refused inputs count as wrong, as for Kev-27B.

## How the texts are built

`bench/tasks/records_count.py` generates every text; running it checks the generator.

- **Questions.** "How many of the loan applications in this document request more than
  300,000 euros" (`TAGGED`: "... are marked [OVER 300K]"). Thirteen options, "None of them"
  and "Exactly 1" to "Exactly 12".
- **No shortcut on the count.** The true count runs 0 to 12, independently of the number of
  files (12 or 24), with as many items in every (files, count) cell. A constant answer is
  exact on 1 item in 13.
- **No shortcut on the amounts.** No amount sits near the threshold (80,000 to 250,000, or
  350,000 to 900,000), so an error comes from counting, not from misreading one amount.
- **Three conditions, same items and gold**, to locate a failure: `TAGGED` (the matching files
  carry a marker: counting alone), `RULE` (plain files: plus comparing amounts), `HAYSTACK`
  (the same files scattered in 32,000 characters of English Wikipedia prose: plus finding
  them).
- **The control**, `records_find`: every text holds exactly one retired applicant. "What is
  that applicant asking to finance", four balanced options, a constant answer scores 25%. It
  measures finding on the very inputs the count is measured on.

This replaces probe P7's count, which offered five ranges and let the number of files decide
the answer: with 31 files it was always "eight or more".

## The bars, set before the first run

So the data could not choose its own bar:

- a model **cannot count** if its exact accuracy on counts 8 to 12, in `TAGGED`, is below 50%;
- its **confidence is a usable warning** if, in every condition, at least 90% of what it
  answers at 0.90 or more is right.

| model | exact on counts 8 to 12, `TAGGED` | can count | right at 0.90, `TAGGED` / `RULE` / `HAYSTACK` | usable warning |
|---|---|---|---|---|
| Jev 1.13.0 | 67% | yes | 98% / 100% / 100% | yes |
| Kev-27B | 48% | no, just | 100% / 89% / 100% | no, just |
| Kev-9B | 17% | no | never reaches 0.90 | - |
| GPT-4o mini | 3% | no | 53% / 71% / 70% | no |
| Qwen3.5 9B | 0% | no | never reaches 0.90 | - |
| Qwen3.5 4B | 0% | no | never reaches 0.90 | - |

Jev is rarely sure when it counts (21% to 28% of its answers reach 0.90), and right when it
is. Its counting errors announce themselves, which is what makes them safe to gate.

## How far off

| model | condition | exact | within 1 | mean error |
|---|---|---|---|---|
| Jev 1.13.0 | `TAGGED` | 62% | 96% | +0.35 |
| | `RULE` | 67% | 97% | -0.25 |
| | `HAYSTACK` | 47% | 76% | -0.82 |
| Kev-27B | `TAGGED` | 63% | 92% | -0.19 |
| | `RULE` | 35% | 65% | -1.19 |
| | `HAYSTACK` | 9% | 42% | -1.83 |
| GPT-4o mini | `RULE` | 19% | 40% | -0.07 |
| Qwen3.5 9B | `RULE` | 15% | 31% | -3.89 |

Mean error is the predicted count minus the true one: negative means the model undercounts.
A refused input counts wrong in *exact* and is left out of the error sizes. The Qwen models
never answer 4 or more correctly: they pick low counts, hence an error near -4. Every model
and condition, split by count and by number of files, is in
[`summary.md`](../results/bench/summary.md).

## Answers read out of leftovers

An LLM held to the one-token answer contract ([setup.md](setup.md#one-contract-for-every-model))
may want to write a word first ("There are..."). *Readable* is the share of answers where the
option labels got at least half of the model's first-token probability.

- Qwen3.5 9B is readable on 0% to 2% of the counts and 4% to 9% of the finds: its answers are
  what is left once the word it wanted is excluded, not a measurement.
- GPT-4o mini is readable on 18% to 60% of the counts and nearly all the finds; Qwen3.5 4B on
  all of them.

Letting the LLMs reason before they answer would be a different measurement, and a different
cost. Jev and Kev answer by construction.

## Kev's input limit

Kev's server refuses a state and question over 8,192 tokens, which every 24-file `HAYSTACK`
text is: 78 of the 156 items in that condition come back with no answer and count wrong.
Kev-27B finds the retiree in all 78 of the others.

## Run it

```bash
cd probes && ../.venv/bin/python fetch_en_corpus.py && cd ..   # data/en-corpus.txt, the HAYSTACK prose
.venv/bin/python bench/tasks/records_count.py                   # checks the generator
.venv/bin/python bench/run.py --model jev-1.13.0 --task records_count --sample 156 --workers 6
.venv/bin/python bench/run.py --model jev-1.13.0 --task records_find --sample 156 --workers 6
```

`--sample 156` is six items per (files, count) cell; any multiple of 26 stays balanced.
Other models and GPUs: [setup.md](setup.md#reproducing-every-stored-run).
