# typed-decision-bench

Jev is TypeSafe's zero-shot classification model. You send it a state (a message, a JSON
object, a document) and typed questions, each with its options and a one-line criterion per
option. It returns a probability for every option, and writes no text. The promise is that
those probabilities can be trusted: set a threshold, automate what clears it, send the rest
to a human.

This repo tests that promise from the outside, on `jev-1.13.0` in September 2026, against
the models you could use instead, hosted or on your own GPU. Every model is used zero-shot,
the way a small team would build a proof of concept: options and criteria written by hand,
no labelled data, no fine-tuning. Every run is stored as raw JSON under `results/`, and
`bench/report.py` rebuilds every table and chart from it without calling a model.

## What we want to know

1. **Does Jev hold up in French?** TypeSafe makes no claim either way.
2. **Can you trust its confidence?** When it says 0.90, is it right 90% of the time, and how
   much can you automate at a given error rate?
3. **How does it compare** to a hosted LLM, or to an open model you run yourself?
4. **Does it hold on real text** that no model could have learned by heart?

## Tests

### 1. Voice-assistant commands, English and French

500 short commands from [MASSIVE](https://github.com/alexa/massive), professionally
translated, so each command exists in both languages. The model picks one of 60 intents
(and, in a second task, one of 18 scenarios). Three conditions: `EN_EN` is an English command
with English criteria, `FR_EN` a French command with English criteria (the common case in a
company), `FR_FR` French on both sides.

| command, English and French | intent (1 of 60) |
|---|---|
| wake me up at five am this week<br>réveille-moi à cinq heures du matin cette semaine | `alarm_set`: "Set a new alarm or wake-up call" |
| delete the alarm i just set<br>supprime l'alarme que je viens de régler | `alarm_remove`: "Cancel, turn off or remove one or more alarms" |

| model | accuracy `EN_EN` | `FR_EN` | `FR_FR` | at 0.90: passes / right | automatable |
|---|---|---|---|---|---|
| Jev 1.13.0 | 85% | 83% | 84% | 73% / 94% | 70% |
| Kev-27B | 84% | 80% | 81% | 42% / 98% | 69% |
| GPT-4o mini | 81% | 74% | 75% | 90% / 82% | 5% |
| Qwen3.5 9B | 80% | 71% | 74% | 18% / 98% | 44% |
| Open-Jev 9B | 74% | 71% | 68% | 12% / 96% | 25% |
| Laya multilingual | 42% | 36% | 35% | 26% / 69% | 0% |

*At 0.90*: the share of answers given with a confidence of 0.90 or more, the strictest
threshold in TypeSafe's docs, and how often those are right. *Automatable*: the largest share
that one threshold, chosen for each model, lets through with at most 5% of it wrong. The 18
scenarios and the other models (Kev-9B, Mistral Nemo, smaller LLMs, a fine-tuned reference)
are in [docs/massive-bench.md](docs/massive-bench.md).

![risk-coverage, 60 intents](figures/risk-coverage-massive_intent.png)

**How to read it.** Each curve is one model. Moving right lowers the threshold: more traffic
is handled automatically (x), and accuracy on what was handled (y) drops. Higher is better.
The markers are the thresholds 0.90, 0.85 and 0.60, left to right. Jev's curve starts in the
middle because it answers exactly 1.00 on about half its commands, and no threshold can split
those.

**What it shows: Jev holds up in French, and its 0.90 means what it says.** It loses about 2
points from English to French (1.6 to 2.4), the others 3 to 8, and 94% of what passes 0.90 is
right. GPT-4o mini lets more through at 0.90, but only 82% of it is right. At a 5% error
budget, Kev-27B automates as much as Jev; Open-Jev 9B and Laya are far behind.

### 2. French written questions to the Government

Real, native French: 1,439 questions from députés to the Government, about 2,000 characters
each, published from March 2026 (Assemblée nationale open data), after the training data of
most models. The gold is the Government's own routing, so nothing was labelled by hand.

- **Routing**: which of the 36 ministries in office must answer. A quarter of the questions
  were moved by the Government away from the ministry the député addressed. The text as
  published names that addressee; a second condition replaces it with "le Gouvernement".
- **Matching**: a published reply and five questions to the same ministry on the same topic.
  Which one does it answer, or none? 500 items; "none" is right when the answered question
  was left out, or when the reply was published empty.

> M. Philippe Gosselin attire l'attention **de M. le ministre de l'action et des comptes
> publics** sur l'adéquation entre les missions croissantes de la MSA (prévention des
> accidents du travail, gestion des maladies professionnelles, etc.) et les moyens humains
> disponibles. Il lui demande comment le Gouvernement compte renforcer [...]

The Government sent it to Agriculture: the MSA is the farmers' social security. On the text
as published, Jev answers public accounts, at 0.91. With the addressee removed ("attire
l'attention **du Gouvernement**"), it answers Agriculture.

| model | routing, as published | routing, addressee removed | automatable | matching | automatable |
|---|---|---|---|---|---|
| Jev 1.13.0 | 75.5% | 66.4% | 39% | 89.6% | 88% |
| Kev-27B | 75.6% | 66.0% | 31% | 90.8% | 83% |
| GPT-4o mini | 74.2% | 56.4% | 4% | 69.0% | 41% |
| Qwen3.5 9B | 72.7% | 48.8% | 2% | 81.2% | 49% |

**How to read it.** The député's own choice scores 75.5% on routing, since a quarter of the
questions were moved. *Automatable* on routing is with the addressee removed, as in test 1.
Laya was not run on them: at its shipped 1,024-token limit it keeps about 770 tokens of input,
less than most questions and a fraction of a matching item. Its README gives 8,192 tokens when
`max_len` is raised, which would fit them; that was not tried. Nor was Open-Jev run. Kev-9B and Mistral Nemo are in
[docs/written-questions-bench.md](docs/written-questions-bench.md).

**What it shows: every model follows the député; once the addressee is gone, Jev and Kev-27B
lead.** On the 352 moved questions, Jev answers the addressed ministry every time, and 324 of
its 353 errors come at 0.90 or more: on the text as published, its 0.90 passes 95% of
traffic at 77% right. With the addressee removed, 41% at 94%. On empty replies, GPT-4o mini
picks question 1 on 33 of 34.

### 3. Counting records in a long text

Synthetic English loan files, 12 or 24 per text: how many ask for more than 300,000 euros?
The options are the exact counts, 0 to 12. `TAGGED` marks the matching files, so it measures
counting alone; `RULE` adds comparing each amount to the threshold; `HAYSTACK` scatters the
same files in 32,000 characters of prose. The control asks, on the same texts, what the one
retired applicant's loan finances.

> File 6938. The applicant is seeking financing of EUR 231,000 in order to consolidate their
> outstanding loans into a single monthly payment. The borrower has been on a permanent
> employment contract for six years. Their sister stands as joint guarantor for the whole
> loan. The Strasbourg branch has passed the file to the credit committee.

| model | count, `TAGGED` | `RULE` | `HAYSTACK` | find, plain | find, `HAYSTACK` |
|---|---|---|---|---|---|
| Jev 1.13.0 | 62% | 67% | 47% | 100% | 100% |
| Kev-27B | 63% | 35% | 9%\* | 100% | 50%\* |
| GPT-4o mini | 17% | 19% | 13% | 85% | 60% |
| Qwen3.5 9B | 18% | 15% | 13% | 55% | 60% |

**How to read it.** Exact counts only: a constant answer scores 7.7%, and 25% on the
control. \* Kev refuses any input over 8,192 tokens, which every 24-file `HAYSTACK` text is:
half of that condition counts as wrong. Kev-27B finds the retiree in all of the rest. Laya
and Open-Jev were not run: the texts are far longer than the 770 tokens Laya keeps at its
shipped limit. Raised to 8,192, most would fit; that was not tried.

**What it shows: counting is where every model breaks, Jev least.** Finding one record is
easy for Jev. On counting alone, Jev and Kev-27B are level; once the count also takes
comparing amounts or finding the files, Jev keeps a wide lead. GPT-4o mini and Qwen3.5 9B
mostly want to write a sentence before they answer, so their counts are read out of
leftovers. Details: [docs/records-bench.md](docs/records-bench.md).

### 4. A real French mortgage offer, and messy messages

The only data here that no model could have seen: a real eleven-page French loan offer, its
identifiers redacted and its amounts replaced by invented ones, and twelve handwritten French
messages. On the offer, 17 questions in one call (is the rate fixed, is the insurance on the
initial or the outstanding capital, is there a co-borrower...), repeated 20 times. On the
messages, seven questions each, where the right answer is often "not stated":

> bjr je suis célibataire je gagne 2580 net par mois je veux acheter une maison ancienne vers
> l'est de paris 210k j'ai 32000 d'apport est ce que j'ai droit au ptz ?

| model | loan offer | messages |
|---|---|---|
| Jev 1.13.0 | 340 / 340 | 252 / 252 |
| Open-Jev 9B | not run | 80 / 84 |
| Laya | does not fit | 123 / 252 |

**What it shows: on text nobody trained on, Jev made no mistake, and Open-Jev 9B came
close.** Open-Jev 9B's four misses are all a cautious "not stated"; Laya gets half its
answers wrong. One document and twelve messages show the task is within reach, not that it
generalises. The other probes (determinism, input size, what happens when no option fits,
several questions in one call, Open-Jev 2B) are in [docs/probes-bench.md](docs/probes-bench.md).

## Speed, cost, and many questions

The tests above ask one question per call. A real application asks many about the same input
(P8's mortgage form has 17), and that is where the models differ most.

### One question per call

| model | runs on | per call, short command | per call, long French text | cost |
|---|---|---|---|---|
| Jev 1.13.0 | TypeSafe's API | 269 ms | 1.0 s | $0.08 to $0.10 per 1,000 calls |
| Kev-27B | one H100 80 GB | 126 ms | 295 ms | the GPU, about $4 an hour |
| GPT-4o mini | OpenAI, via OpenRouter | 1.1 s | 1.1 s | $0.14 to $0.20 per 1,000 calls |
| Qwen3.5 9B | one A100 80 GB | 120 ms | 140 ms | the GPU |
| Open-Jev 9B | one A100 80 GB | 481 ms | not run | the GPU |
| Laya multilingual | a laptop (M3) | 32 ms | not run | the laptop |

**How to read it.** Medians. Hosted calls include the network from France; the others ran one
call at a time, in process. None of this is a throughput: with 8 calls in parallel, Jev served
about 7 a second without hitting a limit. At Kev-27B's one-at-a-time pace (3.4 long-text calls
a second), an H100 costs $0.33 per 1,000 calls, three times Jev, until batching fills it.

### Many questions on one document

The 17 questions of P8 on the loan offer, 20 runs each. GPT-4o mini can take them two ways:
one call per question, which keeps a probability per answer, or all of them in one call,
answered as JSON.

| 17 questions, one document | calls | time | tokens | $ per document | right | wrong at 0.90 or more |
|---|---|---|---|---|---|---|
| Jev 1.13.0, one call | 1 | 324 ms | 13,077 | $0.0006 | 340 / 340 | 0 |
| GPT-4o mini, one call per question | 17 | 2.3 s | 173,029 | $0.026 | 208 / 340 | 100 |
| GPT-4o mini, one JSON call | 1 | 2.1 s | 11,577 | $0.0018 | 260 / 340 | no probability |

**What it shows: with many questions, the interface decides.** One question per call, GPT-4o
mini bills the document 17 times, and says yes, at 0.90 or more, to a deferral, a works budget
and a second loan the offer does not contain. All 17 in one call, it is cheaper and more
accurate, but leaves nothing to set a threshold on. Jev does both in one call, in a sixth of
the time. Its cost grows with the questions, not the input: on a short message, 8 questions in
one call bill 607 tokens, against 2,987 in 8 calls, for the same 250 ms (P16). Details:
[docs/probes-bench.md](docs/probes-bench.md).

### What the Jev-like models support

| | Jev | Kev | Open-Jev | Laya |
|---|---|---|---|---|
| built on | a pretrained model, not named | Qwen: the pretrained-only release (9B), the chat release (27B) | Qwen3.5, the chat release | BERT-style encoders (ModernBERT-large, mmBERT-base) |
| trained with | reinforcement learning for calibrated decisions (RLCD), in place of chat training | supervised learning, then a fitted temperature | supervised learning, then a fitted temperature | RLCD, by its model card |
| question types | choice, yes/no, score | the same three | the same three | the same three |
| several questions in one call | yes, input read once | yes, input read once | yes, input re-read for every option of every question (a cache that reads it once exists, off by default) | yes, input re-read for every question |
| longest input | about 33,000 tokens, then refused | 8,192 tokens, input and one question together, then refused | 4,096 tokens per option, then refused | 512 (English) or 1,024 (multilingual) as shipped, up to 8,192 for multilingual when raised, then silently cut |
| server | hosted API | compatible with Jev's API, batches up to 64 requests: 29 a second for 27B, 80 for 9B, on one H100 | one request at a time | none in the version benchmarked; one request at a time since 0.3.9 |
| licence | closed, hosted only | Apache-2.0 | MIT code, Apache-2.0 weights | Apache-2.0 |

Read from each project's source code at the version benchmarked; the throughput figures are
Kev's own (short texts, six questions each, 64 clients), not measured here. Jev's column is
its [documentation](https://docs.typesafe.ai/introduction/machine-learning-primer) and our
probes (P4, P16). Only Kev has Jev's shape: many questions, the input read once, a server that
batches. Only Laya says it trains the way Jev does, though TypeSafe publishes nothing to check
that against. None has both. Evidence, line by line:
[docs/setup.md](docs/setup.md#what-the-jev-like-models-support-in-their-code).

## Conclusion

Models like Jev, trained to answer typed questions with probabilities rather than text, are
new: Jev has been public for about three weeks, Kev for two. On these tests:

- **Jev is the reference.** It is first, or level with the first, on every test, in both
  languages, with thresholds that work as documented and nothing to host.
- **Kev-27B is the first open model to come close.** It is within 3 points of Jev on the
  commands and the written questions, and automates as much on 60 intents. It falls behind
  on counting, needs an 80 GB GPU, refuses inputs over 8,192 tokens, and its threshold has to
  be set on your own labels. It is also the only open one built like Jev for many questions:
  the input read once, and a server that batches. Kev-9B, which fits a workstation, gives up
  5 to 20 points.
- **Open-Jev and Laya, the open clones getting the attention, are not there yet.** Open-Jev
  9B scores below the Qwen3.5 9B it is built on and automates 25% of commands at a 5% error
  budget, against 44% for that base, though it came close to Jev on the messages. Laya is
  right on 42% of English intents and 35% of French ones, and keeps about 770 tokens
  of input at its shipped limit (8,192 when raised, and slower): it is a base to fine-tune.
- **A general LLM is no substitute.** GPT-4o mini's 0.90 is right 63% to 88% of the time, and
  on many questions it either bills the input once per question or gives no probability. An
  open 4B to 9B LLM comes close on short commands, falls behind on long French text, and runs
  once per question.
- **No threshold protects against a misleading text.** When a question names the wrong
  ministry, every model follows it, Jev with confidence.
- **Self-hosting is rarely about price.** An H100 at about $4 an hour matches Jev's $0.10 per
  1,000 routing calls only above some 40,000 calls an hour, around the clock. The reasons to
  run Kev are data that must stay in-house, no lock-in, and fine-tuning on your own labels.

## Limitations

- **Single-choice questions, two datasets.** Every cross-model number is a single-choice
  question, on MASSIVE commands, synthetic records or French written questions. The written
  questions have no English version: they measure French accuracy, not the drop from
  English. Jev's yes/no and score questions were only probed on Jev.
- **Possible contamination.** MASSIVE has been public since 2022, and the open LLMs have very
  likely seen it; whether Jev has is unknown. The written questions start in March 2026,
  after every model run on them was released, except Kev-27B's base (August 2026). Jev's
  training cutoff is not published.
- **One wording.** Each task was asked with one wording of its criteria, and another may
  suit some models better than others.
- **Zero-shot only.** No model was fine-tuned. That is not how Laya's authors recommend using
  it.
- **Many questions per call, on one document.** Every test above asks one question per call.
  Many questions on one input were measured on Jev and GPT-4o mini only, on the loan offer.
- **Latency and cost do not compare across rows.** Hosted models include the network from
  France; GPU models run one call at a time, in process. See [docs/setup.md](docs/setup.md).
- **One unpublished script.** The script that built the written-question dataset is not in
  the repo; [its README](data/questions-ecrites/README.md) states every rule it applies.
- **A snapshot.** `jev-1.13.0`, September 2026, with no updates planned.

## Licences

The code is under the [MIT licence](LICENSE).

The MASSIVE commands stored under `results/` (`state` in `results/bench/`, `utterance` in the
P2, P12, P14 and P17 probe archives) are a sample of
[MASSIVE 1.1](https://github.com/alexa/massive), © Amazon.com Inc. or its affiliates,
licensed under [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/). They are sampled
from the test and dev splits and otherwise unchanged; the MIT licence does not apply to them.

The written questions and replies under `data/questions-ecrites/`, and in the `state` of the
`ministry_routing` and `reply_matching` results, come from the
[Assemblée nationale open data](https://data.assemblee-nationale.fr), under the
[Licence Ouverte 2.0](https://www.etalab.gouv.fr/licence-ouverte-open-licence/) (Etalab),
with attribution to the Assemblée nationale; the MIT licence does not apply to them.
