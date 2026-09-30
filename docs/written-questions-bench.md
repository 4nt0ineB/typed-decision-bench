# Written questions: routing and matching, in French

The headline table is in the [README](../README.md#2-french-written-questions-to-the-government).
Here: how the two tasks are asked, the questions the Government moved, the answer kinds on
matching, and what limits the gold.

The dataset itself, written questions and replies from the
[Assemblée nationale open data](https://data.assemblee-nationale.fr), is described in
[`data/questions-ecrites/README.md`](../data/questions-ecrites/README.md): source, selection
rules, splits, how the addressee is removed, and how distractors are picked.

## The rows the README leaves out

| model | routing, as published | routing, addressee removed | automatable | matching | automatable |
|---|---|---|---|---|---|
| Kev-9B | 74.6% | 61.8% | 20% | 69.8% | 46% |
| Mistral Nemo | 57.0% | 20.5% | 1% | 23.2% | 1% |

## Routing

- **Question**: "Quel ministère doit répondre à cette question écrite, posée par un député au
  Gouvernement". **Options**: the 36 ministries in office, described by their official
  titles, the same for every question. No way out: every question must go somewhere.
- The wording was fixed on the dev split, with Jev only. A version asking for the *competent*
  ministry "whoever the député addressed" scored lower in both conditions and was dropped.
  Those runs are `results/bench/jev-1.13.0/*.dev.jsonl`.

### The 352 moved questions

A quarter of the test questions were moved by the Government away from the ministry the
député addressed. On those, the député's own choice is wrong by definition.

| model | exact, addressee removed | same family | right on moved questions | answers the addressed ministry anyway | answers it, text as published |
|---|---|---|---|---|---|
| Jev 1.13.0 | 66.4% | 78.5% | 91 | 186 | 352 |
| Kev-27B | 66.0% | 78.1% | 81 | 198 | 349 |
| Kev-9B | 61.8% | 71.6% | 86 | 174 | 349 |
| GPT-4o mini | 56.4% | 68.5% | 73 | 147 | 335 |
| Qwen3.5 9B | 48.8% | 60.1% | 57 | 166 | 333 |
| Mistral Nemo | 20.5% | 24.7% | 20 | 106 | 276 |

The last three columns count moved questions, out of 352.

- **Same family** credits an answer on the gold ministry's parent or sibling: a minister and
  the delegated minister attached to them by their official title. 185 of the 410 moved
  questions in the dataset (45%) stay inside a family, a split set by decrees that no
  question states.
- **Even with the addressee removed**, the models still answer the addressed ministry on 42%
  to 56% of the moved questions (Mistral Nemo 30%). The député had a reason to write there,
  and the text carries it.
- **On the text as published**, an answer that differs from the addressee would flag a
  misroute. No model does it usefully: Jev never departs from the addressee on a moved
  question, Kev-27B three times.
- **Mistral Nemo answers by position**: the second option, the interior ministry, on 53% of
  the questions with the addressee removed.

## Matching

- **Question**: "À laquelle de ces questions écrites cette réponse du ministère
  répond-elle". **Options**: "La question 1" to "La question 5", and `aucune`: "Aucune de
  ces questions : la réponse est vide, ou elle ne répond à aucune d'elles".
- **State**: "Réponse du ministère :", the reply, then "Question 1 :" to "Question 5 :".
  14,000 characters at the median, 24,000 at most.

| model | answered question present (433) | answered question left out (33) | empty reply (34) | says `aucune` when it is there |
|---|---|---|---|---|
| Jev 1.13.0 | 92% | 55% | 97% | 1% |
| Kev-27B | 92% | 61% | 100% | 0% |
| Qwen3.5 9B | 85% | 21% | 97% | 1% |
| Kev-9B | 75% | 64% | 15% | 15% |
| GPT-4o mini | 74% | 70% | 3% | 12% |
| Mistral Nemo | 12% | 100% | 94% | 87% |

- **Empty replies** are the garbage-context case: nothing can be matched. With an empty
  reply, "Réponse du ministère :" is followed directly by "Question 1 :", and GPT-4o mini
  answers question 1 on 33 of 34: it most likely reads that question as the reply. Kev-9B
  answers question 5 on 29 of 34.
- **The answered question left out** is the hard case for every model that actually
  matches: the distractors are about the same issue, answered with similar arguments.
- **Mistral Nemo** answers `aucune` on 441 of 500 items: its 100% on questions left out is
  that habit, not skill.

## What limits this test

- **The routing gold has a ceiling that is not the model's.** Which of a minister and their
  delegated minister answers is set by decrees, and a kept question may have been answered by
  a ministry that did not bother to transfer it. Compare models with each other, and read the
  family score next to the exact one. The text as published does not rank models at all:
  they all follow the addressee.
- **Small subsets.** 33 left-out and 34 empty-reply items give about ±15 points each: read
  them as a direction.
- **Kev's threshold.** Kev ships with a temperature (2.30 for 9B, 1.38 for 27B) that flattens
  its probabilities, so few answers reach 0.90. At that fixed threshold it looks worse than
  at a 5% error budget, where each model gets its own threshold. A company would set Kev's
  threshold on its own labels; Jev's documented thresholds work as they are.
- **French only.** There is no English version of these questions, so they measure French
  accuracy, not the drop from English.

## Run it

```bash
.venv/bin/python bench/run.py --model jev-1.13.0 --task ministry_routing --sample 1439 --workers 8
.venv/bin/python bench/run.py --model jev-1.13.0 --task reply_matching --sample 500 --workers 8
# the dev runs that fixed the wording
.venv/bin/python bench/run.py --model jev-1.13.0 --task ministry_routing --split dev --sample 200 --workers 4
.venv/bin/python bench/run.py --model jev-1.13.0 --task reply_matching --split dev --sample 100 --workers 4
```

Other models and GPUs: [setup.md](setup.md#reproducing-every-stored-run).
