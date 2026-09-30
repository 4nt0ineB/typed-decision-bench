# questions-ecrites

Written questions from French députés to the Government, with the ministry that answered
them and the published replies. French text, gold set by the Government itself: no manual
labeling. Used by `bench/tasks/ministry_routing.py` and `bench/tasks/reply_matching.py`.
The script that built these files is not published; the rules below are the whole method.

**Source.** Assemblée nationale open data, 17th legislature: *Questions écrites* and the
*acteurs, mandats et organes* registry (data.assemblee-nationale.fr), downloaded on
2026-09-29. Licence Ouverte / Open Licence 2.0 (Etalab): free reuse, with attribution to
the Assemblée nationale.

## Selection

- Questions published from **2026-03-01** (last: 2026-09-22), answered by 2026-09-29, under
  the **Lecornu II** government (organe `PO873634`, in office since 2025-10-13). Recent on
  purpose: after the training data of most models compared. Only answered questions
  qualify, which favours ministries that answer quickly.
- **Gold ministry**: the one the question was attributed to on the day its reply was
  published. **Asked**: the one the député addressed. Both must belong to this government,
  so a transfer is a routing correction, never a reshuffle renaming the ministries.
- **Options**: the 36 ministries in office on the question date, keyed by short name
  (`ministries.json`, official titles as descriptions). The set is the same for every
  question. A minister replaced mid-government gets a new organe under the same name;
  they are one option.
- 1,639 questions, 410 transferred (25%). 185 of those transfers (45%) stay inside one
  ministry family, as the official titles link them: a minister and a delegated minister
  attached to them, or two delegated ministers of the same minister. The other 225 move
  the question to another ministry altogether.
- Shuffled with seed 42; the first 200 are the dev split of both tasks, so nothing tuned
  on dev reaches a test score.

## Files

| File | Rows | Content |
|---|---|---|
| `ministries.json` | 36 | option key -> official title |
| `routing.dev.jsonl` | 200 | 58 transferred |
| `routing.test.jsonl` | 1,439 | 352 transferred, 165 rubriques |
| `matching.dev.jsonl` | 100 | 87 match, 6 absent, 7 empty reply |
| `matching.test.jsonl` | 500 | 433 match, 33 absent, 34 empty reply |

**Routing rows**: `id` (Assemblée uid), `asked_on`, `answered_on`, `rubrique` and `analyse`
(the Assemblée's own indexing), `question`, `question_redacted`, `asked`, `gold`,
`transferred`.

`question_redacted` replaces the addressed minister with "le Gouvernement" where the
député addresses them ("interroge M. le ministre de ... sur", "demande à Mme la ministre
de ..."). Ministers mentioned in the narrative are kept: they are content, the clue a
human router reads too. The rule is a pattern on the addressing verbs; on these 1,639
questions no addressing clause was left.

**Matching rows**: `id` (the reply's question), `kind`, `ministry`, `rubrique`, `reply`,
`candidates` (five `{id, question}`, questions redacted as above), `gold` (a candidate id,
or `none`).

- `match`: the answered question and four distractors, in random order.
- `absent` (about 7%): five distractors, gold `none`.
- `empty_reply` (about 7%): the reply was published without text in the source data; gold
  `none`. Kept on purpose, as garbage context nothing can be matched to.
- Distractors come from the same ministry and rubrique, have a non-empty reply, and are
  excluded when their reply is identical to the target's or overlaps it by 0.5 or more
  (word 5-gram Jaccard), or when their question overlaps the target's by 0.2 or more. A
  ministry often publishes one reply for several questions (4,236 same-ministry,
  same-rubrique pairs share an identical reply): without this rule, gold would be
  arbitrary.

## Caveats

- Some matching items stay hard for a human too: several distractors about the same
  crisis or scheme, answered with similar arguments.
- Copy-pasted questions from different députés are kept in routing: they are real traffic.
- Kev-27B's base (Qwen3.8-27B) was released on 2026-08-05, inside this window: the only
  model compared that could have seen some of it.
- Replies and questions are cleaned of HTML tags and whitespace runs; nothing else is
  changed.
