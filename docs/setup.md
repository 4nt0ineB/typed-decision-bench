# Setup: install, models, runs

What every test shares: installing, the models and how each is held to the same contract,
running a task, rented GPUs, and reproducing the stored runs. Each test's own commands are
in its file: [MASSIVE](massive-bench.md), [written questions](written-questions-bench.md),
[records](records-bench.md), [probes](probes-bench.md).

## Install

```bash
python3 -m venv .venv
.venv/bin/pip install -r requirements.txt
cp .env.example .env    # TYPESAFE_API_KEY, and OPENROUTER_API_KEY for the hosted LLMs
```

Jev is pinned to `jev-1.13.0` in `bench/models/jev.py`, never `jev-latest`, an alias that can
move and silently invalidate earlier measurements. Everything in `results/` was measured on
that version. `TYPESAFE_MODEL` overrides it.

## The models

| name in `results/` | what it is | where it ran |
|---|---|---|
| `jev-1.13.0` | [Jev](https://typesafe.ai) ([docs](https://docs.typesafe.ai)), TypeSafe's hosted API | API, from France |
| `kev-9b`, `kev-27b` | [Kev](https://github.com/jaredpalmer/kev) by Jared Palmer ([weights](https://huggingface.co/collections/jaredpalmer/kev-6aad9d0ea49f2589665e07cd)): a LoRA adapter and a pointer head on Qwen3.5-9B-Base and Qwen3.8-27B, serving TypeSafe's API | Modal, L40S and H100 80 GB |
| `open-jev-2b`, `open-jev-9b` | [Open-Jev](https://github.com/Zefan-Cai/Open-Jev) by Zefan Cai ([project page](https://zefan-cai.github.io/open-jev/)): LoRA adapters and decision heads on frozen Qwen3.5 2B and 9B | Modal, A100-80GB |
| `laya-router`, `laya-multilingual` | [Laya](https://github.com/NandhaKishorM/laya) by Convai Innovations ([weights](https://huggingface.co/convaiinnovations/laya)): an open encoder (300-400M) that copies Jev's interface | laptop |
| `qwen3.5-2b`, `-4b`, `-9b`, `minicpm5-2b` | open LLMs; Qwen3.5 2B and 9B are the frozen bases Open-Jev is built on | Modal, A100-80GB, bf16 |
| `qwen3.5-0.8b-mlx-4bit` | the same kind of LLM, 4-bit on Apple's MLX: can a solo developer try the idea on a laptop? | laptop (M3) |
| `gpt-4o-mini-openrouter`, `mistral-nemo-openrouter` | hosted LLMs, through OpenRouter | API, from France |
| `xlm-r-base-massive-intent` | XLM-R base fine-tuned on MASSIVE's English train split: the labelled-data reference, not zero-shot | laptop |
| `<model>-tuned` | not a run: a temperature fitted on 200 labelled dev items and applied to the stored test run | derived |

Kev and Open-Jev each publish their own evaluations: Kev's are in English, on its
development sets; Open-Jev's report accuracy on its own generated data, and no calibration.
TypeSafe's own [workflow evals](https://evals.typesafe.ai/) compare typed workflows against
single prompts on the same models: they rank workflows, this repo ranks models on one
contract. [SemIf, formerly OpenJev](https://openjev.com/), an in-browser demo of the same
constrained-logits trick, is unrelated to Open-Jev.

## What the Jev-like models support, in their code

The README's capability table, cell by cell. Read from Laya 0.3.4 (the installed package),
Open-Jev at `ed45657` and Kev at `f2bb629`, the versions benchmarked. Laya has moved fast
since: 0.3.22 (2026-09-29) adds a server and still encodes the input once per question.

| | Laya | Open-Jev | Kev |
|---|---|---|---|
| built on | ModernBERT-large (English) or mmBERT-base (multilingual), fully fine-tuned, plus a head trained from scratch (model card, "Architecture") | Qwen3.5-2B and 9B, the chat releases, prompted through their chat template (`jev/model.py:65`); a rank-8 LoRA on the attention projections, no MLP (`model.py:46`); the head starts from the model's Yes-minus-No readout (`model.py:27-33`) | Qwen3.5-9B-Base, pretrained only, and Qwen3.8-27B, the chat release; a rank-16 LoRA on attention, MLP and DeltaNet (README, model table and "Training") |
| trained with | RLCD: noise on the logits, a proper scoring rule as the reward, REINFORCE with a group-mean baseline (model card, "Training") | cross-entropy over the options' softmax, one pass over 80,816 rows (`jev/train.py:316`; README, "Training data") | cross-entropy on the right answer, about 12,600 examples for two epochs, then short fine-tunes (README, "Training") |
| how an option is scored | encoder (ModernBERT-large or mmBERT-base) and a small head: each option opens on a marker token, scored, then a softmax | frozen Qwen3.5 with a LoRA and a scalar head: one "is this proposed answer correct?" prompt per option, softmax across options | frozen Qwen with a LoRA and a pointer head: a query token dotted with each option's end token, softmax |
| several questions | one batched pass, one row per question, the input in every row (`laya/common.py:76-85`, `agent.py:259-277`) | up to 4,096, the input in every option sequence of every question; a prefix cache exists, off by default (`jev/api.py:79-84`, `jev/serving.py:42-68`) | any number, the input encoded once and the questions continued from its cache (`kev/model.py:123-124, 395-458`) |
| options of a question | together, sharing the option budget (`common.py:65-81`) | one pass each, so cost grows with the option count (`serving.py:13-35`) | together, in one sequence (`model.py:94-95, 232`) |
| longest input | as shipped, 512 tokens (English) or 1,024 (multilingual); the input is cut silently (`common.py:82-84`); the multilingual one reads up to 8,192 with `max_len=8192` (README, "Long documents") | 4,096 per option sequence, then HTTP 422 (`model.py:69-73`, `server.py:99-100`) | 8,192 for the input and one question together, then HTTP 422 (`model.py:19-20, 96-97`, `serve.py:109-110`) |
| calibration | a temperature per question type and option count; the multilingual checkpoint ships none | one temperature (2B 1.52, 9B 1.90) | one temperature (9B 2.30, 27B 1.38) |
| server | none in 0.3.4 (no server module); since 0.3.9, `laya-serve` on Jev's `/v1/systemone`, one inference thread behind a lock, and a batch route for up to 64 inputs sharing the questions (0.3.22 `serve.py:596-600, 784`) | one request at a time, behind a global lock (`server.py:81, 96-97`) | FastAPI, Jev's `/v1/systemone`, batches up to 64 queued requests (`serve.py:66-73, 118-138`) |

Where a project's README and its code disagree:

- **Kev's input limit.** Its README gives 8,192 tokens for the input and 8,192 for each
  question; the code caps the two together at 8,192.
- **Open-Jev's "batching"** splits the options of one request into chunks. Its HTTP server
  still serves one request at a time.
- **Laya's "single forward pass"** is one batched call, with the input encoded once per
  question: its own table gives 39.5 ms for one question and 158.6 ms for ten. Its summary
  says "calibrated probabilities", while its README says both checkpoints are over-confident as
  shipped.
- **`confidence` is not the same number across them.** Laya's is entropy-based; Open-Jev and
  Kev rescale the top probability, as the bench does for every model that returns none.

Throughput figures are the authors' own: Kev gives 28.9 requests a second for 27B and 79.5
for 9B on one H100, with 64 clients each sending six questions about a new short text, model
time only (README, "Serving"). Laya gives 7.2 ms per question batched on a T4. Open-Jev gives
latencies only.

## One contract for every model

Every model answers one question per call with one option and a probability for each
option, and writes no text.

- **Local LLMs** (`bench/models/hf_logits.py`, `mlx_logits.py`): each option gets a one-token
  label (`A-Z`, then `a-z`, then `0-9`), the prompt ends on an answer prefix, and one forward
  pass gives the next-token scores. The probabilities are the softmax over the label tokens
  alone. Nothing is sampled, so the run is deterministic.
- **Hosted LLMs** (`openai_compat.py`): the same prompt, one user message. The probabilities
  are the softmax over the labels found among the first token's top 20 logprobs, so an
  option outside the top 20 gets 0. OpenRouter picks the provider per call among those that
  honour `logprobs` (the manifest lists which ones served a run), so two runs can hit
  different hardware. The adapter retries 429s, 5xx, dropped connections and timeouts, five
  times with backoff; `retries` on each row says how many it took.
- **`label_mass`**: both LLM adapters store the share of the first-token probability that
  went to the option labels. Below half, the model wanted to write a word, and its answer is
  read out of leftovers; the records reports call the rest *readable*.
- **Open-Jev** scores each option in its own yes/no pass, with prefix caching off as upstream
  ships it, so it costs about four times the tokens of its Qwen base.
- **Kev** runs in process, loaded the way its author's endpoint loads it (bf16, adapter fused
  into the weights, CUDA graphs, the temperature the checkpoint ships with). Its server
  refuses a state and question over 8,192 tokens at this commit (`SERVE_MAX_BRANCH` in
  `kev/model.py`; Kev's README, written later, gives 65,536). A refusal is stored as a row
  with no answer and counted wrong, as Kev's own README counts its over-long documents.
- **Laya** rounds its probabilities to four decimals, so the adapter also stores
  `log_probabilities` from its raw logits, which the `-tuned` rows need.
- **Confidence.** Jev returns one; for a model that does not, the bench applies Jev's
  formula, the chosen probability rescaled from [1/k, 1] to [0, 1]: `(k * p - 1) / (k - 1)`
  (checked at every option count by [P15](probes-bench.md)).

## Running

```bash
# one model on one task: results/bench/<model>/<task>.jsonl and a <task>.json manifest
.venv/bin/python bench/run.py --model qwen3.5-4b --task massive_scenario --sample 100

# every table, results/bench/summary.md, and figures/{risk-coverage,reliability}-<task>.png
.venv/bin/python bench/report.py
```

- A run overwrites that model's previous run of that task. Every condition sees the same
  items.
- Hosted models take `--workers N`, N calls in parallel, rows kept in order. Local models
  refuse anything above 1.
- `--split dev` writes `<task>.dev.jsonl` next to the test run instead; the report ignores it.
  Criteria and temperatures are fitted on dev, never on the rows they are scored on.
- `bench/derive_tuned.py --model <model> --task <task>` fits a temperature on the stored dev
  run and applies it to the stored test run, with no model call, into `<model>-tuned/`.
- The manifest records versions, device, GPU, Hub revisions and the git commit. Every stored
  run's commit ends in `-dirty`, because an untracked file existed: read it as the commit the
  run started from.
- `summary.md` holds every table the docs quote. A few counts in the docs are one-off queries
  over the same rows: label positions, the McNemar test, which option a model falls back on.

## Rented GPUs

Every open-weight model that runs through `transformers` runs on
[Modal](https://modal.com): `bench/modal_run.py` runs `bench/run.py` on a Modal GPU and copies
the results back, and the manifest names the GPU. The first run builds the image and
downloads the weights into a Modal volume; later runs reuse both.

```bash
.venv/bin/modal setup   # once: log in
.venv/bin/modal run bench/modal_run.py --model qwen3.5-9b --task massive_intent --sample 500 --gpu A100-80GB
.venv/bin/modal run bench/modal_run.py --model kev-27b --task reply_matching --sample 500 --gpu H100
.venv/bin/modal run bench/modal_run.py --probe p18_openjev.py --model open-jev-2b
```

- Kev pins `torch<2.9` against the bench's 2.14, so it gets its own image: the one its
  author's endpoint uses, at that script's `KEV_REF`.
- Run one Kev-27B task alone first: its 55 GB of weights reach the volume when that container
  exits, and runs started before then each download their own copy.
- Launch at most three runs at a time: more hits Modal's app-creation rate limit.
- A full pass of one model costs about $0.10 to $0.35 on an A100-80GB.

## Reproducing every stored run

Test runs use seed 42; `--sample` is 500 on MASSIVE and matching, 1439 on routing (the whole
test split), 156 on records.

| models | tasks | how |
|---|---|---|
| `jev-1.13.0`, `gpt-4o-mini-openrouter` | all six | local, `--workers 6` to `8` |
| `mistral-nemo-openrouter` | MASSIVE, routing, matching | local, `--workers 3` to `8` |
| `kev-9b`, `kev-27b` | all six | Modal, `--gpu L40S` and `--gpu H100` |
| `qwen3.5-9b` | all six | Modal, `--gpu A100-80GB` |
| `qwen3.5-4b` | MASSIVE, records | Modal, `--gpu A100-80GB` |
| `qwen3.5-2b`, `minicpm5-2b`, `open-jev-2b`, `open-jev-9b` | MASSIVE | Modal, `--gpu A100-80GB` |
| `laya-router`, `laya-multilingual`, `qwen3.5-0.8b-mlx-4bit` | MASSIVE | local |
| `xlm-r-base-massive-intent` | `massive_intent` | local |

The dev runs: `laya-router` and `open-jev-9b` on both MASSIVE tasks (`--split dev --sample
200 --seed 7`), then `derive_tuned.py` for each; `jev-1.13.0` on routing (`--split dev
--sample 200`) and matching (`--split dev --sample 100`), which fixed their wording.

## Adding a model or a task

- **A model** is one line in `REGISTRY` (`bench/models/__init__.py`), plus an adapter module
  if the family is new; the module docstring there gives the interface. Add a price to
  `PRICES` if it is billed per token. The charts draw only the models in `CHARTED`, the
  README's six; adding one means a colour that still passes the palette checks. The registry name is the
  results directory, so it must pin what it runs, never an alias that moves.
- **A task** is a module in `bench/tasks/` with `NAME`, `CONDITIONS`, `load_pairs(sample, seed,
  split)` and `question(condition)`, and optionally `report(models)` for its own section in
  `summary.md`.

## Latency and cost

The `p50 ms` and `$/1k items` columns of `summary.md` do not compare across rows:

- Jev and the OpenRouter models are hosted APIs called from France, network included, 3 to 8
  calls in parallel. Jev's median moved from 237 ms to 480 ms between two sessions on the
  same code.
- The `transformers` rows time one forward pass in bf16 on a Modal A100-80GB, one call at a
  time; Modal does not pin the variant (SXM4 or PCIe, in the manifest). Qwen3.5 runs without
  the `causal_conv1d` kernel, so its latency is pessimistic, and Open-Jev's with it.
- Kev runs in process on an L40S or an H100, with no network; rows store `model_ms`, Kev's
  own model time, next to `elapsed_ms`. Its server batches concurrent requests, which this
  bench does not measure.
- The 0.8B MLX row times an M3.
- `$/1k items` is per-token billing from `PRICES` (OpenRouter's listing on 2026-09-22); a
  self-hosted model costs GPU hours instead, and none of these is what a served deployment
  would show.

**Calls per second were not measured as capacity.** With 8 calls in parallel, Jev served
about 7 routing calls a second and GPT-4o mini about 6, without a rate limit: that is our
client's pace, not the services' ceiling. The GPU rows ran one call at a time, so their pace
is 1,000 divided by the mean latency: on routing, 3.4 calls a second for Kev-27B on an H100,
6.3 for Qwen3.5 9B on an A100. At that pace an H100 at about $4 an hour costs $0.33 per
1,000 calls, three times Jev's $0.10; it only gets cheaper by batching concurrent requests,
which Kev's server does and this bench did not measure.
