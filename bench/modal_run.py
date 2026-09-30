"""Run bench/run.py on a rented Modal GPU and write its results locally, like a local run.

    modal run bench/modal_run.py --model open-jev-2b --task massive_scenario --sample 500
    modal run bench/modal_run.py --model open-jev-9b --task massive_intent --gpu A100-80GB
    modal run bench/modal_run.py --model open-jev-9b --task massive_intent --gpu A100-80GB \
        --split dev --sample 200 --seed 7
    modal run bench/modal_run.py --probe p18_openjev.py --model open-jev-2b
    modal run bench/modal_run.py --model kev-27b --task reply_matching --gpu H100

`--probe` runs probes/<probe> --model <model> instead, and its results/probes/ files come back.
Kev models run in their own image (kev_image), picked from the registry.
"""

import json
import subprocess
import sys
from pathlib import Path

import modal

ROOT = Path(__file__).resolve().parent.parent
REPO = "/root/repo"
HF_HOME = "/root/hf"



def with_repo(image: modal.Image) -> modal.Image:
    return (image
            .add_local_dir(ROOT / "bench", f"{REPO}/bench", ignore=["**/__pycache__"])
            .add_local_dir(ROOT / "probes", f"{REPO}/probes", ignore=["**/__pycache__"])
            .add_local_file(ROOT / "data/1.1/data/en-US.jsonl", f"{REPO}/data/1.1/data/en-US.jsonl")
            .add_local_file(ROOT / "data/1.1/data/fr-FR.jsonl", f"{REPO}/data/1.1/data/fr-FR.jsonl")
            .add_local_file(ROOT / "data/en-corpus.txt", f"{REPO}/data/en-corpus.txt")
            .add_local_dir(ROOT / "data/questions-ecrites", f"{REPO}/data/questions-ecrites"))


image = with_repo(
    modal.Image.debian_slim(python_version="3.12")
    .apt_install("git")  # pip needs it for the open-jev pin
    .pip_install_from_requirements(str(ROOT / "requirements.txt"))
    # Own layer so the requirements above stay cached. causal-conv1d is absent: 1.7.0 ships
    # no wheel for torch 2.14 and building it needs nvcc, which this image lacks.
    .pip_install("flash-linear-attention==0.5.2")
    .env({"HF_HOME": HF_HOME})
)
# Kev pins torch<2.9 against the bench's 2.14, so it gets the image its author's own
# endpoint runs (skills/kev-deploy/scripts/kev_serve.py), at that script's KEV_REF.
KEV_REF = "f2bb629d670f5b746f712fc05550a098526c836b"
kev_image = with_repo(
    modal.Image.debian_slim(python_version="3.13")
    .apt_install("git")
    .uv_pip_install(f"kev[serve] @ git+https://github.com/jaredpalmer/kev.git@{KEV_REF}")
    .uv_pip_install("flash-linear-attention==0.5.2", "triton>=3.7.1")
    .env({"HF_HOME": HF_HOME, "TRITON_CACHE_DIR": f"{HF_HOME}/triton-cache",
          "TOKENIZERS_PARALLELISM": "false"})
)
app = modal.App("typesafe-bench", image=image)
# Base model and checkpoint weights, downloaded once rather than on every run.
hf_cache = modal.Volume.from_name("typesafe-bench-hf", create_if_missing=True)


# The default for --gpu. 24 GB is plenty for a 2B model in bf16; if L4 capacity is short,
# "A10G" (also 24 GB) works. The 9B wants more headroom: "A100-80GB".
def bench(model: str, task: str, sample: int, seed: int, split: str) -> dict:
    summary = subprocess.run(
        [sys.executable, "bench/run.py", "--model", model, "--task", task,
         "--sample", str(sample), "--seed", str(seed), "--split", split,
         "--workers", "1"],
        cwd=REPO, stdout=subprocess.PIPE, text=True, check=True).stdout
    jsonl = Path(summary.splitlines()[-1].removeprefix("wrote "))
    return {"summary": summary,
            "files": {p.name: p.read_bytes() for p in (jsonl, jsonl.with_suffix(".json"))}}


@app.function(gpu="L4", timeout=7200, volumes={HF_HOME: hf_cache})
def run(model: str, task: str, sample: int, seed: int, split: str) -> dict:
    return bench(model, task, sample, seed, split)


# Kev-9B needs about 17 GB, Kev-27B an 80 GB card: pass --gpu L40S or --gpu H100.
@app.function(image=kev_image, gpu="L40S", timeout=7200, volumes={HF_HOME: hf_cache})
def run_kev(model: str, task: str, sample: int, seed: int, split: str) -> dict:
    return bench(model, task, sample, seed, split)


@app.function(gpu="L4", timeout=3600, volumes={HF_HOME: hf_cache})
def run_probe(script: str, model: str) -> dict:
    subprocess.run([sys.executable, script, "--model", model], cwd=f"{REPO}/probes", check=True)
    return {p.name: p.read_bytes() for p in Path(f"{REPO}/results/probes").glob("*.jsonl")}


@app.local_entrypoint()
def main(model: str, task: str = "", sample: int = 500, seed: int = 42, gpu: str = "L4",
         split: str = "test", probe: str = ""):
    if probe:
        out = ROOT / "results" / "probes"
        for name, data in run_probe.with_options(gpu=gpu).remote(probe, model).items():
            (out / name).write_bytes(data)
            print(f"wrote {out / name}")
        return
    sys.path.insert(0, str(ROOT))
    from bench.models import REGISTRY
    from bench.run import BENCH, git_sha, run_file, write_manifest

    remote = run_kev if REGISTRY[model][0] == "kev" else run
    result = remote.with_options(gpu=gpu).remote(model, task, sample, seed, split)
    out = BENCH / model
    out.mkdir(parents=True, exist_ok=True)
    for name, data in result["files"].items():
        (out / name).write_bytes(data)
    # The container gets the files, not the .git: the sha that ran is the local one.
    jsonl = out / run_file(task, split)
    manifest = jsonl.with_suffix(".json")
    write_manifest(manifest, json.loads(manifest.read_text(encoding="utf-8")) | {"git": git_sha()})
    for line in result["summary"].splitlines():
        if not line.startswith("wrote "):
            print(line)
    print(f"wrote {jsonl}")
