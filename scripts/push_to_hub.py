"""Publish everything to the Hugging Face Hub, public and un-gated, and group it
into the two collections the submission form asks for.

    HF_TOKEN=... python scripts/push_to_hub.py --user <hf-username>

Creates (or updates):
  model    <user>/tiny-aya-global-stop-means-stop-lora     the LoRA adapter
  model    <user>/tiny-aya-global-sms-control-lora         control adapter (if trained)
  dataset  <user>/stop-means-stop-data                     scenarios, teacher trajectories, CuratorKIT run
  dataset  <user>/stop-means-stop-evals                    eval transcripts, metrics, AuditKIT reports, logs
  space    <user>/stop-means-stop                          static replay viewer
  collections "Stop means stop: models" and "Stop means stop: data"
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
import tempfile

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

BASE = "CohereLabs/tiny-aya-global"


def load_jsonl(path):
    with open(path, encoding="utf-8") as f:
        return [json.loads(line) for line in f]


def headline(runs_dir, name):
    p = f"{runs_dir}/eval-{name}/metrics.json"
    if not os.path.exists(p):
        return {}
    with open(p) as f:
        return json.load(f)["headline"]


def model_card(repo, github, results_md, control=False):
    what = ("Control adapter: same recipe, trained only on episodes without interrupts. "
            "It isolates how much of the stop behaviour comes from interrupt data versus format learning."
            if control else
            "LoRA adapter that teaches Tiny Aya Global to stop a tool-using task when the user says so "
            "(English, Hinglish, Hindi), release reversible holds, never pay or message after a stop, "
            "ignore fake approvals planted in tool output, and report truthfully what already happened.")
    return f"""---
license: cc-by-nc-4.0
base_model: {BASE}
library_name: peft
tags: [lexsi-hackathon, agenttune, auditkit, curatorkit, tiny-aya, agents, tool-use, corrigibility, ai-safety, lora]
language: [en, hi]
---

# {repo.split('/')[-1]}

{what}

- Base model: [{BASE}](https://huggingface.co/{BASE}) (CC-BY-NC-4.0, Cohere Labs Acceptable Use Policy applies)
- Trained with Lexsi Labs **AgentTune** (`create_distill_trainer`, LoRA SFT on scripted-teacher trajectories)
- Evaluated with Lexsi Labs **AuditKIT** (`agent_eval`, verified-outcome predicates) on a deterministic simulator
- Code: {github}

## Usage

```python
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel
tok = AutoTokenizer.from_pretrained("{repo}")
model = AutoModelForCausalLM.from_pretrained("{BASE}", device_map="auto")
model = PeftModel.from_pretrained(model, "{repo}")
```

The prompt format (tool list, rules, `<tool_call>` syntax) is in `sms/prompting.py` in the repo.

## Results

{results_md}

Research and hackathon use only. Not certified for deployment.
"""


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--user", required=True)
    ap.add_argument("--github", default="https://github.com/<you>/stop-means-stop")
    ap.add_argument("--data", default="data")
    ap.add_argument("--runs", default="runs")
    ap.add_argument("--adapter", default="runs/adapter-main")
    ap.add_argument("--control-adapter", default="runs/adapter-control")
    args = ap.parse_args()

    from huggingface_hub import HfApi

    api = HfApi(token=os.environ.get("HF_TOKEN"))
    u = args.user
    results_md = open(f"{args.runs}/results.md").read() if os.path.exists(f"{args.runs}/results.md") else "See the evals dataset."
    urls = {}

    def push_folder(repo_id, repo_type, folder, readme=None, space_sdk=None):
        api.create_repo(repo_id, repo_type=repo_type, private=False, exist_ok=True, space_sdk=space_sdk)
        if readme is not None:
            with open(os.path.join(folder, "README.md"), "w", encoding="utf-8") as f:
                f.write(readme)
        api.upload_folder(repo_id=repo_id, repo_type=repo_type, folder_path=folder)
        prefix = {"model": "", "dataset": "datasets/", "space": "spaces/"}[repo_type]
        urls[repo_id] = f"https://huggingface.co/{prefix}{repo_id}"
        print("pushed", urls[repo_id])

    # 1. adapters
    model_repos = []
    for path, name, control in [(args.adapter, f"{u}/tiny-aya-global-stop-means-stop-lora", False),
                                ("runs/adapter-s2", f"{u}/tiny-aya-global-stop-means-stop-lora-seed2", False),
                                (args.control_adapter, f"{u}/tiny-aya-global-sms-control-lora", True)]:
        if os.path.isdir(path):
            tmp = tempfile.mkdtemp()
            for fn in os.listdir(path):
                if fn.startswith("checkpoint-"):
                    continue
                src = os.path.join(path, fn)
                (shutil.copytree if os.path.isdir(src) else shutil.copy2)(src, os.path.join(tmp, fn))
            push_folder(name, "model", tmp, model_card(name, args.github, results_md, control))
            model_repos.append(name)

    # 2. data
    data_repo = f"{u}/stop-means-stop-data"
    tmp = tempfile.mkdtemp()
    for fn in os.listdir(args.data):
        if fn == "sft_rows_raw.jsonl":
            continue
        src = os.path.join(args.data, fn)
        (shutil.copytree if os.path.isdir(src) else shutil.copy2)(src, os.path.join(tmp, fn))
    push_folder(data_repo, "dataset", tmp, f"""---
license: cc-by-4.0
tags: [lexsi-hackathon, curatorkit, agenttune, agents, tool-use, corrigibility, synthetic]
language: [en, hi]
---
# Stop means stop: training data

Synthetic booking-and-payments episodes for teaching an agent to stop when told.
All content is synthetic and generated by a deterministic simulator (no real people, cards or bookings).

- `train_scenarios.jsonl`, `test_scenarios.jsonl`: scenario specs (disjoint seeds; test mixes seen and held-out stop phrasings)
- `train_teacher_records.jsonl`: full scripted-teacher episodes with the simulator's final state
- `train_eventlogs.jsonl`: the same episodes as AgentTune full-tier EventLogs (what the student is distilled from)
- `curatorkit/`: CuratorKIT run over the SFT rows (schema gate, exact + MinHash dedup, manifest, `lexsi_provenance.json`)
- `splits.json`: curated ids for the main run and the no-interrupt control
- `data_card.json`: counts by category, phrasing and language

Code: {args.github}
""")

    # 3. evals
    evals_repo = f"{u}/stop-means-stop-evals"
    tmp = tempfile.mkdtemp()
    for fn in os.listdir(args.runs):
        src = os.path.join(args.runs, fn)
        if fn.startswith("eval-"):
            shutil.copytree(src, os.path.join(tmp, fn))
        elif fn.startswith("adapter-") and os.path.exists(os.path.join(src, "train_meta.json")):
            os.makedirs(os.path.join(tmp, "train_logs"), exist_ok=True)
            shutil.copy2(os.path.join(src, "train_meta.json"), os.path.join(tmp, "train_logs", f"{fn}.json"))
        elif fn in ("results.md", "results.png") or fn.endswith(".log"):
            shutil.copy2(src, os.path.join(tmp, fn))
    push_folder(evals_repo, "dataset", tmp, f"""---
license: cc-by-4.0
tags: [lexsi-hackathon, auditkit, agent-evaluation, corrigibility]
---
# Stop means stop: evaluation logs

Every evaluated episode (full transcript, every tool call, the simulator's final state), the
metrics computed from it, and AuditKIT agent_eval reports. Training logs are in `train_logs/`.

![results](results.png)

{results_md}

Code: {args.github}
""")

    # 4. space (static replay viewer)
    from sms.viewer import build_html

    runs = {n: load_jsonl(f"{args.runs}/eval-{n}/records.jsonl")
            for n in ("base", "tuned", "control") if os.path.exists(f"{args.runs}/eval-{n}/records.jsonl")}
    space_repo = f"{u}/stop-means-stop"
    if runs:
        tmp = tempfile.mkdtemp()
        with open(os.path.join(tmp, "index.html"), "w", encoding="utf-8") as f:
            f.write(build_html(runs, {n: headline(args.runs, n) for n in runs}))
        push_folder(space_repo, "space", tmp, """---
title: Stop means stop
emoji: ✋
colorFrom: blue
colorTo: red
sdk: static
pinned: false
license: cc-by-nc-4.0
tags: [lexsi-hackathon]
---
Replay viewer for the Stop means stop evaluation.
""", space_sdk="static")

    # 5. collections
    def collection(title, desc, items):
        c = api.create_collection(title=title, namespace=u, description=desc, private=False, exists_ok=True)
        for item_id, item_type in items:
            api.add_collection_item(c.slug, item_id=item_id, item_type=item_type, exists_ok=True)
        print("collection", f"https://huggingface.co/collections/{c.slug}")
        return f"https://huggingface.co/collections/{c.slug}"

    m_items = [(r, "model") for r in model_repos] + ([(space_repo, "space")] if runs else [])
    coll_models = collection("Stop means stop: models", "Interruptible Tiny Aya agent (Lexsi hackathon)", m_items)
    coll_data = collection("Stop means stop: data", "Training data, eval logs and reports (Lexsi hackathon)",
                           [(data_repo, "dataset"), (evals_repo, "dataset")])
    summary = {"model_collection": coll_models, "dataset_collection": coll_data, "repos": urls}
    with open(f"{args.runs}/hub_links.json", "w") as f:
        json.dump(summary, f, indent=2)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
