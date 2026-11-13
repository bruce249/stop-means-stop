"""Generate scenarios, roll out the scripted teacher, curate with CuratorKIT,
and write AgentTune EventLogs for distillation. CPU only.

Outputs (under --out, default data/):
  train_scenarios.jsonl / test_scenarios.jsonl
  train_teacher_records.jsonl       full teacher episodes (transcript + final state)
  train_eventlogs.jsonl             AgentTune full-tier trajectories (all curated ids)
  curatorkit/                       CuratorKIT run: sft_messages.jsonl, manifest, provenance
  splits.json                       curated ids for the main run and the no-interrupt control
  data_card.json                    counts and length stats
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from sms.agenttune_bridge import expected_sft_messages, record_to_events, write_events_jsonl  # noqa: E402
from sms.episode import OraclePolicy, run_episodes  # noqa: E402
from sms.scenarios import TEST_MIX, describe, make_split  # noqa: E402

DEFAULT_TRAIN_MIX = {"clean": 260, "no_option": 60, "stop_1": 120, "stop_2": 120, "stop_3": 120,
                     "stop_inject": 140, "distractor": 180}
CONTROL_CATS = {"clean", "no_option", "distractor"}


def write_jsonl(path, rows):
    with open(path, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="data")
    ap.add_argument("--scale", type=float, default=1.0, help="multiply the train mix")
    ap.add_argument("--minhash-threshold", type=float, default=0.97)
    ap.add_argument("--skip-curatorkit", action="store_true")
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)

    mix = {k: max(1, int(v * args.scale)) for k, v in DEFAULT_TRAIN_MIX.items()}
    test = make_split("test", TEST_MIX)
    test_tasks = {s["task"] for s in test}
    train = [s for s in make_split("train", mix) if s["task"] not in test_tasks]  # no task leaks
    write_jsonl(f"{args.out}/train_scenarios.jsonl", train)
    write_jsonl(f"{args.out}/test_scenarios.jsonl", test)

    # The teacher plays every turn of every training episode (no forced prefix needed).
    records = run_episodes(train, OraclePolicy(), mode_fn=lambda s: "free", progress=False)
    write_jsonl(f"{args.out}/train_teacher_records.jsonl", records)

    sft_rows = [{"id": r["scenario_id"], "category": r["category"],
                 "messages": expected_sft_messages(record_to_events(r))} for r in records]
    sft_path = f"{args.out}/sft_rows_raw.jsonl"
    write_jsonl(sft_path, sft_rows)

    kept = [r["id"] for r in sft_rows]
    ck_stats = None
    if not args.skip_curatorkit:
        from curatorkit import Curator, CuratorConfig

        ck_out = f"{args.out}/curatorkit"
        shutil.rmtree(ck_out, ignore_errors=True)
        cfg = CuratorConfig(dataset=sft_path, format="auto", dedup="minhash",
                            minhash_threshold=args.minhash_threshold, clean=False,
                            min_tokens=10, max_tokens=4096, output_dir=ck_out,
                            export_formats=["messages"], name="stop-means-stop-teacher")
        res = Curator(cfg).run()
        kept = [s.id for s in res.passed]
        ck_stats = {"input": len(sft_rows), "passed": len(res.passed), "rejected": len(res.rejected)}

    write_events_jsonl(records, f"{args.out}/train_eventlogs.jsonl")
    cat = {r["id"]: r["category"] for r in sft_rows}
    splits = {
        "main": kept,
        "control_no_interrupts": [i for i in kept if cat[i] in CONTROL_CATS],
    }
    with open(f"{args.out}/splits.json", "w") as f:
        json.dump(splits, f)

    chars = sorted(sum(len(m["content"]) for m in r["messages"]) for r in sft_rows)
    card = {
        "train": describe(train), "test": describe(test), "curatorkit": ck_stats,
        "n_main": len(splits["main"]), "n_control": len(splits["control_no_interrupts"]),
        "conversation_chars": {"p50": chars[len(chars) // 2], "p95": chars[int(len(chars) * 0.95)],
                               "max": chars[-1]},
    }
    with open(f"{args.out}/data_card.json", "w") as f:
        json.dump(card, f, indent=2)
    print(json.dumps(card, indent=2))


if __name__ == "__main__":
    main()
