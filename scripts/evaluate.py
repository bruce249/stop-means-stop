"""Evaluate a policy on the held-out test scenarios and score it.

    python scripts/evaluate.py --policy oracle --name oracle
    python scripts/evaluate.py --policy never_stop --name never_stop
    python scripts/evaluate.py --policy hf --name base
    python scripts/evaluate.py --policy hf --adapter runs/adapter-main --name tuned

Writes runs/eval-<name>/{records.jsonl, metrics.json, auditkit.json}.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from sms.episode import NeverStopPolicy, OraclePolicy, run_episodes  # noqa: E402
from sms.metrics import aggregate  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--policy", choices=["oracle", "never_stop", "hf"], required=True)
    ap.add_argument("--name", required=True)
    ap.add_argument("--model", default="CohereLabs/tiny-aya-global")
    ap.add_argument("--adapter", default=None)
    ap.add_argument("--data", default="data")
    ap.add_argument("--runs", default="runs")
    ap.add_argument("--batch-size", type=int, default=8)
    ap.add_argument("--limit", type=int, default=None, help="evaluate only the first N scenarios")
    ap.add_argument("--no-audit", action="store_true")
    args = ap.parse_args()

    with open(f"{args.data}/test_scenarios.jsonl", encoding="utf-8") as f:
        scenarios = [json.loads(line) for line in f]
    if args.limit:
        # keep the category mix when limiting
        by = {}
        for s in scenarios:
            by.setdefault(s["category"], []).append(s)
        frac = args.limit / len(scenarios)
        scenarios = [s for v in by.values() for s in v[: max(1, round(len(v) * frac))]]

    if args.policy == "oracle":
        policy = OraclePolicy()
    elif args.policy == "never_stop":
        policy = NeverStopPolicy()
    else:
        from sms.hf_policy import HFPolicy

        policy = HFPolicy(args.model, adapter=args.adapter, name=args.name)

    out = f"{args.runs}/eval-{args.name}"
    os.makedirs(out, exist_ok=True)
    t0 = time.time()
    records = run_episodes(scenarios, policy, batch_size=args.batch_size)
    minutes = round((time.time() - t0) / 60, 1)

    with open(f"{out}/records.jsonl", "w", encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    metrics = aggregate(records)
    metrics["run"] = {"name": args.name, "policy": args.policy, "model": args.model,
                      "adapter": args.adapter, "minutes": minutes, "n": len(records)}
    with open(f"{out}/metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)
    print(json.dumps({"run": metrics["run"], **metrics["headline"]}, indent=2))

    if not args.no_audit:
        from sms.auditkit_bridge import audit_records

        rep = audit_records(records, args.name)
        with open(f"{out}/auditkit.json", "w") as f:
            json.dump(rep, f, indent=2, default=str)
        print(rep["summary_text"])


if __name__ == "__main__":
    main()
