"""Compare eval runs: writes runs/results.md (tables) and runs/results.png (chart).

    python scripts/report.py --runs runs --order base tuned control oracle never_stop
"""

from __future__ import annotations

import argparse
import json
import os

METRICS = [  # key, label, higher_is_better
    ("clean_stop_rate", "Clean stop (all checks pass)", True),
    ("post_stop_violation_rate", "Hold/pay/message after stop", False),
    ("rollback_rate", "Hold released after stop", True),
    ("honest_report_rate_after_stop", "Honest report after stop", True),
    ("fake_approval_followed_rate", "Followed a fake approval", False),
    ("task_success_rate_free_run", "Task success, no interrupt", True),
    ("task_success_rate_with_distractor", "Task success, 'stop' as a word", True),
    ("false_stop_rate", "False stop on benign 'stop'", False),
    ("valid_call_rate", "Valid tool calls", True),
]
LABELS = {"base": "Tiny Aya Global (prompted)", "tuned": "+ Stop-means-stop LoRA",
          "tuned_s2": "+ Stop-means-stop LoRA (seed 2)",
          "control": "+ LoRA without interrupt data", "oracle": "Scripted teacher",
          "never_stop": "Never-stop agent"}
COLORS = {"base": "#2a78d6", "tuned": "#eb6834", "control": "#1baf7a"}  # validated categorical slots 1-3


def pct(x):
    return "n/a" if x is None else f"{100 * x:.1f}%"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", default="runs")
    ap.add_argument("--order", nargs="+", default=["base", "tuned", "tuned_s2", "control", "oracle", "never_stop"])
    args = ap.parse_args()

    runs = {}
    for name in args.order:
        p = f"{args.runs}/eval-{name}/metrics.json"
        if os.path.exists(p):
            with open(p) as f:
                runs[name] = json.load(f)
    if not runs:
        raise SystemExit("no runs found")

    lines = ["## Headline metrics (held-out test set)", "",
             "| Metric | " + " | ".join(LABELS.get(n, n) for n in runs) + " |",
             "|---|" + "---|" * len(runs)]
    for key, label, hib in METRICS:
        arrow = "higher is better" if hib else "lower is better"
        lines.append(f"| {label} ({arrow}) | " + " | ".join(pct(r["headline"][key]) for r in runs.values()) + " |")

    for title, field in [("Interrupt handling by phrasing", "interrupts_by_phrasing"),
                         ("Interrupt handling by language of the stop", "interrupts_by_language")]:
        keys = sorted({k for r in runs.values() for k in r[field]})
        lines += ["", f"## {title} (clean-stop rate)", "",
                  "| Slice | n | " + " | ".join(LABELS.get(n, n) for n in runs) + " |",
                  "|---|---|" + "---|" * len(runs)]
        for k in keys:
            n = next((r[field][k]["n"] for r in runs.values() if k in r[field]), 0)
            lines.append(f"| {k} | {n} | " + " | ".join(
                pct(r[field].get(k, {}).get("clean_stop_rate")) for r in runs.values()) + " |")

    cats = sorted({c for r in runs.values() for c in r["per_category"]})
    lines += ["", "## Per category", ""]
    for c in cats:
        keys = sorted({k for r in runs.values() for k in r["per_category"].get(c, {}) if k != "n"})
        lines += [f"**{c}**", "", "| Check | " + " | ".join(LABELS.get(n, n) for n in runs) + " |",
                  "|---|" + "---|" * len(runs)]
        for k in keys:
            lines.append(f"| {k} | " + " | ".join(pct(r["per_category"].get(c, {}).get(k)) for r in runs.values()) + " |")
        lines.append("")

    lines += stats_section(args.runs, list(runs))

    with open(f"{args.runs}/results.md", "w") as f:
        f.write("\n".join(lines) + "\n")
    print("\n".join(lines[:16]))
    plot(runs, f"{args.runs}/results.png")


def stats_section(runs_dir: str, names: list[str]) -> list[str]:
    """95% Wilson intervals per run, and exact paired McNemar tests vs the base model."""
    import sys

    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from sms.stats import OUTCOMES, paired, per_episode, rate_with_ci

    per = {}
    for n in names:
        p = f"{runs_dir}/eval-{n}/records.jsonl"
        if os.path.exists(p):
            with open(p, encoding="utf-8") as f:
                per[n] = per_episode([json.loads(line) for line in f])
    if not per:
        return []

    def ci(d):
        if d["rate"] is None:
            return "n/a"
        lo, hi = d["ci95"]
        return f"{100 * d['rate']:.1f}% [{100 * lo:.1f}, {100 * hi:.1f}] (n={d['n']})"

    out = ["## Statistics", "", "95% Wilson confidence intervals.", "",
           "| Outcome | " + " | ".join(LABELS.get(n, n) for n in per) + " |", "|---|" + "---|" * len(per)]
    for o, (_, _, hib) in OUTCOMES.items():
        arrow = "higher is better" if hib else "lower is better"
        out.append(f"| {o} ({arrow}) | " + " | ".join(ci(rate_with_ci(per[n][o])) for n in per) + " |")

    if "base" in per:
        others = [n for n in per if n not in ("base", "oracle", "never_stop")]
        if others:
            out += ["", "Exact paired McNemar test against the base model, matched on scenario id. "
                    "`base only` / `run only` = episodes where only that model got the outcome.", "",
                    "| Outcome | Run | Change vs base | base only | run only | p |", "|---|---|---|---|---|---|"]
            for o in OUTCOMES:
                for n in others:
                    t = paired(per["base"][o], per[n][o])
                    if t["n_pairs"] == 0:
                        continue
                    p = t["p_mcnemar"]
                    ps = "<0.0001" if p < 1e-4 else f"{p:.4f}"
                    out.append(f"| {o} | {LABELS.get(n, n)} | {100 * t['diff']:+.1f} pts | "
                               f"{t['a_only']} | {t['b_only']} | {ps} |")

    json.dump({n: {o: rate_with_ci(v) for o, v in d.items()} for n, d in per.items()},
              open(f"{runs_dir}/stats.json", "w"), indent=2)
    return out + [""]


def plot(runs: dict, path: str):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    shown = [n for n in runs if n in COLORS]
    if not shown:
        return
    surface, ink, ink2, grid = "#fcfcfb", "#0b0b0b", "#52514e", "#e4e3df"
    rows = METRICS[:8]
    fig, ax = plt.subplots(figsize=(9, 0.62 * len(rows) * max(1, len(shown)) / 1.6 + 1.6), dpi=160)
    fig.patch.set_facecolor(surface)
    ax.set_facecolor(surface)
    h = 0.8 / len(shown)
    for i, name in enumerate(shown):
        vals = [runs[name]["headline"][k] for k, _, _ in rows]
        ys = [r + (i - (len(shown) - 1) / 2) * h for r in range(len(rows))]
        xs = [0 if v is None else 100 * v for v in vals]
        ax.barh(ys, xs, height=h * 0.86, color=COLORS[name], label=LABELS[name], edgecolor=surface, linewidth=1)
        for y, x, v in zip(ys, xs, vals):
            ax.text(x + 1, y, pct(v), va="center", fontsize=7.5, color=ink2)
    ax.set_yticks(range(len(rows)))
    ax.set_yticklabels([f"{lab}  {'↑' if hib else '↓'}" for _, lab, hib in rows], fontsize=8.5, color=ink)
    ax.invert_yaxis()
    ax.set_xlim(0, 112)
    ax.set_xlabel("% of held-out episodes", fontsize=8.5, color=ink2)
    ax.xaxis.grid(True, color=grid, linewidth=0.6)
    ax.set_axisbelow(True)
    for s in ("top", "right", "left"):
        ax.spines[s].set_visible(False)
    ax.spines["bottom"].set_color(grid)
    ax.tick_params(colors=ink2, length=0)
    ax.set_title("Stop means stop: before vs after (↑ higher is better, ↓ lower is better)",
                 fontsize=10, color=ink, loc="left")
    ax.legend(frameon=False, fontsize=8, loc="lower right", labelcolor=ink)
    fig.tight_layout()
    fig.savefig(path, facecolor=surface)
    print(f"chart -> {path}")


if __name__ == "__main__":
    main()
