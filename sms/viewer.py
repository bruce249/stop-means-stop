"""Builds the static replay viewer used as the Hugging Face Space.

Pick an episode, see where the user said stop, and compare what each model did
next, turn by turn, with the verified verdict for each run.
"""

from __future__ import annotations

import json

from .metrics import episode_verdict

RUN_LABELS = {"base": "Tiny Aya Global (prompted)", "tuned": "+ Stop-means-stop LoRA",
              "control": "+ LoRA without interrupt data"}


def _compact(rec: dict) -> dict:
    v = episode_verdict(rec)
    turns = []
    for i, t in enumerate(rec["turns"]):
        user_line = None
        obs_idx = 2 + 2 * i
        if obs_idx < len(rec["messages"]):
            text = rec["messages"][obs_idx]["content"]
            if "\n[user] " in text:
                user_line = text.split("\n[user] ", 1)[1]
        res = t.get("result", {})
        turns.append({"forced": t["forced"], "phase": t["phase"], "name": t["name"], "args": t["arguments"],
                      "error": t["parse_error"], "raw": t["text"][:400] if t["parse_error"] else None,
                      "notice": res.get("notice"), "user": user_line})
    return {"turns": turns, "verdict": {k: v[k] for k in v if isinstance(v[k], (bool, type(None)))},
            "end": rec["end_reason"]}


def build_html(runs: dict[str, list[dict]], headline: dict[str, dict]) -> str:
    names = [n for n in RUN_LABELS if n in runs]
    first = runs[names[0]]
    episodes = []
    by_run = {n: {r["scenario_id"]: r for r in runs[n]} for n in names}
    for rec in first:
        sid = rec["scenario_id"]
        episodes.append({
            "id": sid, "category": rec["category"], "task": rec["messages"][0]["content"].split("Task: ", 1)[-1],
            "event": rec["event"], "runs": {n: _compact(by_run[n][sid]) for n in names if sid in by_run[n]},
        })
    data = {"runs": names, "labels": {n: RUN_LABELS[n] for n in names}, "episodes": episodes,
            "headline": {n: headline.get(n, {}) for n in names}}
    return TEMPLATE.replace("__DATA__", json.dumps(data, ensure_ascii=False))


TEMPLATE = r"""<!doctype html>
<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>Stop means stop</title>
<style>
:root{--bg:#fcfcfb;--card:#ffffff;--ink:#0b0b0b;--ink2:#52514e;--line:#e4e3df;--accent:#2a78d6;--bad:#c23b3a;--good:#1d7f3f;--forced:#f1f0ec;--userbg:#fff4e8}
@media (prefers-color-scheme:dark){:root{--bg:#1a1a19;--card:#232322;--ink:#f0efec;--ink2:#c3c2b7;--line:#3a3a38;--accent:#3987e5;--bad:#e66767;--good:#4caf6e;--forced:#2a2a28;--userbg:#3a2c1c}}
*{box-sizing:border-box}body{margin:0;background:var(--bg);color:var(--ink);font:15px/1.5 system-ui,-apple-system,Segoe UI,Roboto,sans-serif}
main{max-width:1180px;margin:0 auto;padding:24px 16px 64px}
h1{font-size:24px;margin:0 0 4px}p.sub{color:var(--ink2);margin:0 0 20px}
.controls{display:flex;flex-wrap:wrap;gap:12px;margin-bottom:16px}
select{font:inherit;padding:6px 10px;border:1px solid var(--line);border-radius:8px;background:var(--card);color:var(--ink)}
.task{background:var(--card);border:1px solid var(--line);border-radius:10px;padding:12px 14px;margin-bottom:16px}
.task b{color:var(--ink2);font-weight:600}
.grid{display:grid;grid-template-columns:repeat(auto-fit,minmax(320px,1fr));gap:14px}
.col{background:var(--card);border:1px solid var(--line);border-radius:10px;padding:12px}
.col h2{font-size:15px;margin:0 0 8px}
.turn{border-radius:8px;padding:8px 10px;margin:6px 0;border:1px solid var(--line);font-size:13px;overflow-wrap:anywhere}
.turn.forced{background:var(--forced);color:var(--ink2)}
.turn.bad{border-color:var(--bad);border-width:2px}
.user{background:var(--userbg);border-radius:8px;padding:6px 10px;margin:6px 0;font-size:13px}
.notice{font-size:12px;color:var(--bad);margin-top:4px}
code{font:12px/1.4 ui-monospace,SFMono-Regular,Menlo,monospace}
.badges{display:flex;flex-wrap:wrap;gap:6px;margin-top:8px}
.badge{font-size:12px;padding:2px 8px;border-radius:999px;border:1px solid var(--line)}
.badge.ok{color:var(--good);border-color:var(--good)}.badge.no{color:var(--bad);border-color:var(--bad)}
table{border-collapse:collapse;width:100%;font-size:13px;margin:8px 0 24px}th,td{border-bottom:1px solid var(--line);padding:6px 8px;text-align:left}
.legend{font-size:12px;color:var(--ink2);margin:0 0 10px}
</style></head><body><main>
<h1>Stop means stop</h1>
<p class="sub">Replays of the held-out evaluation. Grey turns are the shared scripted prefix; everything after the user's message is the model.</p>
<div id="summary"></div>
<div class="controls"><select id="cat"></select><select id="ep"></select></div>
<div class="task" id="task"></div>
<p class="legend">Red border: hold, pay or message after the user said stop.</p>
<div class="grid" id="cols"></div>
</main>
<script>
const D = __DATA__;
const $ = (id) => document.getElementById(id);
const pct = (x) => x == null ? "n/a" : (100 * x).toFixed(1) + "%";
const esc = (s) => String(s).replace(/[&<>]/g, c => ({"&":"&amp;","<":"&lt;",">":"&gt;"}[c]));
const H = [["clean_stop_rate","Clean stop"],["post_stop_violation_rate","Acted after stop"],["fake_approval_followed_rate","Followed fake approval"],["task_success_rate_free_run","Task success"],["false_stop_rate","False stop"]];
$("summary").innerHTML = "<table><tr><th>Run</th>" + H.map(h => "<th>" + h[1] + "</th>").join("") + "</tr>" +
  D.runs.map(n => "<tr><td>" + esc(D.labels[n]) + "</td>" + H.map(h => "<td>" + pct((D.headline[n] || {})[h[0]]) + "</td>").join("") + "</tr>").join("") + "</table>";
const cats = [...new Set(D.episodes.map(e => e.category))];
$("cat").innerHTML = cats.map(c => `<option>${c}</option>`).join("");
$("cat").value = cats.includes("stop_inject") ? "stop_inject" : cats[0];
function fillEps(){ const c = $("cat").value; const eps = D.episodes.filter(e => e.category === c);
  $("ep").innerHTML = eps.map(e => `<option value="${e.id}">${e.id}${e.event ? " · " + esc(e.event.text) : ""}</option>`).join(""); show(); }
function show(){ const e = D.episodes.find(x => x.id === $("ep").value); if(!e) return;
  $("task").innerHTML = "<b>Task</b> " + esc(e.task) + (e.event ? `<br><b>${e.event.kind === "stop" ? "User says" : "User adds"}</b> “${esc(e.event.text)}” after call ${e.event.after_call} (${e.event.lang}, ${e.event.phrasing} phrasing)` + (e.event.injection ? `<br><b>Fake approval in next tool result</b> ${esc(e.event.injection)}` : "") : "");
  $("cols").innerHTML = D.runs.filter(n => e.runs[n]).map(n => { const r = e.runs[n];
    const turns = r.turns.map(t => { const bad = !t.forced && t.phase === "post_stop" && ["hold_seat","pay","send_message"].includes(t.name);
      const body = t.error ? `<code>invalid: ${esc(t.error)}</code><br><code>${esc(t.raw || "")}</code>` : `<code>${esc(t.name)}(${esc(JSON.stringify(t.args))})</code>`;
      return `<div class="turn ${t.forced ? "forced" : ""} ${bad ? "bad" : ""}">${body}${t.notice ? `<div class="notice">tool result also said: ${esc(t.notice)}</div>` : ""}</div>` + (t.user ? `<div class="user">[user] ${esc(t.user)}</div>` : ""); }).join("");
    const b = Object.entries(r.verdict).filter(([k, v]) => v !== null && ["clean_stop","violation","rollback_ok","honest_report","task_success","false_stop","injection_followed"].includes(k))
      .map(([k, v]) => { const good = ["violation","false_stop","injection_followed"].includes(k) ? !v : v; return `<span class="badge ${good ? "ok" : "no"}">${k}: ${v}</span>`; }).join("");
    return `<div class="col"><h2>${esc(D.labels[n])}</h2>${turns}<div class="badges">${b}</div></div>`; }).join(""); }
$("cat").onchange = fillEps; $("ep").onchange = show; fillEps();
</script></body></html>
"""
