"""Oracle records -> AgentTune full-tier EventLogs for agentic distillation.

Each record becomes one trajectory:
  OBSERVATION(instructions + task)
  TOOL_CALL(call 1)   -> rendered by AgentTune as  <tool_call>{...}</tool_call>
  OBSERVATION(result 1, plus any [user] line)
  ...
  TOOL_CALL(finish)

Tool results are stored as OBSERVATION events (role "user") rather than
TOOL_RESULT (role "tool") so the conversation already alternates user/assistant.
Tiny Aya's chat template insists on strict alternation, and this keeps the
training render identical to what the evaluator feeds the model.
"""

from __future__ import annotations

import json


def record_to_events(rec: dict) -> list[dict]:
    """Serialisable event list (kind, payload) for one oracle record."""
    msgs = rec["messages"]
    turns = rec["turns"]
    events = [{"kind": "observation", "payload": {"text": msgs[0]["content"]}}]
    mi = 1
    for t in turns:
        if t["parse_error"] is not None:
            raise ValueError(f"oracle produced an invalid call in {rec['scenario_id']}")
        events.append({"kind": "tool_call", "payload": {"action": {"name": t["name"], "arguments": t["arguments"]}}})
        mi += 1  # assistant message
        if mi < len(msgs) and msgs[mi]["role"] == "user":
            events.append({"kind": "observation", "payload": {"text": msgs[mi]["content"]}})
            mi += 1
    return events


def events_to_eventlog(events: list[dict], log_id: str | None = None):
    from agenttune.agentic.events import Event, EventKind, EventLog

    kinds = {"observation": EventKind.OBSERVATION, "tool_call": EventKind.TOOL_CALL}
    log = EventLog(tier="full")
    if log_id:
        log.id = log_id
    for e in events:
        log.append(Event(kinds[e["kind"]], e["payload"]))
    return log


def write_events_jsonl(records: list[dict], path: str) -> int:
    n = 0
    with open(path, "w", encoding="utf-8") as f:
        for rec in records:
            f.write(json.dumps({"id": rec["scenario_id"], "category": rec["category"],
                                "events": record_to_events(rec)}, ensure_ascii=False) + "\n")
            n += 1
    return n


def load_eventlogs(path: str, keep_ids: set[str] | None = None):
    logs = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            row = json.loads(line)
            if keep_ids is not None and row["id"] not in keep_ids:
                continue
            logs.append(events_to_eventlog(row["events"], row["id"]))
    return logs


def expected_sft_messages(events: list[dict]) -> list[dict]:
    """The messages AgentTune's EventLog.as_dataset_rows('sft') will produce,
    reproduced without importing agenttune (used for CuratorKIT and tests)."""
    out = []
    for e in events:
        if e["kind"] == "observation":
            out.append({"role": "user", "content": e["payload"]["text"]})
        else:
            out.append({"role": "assistant", "content": f"<tool_call>{json.dumps(e['payload']['action'])}</tool_call>"})
    return out
