"""Episode runner: steps a policy through the simulator, injecting user
interrupts, benign distractors and fake approvals at fixed points.

Evaluation protocol
  * clean / no_option episodes run free: the policy does everything.
  * stop / stop_inject / distractor episodes use a forced prefix: the scripted
    oracle plays every turn up to the event, then the policy takes over. Base and
    fine-tuned models therefore face identical states when the stop arrives, and
    a model that cannot book a flight cannot look "safe" by never getting far
    enough to be interrupted.
"""

from __future__ import annotations

from dataclasses import asdict
from typing import Callable, Iterable, Protocol

from .env import BookingEnv
from .oracle import oracle_text
from .prompting import first_user_message, format_observation, parse_tool_call

MAX_TURNS = 10
MAX_INVALID_STREAK = 3


class Policy(Protocol):
    name: str

    def act_batch(self, episodes: list["Episode"]) -> list[str]: ...


def default_mode(scenario: dict) -> str:
    return "forced" if scenario["event"] else "free"


class Episode:
    def __init__(self, scenario: dict, mode: str = "free", max_turns: int = MAX_TURNS):
        self.s = scenario
        self.mode = mode
        self.max_turns = max_turns
        self.env = BookingEnv(scenario)
        self.messages: list[dict] = [{"role": "user", "content": first_user_message(scenario)}]
        self.turns: list[dict] = []
        ev = scenario["event"]
        self.event_point = ev["after_call"] if ev else None
        self.forced_until = self.event_point if (mode == "forced" and ev) else 0
        self.calls_done = 0
        self.event_delivered = False
        self.stop_received = False
        self.state_at_stop: dict | None = None
        self.injection_seen = False
        self.invalid_streak = 0
        self.done = False
        self.end_reason: str | None = None

    # ----------------------------------------------------------------- driving
    @property
    def forced_now(self) -> bool:
        return not self.done and self.calls_done < self.forced_until

    def advance_forced(self) -> None:
        while self.forced_now:
            self.apply(oracle_text(self.s, self.env, self.stop_received), forced=True)

    def apply(self, text: str, forced: bool = False) -> None:
        if self.done:
            return
        name, args, err = parse_tool_call(text)
        if err is None:
            err = BookingEnv.validate(name, args)
        phase = "post_stop" if self.stop_received else "pre_stop"
        turn = {"text": text, "forced": forced, "phase": phase, "name": name, "arguments": args,
                "parse_error": err}
        self.messages.append({"role": "assistant", "content": text})

        if err is not None:
            self.invalid_streak += 1
            result = {"ok": False, "error": f"invalid tool call: {err}"}
            turn["result"] = result
            self.turns.append(turn)
            if self.invalid_streak >= MAX_INVALID_STREAK:
                return self._end("invalid_calls")
            self.messages.append({"role": "user", "content": format_observation("invalid", result)})
            return self._check_turn_budget()

        self.invalid_streak = 0
        result = self.env.call(name, args, phase=phase, forced=forced)
        if name == "finish":
            turn["result"] = result
            self.turns.append(turn)
            return self._end("finish")

        self.calls_done += 1
        ev = self.s["event"]
        if self.stop_received and not self.injection_seen and ev and ev.get("injection"):
            result = {**result, "notice": ev["injection"]}
            self.injection_seen = True
        user_text = None
        if ev and not self.event_delivered and self.calls_done == self.event_point:
            user_text = ev["text"]
            self.event_delivered = True
            if ev["kind"] == "stop":
                self.stop_received = True
                self.state_at_stop = {
                    "active_holds": list(self.env.active_holds),
                    "payment_made": self.env.payment_made,
                    "message_sent": self.env.message_sent,
                    "turn_index": len(self.turns) + 1,
                }
        turn["result"] = result
        self.turns.append(turn)
        self.messages.append({"role": "user", "content": format_observation(name, result, user_text)})
        self._check_turn_budget()

    def _check_turn_budget(self) -> None:
        if len(self.turns) >= self.max_turns:
            self._end("max_turns")

    def _end(self, reason: str) -> None:
        self.done = True
        self.end_reason = reason

    # ---------------------------------------------------------------- export
    def record(self) -> dict:
        return {
            "scenario_id": self.s["scenario_id"],
            "category": self.s["category"],
            "split": self.s["split"],
            "task_lang": self.s["task_lang"],
            "event": self.s["event"],
            "mode": self.mode,
            "target_flight": self.s["target_flight"],
            "contact": self.s["contact"],
            "messages": self.messages,
            "turns": self.turns,
            "actions": [asdict(a) for a in self.env.actions],
            "final_state": self.env.snapshot(),
            "state_at_stop": self.state_at_stop,
            "stop_received": self.stop_received,
            "event_delivered": self.event_delivered,
            "injection_seen": self.injection_seen,
            "end_reason": self.end_reason,
        }


def run_episodes(
    scenarios: Iterable[dict],
    policy: Policy,
    mode_fn: Callable[[dict], str] = default_mode,
    batch_size: int = 8,
    max_turns: int = MAX_TURNS,
    progress: bool = True,
) -> list[dict]:
    """Run episodes in lockstep batches so an HF policy can batch generation."""
    pending = [Episode(s, mode_fn(s), max_turns) for s in scenarios]
    total = len(pending)
    finished: list[Episode] = []
    active: list[Episode] = []
    bar = None
    if progress:
        try:
            from tqdm.auto import tqdm
            bar = tqdm(total=total, desc=f"episodes[{policy.name}]")
        except ImportError:
            bar = None
    while pending or active:
        while pending and len(active) < batch_size:
            active.append(pending.pop(0))
        for ep in active:
            ep.advance_forced()
        need = [ep for ep in active if not ep.done]
        if need:
            texts = policy.act_batch(need)
            for ep, text in zip(need, texts):
                ep.apply(text, forced=False)
        still = []
        for ep in active:
            if ep.done:
                finished.append(ep)
                if bar:
                    bar.update(1)
            else:
                still.append(ep)
        active = still
    if bar:
        bar.close()
    finished.sort(key=lambda e: e.s["scenario_id"])
    return [e.record() for e in finished]


class OraclePolicy:
    """Upper bound: the scripted teacher."""

    name = "oracle"

    def act_batch(self, episodes: list[Episode]) -> list[str]:
        return [oracle_text(ep.s, ep.env, ep.stop_received) for ep in episodes]


class NeverStopPolicy:
    """Lower bound: competent at the task, deaf to the user. Shows what each
    safety metric looks like when the agent simply keeps going."""

    name = "never_stop"

    def act_batch(self, episodes: list[Episode]) -> list[str]:
        return [oracle_text(ep.s, ep.env, False) for ep in episodes]
