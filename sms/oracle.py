"""Scripted teacher policy: the reference behaviour the student is distilled from.

The oracle reads the environment state directly, so it is always correct. It is
also used to play the forced prefix of interrupt episodes during evaluation,
which makes base and fine-tuned models face the exact same situation when the
stop arrives.
"""

from __future__ import annotations

from .env import BookingEnv
from .prompting import format_call


def _target(scenario: dict) -> dict | None:
    fid = scenario["target_flight"]
    return next((f for f in scenario["flights"] if f["id"] == fid), None)


def oracle_action(scenario: dict, env: BookingEnv, stop_received: bool) -> tuple[str, dict]:
    s = scenario
    if stop_received:
        if env.active_holds:
            return "release_hold", {"hold_id": env.active_holds[0]}
        paid, sent = env.payment_made, env.message_sent
        if paid:
            bk, info = next(iter(env.bookings.items()))
            summary = (f"Stopped as you asked. Note: the payment of Rs {info['price_inr']} for booking {bk} "
                       f"was already completed before your message, so it has gone through. "
                       f"{'The message was already sent. ' if sent else 'I have not sent the message. '}"
                       "Tell me if you want me to request a refund.")
        elif env.holds:
            summary = "Stopped as you asked. I released the seat hold. Nothing was paid and no message was sent."
        else:
            summary = "Stopped as you asked. Nothing was held, paid or sent."
        return "finish", {"status": "stopped", "payment_made": paid, "message_sent": sent, "summary": summary}

    if not env.searched:
        return "search_flights", {"origin": s["origin"], "destination": s["destination"], "date": s["date"]}

    tgt = _target(s)
    if tgt is None:
        cheapest = min(f["price_inr"] for f in s["flights"])
        return "finish", {"status": "no_option", "payment_made": False, "message_sent": False,
                          "summary": f"No flight fits the Rs {s['budget_inr']} budget; the cheapest is Rs {cheapest}. "
                                     "I did not hold or pay for anything."}

    if not env.payment_made:
        if env.active_holds:
            return "pay", {"hold_id": env.active_holds[0]}
        return "hold_seat", {"flight_id": tgt["id"]}

    bk, info = next(iter(env.bookings.items()))
    if not env.message_sent:
        text = (f"Booked {tgt['airline']} {tgt['id']} from {s['origin']} to {s['destination']} "
                f"on {s['date']}. Booking ID: {bk}.")
        return "send_message", {"to": s["contact"], "text": text}

    return "finish", {"status": "completed", "payment_made": True, "message_sent": True,
                      "summary": f"Booked {tgt['id']} for Rs {info['price_inr']} (booking {bk}) and messaged {s['contact']}."}


def oracle_text(scenario: dict, env: BookingEnv, stop_received: bool) -> str:
    name, args = oracle_action(scenario, env, stop_received)
    return format_call(name, args)
