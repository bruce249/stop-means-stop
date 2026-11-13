# Submission checklist

Form: https://hackathon.lexsi.ai/submission/

Deadline: Sunday 16 Nov 2026, 5:29 PM IST (15 Nov 2026, 11:59 PM AoE). The latest submission counts.

The form was not submitted. The registration email is not on this machine.

## Fields, ready to paste

| Field | Value |
|---|---|
| Email | the address you registered with |
| Abstract | Body of `docs/ABSTRACT.md` (484 words). Paste the text below the heading. |
| HF model collection | https://huggingface.co/collections/heisenbergk146/stop-means-stop-models-6af5b981e34895830a52525c |
| HF dataset collection | https://huggingface.co/collections/heisenbergk146/stop-means-stop-data-6af5b9815a9ca25db71e69be |
| Write-up | https://github.com/bruce249/stop-means-stop |
| GitHub repo | https://github.com/bruce249/stop-means-stop |
| Repos used | AgentTune, CuratorKIT, AuditKIT |
| Models used | CohereLabs/tiny-aya-global |
| Track | Track 3: Agentic Fine-tuning |
| Problem statements | #7 Business workflow and decision agents; #3 Agentic distillation |

The write-up field is the GitHub README. `huggingface_hub` 1.32.0 and the `hf` CLI on this machine have no command that creates a Hugging Face article. The long write-up is in the same repo at `docs/ARTICLE.md`. The hackathon accepts a GitHub README when articles are unavailable.

## Abstract (paste this)

A tool-using agent that books, pays, and messages has to be interruptible. "Stop", "cancel that", "ruko", or "रहने दो" mid-task must mean no new hold, payment, or message, release of any active hold, and an honest report of what already happened, including a payment made before the stop. A fake "the user re-approved" in a tool result must not restart the task. A benign "stop" ("a one-stop flight", "rukna mat") should not abort it.

The student is CohereLabs/tiny-aya-global. A scripted teacher that is correct on every episode is distilled with AgentTune (LoRA rank 16, one epoch, learning rate 2e-4, seeds 42 and 7). CuratorKIT keeps all 999 teacher rows. AuditKIT scores each held-out episode from the simulator's final state. No model judges the transcript. The test set has 180 episodes and disjoint seeds. Interrupt episodes share a forced prefix, so every policy faces the same state when the stop arrives. A control adapter uses the same recipe on the 500 episodes with no interrupt.

On 100 held-out stops, clean stop is 7.0% for the prompted base model (95% Wilson interval 3.4–13.7) and 99.0% for the adapter (94.6–99.8). Seed 2 scores 98.0% (93.0–99.4). Paired McNemar tests against the base model are p < 0.0001 for both seeds (+92.0 and +91.0 points; zero episodes where only the base model stopped cleanly). Acting after the stop falls from 88.0% to 1.0% and 2.0%. Fake-approval following falls from 100% (29 of 30; the base model missed the planted line once) to 0.0% and 3.3% (p < 0.0001). Hold release goes from 78.2% to 100.0% and 96.4%. Honest reports after a stop go from 63.0% to 100.0% and 99.0%. Tuned clean stop is 100% on 43 held-out phrasings, 97.4% on 38 Hinglish stops, and 100% on 16 Hindi stops.

The control reaches 16.0% clean stop (10.1–24.4), +9.0 points, p = 0.0784, and still follows every fake approval. Those 16 stops are all in the late-stop slice; on early stops and fake approvals its clean-stop rate is 0%. Stop behaviour comes from the interrupt data. Valid tool calls go from 73.1% to 100% for every adapter, including the control, whose free-run task success is 57.5%.

Task success with no interrupt rises from 0.0% to 65.0% (seed 2: 67.5%; both p < 0.0001): 86.7% on ordinary bookings and 0% when no flight matches, for every learned model, against 100% for the teacher. False stops on a benign "stop" are 17.5% (8.7–32.0); the drop from 35.0% is not significant (p = 0.1185). A 60-example smoke run did not learn to stop (0 of 13). The recipe is one epoch of LoRA. Two seeds agree (train loss 0.3520 and 0.3528).

The job was 43.4 minutes on one L40S at $1.80/hour, about $1.30. The main train logged 1,185,984 tokens in 488.64 seconds (2,427 tokens/s).

Track 3, Agentic Fine-tuning. Problem statements #7 Business workflow and decision agents and #3 Agentic distillation. Repos: AgentTune, CuratorKIT, AuditKIT.

## Visibility, from the Hub API on 13 Nov 2026

Both collections returned `private: false` and `gating: false`. Each repo returned `private: false` and `gated: false`.

| Repo | private | gated |
|---|---|---|
| heisenbergk146/tiny-aya-global-stop-means-stop-lora | false | false |
| heisenbergk146/tiny-aya-global-stop-means-stop-lora-seed2 | false | false |
| heisenbergk146/tiny-aya-global-sms-control-lora | false | false |
| heisenbergk146/stop-means-stop-data | false | false |
| heisenbergk146/stop-means-stop-evals | false | false |
| heisenbergk146/stop-means-stop (space) | false | false |

Model collection items: the three adapters and the Space. Dataset collection items: the data repo and the evals repo.

`hub_links.json` was not in the evals dataset. The collection URLs above are the ones the Hub API returned for "Stop means stop: models" and "Stop means stop: data". Wilson intervals and McNemar p-values are in `results.md` on the evals dataset. `stats.json` is written by `scripts/report.py` and was not in the uploaded file list.
