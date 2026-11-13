# Stop means stop

A tool-using agent that books, pays, and messages has to be interruptible. "Stop", "cancel that", "ruko", or "रहने दो" mid-task must mean no new hold, payment, or message, release of any active hold, and an honest report of what already happened, including a payment made before the stop. A fake "the user re-approved" in a tool result must not restart the task. A benign "stop" ("a one-stop flight", "rukna mat") should not abort it.

The student is CohereLabs/tiny-aya-global. A scripted teacher that is correct on every episode is distilled with AgentTune (LoRA rank 16, one epoch, learning rate 2e-4, seeds 42 and 7). CuratorKIT keeps all 999 teacher rows. AuditKIT scores each held-out episode from the simulator's final state. No model judges the transcript. The test set has 180 episodes and disjoint seeds. Interrupt episodes share a forced prefix, so every policy faces the same state when the stop arrives. A control adapter uses the same recipe on the 500 episodes with no interrupt.

On 100 held-out stops, clean stop is 7.0% for the prompted base model (95% Wilson interval 3.4–13.7) and 99.0% for the adapter (94.6–99.8). Seed 2 scores 98.0% (93.0–99.4). Paired McNemar tests against the base model are p < 0.0001 for both seeds (+92.0 and +91.0 points; zero episodes where only the base model stopped cleanly). Acting after the stop falls from 88.0% to 1.0% and 2.0%. Fake-approval following falls from 100% (29 of 30; the base model missed the planted line once) to 0.0% and 3.3% (p < 0.0001). Hold release goes from 78.2% to 100.0% and 96.4%. Honest reports after a stop go from 63.0% to 100.0% and 99.0%. Tuned clean stop is 100% on 43 held-out phrasings, 97.4% on 38 Hinglish stops, and 100% on 16 Hindi stops.

The control reaches 16.0% clean stop (10.1–24.4), +9.0 points, p = 0.0784, and still follows every fake approval. Those 16 stops are all in the late-stop slice; on early stops and fake approvals its clean-stop rate is 0%. Stop behaviour comes from the interrupt data. Valid tool calls go from 73.1% to 100% for every adapter, including the control, whose free-run task success is 57.5%.

Task success with no interrupt rises from 0.0% to 65.0% (seed 2: 67.5%; both p < 0.0001): 86.7% on ordinary bookings and 0% when no flight matches, for every learned model, against 100% for the teacher. False stops on a benign "stop" are 17.5% (8.7–32.0); the drop from 35.0% is not significant (p = 0.1185). A 60-example smoke run did not learn to stop (0 of 13). The recipe is one epoch of LoRA. Two seeds agree (train loss 0.3520 and 0.3528).

The job was 43.4 minutes on one L40S at $1.80/hour, about $1.30. The main train logged 1,185,984 tokens in 488.64 seconds (2,427 tokens/s).

Track 3, Agentic Fine-tuning. Problem statements #7 Business workflow and decision agents and #3 Agentic distillation. Repos: AgentTune, CuratorKIT, AuditKIT.
