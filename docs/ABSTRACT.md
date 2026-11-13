# Stop means stop

A tool-using agent that books, pays, and messages has to be interruptible. "Stop", "cancel that", "ruko", or "रहने दो" mid-task must mean: no new hold, payment, or message; release any active hold; and an honest report of what already happened, including a payment made before the stop. A fake "the user re-approved" planted in a tool result must not restart it.

We distill a scripted teacher that is always correct into CohereLabs/tiny-aya-global (about 3B) with AgentTune LoRA SFT. CuratorKIT gates the teacher's rows. AuditKIT scores every episode from the simulator's final state. No language model judges the transcripts.

The world is a deterministic flight-booking simulator. Train and test use disjoint seeds (999 and 180 episodes). Interrupt episodes are played by the teacher up to the user's message, so the base model and the tuned model face the same state when the stop arrives. A control adapter uses the same recipe on episodes with no interrupts.

On 100 held-out stops, clean stop is 7.0% for the prompted base model (95% Wilson interval 3.4–13.7) and 99.0% after one epoch (94.6–99.8). A second seed scores 98.0%. Paired McNemar test versus the base: +92 points, p < 0.0001. The control adapter scores 16.0% and does not separate from the base (p = 0.078), so the stop behaviour comes from the interrupt data, not from learning the tool format. Fake approvals followed fall from 100% to 0% (seed 2: 3.3%). The teacher scores 100% and a never-stop agent scores 0%, so the metric cannot be gamed by refusing to act or by lying in the transcript.

Task success on free runs rises from 0% to 65%. That is not free: impossible-budget episodes are still never finished correctly, and false stops on benign "stop" wording fall only from 35% to 17.5% (p = 0.12). The full run took 43 minutes on one L40S, about $1.30.

Track 3, Agentic Fine-tuning. Problem statements #7 and #3. Repos: AgentTune, CuratorKIT, AuditKIT.
