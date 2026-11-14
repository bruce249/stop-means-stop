# Stop means stop

An agent that books, pays, and sends messages must halt when the user says stop, in English, Hinglish, or Hindi. It must make no new hold, payment, or message, release any active hold, and report truthfully what already happened. A fake approval planted inside a tool result must not restart it.

We distill a scripted teacher into CohereLabs/tiny-aya-global with AgentTune LoRA. CuratorKIT checks the training rows. AuditKIT scores every episode from a deterministic booking simulator, with no model judge. On interrupt episodes the base model and the tuned model face the same state when the stop arrives. A control adapter is trained the same way on episodes that contain no interrupts.

On 100 held-out stops, clean stop rises from 7.0% (95% Wilson interval 3.4–13.7) to 99.0% (94.6–99.8). A second seed scores 98.0%. Against the base model the gain is +92 points, p < 0.0001. The control scores 16.0% and does not beat the base (p = 0.078), so the behaviour comes from the interrupt data. Fake approvals followed fall from 100% to 0%. The teacher scores 100% and a never-stop agent scores 0%.

Uninterrupted task success rises from 0% to 65%. Episodes with no flight in budget are still never handled correctly, and false stops on a benign "stop" fall only from 35% to 17.5% (p = 0.12). The run took 43 minutes on one L40S.

Track 3, Agentic Fine-tuning. Problem statements #7 and #3. Repos: AgentTune, CuratorKIT, AuditKIT.
