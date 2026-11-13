"""Distill the scripted teacher into Tiny Aya with AgentTune (LoRA SFT).

    python scripts/train.py --split main --out runs/adapter-main
    python scripts/train.py --split control_no_interrupts --out runs/adapter-control

Uses agenttune.create_distill_trainer on full-tier EventLogs. --backend trl runs
the same rows through plain TRL in case the AgentTune path breaks on your box.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

LORA_TARGETS = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="CohereLabs/tiny-aya-global")
    ap.add_argument("--data", default="data")
    ap.add_argument("--split", default="main", choices=["main", "control_no_interrupts"])
    ap.add_argument("--out", default="runs/adapter-main")
    ap.add_argument("--epochs", type=float, default=1.0)
    ap.add_argument("--lr", type=float, default=2e-4)
    ap.add_argument("--batch-size", type=int, default=2)
    ap.add_argument("--grad-accum", type=int, default=4)
    ap.add_argument("--lora-r", type=int, default=16)
    ap.add_argument("--max-length", type=int, default=2048)
    ap.add_argument("--max-examples", type=int, default=None)
    ap.add_argument("--backend", default="agenttune", choices=["agenttune", "trl"])
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    import torch

    from sms.agenttune_bridge import load_eventlogs

    with open(f"{args.data}/splits.json") as f:
        ids = json.load(f)[args.split]
    if args.max_examples:
        ids = ids[: args.max_examples]
    logs = load_eventlogs(f"{args.data}/train_eventlogs.jsonl", keep_ids=set(ids))
    print(f"[train] {len(logs)} teacher trajectories, split={args.split}")

    bf16 = torch.cuda.is_available() and torch.cuda.get_device_capability(0)[0] >= 8
    dtype = torch.bfloat16 if bf16 else torch.float16
    sft_kwargs = dict(
        num_train_epochs=args.epochs, learning_rate=args.lr,
        per_device_train_batch_size=args.batch_size, gradient_accumulation_steps=args.grad_accum,
        max_length=args.max_length, bf16=bf16, fp16=not bf16,
        gradient_checkpointing=True, gradient_checkpointing_kwargs={"use_reentrant": False},
        lr_scheduler_type="cosine", warmup_steps=10, logging_steps=5,
        save_strategy="no", report_to="none", seed=args.seed, data_seed=args.seed,
    )
    t0 = time.time()

    if args.backend == "agenttune":
        from agenttune import create_distill_trainer
        from agenttune.agentic.distill_trainer import _DefaultSFTTrainerFactory

        class Fp32AdapterFactory(_DefaultSFTTrainerFactory):
            """AgentTune's default TRL factory, plus: keep trainable LoRA weights in
            fp32 so fp16 mixed precision works on T4s (no-op on bf16 GPUs)."""

            def __call__(self, **kw):
                trainer = super().__call__(**kw)
                for _, p in trainer.model.named_parameters():
                    if p.requires_grad and p.dtype != torch.float32:
                        p.data = p.data.float()
                return trainer

        factory = Fp32AdapterFactory(lora_r=args.lora_r, lora_alpha=2 * args.lora_r, lora_dropout=0.05,
                                     lora_targets=LORA_TARGETS, dtype=dtype, device_map={"": 0})
        trainer = create_distill_trainer(student=args.model, trajectories=logs, output_dir=args.out,
                                         trainer_factory=factory, **sft_kwargs)
        trainer.train()
        sft = factory.sft_trainer
        tok = factory.tokenizer
    else:
        from datasets import Dataset
        from peft import LoraConfig
        from transformers import AutoModelForCausalLM, AutoTokenizer
        from trl import SFTConfig, SFTTrainer

        rows = [r for log in logs for r in log.as_dataset_rows("sft")]
        tok = AutoTokenizer.from_pretrained(args.model)
        if tok.pad_token is None:
            tok.pad_token = tok.eos_token
        model = AutoModelForCausalLM.from_pretrained(args.model, dtype=dtype, device_map={"": 0})
        peft_cfg = LoraConfig(r=args.lora_r, lora_alpha=2 * args.lora_r, lora_dropout=0.05,
                              target_modules=LORA_TARGETS, task_type="CAUSAL_LM")
        sft = SFTTrainer(model=model, args=SFTConfig(output_dir=args.out, **sft_kwargs),
                         train_dataset=Dataset.from_list([{"messages": r["messages"]} for r in rows]),
                         processing_class=tok, peft_config=peft_cfg)
        for _, p in sft.model.named_parameters():
            if p.requires_grad and p.dtype != torch.float32:
                p.data = p.data.float()
        sft.train()

    os.makedirs(args.out, exist_ok=True)
    sft.save_model(args.out)
    tok.save_pretrained(args.out)
    meta = {"base_model": args.model, "split": args.split, "n_trajectories": len(logs),
            "backend": args.backend, "minutes": round((time.time() - t0) / 60, 1),
            "dtype": str(dtype), "lora_targets": LORA_TARGETS, "args": vars(args),
            "log_history": sft.state.log_history}
    with open(f"{args.out}/train_meta.json", "w") as f:
        json.dump(meta, f, indent=2, default=str)
    print(f"[train] done in {meta['minutes']} min -> {args.out}")


if __name__ == "__main__":
    main()
