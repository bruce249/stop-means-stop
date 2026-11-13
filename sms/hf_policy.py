"""Hugging Face transformers policy (base model, or base + LoRA adapter).

Prompts are rendered with the model's own chat template from the exact
messages the episode holds, which are byte-identical to the AgentTune SFT rows
used in training.
"""

from __future__ import annotations

import torch


def pick_dtype():
    if not torch.cuda.is_available():
        return torch.float32
    major, _ = torch.cuda.get_device_capability(0)
    return torch.bfloat16 if major >= 8 else torch.float16


class HFPolicy:
    def __init__(self, model_id: str, adapter: str | None = None, name: str | None = None,
                 max_new_tokens: int = 220, merge: bool = True, token: str | None = None):
        from transformers import AutoModelForCausalLM, AutoTokenizer

        self.name = name or (adapter or model_id).split("/")[-1]
        self.tok = AutoTokenizer.from_pretrained(adapter or model_id, token=token)
        if self.tok.pad_token is None:
            self.tok.pad_token = self.tok.eos_token
        self.tok.padding_side = "left"
        device_map = {"": 0} if torch.cuda.is_available() else None
        model = AutoModelForCausalLM.from_pretrained(model_id, dtype=pick_dtype(), device_map=device_map,
                                                     token=token)
        if adapter:
            from peft import PeftModel

            model = PeftModel.from_pretrained(model, adapter, token=token)
            if merge:
                model = model.merge_and_unload()
        model.eval()
        self.model = model
        self.max_new_tokens = max_new_tokens

    def _encode(self, messages: list[dict]) -> list[int]:
        enc = self.tok.apply_chat_template(messages, add_generation_prompt=True, tokenize=True,
                                           return_dict=True)
        ids = enc["input_ids"]
        return list(ids[0]) if ids and isinstance(ids[0], list) else list(ids)

    @torch.no_grad()
    def act_batch(self, episodes) -> list[str]:
        seqs = [self._encode(ep.messages) for ep in episodes]
        width = max(len(s) for s in seqs)
        pad = self.tok.pad_token_id
        input_ids = torch.full((len(seqs), width), pad, dtype=torch.long)
        attn = torch.zeros((len(seqs), width), dtype=torch.long)
        for i, s in enumerate(seqs):
            input_ids[i, width - len(s):] = torch.tensor(s)
            attn[i, width - len(s):] = 1
        dev = self.model.device
        out = self.model.generate(
            input_ids=input_ids.to(dev), attention_mask=attn.to(dev),
            max_new_tokens=self.max_new_tokens, do_sample=False,
            stop_strings=["</tool_call>"], tokenizer=self.tok, pad_token_id=pad,
        )
        texts = []
        for row in out[:, width:]:
            text = self.tok.decode(row, skip_special_tokens=True)
            cut = text.find("</tool_call>")
            texts.append(text[: cut + len("</tool_call>")] if cut != -1 else text.strip())
        return texts
