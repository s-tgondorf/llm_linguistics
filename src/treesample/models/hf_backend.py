"""transformers backend: CPU-friendly, supports `revision=` for checkpoint branches."""
from __future__ import annotations

import torch
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer

from .base import TextGenerator


class HFGenerator(TextGenerator):
    def __init__(
        self,
        hf_id: str,
        revision: str = "main",
        dtype: str = "float32",
        device: str = "cpu",
        trust_remote_code: bool = False,
    ):
        self.device = device
        self.tokenizer = AutoTokenizer.from_pretrained(
            hf_id, revision=revision, trust_remote_code=trust_remote_code
        )
        self.model = AutoModelForCausalLM.from_pretrained(
            hf_id,
            revision=revision,
            torch_dtype=getattr(torch, dtype),
            trust_remote_code=trust_remote_code,
        )
        self.model.to(device).eval()

        tok = self.tokenizer
        self.eos_id = tok.eos_token_id
        self.pad_id = tok.pad_token_id if tok.pad_token_id is not None else tok.eos_token_id
        # Token used to start unconditional generation (empty prompt).
        self.start_id = tok.bos_token_id if tok.bos_token_id is not None else tok.eos_token_id

    def _encode(self, prompt: str) -> torch.Tensor:
        if prompt == "":
            ids = [self.start_id]
        else:
            ids = self.tokenizer(prompt)["input_ids"]
        return torch.tensor([ids], device=self.device)

    @torch.inference_mode()
    def generate(
        self,
        prompts: list[str],
        n: int,
        max_new_tokens: int = 40,
        temperature: float = 1.0,
        top_p: float = 1.0,
        top_k: int = -1,
        batch_size: int = 32,
        seed: int | None = None,
        **_,
    ) -> list[dict]:
        if seed is not None:
            torch.manual_seed(seed)

        gen_kwargs = dict(max_new_tokens=max_new_tokens, pad_token_id=self.pad_id)
        if temperature > 0:
            gen_kwargs.update(
                do_sample=True,
                temperature=temperature,
                top_p=top_p,
                top_k=max(top_k, 0),  # HF: 0 disables top-k
            )
        else:
            gen_kwargs.update(do_sample=False)

        records = []
        for p_idx, prompt in enumerate(tqdm(prompts, desc="prompts")):
            ids = self._encode(prompt)
            in_len = ids.shape[1]
            attn = torch.ones_like(ids)
            done = 0
            while done < n:
                k = min(batch_size, n - done)
                out = self.model.generate(
                    ids, attention_mask=attn, num_return_sequences=k, **gen_kwargs
                )
                for j in range(k):
                    new = out[j, in_len:].tolist()
                    hit_eos = self.eos_id in new
                    if hit_eos:
                        new = new[: new.index(self.eos_id)]
                    records.append(
                        {
                            "prompt_id": p_idx,
                            "sample_id": f"{p_idx}-{done + j}",
                            "prompt": prompt,
                            "text": self.tokenizer.decode(new, skip_special_tokens=True),
                            "num_new_tokens": len(new),
                            "hit_eos": hit_eos,
                        }
                    )
                done += k
        return records

    def close(self) -> None:
        del self.model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
