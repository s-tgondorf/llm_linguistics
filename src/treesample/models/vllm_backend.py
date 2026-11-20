"""vLLM backend: fast batched sampling on GPUs (LLM(model=hf_id, revision=revision, ...))."""
"""vLLM backend: fast batched sampling on GPUs (LLM(model=hf_id, revision=revision, ...))."""
from __future__ import annotations

from .base import TextGenerator


class VLLMGenerator(TextGenerator):
    def __init__(
        self,
        hf_id: str,
        revision: str = "main",
        dtype: str = "bfloat16",
        seed: int = 0,
        trust_remote_code: bool = False,
        **llm_kwargs,
    ):
        try:
            from vllm import LLM
        except ImportError as e:
            raise ImportError(
                "backend=vllm needs vllm (Linux + NVIDIA GPU): pip install -r requirements-gpu.txt. "
                "On a CPU machine use backend=hf."
            ) from e

        self.llm = LLM(
            model=hf_id,
            revision=revision,
            tokenizer_revision=revision,
            dtype=dtype,
            seed=seed,
            trust_remote_code=trust_remote_code,
            **llm_kwargs,
        )
        tok = self.llm.get_tokenizer()
        self.start_id = tok.bos_token_id if tok.bos_token_id is not None else tok.eos_token_id

    def generate(
        self,
        prompts: list[str],
        n: int,
        max_new_tokens: int = 40,
        temperature: float = 1.0,
        top_p: float = 1.0,
        top_k: int = -1,
        **_,
    ) -> list[dict]:
        from vllm import SamplingParams

        params = SamplingParams(
            n=n,
            temperature=temperature,
            top_p=top_p,
            top_k=top_k,
            max_tokens=max_new_tokens,
        )
        # Empty prompt = unconditional generation from the start token.
        inputs = [
            {"prompt_token_ids": [self.start_id]} if p == "" else {"prompt": p} for p in prompts
        ]
        request_outputs = self.llm.generate(inputs, params)  # returned in input order

        records = []
        for p_idx, (prompt, ro) in enumerate(zip(prompts, request_outputs)):
            for j, comp in enumerate(ro.outputs):
                records.append(
                    {
                        "prompt_id": p_idx,
                        "sample_id": f"{p_idx}-{j}",
                        "prompt": prompt,
                        "text": comp.text,
                        "num_new_tokens": len(comp.token_ids),
                        "hit_eos": comp.finish_reason == "stop",
                    }
                )
        return records

    def close(self) -> None:
        del self.llm
