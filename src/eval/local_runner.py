"""Laptop-friendly text generation with HF transformers only.

Why this exists separately from `src/eval/evaluate.py`:

- `evaluate.py` prefers vLLM, which doesn't install cleanly on a laptop and
  wants more VRAM than most laptops have.
- This module uses plain `transformers.AutoModelForCausalLM.generate`.
  Works on CPU (slow but functional) and on consumer GPUs (6GB RTX 4050
  class — enough for Qwen3-0.6B inference in fp16).

Importing this module is laptop-safe: all heavy imports are lazy. A unit
test can patch the LocalRunner internals without pulling in torch at all.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any, Callable

from src.common.device import resolve_device

log = logging.getLogger(__name__)


@dataclass(frozen=True)
class GenConfig:
    """Generation hyperparameters. Deterministic by default (temperature=0)."""
    max_new_tokens: int = 512
    temperature: float = 0.0      # 0 → greedy; >0 → sampling
    top_p: float = 1.0
    repetition_penalty: float = 1.0
    batch_size: int = 1           # keep small for 6GB VRAM / CPU


class LocalRunner:
    """Model + tokenizer + generate(), with explicit device control.

    Create once per process. Reuse across many prompt lists; the model stays
    resident in memory.
    """

    def __init__(
        self,
        model_path: str,
        *,
        device: str = "auto",
        dtype: str = "auto",
        trust_remote_code: bool = True,
    ):
        self.model_path = model_path
        self.device = resolve_device(device)
        self._dtype_pref = dtype
        self._trust_remote_code = trust_remote_code
        self._model = None
        self._tokenizer = None

    # ---- lazy loaders -------------------------------------------------

    def _load(self) -> None:
        if self._model is not None:
            return
        import torch  # type: ignore
        from transformers import AutoModelForCausalLM, AutoTokenizer  # type: ignore

        dtype = self._resolve_dtype(torch)
        log.info(
            "Loading %s on %s (dtype=%s)",
            self.model_path, self.device, dtype,
        )

        self._tokenizer = AutoTokenizer.from_pretrained(
            self.model_path, trust_remote_code=self._trust_remote_code
        )
        if self._tokenizer.pad_token_id is None:
            # Qwen tokenizers typically have eos but no pad; reuse eos.
            self._tokenizer.pad_token = self._tokenizer.eos_token

        self._model = AutoModelForCausalLM.from_pretrained(
            self.model_path,
            torch_dtype=dtype,
            trust_remote_code=self._trust_remote_code,
        )
        self._model.to(self.device)
        self._model.eval()

    def _resolve_dtype(self, torch_mod: Any):
        pref = self._dtype_pref
        if pref == "auto":
            # fp16 on cuda; fp32 on cpu (fp16 on cpu is slower and unstable).
            return torch_mod.float16 if self.device == "cuda" else torch_mod.float32
        return {
            "float16": torch_mod.float16,
            "bfloat16": torch_mod.bfloat16,
            "float32": torch_mod.float32,
        }[pref]

    # ---- public API ---------------------------------------------------

    def generate(
        self,
        prompts: list[str],
        cfg: GenConfig = GenConfig(),
        *,
        progress: Callable[[int, int], None] | None = None,
    ) -> list[str]:
        """Generate one completion per prompt. Deterministic when temperature=0.

        `progress(done, total)` is called after each batch, for CLI UX.
        """
        self._load()
        import torch  # type: ignore

        out: list[str] = []
        bs = max(1, cfg.batch_size)
        total = len(prompts)
        do_sample = cfg.temperature > 0.0

        with torch.no_grad():
            for i in range(0, total, bs):
                chunk = prompts[i:i + bs]
                enc = self._tokenizer(
                    chunk,
                    return_tensors="pt",
                    padding=True,
                    truncation=True,
                ).to(self.device)
                gen = self._model.generate(
                    **enc,
                    max_new_tokens=cfg.max_new_tokens,
                    do_sample=do_sample,
                    temperature=cfg.temperature if do_sample else 1.0,
                    top_p=cfg.top_p if do_sample else 1.0,
                    repetition_penalty=cfg.repetition_penalty,
                    pad_token_id=self._tokenizer.pad_token_id,
                )
                # Drop the prompt tokens from each row.
                prompt_lens = enc["input_ids"].shape[1]
                for row in gen:
                    text = self._tokenizer.decode(
                        row[prompt_lens:], skip_special_tokens=True
                    )
                    out.append(text)
                if progress is not None:
                    progress(min(i + bs, total), total)
        return out
