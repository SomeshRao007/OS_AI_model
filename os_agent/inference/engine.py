"""Inference engine wrapping llama-cpp-python for the OS agent.

Single shared LLM instance for all agent domains. Uses raw create_completion()
with manual ChatML formatting — NOT create_chat_completion(), which mangles
<think> tags during auto-detection.
"""

import logging
import re
import subprocess
from pathlib import Path
from typing import Generator

import yaml
from llama_cpp import Llama

log = logging.getLogger("ai-daemon.engine")

# Project root: two levels up from os_agent/inference/engine.py
_PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent

# Config discovery: env var (set by systemd/neurosh launcher) → dev path
import os as _os
_CONFIG_PATH = Path(
    _os.environ.get("AI_DAEMON_CONFIG",
                     str(_PROJECT_ROOT / "os_agent" / "config" / "daemon.yaml"))
)

# Minimum chars to buffer before deciding if output starts with <think>
_THINK_TAG = "<think>"
_THINK_CLOSE = "</think>"
_THINK_DETECT_LEN = len(_THINK_TAG)


class InferenceEngine:
    """Single shared LLM instance for all agent domains.

    Loads the GGUF model once and provides synchronous + streaming
    inference with ChatML prompt formatting.
    """

    def __init__(self, config: dict | None = None):
        """Load model and store generation parameters.

        Args:
            config: Optional config dict matching daemon.yaml structure.
                    If None, loads from os_agent/config/daemon.yaml.
        """
        if config is None:
            config = self._load_config()

        model_cfg = config["model"]
        gen_cfg = config["generation"]

        raw_path = Path(model_cfg["path"])
        model_path = raw_path if raw_path.is_absolute() else _PROJECT_ROOT / raw_path
        if not model_path.exists():
            raise FileNotFoundError(f"Model not found: {model_path}")

        self._model = Llama(
            model_path=str(model_path),
            n_gpu_layers=model_cfg.get("n_gpu_layers", -1),
            n_ctx=model_cfg.get("n_ctx", 1024),
            verbose=False,
        )

        self._temperature = gen_cfg.get("temperature", 0.6)
        self._top_p = gen_cfg.get("top_p", 0.95)
        self._top_k = gen_cfg.get("top_k", 20)
        self._repeat_penalty = gen_cfg.get("repeat_penalty", 1.0)
        self._max_tokens = gen_cfg.get("max_tokens", 1024)
        self._seed = gen_cfg.get("seed", -1)
        self._stop_tokens = gen_cfg.get("stop_tokens", ["<|im_end|>", "<|endoftext|>"])
        self._last_completion_tokens = 0

    # Keys that cannot be hot-applied — they require rebuilding the llama_cpp
    # context. KCM shows a "reload required" banner for these.
    RELOAD_REQUIRED_KEYS = ("n_ctx", "n_gpu_layers")

    # Keys that can be updated in place without touching llama_cpp.
    HOT_APPLY_KEYS = (
        "temperature", "top_p", "top_k", "repeat_penalty", "max_tokens", "seed",
    )

    def update_generation_params(self, params: dict) -> dict:
        """Apply new sampling parameters in place. Returns dict of keys actually changed.

        Unknown keys and reload-required keys are ignored (caller must trigger
        a full reload for those). Safe to call while the engine is loaded.
        """
        changed = {}
        for key in self.HOT_APPLY_KEYS:
            if key not in params:
                continue
            value = params[key]
            attr = f"_{key}"
            if getattr(self, attr, None) != value:
                setattr(self, attr, value)
                changed[key] = value
        if changed:
            log.info("Hot-applied generation params: %s", changed)
        return changed

    @property
    def loaded(self) -> bool:
        """True if the model is loaded and ready for inference."""
        return self._model is not None

    @property
    def last_completion_tokens(self) -> int:
        """Token count from the most recent create_completion() call."""
        return self._last_completion_tokens

    def unload(self) -> None:
        """Free the model from VRAM/RAM. Safe to call multiple times.

        Steps beyond just close():
        1. close() — frees C-side llama memory (llama_free)
        2. del — drops Python reference so GC can collect
        3. gc.collect() — force immediate collection of cyclic refs
        4. malloc_trim — tell glibc to return freed pages to the OS
           (without this, glibc keeps freed pages in its arena and the
           process RSS stays high even though the memory is unused)
        """
        if self._model is not None:
            import ctypes
            import gc

            self._model.close()
            del self._model
            self._model = None
            gc.collect()

            # Force glibc to release freed pages back to the OS
            try:
                ctypes.CDLL("libc.so.6").malloc_trim(0)
            except (OSError, AttributeError):
                pass  # Non-glibc systems (musl, macOS) — skip

            log.info("Model unloaded from memory")

    def infer(self, system_prompt: str, user_message: str, max_tokens: int | None = None) -> str:
        """Run synchronous inference. Returns cleaned response text."""
        if self._model is None:
            raise RuntimeError("Model has been unloaded — cannot run inference")
        prompt = self._build_prompt(system_prompt, user_message)

        result = self._model.create_completion(
            prompt=prompt,
            temperature=self._temperature,
            top_p=self._top_p,
            top_k=self._top_k,
            repeat_penalty=self._repeat_penalty,
            max_tokens=max_tokens or self._max_tokens,
            seed=self._seed if self._seed is not None and self._seed >= 0 else None,
            stop=self._stop_tokens,
        )

        self._last_completion_tokens = result.get("usage", {}).get("completion_tokens", 0)
        raw = result["choices"][0]["text"]
        return self._strip_thinking(raw)

    def infer_streaming(
        self, system_prompt: str, user_message: str, max_tokens: int | None = None
    ) -> Generator[str, None, None]:
        """Yield response tokens one at a time, stripping thinking blocks.

        Buffers the start of generation to detect and discard <think>...</think>
        blocks, then switches to pass-through for remaining tokens.
        """
        if self._model is None:
            raise RuntimeError("Model has been unloaded — cannot run inference")
        prompt = self._build_prompt(system_prompt, user_message)

        stream = self._model.create_completion(
            prompt=prompt,
            temperature=self._temperature,
            top_p=self._top_p,
            top_k=self._top_k,
            repeat_penalty=self._repeat_penalty,
            max_tokens=max_tokens or self._max_tokens,
            seed=self._seed if self._seed is not None and self._seed >= 0 else None,
            stop=self._stop_tokens,
            stream=True,
        )

        # Phase 1: buffer initial tokens to detect <think> block
        buffer = ""
        in_think_block = False

        for chunk in stream:
            token = chunk["choices"][0]["text"]

            if not in_think_block and _THINK_CLOSE not in buffer:
                buffer += token

                # Haven't seen enough chars yet to decide
                if len(buffer) < _THINK_DETECT_LEN:
                    continue

                # Check if output starts with a think block
                if buffer.startswith(_THINK_TAG):
                    in_think_block = True
                    # Check if the close tag is already in the buffer
                    close_idx = buffer.find(_THINK_CLOSE)
                    if close_idx != -1:
                        # Think block complete, yield anything after it
                        remainder = buffer[close_idx + len(_THINK_CLOSE):].lstrip()
                        if remainder:
                            yield remainder
                        in_think_block = False
                        buffer = ""
                    continue

                # No think block — flush the buffer and switch to pass-through
                yield buffer
                buffer = ""
                continue

            if in_think_block:
                buffer += token
                close_idx = buffer.find(_THINK_CLOSE)
                if close_idx != -1:
                    remainder = buffer[close_idx + len(_THINK_CLOSE):].lstrip()
                    if remainder:
                        yield remainder
                    in_think_block = False
                    buffer = ""
                continue

            # Phase 2: pass-through
            yield token

        # Flush any remaining buffer (edge case: short response with no think block)
        if buffer and not in_think_block:
            cleaned = self._strip_thinking(buffer)
            if cleaned:
                yield cleaned

    def infer_with_rag(
        self, system_prompt: str, query: str, max_tokens: int | None = None
    ) -> str:
        """Run inference with RAG context injected into system_prompt.

        Detects the command the query is about and appends a one-line flag
        summary to the system prompt before calling infer(). This gives the
        model ground-truth flag information without changing anything else.
        """
        from os_agent.inference.rag import build_rag_context

        rag_ctx = build_rag_context(query)
        if rag_ctx:
            system_prompt = system_prompt + f"\n\nCOMMAND REFERENCE: {rag_ctx}"
        return self.infer(system_prompt, query, max_tokens)

    def infer_validated(
        self, system_prompt: str, query: str, max_tokens: int | None = None
    ) -> dict:
        """Run RAG-augmented inference then validate the extracted command.

        Returns a dict with keys:
          - response (str): full model response
          - command (str | None): extracted bash command, or None
          - blocked (bool): True if validator blocked the command
          - error (str): validation error message (only when blocked=True)
          - suggestion (str): fix hint (only when blocked=True)
        """
        from os_agent.tools.parser import extract_command
        from os_agent.inference.validator import validate

        response = self.infer_with_rag(system_prompt, query, max_tokens)
        command = extract_command(response)

        if command:
            result = validate(command)
            if not result["ok"]:
                return {
                    "response": response,
                    "command": command,
                    "blocked": True,
                    "error": result["error"],
                    "suggestion": result.get("suggestion", ""),
                }

        return {"response": response, "command": command, "blocked": False}

    def get_vram_usage(self) -> dict[str, int]:
        """Return current VRAM stats: {used, total, free} in MB."""
        result = subprocess.run(
            [
                "nvidia-smi",
                "--query-gpu=memory.used,memory.total,memory.free",
                "--format=csv,noheader,nounits",
            ],
            capture_output=True,
            text=True,
        )
        if result.returncode != 0:
            return {"used": 0, "total": 0, "free": 0}

        try:
            parts = result.stdout.strip().split(", ")
            if len(parts) < 3:
                return {"used": 0, "total": 0, "free": 0}
            return {"used": int(parts[0]), "total": int(parts[1]), "free": int(parts[2])}
        except (ValueError, IndexError):
            return {"used": 0, "total": 0, "free": 0}

    def _build_prompt(self, system_prompt: str, user_message: str) -> str:
        """Format system + user message into ChatML template."""
        return (
            f"<|im_start|>system\n{system_prompt}<|im_end|>\n"
            f"<|im_start|>user\n{user_message}<|im_end|>\n"
            f"<|im_start|>assistant\n"
        )

    @staticmethod
    def _strip_thinking(text: str) -> str:
        """Remove <think>...</think> blocks from completed text."""
        cleaned = re.sub(r"<think>.*?</think>\s*", "", text, flags=re.DOTALL)
        # Safety net: handle unclosed think block (max_tokens cutoff)
        cleaned = re.sub(r"<think>.*", "", cleaned, flags=re.DOTALL)
        return cleaned.strip()

    @staticmethod
    def _load_config() -> dict:
        """Load and return daemon.yaml as a dict."""
        with open(_CONFIG_PATH, encoding="utf-8") as f:
            return yaml.safe_load(f)
