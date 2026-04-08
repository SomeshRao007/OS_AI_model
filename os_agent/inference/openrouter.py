"""OpenRouter API client — OpenAI-compatible cloud inference.

Provides the same infer()/infer_streaming() interface as InferenceEngine
so agents can use either backend transparently. Handles SSE streaming,
<think> block stripping, retry with backoff, and API key management.
"""

from __future__ import annotations

import json
import logging
import os
import re
import stat
import time
from pathlib import Path
from typing import Generator

import requests
import yaml

log = logging.getLogger("ai-daemon.openrouter")

_API_URL = "https://openrouter.ai/api/v1/chat/completions"
_SYNC_TIMEOUT = 30
_STREAM_TIMEOUT = 60
_MAX_RETRIES = 3
_BACKOFF_BASE = 1  # seconds: 1, 2, 4

# Reuse the same think-stripping logic as engine.py
_THINK_RE = re.compile(r"<think>.*?</think>\s*", flags=re.DOTALL)
_THINK_UNCLOSED_RE = re.compile(r"<think>.*", flags=re.DOTALL)

_SECRETS_PATH = Path.home() / ".config" / "ai-daemon" / "secrets.yaml"


def load_api_key() -> str | None:
    """Load OpenRouter API key from secrets.yaml or env var.

    Priority: secrets.yaml -> OPENROUTER_API_KEY env var.
    Returns None if no key is configured.
    """
    if _SECRETS_PATH.exists():
        # Warn if permissions are too open
        file_mode = _SECRETS_PATH.stat().st_mode
        if file_mode & (stat.S_IRGRP | stat.S_IROTH):
            log.warning(
                "secrets.yaml at %s is readable by group/others. "
                "Run: chmod 600 %s", _SECRETS_PATH, _SECRETS_PATH
            )

        with open(_SECRETS_PATH) as f:
            secrets = yaml.safe_load(f) or {}
        if isinstance(secrets, str):
            # File contains just the raw key, not a YAML dict
            return secrets
        if isinstance(secrets, dict):
            key = secrets.get("openrouter_api_key")
            if key:
                return key

    return os.environ.get("OPENROUTER_API_KEY")


class OpenRouterClient:
    """OpenRouter API client with retry and streaming support."""

    def __init__(
        self,
        api_key: str,
        model_id: str = "deepseek/deepseek-chat",
        temperature: float = 0.3,
        max_tokens: int = 1024,
    ) -> None:
        self._api_key = api_key
        self._model_id = model_id
        self._temperature = temperature
        self._max_tokens = max_tokens
        self._last_prompt_tokens = 0
        self._last_completion_tokens = 0
        self._last_cost = 0.0
        self._session = requests.Session()
        self._session.headers.update({
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
            "HTTP-Referer": "https://github.com/aios",
            "X-Title": "AI OS Daemon",
        })

    @property
    def model_id(self) -> str:
        return self._model_id

    @property
    def last_prompt_tokens(self) -> int:
        return self._last_prompt_tokens

    @property
    def last_completion_tokens(self) -> int:
        return self._last_completion_tokens

    @property
    def last_cost(self) -> float:
        return self._last_cost

    def close(self) -> None:
        """Close the HTTP session to free connections."""
        self._session.close()

    def infer(
        self,
        system_prompt: str,
        user_message: str,
        max_tokens: int | None = None,
    ) -> str:
        """Synchronous inference via OpenRouter API.

        Returns the response text with <think> blocks stripped.
        """
        payload = self._build_payload(system_prompt, user_message,
                                      max_tokens or self._max_tokens,
                                      stream=False)

        response = self._request_with_retry(payload, timeout=_SYNC_TIMEOUT)
        usage = response.get("usage", {})
        self._last_prompt_tokens = usage.get("prompt_tokens", 0)
        self._last_completion_tokens = usage.get("completion_tokens", 0)
        self._last_cost = float(usage.get("cost", 0.0) or 0.0)
        raw = response["choices"][0]["message"]["content"]
        return self._strip_thinking(raw)

    def infer_streaming(
        self,
        system_prompt: str,
        user_message: str,
        max_tokens: int | None = None,
    ) -> Generator[str, None, None]:
        """Streaming inference via SSE. Yields text chunks with <think> blocks stripped.

        Buffers the start of generation to detect and discard <think>...</think>
        blocks, then switches to pass-through for remaining tokens.
        """
        payload = self._build_payload(system_prompt, user_message,
                                      max_tokens or self._max_tokens,
                                      stream=True)

        resp = self._session.post(
            _API_URL,
            json=payload,
            stream=True,
            timeout=_STREAM_TIMEOUT,
        )
        resp.raise_for_status()

        buffer = ""
        in_think_block = False
        think_detected = False

        for line in resp.iter_lines(decode_unicode=True):
            if not line or not line.startswith("data: "):
                continue

            data_str = line[6:]  # strip "data: " prefix
            if data_str.strip() == "[DONE]":
                break

            chunk_data = json.loads(data_str)
            delta = chunk_data.get("choices", [{}])[0].get("delta", {})
            token = delta.get("content", "")
            if not token:
                continue

            # Think-block detection (same logic as engine.py)
            if not think_detected:
                buffer += token
                if len(buffer) < 7:  # len("<think>")
                    continue

                if buffer.startswith("<think>"):
                    in_think_block = True
                    think_detected = True
                    close_idx = buffer.find("</think>")
                    if close_idx != -1:
                        remainder = buffer[close_idx + 8:].lstrip()
                        if remainder:
                            yield remainder
                        in_think_block = False
                        buffer = ""
                    continue

                # No think block — flush buffer
                think_detected = True
                yield buffer
                buffer = ""
                continue

            if in_think_block:
                buffer += token
                close_idx = buffer.find("</think>")
                if close_idx != -1:
                    remainder = buffer[close_idx + 8:].lstrip()
                    if remainder:
                        yield remainder
                    in_think_block = False
                    buffer = ""
                continue

            yield token

        # Flush remaining buffer
        if buffer and not in_think_block:
            cleaned = self._strip_thinking(buffer)
            if cleaned:
                yield cleaned

    def test_connection(self) -> tuple[bool, str]:
        """Make a minimal API call to validate the key and model.

        Returns (success, error_message). On success error_message is empty.
        """
        payload = {
            "model": self._model_id,
            "messages": [{"role": "user", "content": "hi"}],
            "max_tokens": 1,
        }

        try:
            resp = self._session.post(
                _API_URL, json=payload, timeout=_SYNC_TIMEOUT,
            )
            if resp.status_code == 401:
                return False, "Invalid API key"
            if resp.status_code == 404:
                return False, f"Model not found: {self._model_id}"
            if resp.status_code == 402:
                return False, "Insufficient credits on OpenRouter account"
            resp.raise_for_status()

            data = resp.json()
            if "choices" in data:
                return True, ""
            return False, f"Unexpected response format: {list(data.keys())}"

        except requests.exceptions.Timeout:
            return False, "Connection timed out"
        except requests.exceptions.ConnectionError:
            return False, "Cannot reach OpenRouter API — check internet connection"
        except requests.exceptions.HTTPError as e:
            return False, f"HTTP error: {e.response.status_code}"
        except Exception as e:
            return False, f"Unexpected error: {e}"

    def _build_payload(
        self,
        system_prompt: str,
        user_message: str,
        max_tokens: int,
        stream: bool,
    ) -> dict:
        """Build the OpenAI-compatible request payload."""
        payload: dict = {
            "model": self._model_id,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_message},
            ],
            "max_tokens": max_tokens,
            "temperature": self._temperature,
            "stream": stream,
            "usage": {"include": True},
        }
        return payload

    def _request_with_retry(self, payload: dict, timeout: int) -> dict:
        """POST with exponential backoff on 5xx / timeout errors."""
        last_error: Exception | None = None

        for attempt in range(_MAX_RETRIES):
            try:
                resp = self._session.post(
                    _API_URL, json=payload, timeout=timeout,
                )
                # Don't retry client errors (4xx) — they won't self-resolve
                if 400 <= resp.status_code < 500:
                    resp.raise_for_status()
                if resp.status_code >= 500:
                    last_error = requests.exceptions.HTTPError(
                        f"Server error: {resp.status_code}", response=resp
                    )
                    # Fall through to retry
                else:
                    return resp.json()

            except (requests.exceptions.Timeout,
                    requests.exceptions.ConnectionError) as e:
                last_error = e

            if attempt < _MAX_RETRIES - 1:
                wait = _BACKOFF_BASE * (2 ** attempt)
                log.warning("OpenRouter request failed (attempt %d/%d), "
                            "retrying in %ds: %s",
                            attempt + 1, _MAX_RETRIES, wait, last_error)
                time.sleep(wait)

        raise last_error  # type: ignore[misc]

    @staticmethod
    def _strip_thinking(text: str) -> str:
        """Remove <think>...</think> blocks from completed text."""
        cleaned = _THINK_RE.sub("", text)
        cleaned = _THINK_UNCLOSED_RE.sub("", cleaned)
        return cleaned.strip()
