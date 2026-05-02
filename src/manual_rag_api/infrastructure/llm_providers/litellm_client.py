"""LiteLLM wrapper with cost tracking and built-in retry logic."""

import logging
import os
import re
import time
from typing import Any, Dict, List, Optional

from litellm import completion, completion_cost
from litellm.exceptions import RateLimitError


def _parse_retry_after(error_msg: str) -> Optional[float]:
    """
    Extract the suggested wait time from a Groq / OpenAI rate-limit message.
    e.g. "Please try again in 3.924s." → 3.924
    Returns None if the pattern is not found.
    """
    m = re.search(r"try again in\s+([\d.]+)s", error_msg, re.IGNORECASE)
    return float(m.group(1)) if m else None

logger = logging.getLogger(__name__)


class LitellmConfigError(Exception):
    """Raised when required environment/configuration for LiteLLM is missing."""


class CostTracker:
    """Accumulate costs across multiple API calls."""

    def __init__(self):
        self.reset()

    def add_call(self, response: Any, model_name: str, call_type: str = "unknown"):
        try:
            # Pass model explicitly so LiteLLM can resolve cost for provider-prefixed
            # names like "groq/llama-3.3-70b-versatile" (the response object strips
            # the prefix, making automatic lookup fail).
            cost = completion_cost(completion_response=response, model=model_name)
            usage = response.get("usage", {})
            prompt_tokens = usage.get("prompt_tokens", 0)
            completion_tokens = usage.get("completion_tokens", 0)
            total_tokens = usage.get("total_tokens", 0)

            self.total_cost += cost
            self.total_prompt_tokens += prompt_tokens
            self.total_completion_tokens += completion_tokens
            self.total_tokens += total_tokens
            self.call_count += 1
            self.cost_breakdown.append(
                {
                    "call_type": call_type,
                    "model": model_name,
                    "cost": cost,
                    "prompt_tokens": prompt_tokens,
                    "completion_tokens": completion_tokens,
                    "total_tokens": total_tokens,
                }
            )
            logger.info(
                f"API call | cost=${cost:.6f} | tokens={total_tokens} "
                f"| model={model_name} | type={call_type}"
            )
        except Exception as e:
            logger.warning(f"Could not calculate cost: {e}")

    def get_summary(self) -> Dict[str, Any]:
        return {
            "total_cost": self.total_cost,
            "total_prompt_tokens": self.total_prompt_tokens,
            "total_completion_tokens": self.total_completion_tokens,
            "total_tokens": self.total_tokens,
            "call_count": self.call_count,
            "cost_breakdown": self.cost_breakdown,
        }

    def reset(self):
        self.total_cost = 0.0
        self.total_prompt_tokens = 0
        self.total_completion_tokens = 0
        self.total_tokens = 0
        self.call_count = 0
        self.cost_breakdown = []


class LitellmClient:
    """
    Provider-agnostic LLM client with cost tracking and automatic retry.

    Improvements over reference project:
    - Built-in retry with exponential backoff (num_retries=3)
    - Validates at least one API key is set on init
    - Cost tracker exposed via get_cost_summary()
    """

    _KNOWN_KEYS = [
        "OPENAI_API_KEY",
        "ANTHROPIC_API_KEY",
        "GOOGLE_API_KEY",
        "GROQ_API_KEY",
        "OPENROUTER_API_KEY",
        "AZURE_OPENAI_KEY",
        "COHERE_API_KEY",
        "TOGETHER_API_KEY",
    ]

    def __init__(self, model_name: Optional[str] = None) -> None:
        self.model_name = model_name
        self.cost_tracker = CostTracker()

        if not any(os.getenv(k) for k in self._KNOWN_KEYS):
            raise LitellmConfigError(
                "No LLM API key found. Set one of: "
                + ", ".join(self._KNOWN_KEYS)
            )
        logger.info(f"LitellmClient ready — model={model_name or '<none>'}")

    def chat(
        self,
        messages: List[Dict[str, Any]],
        *,
        model_name: Optional[str] = None,
        temperature: Optional[float] = None,
        top_p: Optional[float] = None,
        max_tokens: Optional[int] = None,
        call_type: str = "chat",
        **kwargs: Any,
    ) -> Any:
        """
        Send a chat request via LiteLLM with rate-limit-aware retry.

        Unlike LiteLLM's built-in num_retries (tenacity with ~0s wait), this
        loop extracts the provider's suggested retry-after from the 429 message
        and sleeps for that long before retrying.  Three retries; exponential
        fallback (10s, 20s, 40s) when no retry-after is present.
        """
        model = model_name or self.model_name
        if not model:
            raise LitellmConfigError("No model_name provided to chat() or __init__.")

        payload: Dict[str, Any] = {
            "model":    model,
            "messages": messages,
            "timeout":  kwargs.pop("timeout", 60),   # default: 60 s hard limit
        }
        if temperature is not None:
            payload["temperature"] = temperature
        if top_p is not None:
            payload["top_p"] = top_p
        if max_tokens is not None:
            payload["max_tokens"] = max_tokens
        payload.update(kwargs)

        last_exc: Optional[Exception] = None
        for attempt in range(4):          # 1 try + 3 retries
            try:
                response = completion(**payload)
                self.cost_tracker.add_call(response, model, call_type)
                return response
            except RateLimitError as exc:
                last_exc = exc
                if attempt == 3:
                    break                 # exhausted retries — re-raise below
                wait = _parse_retry_after(str(exc)) or (10 * (2 ** attempt))
                wait = min(wait + 2, 70)  # +2s buffer; cap at 70s (just over 1 min)
                logger.warning(
                    "Rate limit hit (attempt %d/3) — sleeping %.1fs…", attempt + 1, wait
                )
                time.sleep(wait)

        raise last_exc  # type: ignore[misc]

    def stream_chat(
        self,
        messages: List[Dict[str, Any]],
        *,
        temperature: Optional[float] = None,
        max_tokens:  Optional[int]   = None,
        **kwargs: Any,
    ):
        """
        Generator that yields raw text delta strings as they arrive.

        Uses LiteLLM's stream=True mode.  Cost tracking is skipped because
        most providers don't include usage stats in streaming chunks.
        The caller sees tokens as soon as they arrive — no waiting for the
        full response.
        """
        model = self.model_name
        if not model:
            raise LitellmConfigError("No model_name provided.")

        payload: Dict[str, Any] = {
            "model":    model,
            "messages": messages,
            "stream":   True,
            "timeout":  kwargs.pop("timeout", 60),   # default: 60 s hard limit
        }
        if temperature is not None:
            payload["temperature"] = temperature
        if max_tokens is not None:
            payload["max_tokens"] = max_tokens
        payload.update(kwargs)

        response = completion(**payload)
        for chunk in response:
            delta = chunk.choices[0].delta.content
            if delta:
                yield delta

    def get_cost_summary(self) -> Dict[str, Any]:
        return self.cost_tracker.get_summary()

    def reset_cost_tracking(self):
        self.cost_tracker.reset()
