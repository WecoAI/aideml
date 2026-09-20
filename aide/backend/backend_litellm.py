"""Backend for LiteLLM, a unified gateway to 100+ LLM providers.

LiteLLM speaks the OpenAI wire format, so models are selected with a
``litellm/`` prefix (e.g. ``litellm/gpt-4o``, ``litellm/anthropic/claude-3-5-sonnet-20241022``,
``litellm/gemini/gemini-1.5-pro``). The prefix is stripped here and the
remainder is passed to ``litellm.completion`` which routes by provider prefix.

Credentials are resolved by LiteLLM from each provider's own environment
variable (``OPENAI_API_KEY``, ``ANTHROPIC_API_KEY``, ``GEMINI_API_KEY`` ...).
To target a LiteLLM proxy instead, set ``LITELLM_API_BASE`` (or
``LITELLM_BASE_URL``) and optionally ``LITELLM_API_KEY``.
"""

import json
import logging
import os
import time

from .utils import FunctionSpec, OutputType, opt_messages_to_list, backoff_create
from funcy import notnone, select_values
import litellm

logger = logging.getLogger("aide")

_MODEL_PREFIX = "litellm/"

LITELLM_TIMEOUT_EXCEPTIONS = (
    litellm.exceptions.RateLimitError,
    litellm.exceptions.APIConnectionError,
    litellm.exceptions.Timeout,
    litellm.exceptions.InternalServerError,
    litellm.exceptions.ServiceUnavailableError,
)


def _strip_prefix(model: str) -> str:
    """Drop the routing ``litellm/`` prefix, leaving LiteLLM's own model id."""
    if model.startswith(_MODEL_PREFIX):
        return model[len(_MODEL_PREFIX) :]
    return model


def query(
    system_message: str | None,
    user_message: str | None,
    func_spec: FunctionSpec | None = None,
    **model_kwargs,
) -> tuple[OutputType, float, int, int, dict]:
    """
    Query any provider through LiteLLM, optionally with function calling.

    LiteLLM returns OpenAI-shaped responses, so parsing mirrors the OpenAI
    chat-completions backend.
    """
    filtered_kwargs: dict = select_values(notnone, model_kwargs)  # type: ignore

    # Strip the routing prefix so LiteLLM sees its native model id.
    filtered_kwargs["model"] = _strip_prefix(filtered_kwargs["model"])

    # Silently drop kwargs a given provider doesn't support (e.g. Anthropic
    # rejects `seed`/`logprobs`, Gemini rejects OpenAI-schema tool params).
    # User can override by passing drop_params=False.
    filtered_kwargs.setdefault("drop_params", True)

    # Optional LiteLLM-proxy targeting. Omitted when blank so LiteLLM falls
    # back to each provider's own credentials / env vars.
    api_base = os.getenv("LITELLM_API_BASE") or os.getenv("LITELLM_BASE_URL")
    api_key = os.getenv("LITELLM_API_KEY")
    if api_base:
        filtered_kwargs.setdefault("api_base", api_base)
    if api_key:
        filtered_kwargs.setdefault("api_key", api_key)

    messages = opt_messages_to_list(system_message, user_message)

    if func_spec is not None:
        filtered_kwargs["tools"] = [func_spec.as_openai_tool_dict]
        filtered_kwargs["tool_choice"] = func_spec.openai_tool_choice_dict

    logger.info(f"LiteLLM API request: system={system_message}, user={user_message}")

    t0 = time.time()
    completion = backoff_create(
        litellm.completion,
        LITELLM_TIMEOUT_EXCEPTIONS,
        messages=messages,
        **filtered_kwargs,
    )
    req_time = time.time() - t0

    message = completion.choices[0].message

    if func_spec is not None and getattr(message, "tool_calls", None):
        tool_call = message.tool_calls[0]
        if tool_call.function.name == func_spec.name:
            try:
                output = json.loads(tool_call.function.arguments)
            except json.JSONDecodeError as ex:
                logger.error(
                    "Error decoding function arguments:\n"
                    f"{tool_call.function.arguments}"
                )
                raise ex
        else:
            logger.warning(
                f"Function name mismatch: expected {func_spec.name}, "
                f"got {tool_call.function.name}. Fallback to text."
            )
            output = message.content
    else:
        output = message.content

    in_tokens = completion.usage.prompt_tokens
    out_tokens = completion.usage.completion_tokens

    info = {
        "system_fingerprint": getattr(completion, "system_fingerprint", None),
        "model": completion.model,
        "created": getattr(completion, "created", None),
    }

    logger.info(
        f"LiteLLM API call completed - {completion.model} - {req_time:.2f}s - {in_tokens + out_tokens} tokens (in: {in_tokens}, out: {out_tokens})"
    )
    logger.info(f"LiteLLM API response: {output}")

    return output, req_time, in_tokens, out_tokens, info
