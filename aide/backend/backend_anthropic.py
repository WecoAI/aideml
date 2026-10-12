"""Backend for Anthropic API."""

import logging
import time

from .utils import FunctionSpec, OutputType, opt_messages_to_list, backoff_create
from funcy import notnone, once, select_values
import anthropic
import jsonschema

logger = logging.getLogger("aide")

_client: anthropic.Anthropic = None  # type: ignore

ANTHROPIC_TIMEOUT_EXCEPTIONS = (
    anthropic.RateLimitError,
    anthropic.APIConnectionError,
    anthropic.APITimeoutError,
    anthropic.InternalServerError,
)

ANTHROPIC_MODEL_ALIASES = {
    "claude-3.5-sonnet": "claude-3-5-sonnet-20241022",
    "claude-3.7-sonnet": "claude-3-7-sonnet-20250219",
}


# Verified model IDs; older models retain their sampling and forced-tool behavior.
ADAPTIVE_MODELS = {
    "claude-fable-5-1",
    "claude-opus-5-5",
    "claude-sonnet-5-5",
    "claude-haiku-5-5",
}
AUTO_TOOL_MODELS = ADAPTIVE_MODELS - {"claude-haiku-5-5"}


@once
def _setup_anthropic_client():
    global _client
    _client = anthropic.Anthropic(max_retries=0)


def query(
    system_message: str | None,
    user_message: str | None,
    func_spec: FunctionSpec | None = None,
    **model_kwargs,
) -> tuple[OutputType, float, int, int, dict]:
    """
    Query Anthropic's API, optionally with tool use (Anthropic's equivalent to function calling).
    """
    _setup_anthropic_client()

    filtered_kwargs: dict = select_values(notnone, model_kwargs)  # type: ignore
    if "max_tokens" not in filtered_kwargs:
        filtered_kwargs["max_tokens"] = 4096  # default for Claude models

    model_name = filtered_kwargs.get("model", "")
    logger.debug(f"Anthropic query called with model='{model_name}'")

    if model_name in ANTHROPIC_MODEL_ALIASES:
        model_name = ANTHROPIC_MODEL_ALIASES[model_name]
        filtered_kwargs["model"] = model_name
        logger.debug(f"Using aliased model name: {model_name}")

    effort = filtered_kwargs.pop("reasoning_effort", None)
    if effort is not None:
        filtered_kwargs["output_config"] = {
            **filtered_kwargs.get("output_config", {}),
            "effort": effort,
        }
    sampling = {
        parameter: filtered_kwargs.pop(parameter)
        for parameter in ("temperature", "top_p", "top_k")
        if parameter in filtered_kwargs
    }
    if model_name not in ADAPTIVE_MODELS and sampling:
        # New SDKs removed these named arguments; older models still accept them.
        filtered_kwargs["extra_body"] = {
            **filtered_kwargs.get("extra_body", {}),
            **sampling,
        }

    if func_spec is not None:
        filtered_kwargs["tools"] = [func_spec.as_anthropic_tool_dict]
        if model_name in AUTO_TOOL_MODELS:
            filtered_kwargs["tool_choice"] = {"type": "auto"}
            instruction = (
                f"Return your answer by calling the {func_spec.name} tool exactly once."
            )
            system_message = (
                f"{system_message}\n\n{instruction}" if system_message else instruction
            )
        else:
            filtered_kwargs["tool_choice"] = func_spec.anthropic_tool_choice_dict

    # Anthropic doesn't allow not having user messages
    # if we only have system msg -> use it as user msg
    if system_message is not None and user_message is None:
        system_message, user_message = user_message, system_message

    # Anthropic passes system messages as a separate argument
    if system_message is not None:
        filtered_kwargs["system"] = system_message

    messages = opt_messages_to_list(None, user_message)

    logger.info(f"Anthropic API request: system={system_message}, user={user_message}")

    t0 = time.time()
    message = backoff_create(
        _client.messages.create,
        ANTHROPIC_TIMEOUT_EXCEPTIONS,
        messages=messages,
        **filtered_kwargs,
    )
    req_time = time.time() - t0

    if message.stop_reason in (
        "max_tokens",
        "refusal",
        "model_context_window_exceeded",
    ):
        raise ValueError(f"Anthropic response did not complete: {message.stop_reason}")
    if func_spec is not None:
        blocks = [block for block in message.content if block.type == "tool_use"]
        if len(blocks) != 1 or blocks[0].name != func_spec.name:
            raise ValueError(f"Expected one call to {func_spec.name}")
        output = blocks[0].input
        jsonschema.validate(output, func_spec.json_schema)
    else:
        output = "".join(
            block.text for block in message.content if block.type == "text"
        )
        if not output:
            raise ValueError("Anthropic returned no text")

    in_tokens = message.usage.input_tokens
    out_tokens = message.usage.output_tokens

    info = {
        "stop_reason": message.stop_reason,
        "model": message.model,
    }

    logger.info(
        f"Anthropic API call completed - {message.model} - {req_time:.2f}s - {in_tokens + out_tokens} tokens (in: {in_tokens}, out: {out_tokens})"
    )
    logger.info(f"Anthropic API response: {output}")

    return output, req_time, in_tokens, out_tokens, info
