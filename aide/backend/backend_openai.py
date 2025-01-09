"""Backend for OpenAI API."""

import json
import logging
import time

from .utils import FunctionSpec, OutputType, opt_messages_to_list, backoff_create
from funcy import notnone, once, select_values
import openai

logger = logging.getLogger("aide")

_client: openai.OpenAI = None  # type: ignore

OPENAI_TIMEOUT_EXCEPTIONS = (
    openai.RateLimitError,
    openai.APIConnectionError,
    openai.APITimeoutError,
    openai.InternalServerError,
)

# (docs) https://platform.openai.com/docs/guides/function-calling/supported-models
SUPPORTED_FUNCTION_CALL_MODELS = {
    "gpt-4o",
    "gpt-4o-2024-08-06",
    "gpt-4o-2024-05-13",
    "gpt-4o-mini",
    "gpt-4o-mini-2024-07-18",
    "gpt-4-turbo",
    "gpt-4-turbo-2024-04-09",
    "gpt-4-turbo-preview",
    "gpt-4-0125-preview",
    "gpt-4-1106-preview",
    "gpt-3.5-turbo",
    "gpt-3.5-turbo-0125",
    "gpt-3.5-turbo-1106",
}


@once
def _setup_openai_client():
    global _client
    _client = openai.OpenAI(max_retries=0)


def is_function_call_supported(model_name: str) -> bool:
    """Return True if the model supports function calling."""
    return model_name in SUPPORTED_FUNCTION_CALL_MODELS


def query(
    system_message: str | None,
    user_message: str | None,
    func_spec: FunctionSpec | None = None,
    **model_kwargs,
) -> tuple[OutputType, float, int, int, dict]:
    """
    Query the OpenAI API, optionally with function calling.
    Function calling support is only checked for feedback/review operations.
    """
    _setup_openai_client()
    filtered_kwargs: dict = select_values(notnone, model_kwargs)
    model_name = filtered_kwargs.get("model", "")
    logger.debug(f"OpenAI query called with model='{model_name}'")

    messages = opt_messages_to_list(system_message, user_message)

    if func_spec is not None:
        # Only check function call support for feedback/search operations
        if func_spec.name == "submit_review":
            if not is_function_call_supported(model_name):
                logger.warning(
                    f"Review function calling was requested, but model '{model_name}' "
                    "does not support function calling. Falling back to plain text generation."
                )
                filtered_kwargs.pop("tools", None)
                filtered_kwargs.pop("tool_choice", None)
            else:
                filtered_kwargs["tools"] = [func_spec.as_openai_tool_dict]
                filtered_kwargs["tool_choice"] = func_spec.openai_tool_choice_dict

    t0 = time.time()
    completion = backoff_create(
        _client.chat.completions.create,
        OPENAI_TIMEOUT_EXCEPTIONS,
        messages=messages,
        **filtered_kwargs,
    )
    req_time = time.time() - t0

    choice = completion.choices[0]

    if func_spec is None or "tools" not in filtered_kwargs:
        output = choice.message.content
    else:
        tool_calls = getattr(choice.message, "tool_calls", None)

        if not tool_calls:
            logger.warning(
                f"No function call used despite function spec. Fallback to text. "
                f"Message content: {choice.message.content}"
            )
            output = choice.message.content
        else:
            first_call = tool_calls[0]
            assert first_call.function.name == func_spec.name, (
                f"Function name mismatch: expected {func_spec.name}, "
                f"got {first_call.function.name}"
            )
            try:
                output = json.loads(first_call.function.arguments)
            except json.JSONDecodeError as e:
                logger.error(
                    f"Error decoding function arguments:\n{first_call.function.arguments}"
                )
                raise e

    in_tokens = completion.usage.prompt_tokens
    out_tokens = completion.usage.completion_tokens

    info = {
        "system_fingerprint": completion.system_fingerprint,
        "model": completion.model,
        "created": completion.created,
    }

    return output, req_time, in_tokens, out_tokens, info
