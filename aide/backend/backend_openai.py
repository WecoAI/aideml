"""Backend for OpenAI API."""

import json
import logging
import re
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


@once
def _setup_openai_client():
    global _client
    _client = openai.OpenAI(max_retries=0)


def query(
    system_message: str | None,
    user_message: str | None,
    func_spec: FunctionSpec | None = None,
    **model_kwargs,
) -> tuple[OutputType, float, int, int, dict]:
    """
    Query the OpenAI API, optionally with function calling.
    If the model doesn't support function calling, gracefully degrade to text generation.
    """
    _setup_openai_client()
    filtered_kwargs: dict = select_values(notnone, model_kwargs)
    if "max_tokens" in filtered_kwargs:
        filtered_kwargs["max_output_tokens"] = filtered_kwargs.pop("max_tokens")

    if (
        re.match(r"^o\d", filtered_kwargs["model"])
        or filtered_kwargs["model"] == "codex-mini-latest"
    ):
        filtered_kwargs.pop("temperature", None)

    # Convert system/user messages to the format required by the client
    messages = opt_messages_to_list(system_message, user_message)
    # Convert to the responses API format
    for i in range(len(messages)):
        messages[i]["content"] = [
            {"type": "input_text", "text": messages[i]["content"]}
        ]

    # If function calling is requested, attach the function spec
    if func_spec is not None:
        filtered_kwargs["tools"] = [func_spec.as_openai_responses_tool_dict]
        filtered_kwargs["tool_choice"] = func_spec.openai_responses_tool_choice_dict

    t0 = time.time()

    # Attempt the API call
    try:
        response = backoff_create(
            _client.responses.create,
            OPENAI_TIMEOUT_EXCEPTIONS,
            input=messages,
            **filtered_kwargs,
        )
    except openai.BadRequestError as e:
        # Check whether the error indicates that function calling is not supported
        if "function calling" in str(e).lower() or "tools" in str(e).lower():
            logger.warning(
                "Function calling was attempted but is not supported by this model. "
                "Falling back to plain text generation."
            )
            # Remove function-calling parameters and retry
            filtered_kwargs.pop("tools", None)
            filtered_kwargs.pop("tool_choice", None)

            # Retry without function calling
            response = backoff_create(
                _client.responses.create,
                OPENAI_TIMEOUT_EXCEPTIONS,
                input=messages,
                **filtered_kwargs,
            )
        else:
            # If it's some other error, re-raise
            raise

    req_time = time.time() - t0

    # Parse the output from responses API
    if (
        hasattr(response, "output")
        and response.output is not None
        and len(response.output) > 0
    ):
        # Look for function calls in the response output items
        function_call_item = None
        for output_item in response.output:
            if hasattr(output_item, "type") and output_item.type == "function_call":
                function_call_item = output_item
                break

        if function_call_item is not None:
            # Function call found
            if func_spec is not None and function_call_item.name == func_spec.name:
                try:
                    output = json.loads(function_call_item.arguments)
                except json.JSONDecodeError as ex:
                    logger.error(
                        "Error decoding function arguments:\n"
                        f"{function_call_item.arguments}"
                    )
                    raise ex
            else:
                # Function name mismatch or no func_spec
                if func_spec is not None:
                    logger.warning(
                        f"Function name mismatch: expected {func_spec.name}, "
                        f"got {function_call_item.name}. Fallback to text."
                    )
                output = response.output_text
        else:
            # No function call, use regular text output
            output = response.output_text
    else:
        # Fallback to output_text
        output = response.output_text

    in_tokens = response.usage.input_tokens
    out_tokens = response.usage.output_tokens

    info = {
        "system_fingerprint": getattr(response, "system_fingerprint", None),
        "model": response.model,
        "created": getattr(response, "created", None),
    }

    return output, req_time, in_tokens, out_tokens, info
