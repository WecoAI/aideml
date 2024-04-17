"""Backend for OpenAI API."""

import json
import logging
import time

from .utils import FunctionSpec, OutputType, opt_messages_to_list
from funcy import notnone, once, retry, select_values
from openai import OpenAI, RateLimitError

logger = logging.getLogger("aide")

_client: OpenAI = None  # type: ignore

RATELIMIT_RETRIES = 5
retry_exp = retry(RATELIMIT_RETRIES, errors=RateLimitError, timeout=lambda a: 2 ** (a + 1))  # type: ignore


@once
def _setup_openai_client():
    global _client
    _client = OpenAI(max_retries=3)


@retry_exp
def query(
    system_message: str | None,
    user_message: str | None,
    func_spec: FunctionSpec | None = None,
    **model_kwargs,
) -> tuple[OutputType, float, int, int, dict]:
    _setup_openai_client()
    filtered_kwargs: dict = select_values(notnone, model_kwargs)  # type: ignore

    messages = opt_messages_to_list(system_message, user_message)

    if func_spec is not None:
        filtered_kwargs["tools"] = [func_spec.as_openai_tool_dict]
        # force the model the use the function
        filtered_kwargs["tool_choice"] = func_spec.openai_tool_choice_dict

    t0 = time.time()
    completion = _client.chat.completions.create(messages=messages, **filtered_kwargs)  # type: ignore
    req_time = time.time() - t0

    choice = completion.choices[0]

    if func_spec is None:
        output = choice.message.content
    else:
        assert (
            choice.message.tool_calls
        ), f"function_call is empty, it is not a function call: {choice.message}"
        assert (
            choice.message.tool_calls[0].function.name == func_spec.name
        ), "Function name mismatch"
        try:
            output = json.loads(choice.message.tool_calls[0].function.arguments)
        except json.JSONDecodeError as e:
            logger.error(
                f"Error decoding the function arguments: {choice.message.tool_calls[0].function.arguments}"
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
