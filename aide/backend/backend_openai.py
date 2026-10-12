"""Backend for OpenAI API."""

import json
import logging
import os
import re
import time

from .utils import FunctionSpec, OutputType, opt_messages_to_list, backoff_create
from funcy import notnone, once, select_values
import openai
import jsonschema

logger = logging.getLogger("aide")

OPENAI_BASE_URL = "https://api.openai.com/v1"

_client: openai.OpenAI = None  # type: ignore
_custom_client: openai.OpenAI = None  # type: ignore

OPENAI_TIMEOUT_EXCEPTIONS = (
    openai.RateLimitError,
    openai.APIConnectionError,
    openai.APITimeoutError,
    openai.InternalServerError,
)


@once
def _setup_openai_client():
    global _client
    # Use real OpenAI API with proper API key, explicitly override base_url
    api_key = os.getenv("OPENAI_API_KEY")
    _client = openai.OpenAI(api_key=api_key, base_url=OPENAI_BASE_URL, max_retries=0)


@once
def _setup_custom_client():
    global _custom_client
    # Only create custom client if base URL is set
    base_url = os.getenv("OPENAI_BASE_URL")
    api_key = os.getenv("OPENAI_API_KEY")
    if base_url:
        _custom_client = openai.OpenAI(
            api_key=api_key, base_url=base_url, max_retries=0
        )


def query(
    system_message: str | None,
    user_message: str | None,
    func_spec: FunctionSpec | None = None,
    **model_kwargs,
) -> tuple[OutputType, float, int, int, dict]:
    """
    Query the OpenAI API, optionally with function calling.
    """
    # Setup clients
    _setup_openai_client()

    filtered_kwargs: dict = select_values(notnone, model_kwargs)
    model_name = filtered_kwargs.get("model", "")
    is_openai_model = re.match(r"^(gpt-|o\d|codex-mini-latest$)", model_name)
    use_chat_api = bool(os.getenv("OPENAI_BASE_URL")) and not is_openai_model

    messages = opt_messages_to_list(system_message, user_message)
    if use_chat_api:
        _setup_custom_client()
        # OpenAI-compatible local servers use Chat Completions parameters.
        if func_spec is not None:
            filtered_kwargs["tools"] = [func_spec.as_openai_tool_dict]
            filtered_kwargs["tool_choice"] = func_spec.openai_tool_choice_dict
    else:
        if "max_tokens" in filtered_kwargs:
            filtered_kwargs["max_output_tokens"] = filtered_kwargs.pop("max_tokens")
        effort = filtered_kwargs.pop("reasoning_effort", None)
        if effort is not None:
            filtered_kwargs["reasoning"] = {
                **filtered_kwargs.get("reasoning", {}),
                "effort": effort,
            }
        is_reasoning_model = (
            re.match(r"^(o\d|gpt-[56](?:[.-]|$))", model_name)
            or model_name == "codex-mini-latest"
        )
        if (
            is_reasoning_model
            and filtered_kwargs.get("reasoning", {}).get("effort") != "none"
        ):
            for parameter in ("temperature", "top_p", "top_logprobs"):
                filtered_kwargs.pop(parameter, None)
            if "include" in filtered_kwargs:
                filtered_kwargs["include"] = [
                    item
                    for item in filtered_kwargs["include"]
                    if item != "message.output_text.logprobs"
                ]
        for message in messages:
            message["content"] = [{"type": "input_text", "text": message["content"]}]
        if func_spec is not None:
            filtered_kwargs["tools"] = [func_spec.as_openai_responses_tool_dict]
            filtered_kwargs["tool_choice"] = func_spec.openai_responses_tool_choice_dict

    logger.info(f"OpenAI API request: system={system_message}, user={user_message}")

    t0 = time.time()

    if use_chat_api:
        response = backoff_create(
            _custom_client.chat.completions.create,
            OPENAI_TIMEOUT_EXCEPTIONS,
            messages=messages,
            **filtered_kwargs,
        )
    else:
        response = backoff_create(
            _client.responses.create,
            OPENAI_TIMEOUT_EXCEPTIONS,
            input=messages,
            **filtered_kwargs,
        )

    req_time = time.time() - t0

    if use_chat_api:
        message = response.choices[0].message
        if func_spec is not None:
            calls = message.tool_calls or []
            if len(calls) != 1 or calls[0].function.name != func_spec.name:
                raise ValueError(f"Expected one call to {func_spec.name}")
            output = json.loads(calls[0].function.arguments)
        else:
            output = message.content
        in_tokens = response.usage.prompt_tokens
        out_tokens = response.usage.completion_tokens
    else:
        if response.status != "completed":
            raise ValueError(f"OpenAI response did not complete: {response.status}")
        if func_spec is not None:
            calls = [item for item in response.output if item.type == "function_call"]
            if len(calls) != 1 or calls[0].name != func_spec.name:
                raise ValueError(f"Expected one call to {func_spec.name}")
            output = json.loads(calls[0].arguments)
        else:
            output = response.output_text
        in_tokens = response.usage.input_tokens
        out_tokens = response.usage.output_tokens

    if func_spec is not None:
        jsonschema.validate(output, func_spec.json_schema)
    elif not output:
        raise ValueError("OpenAI returned no text")

    info = {
        "system_fingerprint": getattr(response, "system_fingerprint", None),
        "model": response.model,
        "created": getattr(response, "created", None),
    }

    logger.info(
        f"OpenAI API call completed - {response.model} - {req_time:.2f}s - {in_tokens + out_tokens} tokens (in: {in_tokens}, out: {out_tokens})"
    )
    logger.info(f"OpenAI API response: {output}")

    return output, req_time, in_tokens, out_tokens, info
