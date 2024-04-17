"""Backend for Anthropic API."""

import time

from anthropic import Anthropic, RateLimitError
from .utils import FunctionSpec, OutputType, opt_messages_to_list
from funcy import notnone, once, retry, select_values

_client: Anthropic = None  # type: ignore

RATELIMIT_RETRIES = 5
retry_exp = retry(RATELIMIT_RETRIES, errors=RateLimitError, timeout=lambda a: 2 ** (a + 1))  # type: ignore


@once
def _setup_anthropic_client():
    global _client
    _client = Anthropic()


@retry_exp
def query(
    system_message: str | None,
    user_message: str | None,
    func_spec: FunctionSpec | None = None,
    **model_kwargs,
) -> tuple[OutputType, float, int, int, dict]:
    _setup_anthropic_client()

    filtered_kwargs: dict = select_values(notnone, model_kwargs)  # type: ignore
    if "max_tokens" not in filtered_kwargs:
        filtered_kwargs["max_tokens"] = 4096  # default for Claude models

    if func_spec is not None:
        raise NotImplementedError(
            "Anthropic does not support function calling for now."
        )

    # Anthropic doesn't allow not having a user messages
    # if we only have system msg -> use it as user msg
    if system_message is not None and user_message is None:
        system_message, user_message = user_message, system_message

    # Anthropic passes the system messages as a separate argument
    if system_message is not None:
        filtered_kwargs["system"] = system_message

    messages = opt_messages_to_list(None, user_message)

    t0 = time.time()
    message = _client.messages.create(messages=messages, **filtered_kwargs)  # type: ignore
    req_time = time.time() - t0

    assert len(message.content) == 1 and message.content[0].type == "text"

    output: str = message.content[0].text
    in_tokens = message.usage.input_tokens
    out_tokens = message.usage.output_tokens

    info = {
        "stop_reason": message.stop_reason,
    }

    return output, req_time, in_tokens, out_tokens, info
