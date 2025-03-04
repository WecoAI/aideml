"""Backend for Anthropic API."""

import logging
import time

from .utils import FunctionSpec, OutputType, opt_messages_to_list, backoff_create
from funcy import notnone, once, select_values
import anthropic

logger = logging.getLogger("aide")

_client: anthropic.Anthropic = None  # type: ignore

ANTHROPIC_TIMEOUT_EXCEPTIONS = (
    anthropic.RateLimitError,
    anthropic.APIConnectionError,
    anthropic.APITimeoutError,
    anthropic.InternalServerError,
)

# Define thinking-enabled model aliases
# Format: "alias": ("actual_model_name", thinking_budget)
ANTHROPIC_MODEL_ALIASES = {
    "claude-3.5-sonnet": "claude-3-5-sonnet-20241022",
    "claude-3.7-sonnet": "claude-3-7-sonnet-20250219",
    "claude-3.7-sonnet-thinking": (
        "claude-3-7-sonnet-20250219",
        16000,
    ),  # With default 16K thinking budget
}


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

    Extended thinking is automatically enabled when using model aliases with "-thinking" suffix.
    """
    _setup_anthropic_client()

    filtered_kwargs: dict = select_values(notnone, model_kwargs)  # type: ignore
    if "max_tokens" not in filtered_kwargs:
        filtered_kwargs["max_tokens"] = 8192  # default for Claude models

    model_name = filtered_kwargs.get("model", "")
    logger.debug(f"Anthropic query called with model='{model_name}'")

    # Check if this is a thinking-enabled model alias
    thinking_enabled = False
    thinking_budget = None

    if model_name in ANTHROPIC_MODEL_ALIASES:
        alias_value = ANTHROPIC_MODEL_ALIASES[model_name]

        # Check if this is a tuple with thinking budget
        if isinstance(alias_value, tuple):
            model_name = alias_value[0]
            thinking_budget = alias_value[1]
            thinking_enabled = True
            logger.debug(
                f"Using thinking-enabled model: {model_name} with budget: {thinking_budget}"
            )
        else:
            model_name = alias_value

        filtered_kwargs["model"] = model_name
        logger.debug(f"Using aliased model name: {model_name}")

    # Configure extended thinking if enabled via alias
    if thinking_enabled and thinking_budget is not None:
        if thinking_budget >= filtered_kwargs["max_tokens"]:
            logger.warning("thinking_budget must be less than max_tokens, adjusting")
            thinking_budget = filtered_kwargs["max_tokens"] - 1

        filtered_kwargs["thinking"] = {
            "type": "enabled",
            "budget_tokens": thinking_budget,
        }
        filtered_kwargs["temperature"] = 1  # temp must be 1 when thinking enabled

    if func_spec is not None and func_spec.name == "submit_review":
        filtered_kwargs["tools"] = [func_spec.as_anthropic_tool_dict]
        # Force tool use
        filtered_kwargs["tool_choice"] = func_spec.anthropic_tool_choice_dict

    # Anthropic doesn't allow not having user messages
    # if we only have system msg -> use it as user msg
    if system_message is not None and user_message is None:
        system_message, user_message = user_message, system_message

    # Anthropic passes system messages as a separate argument
    if system_message is not None:
        filtered_kwargs["system"] = system_message

    messages = opt_messages_to_list(None, user_message)

    t0 = time.time()
    logger.info(f"Sending request to Anthropic API with model: {model_name}")
    if thinking_enabled:
        logger.info(f"Thinking mode enabled with budget: {thinking_budget} tokens")

    # Enable streaming if thinking is enabled
    if thinking_enabled:
        # Process the stream
        with backoff_create(
            _client.beta.messages.stream,
            ANTHROPIC_TIMEOUT_EXCEPTIONS,
            messages=messages,
            betas=["output-128k-2025-02-19"],
            **filtered_kwargs,
        ) as stream:
            for event in stream:
                if event.type == "content_block_start":
                    # print(f"\nStarting {event.content_block.type} block...")
                    pass
                elif event.type == "content_block_delta":
                    if event.delta.type == "thinking_delta":
                        text = event.delta.thinking.replace("\n", "")
                        print(text, end="", flush=True)
                    elif event.delta.type == "text_delta":
                        text = event.delta.text.replace("\n", "")
                        print(text, end="", flush=True)
                elif event.type == "content_block_stop":
                    # print("\nBlock complete.")
                    pass

        message = stream.get_final_message()
    else:
        # Non-streaming approach (original)
        message = backoff_create(
            _client.messages.create,
            ANTHROPIC_TIMEOUT_EXCEPTIONS,
            messages=messages,
            **filtered_kwargs,
        )

    req_time = time.time() - t0

    # Handle tool calls if present
    if (
        func_spec is not None
        and "tools" in filtered_kwargs
        and len(message.content) > 0
        and message.content[0].type == "tool_use"
    ):
        block = message.content[0]  # This is a "ToolUseBlock"
        # block has attributes: type, id, name, input
        assert (
            block.name == func_spec.name
        ), f"Function name mismatch: expected {func_spec.name}, got {block.name}"
        output = block.input  # Anthropic calls the parameters "input"
    else:
        # For non-tool responses, handle text content
        if len(message.content) == 0:
            logger.warning("Received empty content from Anthropic API")
            output = ""
        else:
            # Check if thinking content is present (for logging purposes)
            thinking_content = None
            for item in message.content:
                if item.type == "thinking":
                    thinking_content = item
                    logger.info(f"Thinking content: {thinking_content.thinking}")

            # Collect and concatenate all text content items
            text_contents = []
            for content_item in message.content:
                if content_item.type == "text":
                    text_contents.append(content_item.text)

            if not text_contents:
                logger.warning(
                    f"No text content found in response. Content types: {[item.type for item in message.content]}"
                )
                output = ""
            else:
                output = "".join(text_contents)
                if len(text_contents) > 1:
                    logger.info(
                        f"Concatenated {len(text_contents)} text content blocks"
                    )

    in_tokens = message.usage.input_tokens
    out_tokens = message.usage.output_tokens

    info = {
        "stop_reason": message.stop_reason,
        "model": message.model,
    }

    logger.info(f"Request completed in {req_time:.2f} seconds")
    logger.info(f"Tokens: {in_tokens} input, {out_tokens} output")

    return output, req_time, in_tokens, out_tokens, info
