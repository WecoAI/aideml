"""Provider contracts exercised through real SDKs with mocked HTTP responses."""

import json
import importlib

import anthropic
import jsonschema
import openai
import pytest
from omegaconf import OmegaConf

from aide import backend
from aide.agent import (
    Agent,
    determine_task_metric,
    review_func_spec,
    task_metric_func_spec,
)
from aide.backend import backend_anthropic, backend_openai, utils
from aide.backend.utils import FunctionSpec
from aide.interpreter import ExecutionResult
from aide.journal import Journal, Node
from aide.journal2report import journal2report
from aide.utils.config import Config, StageConfig

OPENAI_MODELS = ["gpt-6-astra", "gpt-6.1-sol", "gpt-6-luna", "gpt-6-sol"]
CLAUDE_MODELS = [
    "claude-fable-5-1",
    "claude-opus-5-5",
    "claude-sonnet-5-5",
    "claude-haiku-5-5",
]
CASES = [
    (None, "Plan\n```python\nprint(1)\n```"),
    (task_metric_func_spec, {"metric_name": "RMSE", "lower_is_better": True}),
    (
        review_func_spec,
        {
            "is_bug": True,
            "summary": "Training failed.",
            "metric": None,
            "lower_is_better": True,
        },
    ),
]


def provider_response(provider, model, spec, value):
    if provider == "openai":
        output = [{"type": "reasoning", "id": "rs_1", "summary": []}]
        if spec:
            output.append(
                {
                    "type": "function_call",
                    "id": "fc_1",
                    "call_id": "call_1",
                    "name": spec.name,
                    "arguments": json.dumps(value),
                    "status": "completed",
                }
            )
        else:
            output.append(
                {
                    "type": "message",
                    "role": "assistant",
                    "id": "msg_1",
                    "status": "completed",
                    "content": [
                        {"type": "output_text", "text": value, "annotations": []}
                    ],
                }
            )
        return {
            "id": "resp_1",
            "object": "response",
            "created_at": 1,
            "status": "completed",
            "model": model,
            "output": output,
            "usage": {"input_tokens": 12, "output_tokens": 8, "total_tokens": 20},
        }
    content = [{"type": "thinking", "thinking": "", "signature": "signature"}]
    if spec:
        content += [
            {"type": "text", "text": "Submitting the result."},
            {"type": "tool_use", "id": "tool_1", "name": spec.name, "input": value},
        ]
    else:
        content += [
            {"type": "text", "text": value[:5]},
            {"type": "text", "text": value[5:]},
        ]
    return {
        "id": "msg_1",
        "type": "message",
        "role": "assistant",
        "model": model,
        "content": content,
        "stop_reason": "tool_use" if spec else "end_turn",
        "stop_sequence": None,
        "usage": {"input_tokens": 12, "output_tokens": 8},
    }


def mock_response(status_code, *, json):
    return status_code, json


def mock_sdk_client(sdk, handler, **kwargs):
    # SDK major versions can use different HTTP transport packages.
    transport_name = next(
        base.__module__.split(".")[0]
        for base in sdk.DefaultHttpxClient.__mro__
        if base.__module__.split(".")[0] in {"httpx", "httpx2"}
    )
    transport = importlib.import_module(transport_name)

    def dispatch(request):
        status_code, payload = handler(request)
        return transport.Response(status_code, json=payload)

    cls = openai.OpenAI if sdk is openai else anthropic.Anthropic
    return cls(
        api_key="test-key",
        max_retries=0,
        http_client=sdk.DefaultHttpxClient(transport=transport.MockTransport(dispatch)),
        **kwargs,
    )


@pytest.fixture
def install_client(monkeypatch):
    clients = []

    def install(provider, handler):
        module = backend_openai if provider == "openai" else backend_anthropic
        sdk = openai if provider == "openai" else anthropic
        client = mock_sdk_client(sdk, handler)
        clients.append(client)
        monkeypatch.setattr(module, "_client", client)
        monkeypatch.setattr(module, f"_setup_{provider}_client", lambda: None)
        return client

    yield install
    for client in clients:
        client.close()


@pytest.mark.parametrize("model", OPENAI_MODELS + CLAUDE_MODELS)
@pytest.mark.parametrize("spec,value", CASES, ids=["code", "task-metric", "review"])
def test_current_models_serialize_and_parse(install_client, model, spec, value):
    provider = backend.determine_provider(model)
    captured = {}

    def handler(request):
        captured.update(json.loads(request.content))
        assert request.url.path == (
            "/v1/responses" if provider == "openai" else "/v1/messages"
        )
        return mock_response(200, json=provider_response(provider, model, spec, value))

    install_client(provider, handler)
    result = backend.query(
        "Generate or review a solution.",
        None,
        model=model,
        temperature=0.5,
        top_p=0.9,
        max_tokens=16384,
        reasoning_effort="low",
        func_spec=spec,
    )
    assert result == value
    assert "temperature" not in captured and "top_p" not in captured
    if provider == "openai":
        assert captured["max_output_tokens"] == 16384
        assert captured["reasoning"] == {"effort": "low"}
        if spec:
            assert captured["tools"][0]["parameters"] == spec.json_schema
            assert captured["tools"][0]["strict"] is False
    else:
        assert captured["max_tokens"] == 16384
        assert captured["output_config"] == {"effort": "low"}
        if spec:
            assert captured["tools"][0]["input_schema"] == spec.json_schema
            if model != "claude-haiku-5-5":
                assert captured["tool_choice"] == {"type": "auto"}
                assert spec.name in captured["messages"][0]["content"]
            else:
                assert captured["tool_choice"] == spec.anthropic_tool_choice_dict


@pytest.mark.parametrize("model", ["gpt-4.1-mini", "gpt-6-luna", "claude-haiku-4-5"])
def test_older_models_and_explicit_nonreasoning_sampling(install_client, model):
    provider = backend.determine_provider(model)
    captured = {}

    def handler(request):
        captured.update(json.loads(request.content))
        return mock_response(200, json=provider_response(provider, model, None, "code"))

    install_client(provider, handler)
    kwargs = {"reasoning_effort": "none"} if model == "gpt-6-luna" else {}
    assert backend.query("Code", None, model, temperature=0.5, **kwargs) == "code"
    assert captured["temperature"] == 0.5


@pytest.mark.parametrize(
    "provider,model", [("openai", "gpt-6.1-sol"), ("anthropic", "claude-opus-5-5")]
)
@pytest.mark.parametrize(
    "failure", ["missing-tool", "wrong-tool", "invalid-schema", "truncated", "refusal"]
)
def test_structured_failures_are_explicit(install_client, provider, model, failure):
    spec, value = CASES[1]
    response = provider_response(provider, model, spec, value)
    blocks = response["output"] if provider == "openai" else response["content"]
    if failure == "missing-tool":
        blocks.pop()
    elif failure == "wrong-tool":
        blocks[-1]["name"] = "wrong_tool"
    elif failure == "invalid-schema":
        blocks[-1]["arguments" if provider == "openai" else "input"] = (
            '"wrong"' if provider == "openai" else "wrong"
        )
    elif provider == "openai":
        response["status"] = "incomplete" if failure == "truncated" else "failed"
    else:
        response["stop_reason"] = "max_tokens" if failure == "truncated" else "refusal"
    install_client(provider, lambda _: mock_response(200, json=response))
    with pytest.raises((ValueError, jsonschema.ValidationError)):
        backend.query("Review", None, model, func_spec=spec)


@pytest.mark.parametrize(
    "provider,model", [("openai", "gpt-6.1-sol"), ("anthropic", "claude-opus-5-5")]
)
def test_provider_retries_are_finite(install_client, monkeypatch, provider, model):
    def no_wait(**kwargs):
        while True:
            yield 0

    monkeypatch.setattr(utils.backoff, "expo", no_wait)
    attempts = []

    def handler(request):
        attempts.append(request)
        return mock_response(
            429, json={"error": {"type": "rate_limit_error", "message": "slow down"}}
        )

    install_client(provider, handler)
    error = openai.RateLimitError if provider == "openai" else anthropic.RateLimitError
    with pytest.raises(error):
        backend.query("Code", None, model)
    assert len(attempts) == utils.BACKOFF_MAX_TRIES == 5


def test_arbitrary_optional_schema_is_preserved(install_client):
    spec = FunctionSpec(
        "custom_result",
        {
            "type": "object",
            "properties": {
                "value": {"type": "integer"},
                "optional": {"type": "string"},
            },
            "required": ["value"],
        },
        "Return a result.",
    )
    for provider, model in [
        ("openai", "gpt-6-astra"),
        ("anthropic", "claude-sonnet-5-5"),
    ]:
        install_client(
            provider,
            lambda _, p=provider, m=model: mock_response(
                200, json=provider_response(p, m, spec, {"value": 7})
            ),
        )
        assert backend.query("Calculate", None, model, func_spec=spec) == {"value": 7}


def test_local_chat_token_limit_and_tool_path(monkeypatch):
    captured = {}

    def handler(request):
        captured.update(json.loads(request.content))
        return mock_response(
            200,
            json={
                "id": "chat_1",
                "object": "chat.completion",
                "created": 1,
                "model": "local-model",
                "choices": [
                    {
                        "index": 0,
                        "finish_reason": "tool_calls",
                        "message": {
                            "role": "assistant",
                            "content": None,
                            "tool_calls": [
                                {
                                    "id": "call_1",
                                    "type": "function",
                                    "function": {
                                        "name": task_metric_func_spec.name,
                                        "arguments": json.dumps(CASES[1][1]),
                                    },
                                }
                            ],
                        },
                    }
                ],
                "usage": {
                    "prompt_tokens": 12,
                    "completion_tokens": 8,
                    "total_tokens": 20,
                },
            },
        )

    monkeypatch.setenv("OPENAI_BASE_URL", "http://localhost:1234/v1")
    monkeypatch.setattr(backend_openai, "_setup_openai_client", lambda: None)
    monkeypatch.setattr(backend_openai, "_setup_custom_client", lambda: None)
    monkeypatch.setattr(
        backend_openai,
        "_custom_client",
        mock_sdk_client(openai, handler, base_url="http://localhost:1234/v1"),
    )
    assert (
        backend.query(
            "Select",
            None,
            "local-model",
            max_tokens=512,
            func_spec=task_metric_func_spec,
        )
        == CASES[1][1]
    )
    assert captured["max_tokens"] == 512
    assert "max_output_tokens" not in captured


@pytest.mark.parametrize("model", ["gpt-6.1-sol", "claude-opus-5-5"])
def test_agent_and_report_forward_stage_options(install_client, model):
    provider = backend.determine_provider(model)
    captured = []

    def handler(request):
        data = json.loads(request.content)
        captured.append(data)
        tools = data.get("tools", [])
        spec, value = CASES[0]
        if tools:
            name = tools[0]["name"]
            spec, value = next(
                case for case in CASES if case[0] and case[0].name == name
            )
        return mock_response(200, json=provider_response(provider, model, spec, value))

    install_client(provider, handler)
    raw_cfg = OmegaConf.load("aide/utils/config.yaml")
    raw_cfg.data_dir = "."
    raw_cfg.exp_name = "test"
    cfg = OmegaConf.merge(OmegaConf.structured(Config), raw_cfg)
    for stage in [cfg.agent.code, cfg.agent.feedback, cfg.report]:
        stage.model = model
        stage.max_tokens = 16384
        stage.reasoning_effort = "low"
    assert determine_task_metric("Use RMSE", cfg.agent).name == "RMSE"
    agent = Agent(task_desc="Use RMSE", cfg=cfg, journal=Journal())
    assert agent.plan_and_code_query("Code")[1] == "print(1)\n"
    node = Node(code="pass")
    agent.parse_exec_result(
        node, ExecutionResult(term_out=["failed"], exec_time=0.1, exc_type=None)
    )
    assert node.is_buggy
    assert journal2report(Journal(), {"Task": "Use RMSE"}, cfg.report)
    assert len(captured) == 4
    for data in captured:
        assert (
            data["max_output_tokens" if provider == "openai" else "max_tokens"] == 16384
        )
        assert data["reasoning" if provider == "openai" else "output_config"] == {
            "effort": "low"
        }


def test_old_stage_config_remains_valid():
    stage = OmegaConf.merge(
        OmegaConf.structured(StageConfig), {"model": "gpt-4.1", "temp": 0.5}
    )
    assert stage.max_tokens is None and stage.reasoning_effort is None


@pytest.mark.parametrize(
    "provider,model", [("openai", "gpt-6.1-sol"), ("anthropic", "claude-opus-5-5")]
)
def test_bad_tool_request_is_not_retried_or_downgraded(install_client, provider, model):
    attempts = []

    def handler(request):
        attempts.append(json.loads(request.content))
        return mock_response(
            400,
            json={
                "error": {
                    "type": "invalid_request_error",
                    "message": "Invalid tools schema",
                }
            },
        )

    install_client(provider, handler)
    error = (
        openai.BadRequestError if provider == "openai" else anthropic.BadRequestError
    )
    with pytest.raises(error):
        backend.query("Choose RMSE", None, model, func_spec=task_metric_func_spec)
    assert len(attempts) == 1
    assert attempts[0]["tools"]


def test_reasoning_logprobs_are_removed(install_client):
    captured = {}

    def handler(request):
        captured.update(json.loads(request.content))
        return mock_response(
            200, json=provider_response("openai", "gpt-6-astra", None, "code")
        )

    install_client("openai", handler)
    backend.query(
        "Code",
        None,
        "gpt-6-astra",
        top_logprobs=2,
        include=["message.output_text.logprobs", "file_search_call.results"],
    )
    assert "top_logprobs" not in captured
    assert captured["include"] == ["file_search_call.results"]


@pytest.mark.parametrize(
    "provider,model", [("openai", "gpt-6.1-sol"), ("anthropic", "claude-opus-5-5")]
)
def test_thinking_without_visible_text_raises(install_client, provider, model):
    response = provider_response(provider, model, None, "")
    install_client(provider, lambda _: mock_response(200, json=response))
    with pytest.raises(ValueError, match="no text"):
        backend.query("Code", None, model)
