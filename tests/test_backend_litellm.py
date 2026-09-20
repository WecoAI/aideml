from types import SimpleNamespace

from aide.backend import backend_litellm
from aide.backend import determine_provider
from aide.backend.utils import FunctionSpec


def _completion(content=None, tool_calls=None, model="litellm-test"):
    message = SimpleNamespace(content=content, tool_calls=tool_calls)
    return SimpleNamespace(
        choices=[SimpleNamespace(message=message)],
        usage=SimpleNamespace(prompt_tokens=3, completion_tokens=2),
        model=model,
        system_fingerprint="fp_test",
        created=1234567890,
    )


def test_determine_provider_routes_litellm_prefix():
    assert determine_provider("litellm/gpt-4o") == "litellm"
    assert (
        determine_provider("litellm/anthropic/claude-3-5-sonnet-20241022") == "litellm"
    )
    # Bare model ids still route to their native backends.
    assert determine_provider("gpt-4o") == "openai"
    assert determine_provider("claude-3-5-sonnet-20241022") == "anthropic"


def test_litellm_strips_prefix_and_defaults_drop_params(monkeypatch):
    captured = {}

    def completion(**kwargs):
        captured.update(kwargs)
        return _completion(content="hello")

    monkeypatch.setattr(backend_litellm.litellm, "completion", completion)

    output, _, in_tok, out_tok, info = backend_litellm.query(
        system_message="You are helpful.",
        user_message="Say hello.",
        func_spec=None,
        model="litellm/gpt-4o",
        temperature=0.5,
    )

    assert output == "hello"
    assert in_tok == 3 and out_tok == 2
    assert info["model"] == "litellm-test"
    # Prefix stripped before reaching LiteLLM.
    assert captured["model"] == "gpt-4o"
    # drop_params defaults on for cross-provider compatibility.
    assert captured["drop_params"] is True
    assert captured["temperature"] == 0.5
    assert captured["messages"] == [
        {"role": "system", "content": "You are helpful."},
        {"role": "user", "content": "Say hello."},
    ]


def test_litellm_drop_params_opt_out(monkeypatch):
    captured = {}

    def completion(**kwargs):
        captured.update(kwargs)
        return _completion(content="ok")

    monkeypatch.setattr(backend_litellm.litellm, "completion", completion)

    backend_litellm.query(
        system_message=None,
        user_message="hi",
        model="litellm/gpt-4o",
        drop_params=False,
    )

    assert captured["drop_params"] is False


def test_litellm_supports_arbitrary_function_specs(monkeypatch):
    captured = {}
    func_spec = FunctionSpec(
        name="submit_task_metric",
        json_schema={
            "type": "object",
            "properties": {"lower_is_better": {"type": "boolean"}},
            "required": ["lower_is_better"],
        },
        description="Submit a task metric.",
    )
    tool_call = SimpleNamespace(
        function=SimpleNamespace(
            name=func_spec.name,
            arguments='{"lower_is_better": true}',
        )
    )

    def completion(**kwargs):
        captured.update(kwargs)
        return _completion(tool_calls=[tool_call])

    monkeypatch.setattr(backend_litellm.litellm, "completion", completion)

    output, _, _, _, _ = backend_litellm.query(
        system_message="Choose a metric.",
        user_message=None,
        func_spec=func_spec,
        model="litellm/gpt-4o",
    )

    assert output == {"lower_is_better": True}
    assert captured["tools"] == [func_spec.as_openai_tool_dict]
    assert captured["tool_choice"] == func_spec.openai_tool_choice_dict


def test_litellm_forwards_proxy_env(monkeypatch):
    captured = {}

    def completion(**kwargs):
        captured.update(kwargs)
        return _completion(content="via-proxy")

    monkeypatch.setattr(backend_litellm.litellm, "completion", completion)
    monkeypatch.setenv("LITELLM_API_BASE", "http://localhost:4000/v1")
    monkeypatch.setenv("LITELLM_API_KEY", "sk-proxy")

    backend_litellm.query(
        system_message=None,
        user_message="hi",
        model="litellm/gpt-4o",
    )

    assert captured["api_base"] == "http://localhost:4000/v1"
    assert captured["api_key"] == "sk-proxy"


def test_litellm_omits_creds_when_env_absent(monkeypatch):
    captured = {}

    def completion(**kwargs):
        captured.update(kwargs)
        return _completion(content="native")

    monkeypatch.setattr(backend_litellm.litellm, "completion", completion)
    monkeypatch.delenv("LITELLM_API_BASE", raising=False)
    monkeypatch.delenv("LITELLM_BASE_URL", raising=False)
    monkeypatch.delenv("LITELLM_API_KEY", raising=False)

    backend_litellm.query(
        system_message=None,
        user_message="hi",
        model="litellm/anthropic/claude-3-5-sonnet-20241022",
    )

    # No proxy creds forwarded -> LiteLLM falls back to provider env vars.
    assert "api_base" not in captured
    assert "api_key" not in captured
    assert captured["model"] == "anthropic/claude-3-5-sonnet-20241022"
