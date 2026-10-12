from types import SimpleNamespace

from aide.backend import backend_anthropic
from aide.backend.utils import FunctionSpec


def test_anthropic_supports_arbitrary_function_specs(monkeypatch):
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
    tool_block = SimpleNamespace(
        type="tool_use",
        name=func_spec.name,
        input={"lower_is_better": True},
    )
    message = SimpleNamespace(
        content=[tool_block],
        usage=SimpleNamespace(input_tokens=2, output_tokens=1),
        stop_reason="tool_use",
        model="claude-test",
    )

    def create(**kwargs):
        captured.update(kwargs)
        return message

    monkeypatch.setattr(backend_anthropic, "_setup_anthropic_client", lambda: None)
    monkeypatch.setattr(
        backend_anthropic,
        "_client",
        SimpleNamespace(messages=SimpleNamespace(create=create)),
    )

    output, _, _, _, _ = backend_anthropic.query(
        system_message="Choose a metric.",
        user_message=None,
        func_spec=func_spec,
        model="claude-test",
    )

    assert output == {"lower_is_better": True}
    assert captured["tools"] == [func_spec.as_anthropic_tool_dict]
    assert captured["tool_choice"] == {"type": "auto"}
    assert func_spec.name in captured["messages"][0]["content"]
