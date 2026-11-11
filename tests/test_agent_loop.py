"""Agent 循环测试：正常、预算、伪造引用、越权路径、无证据、超时与历史。"""

from __future__ import annotations

import pytest
from fakes import FakeTools, ScriptedLLM, tool_call

from codeagent.agent.loop import AgentLimits, CodeAgent
from codeagent.agent.tools import AgentToolError


def search_result(
    path: str = "pkg/b.py",
    start: int = 1,
    end: int = 3,
    qualname: str = "ToolExecutor",
    text: str = "class ToolExecutor: ...",
) -> dict[str, object]:
    return {
        "tool": "search_code",
        "query": "ToolExecutor",
        "count": 1,
        "hits": [
            {
                "path": path,
                "start_line": start,
                "end_line": end,
                "qualname": qualname,
                "kind": "class",
                "score": 0.5,
                "text": text,
            }
        ],
    }


def make_agent(llm: ScriptedLLM, tools: FakeTools | None = None, **limits) -> CodeAgent:
    fake_tools = tools or FakeTools(results={"search_code": search_result()})
    return CodeAgent(fake_tools, llm, AgentLimits(**limits))


def planner_turn(*calls: dict[str, object]) -> dict[str, object]:
    return {"text": "", "tool_calls": list(calls)}


def test_normal_loop_maps_citations() -> None:
    llm = ScriptedLLM(
        tool_responses=[
            planner_turn(tool_call("search_code", {"query": "ToolExecutor"})),
            planner_turn(),
        ],
        answer="ToolExecutor 是工具执行类 [1]\n\n它定义在 pkg/b.py [1]。",
    )
    result = make_agent(llm).ask("ToolExecutor 在哪？")

    assert result["refusal"] is None
    assert result["check"] == "1/1 通过"
    assert result["citations"][0]["n"] == 1
    assert result["citations"][0]["path"] == "pkg/b.py"
    assert result["citations"][0]["start_line"] == 1
    assert result["citations"][0]["qualname"] == "ToolExecutor"
    assert result["stats"]["turns"] == 2
    assert result["stats"]["tool_calls"] == 1
    assert result["stats"]["budget_exhausted"] is False

    planner = llm.messages_seen[1][1]
    assert any(message["role"] == "tool" for message in planner)
    assert any(message["role"] == "assistant" and message.get("tool_calls") for message in planner)
    synth = llm.synth_messages()
    assert synth[0]["role"] == "system"
    assert "【检索上下文】" in synth[-1]["content"]
    assert "[1] pkg/b.py:1-3" in synth[-1]["content"]
    assert "abs_path" not in result["contexts"][0]


def test_planner_without_tools_refuses() -> None:
    llm = ScriptedLLM(tool_responses=[planner_turn()], answer="不应被调用 [1]")
    result = make_agent(llm).ask("这是什么？")
    assert result["refusal"]["code"] == "no_evidence"
    assert result["citations"] == []
    assert llm.synth_messages() == []
    assert result["stats"]["tool_calls"] == 0


def test_tool_budget_exhausted_still_synthesizes() -> None:
    responses = [
        planner_turn(
            tool_call("search_code", {"query": "ToolExecutor"}),
            tool_call("search_code", {"query": "ToolExecutor"}),
        )
        for _ in range(6)
    ]
    llm = ScriptedLLM(tool_responses=responses, answer="结论 [1]\n\n展开 [1]。")
    result = make_agent(llm, max_tool_calls=3).ask("ToolExecutor 在哪？")

    assert result["stats"]["budget_exhausted"] is True
    assert result["stats"]["tool_calls"] == 3
    assert len(result["contexts"]) == 1  # 重复命中去重，编号仍连续
    assert result["citations"][0]["n"] == 1
    assert result["refusal"] is None


def test_turn_budget_exhausted() -> None:
    responses = [planner_turn(tool_call("search_code", {"query": "x"})) for _ in range(6)]
    llm = ScriptedLLM(tool_responses=responses, answer="结论 [1]\n\n展开 [1]。")
    result = make_agent(llm, max_tool_calls=20, max_turns=3).ask("问题")
    assert result["stats"]["turns"] == 3
    assert result["stats"]["tool_calls"] == 3
    assert result["stats"]["budget_exhausted"] is False
    assert len(result["contexts"]) == 1


def test_forged_citation_falls_back_extractive() -> None:
    llm = ScriptedLLM(
        tool_responses=[
            planner_turn(tool_call("search_code", {"query": "ToolExecutor"})),
            planner_turn(),
        ],
        answer="伪造结论 [99]\n\n伪造展开 [99]。",
    )
    result = make_agent(llm).ask("ToolExecutor 在哪？")

    assert result["stats"]["fallback"] == "extractive"
    assert result["stats"]["invalid_citations"] == [99]
    assert result["refusal"] is None
    assert result["citations"] and result["citations"][0]["n"] == 1
    text = result["answer"]["lead"] + result["answer"]["body"]
    assert "[99]" not in text
    assert "[1]" in text


def test_mixed_citation_keeps_valid_and_strips_invalid() -> None:
    llm = ScriptedLLM(
        tool_responses=[
            planner_turn(tool_call("search_code", {"query": "ToolExecutor"})),
            planner_turn(),
        ],
        answer="结论见 [1]，伪造 [99] 不应保留。\n\n展开 [1]。",
    )
    result = make_agent(llm).ask("ToolExecutor 在哪？")

    assert result["stats"]["fallback"] is None
    assert result["invalid_citations"] == [99]
    assert [item["n"] for item in result["citations"]] == [1]
    text = result["answer"]["lead"] + result["answer"]["body"]
    assert "[1]" in text
    assert "[99]" not in text


@pytest.mark.parametrize("bad_path", ["../secret.py", "/etc/passwd"])
def test_out_of_root_evidence_is_rejected(bad_path: str) -> None:
    tools = FakeTools(results={"search_code": search_result(path=bad_path)})
    llm = ScriptedLLM(
        tool_responses=[
            planner_turn(tool_call("search_code", {"query": "secret"})),
            planner_turn(),
        ],
        answer="越权答案 [1]",
    )
    result = make_agent(llm, tools).ask("secret 在哪？")
    assert result["contexts"] == []
    assert result["refusal"]["code"] == "no_evidence"
    assert result["citations"] == []
    assert llm.synth_messages() == []


def test_empty_search_refuses_without_synthesize() -> None:
    tools = FakeTools(results={"search_code": {"tool": "search_code", "query": "x", "count": 0, "hits": []}})
    llm = ScriptedLLM(
        tool_responses=[
            planner_turn(tool_call("search_code", {"query": "不存在"})),
            planner_turn(),
        ],
        answer="编造答案 [1]",
    )
    result = make_agent(llm, tools).ask("不存在的东西？")
    assert result["refusal"]["code"] == "no_evidence"
    assert result["answer"]["lead"]
    assert llm.synth_messages() == []


def test_synthesize_failure_falls_back() -> None:
    llm = ScriptedLLM(
        tool_responses=[
            planner_turn(tool_call("search_code", {"query": "ToolExecutor"})),
            planner_turn(),
        ],
        answer="回答 [1]",
        fail_chat=True,
    )
    result = make_agent(llm).ask("ToolExecutor 在哪？")
    assert result["refusal"] is None
    assert result["stats"]["fallback"] == "extractive"
    assert result["stats"]["degraded"] is True
    assert "synth down" in str(result["stats"]["llm_error"])
    assert result["citations"][0]["path"] == "pkg/b.py"


def test_planner_llm_failure_refuses() -> None:
    llm = ScriptedLLM(fail_tools=True)
    result = make_agent(llm).ask("问题")
    assert result["refusal"]["code"] == "llm_error"
    assert result["stats"]["degraded"] is True


def test_timeout_stops_planning() -> None:
    class Clock:
        def __init__(self) -> None:
            self.value = 0.0

        def __call__(self) -> float:
            return self.value

    class SlowLLM(ScriptedLLM):
        def __init__(self, clock: Clock) -> None:
            super().__init__(
                tool_responses=[
                    planner_turn(tool_call("search_code", {"query": "ToolExecutor"})),
                    planner_turn(),
                ],
                answer="结论 [1]\n\n展开 [1]。",
            )
            self.clock = clock

        def chat_tools(self, messages, tools, max_tokens=1024):
            result = super().chat_tools(messages, tools, max_tokens)
            self.clock.value += 200.0
            return result

    clock = Clock()
    llm = SlowLLM(clock)
    agent = CodeAgent(FakeTools(results={"search_code": search_result()}), llm, AgentLimits(), clock=clock)
    result = agent.ask("ToolExecutor 在哪？")
    assert result["stats"]["timed_out"] is True
    assert result["stats"]["turns"] == 1
    assert result["stats"]["tool_calls"] == 1
    assert result["refusal"] is None


def test_history_is_passed_to_both_phases() -> None:
    history = [
        {"role": "user", "content": "上一轮问题"},
        {"role": "assistant", "content": "上一轮回答"},
    ]
    llm = ScriptedLLM(
        tool_responses=[
            planner_turn(tool_call("search_code", {"query": "ToolExecutor"})),
            planner_turn(),
        ],
        answer="结论 [1]\n\n展开 [1]。",
    )
    result = make_agent(llm).ask("继续问 ToolExecutor", history)
    planner = llm.messages_seen[1][1]
    assert any(message.get("content") == "上一轮问题" for message in planner)
    synth = llm.synth_messages()
    assert any(message.get("content") == "上一轮回答" for message in synth)
    assert result["citations"]


def test_unknown_tool_and_bad_arguments_are_reported() -> None:
    bad_json = {
        "id": "call_bad",
        "type": "function",
        "function": {"name": "search_code", "arguments": "{not-json"},
    }
    llm = ScriptedLLM(
        tool_responses=[
            planner_turn(tool_call("nope", {}), bad_json),
            planner_turn(),
        ],
        answer="答案 [1]",
    )
    result = make_agent(llm).ask("问题")
    assert result["tool_calls"][0]["ok"] is False
    assert "未知工具" in str(result["tool_calls"][0]["error"])
    assert result["tool_calls"][1]["ok"] is False
    assert "JSON" in str(result["tool_calls"][1]["error"])
    assert result["refusal"]["code"] == "no_evidence"


def test_tool_exception_is_structured() -> None:
    tools = FakeTools(results={"search_code": AgentToolError("not_found", "没找到")})
    llm = ScriptedLLM(
        tool_responses=[planner_turn(tool_call("search_code", {"query": "x"})), planner_turn()],
        answer="答案 [1]",
    )
    result = make_agent(llm, tools).ask("问题")
    assert result["tool_calls"][0]["ok"] is False
    assert result["tool_calls"][0]["error"] == "没找到"
    assert result["refusal"]["code"] == "no_evidence"
