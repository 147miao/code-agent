"""citations 测试：解析、通用范围校验、结构化答案拆分、代码引用校验。"""
from __future__ import annotations

from pathlib import Path

from codeagent.citations import (
    parse_citations,
    parse_structured_answer,
    verify_citations,
    verify_code_citations,
)


def test_parse_and_verify_citations() -> None:
    assert parse_citations("见 [1] 和 [2][1]，还有 [10]") == [1, 2, 10]
    assert parse_citations("无引用") == []
    assert verify_citations("无引用", 3) == ([], [], "0/0 通过（无引用）")
    assert verify_citations("见 [1][3]", 2) == ([1], [3], "1/2 通过")


def test_parse_structured_answer_lead_and_body() -> None:
    assert parse_structured_answer("结论：A 是 B。\n\n展开说明。") == ("A 是 B。", "展开说明。")
    assert parse_structured_answer("foo 负责请求。后续说明。") == ("foo 负责请求。", "后续说明。")
    assert parse_structured_answer("  单行结论  ") == ("单行结论", "")
    assert parse_structured_answer("") == ("", "")


def test_parse_structured_answer_strips_code_fence() -> None:
    fence = chr(96) * 3
    text = f"{fence}text\n结论：A 是 B。\n\n展开说明。\n{fence}"
    assert parse_structured_answer(text) == ("A 是 B。", "展开说明。")


def test_verify_code_citations_filters_invalid_numbers() -> None:
    contexts = [
        {
            "n": 1,
            "path": "src/a.py",
            "abs_path": "/repo/src/a.py",
            "start_line": 2,
            "end_line": 5,
            "qualname": "foo",
        }
    ]
    citations, invalid, check = verify_code_citations(
        "见 [1] 和 [2]。",
        contexts,
        file_exists=lambda _path: True,
        read_lines=lambda _path: 5,
    )
    assert citations == [{"n": 1, "path": "src/a.py", "start_line": 2, "end_line": 5, "qualname": "foo"}]
    assert invalid == [2]
    assert check == "1/2 通过"

    # 行号超过 EOF -> 无效
    citations, invalid, check = verify_code_citations(
        "见 [1]",
        contexts,
        file_exists=lambda _path: True,
        read_lines=lambda _path: 4,
    )
    assert citations == [] and invalid == [1] and check == "0/1 通过"

    # 非法行号区间 -> 无效
    bad_range = [{**contexts[0], "start_line": 6, "end_line": 5}]
    citations, invalid, _ = verify_code_citations(
        "[1]", bad_range, file_exists=lambda _path: True, read_lines=lambda _path: 10
    )
    assert citations == [] and invalid == [1]

    # 已解析的编号序列去重，编号为 0 的忽略
    citations, invalid, check = verify_code_citations(
        [1, 1, 0], contexts, file_exists=lambda _path: True, read_lines=lambda _path: 10
    )
    assert len(citations) == 1 and invalid == [] and check == "1/1 通过"


def test_verify_code_citations_uses_real_file_by_default(tmp_path: Path) -> None:
    target = tmp_path / "a.py"
    target.write_text("line1\nline2\nline3\n", encoding="utf-8")
    contexts = [
        {
            "path": "a.py",
            "abs_path": str(target),
            "start_line": 1,
            "end_line": 3,
            "qualname": "f",
        }
    ]
    citations, invalid, check = verify_code_citations("[1]", contexts)
    assert len(citations) == 1 and invalid == [] and check == "1/1 通过"

    missing = [{**contexts[0], "abs_path": str(tmp_path / "missing.py")}]
    assert verify_code_citations("[1]", missing) == ([], [1], "0/1 通过")
