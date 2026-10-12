"""The example templates, driven by a scripted model so they run without an API key."""

import importlib.util
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

from agentloom import FINAL_ANSWER, Approval
from agentloom.testing import ScriptedProvider, call

EXAMPLES = Path(__file__).resolve().parent.parent / "examples"


def load(name: str):
    sys.path.insert(0, str(EXAMPLES))  # the examples import their sibling _provider module
    try:
        spec = importlib.util.spec_from_file_location(f"example_{name}", EXAMPLES / f"{name}.py")
        module = importlib.util.module_from_spec(spec)
        sys.modules[spec.name] = module  # dataclasses resolve annotations through sys.modules
        spec.loader.exec_module(module)
        return module
    finally:
        sys.path.remove(str(EXAMPLES))


def results(provider: ScriptedProvider, request: int = -1):
    return provider.requests[request].messages[-1].tool_results


async def test_support_bot_scopes_orders_to_the_customer_and_gates_refunds():
    bot = load("06_support_bot")
    asked = []

    async def supervisor(tool_call, ctx):
        asked.append(tool_call.arguments["amount_cents"])
        return Approval(tool_call.arguments["amount_cents"] <= 2_000, "too large")

    provider = ScriptedProvider(
        [
            [call("list_orders"), call("order_status", order_id="A-2001")],
            call("refund", order_id="a-1001", amount_cents=1_500, reason="damaged"),
            call("refund", order_id="A-1002", amount_cents=12_900, reason="late"),
            call("refund", order_id="A-1001", amount_cents=0, reason="again"),
            "Done.",
        ]
    )
    agent = bot.build_agent(provider, approval=supervisor)

    result = await agent.run("Refund me.", deps=bot.Customer("c_42", "Ada"))

    assert result.ok and "helping Ada" in provider.requests[0].system
    listed, foreign = results(provider, 1)
    assert "A-1001" in listed.content and "A-2001" not in listed.content
    assert foreign.is_error and "No order" in foreign.content  # another customer's order
    assert results(provider, 2)[0].content == "Refunded $15.00 on A-1001."
    assert "not approved: too large" in results(provider, 3)[0].content
    assert "between 1 and 3000 cents" in results(provider, 4)[0].content  # approved, then refused
    assert asked == [1_500, 12_900, 0]


@pytest.mark.skipif(shutil.which("git") is None, reason="needs git")
async def test_code_review_reads_the_diff_and_returns_a_typed_review(tmp_path):
    reviewer = load("07_code_review")

    def git(*args: str) -> None:
        identity = ["-c", "user.name=t", "-c", "user.email=t@example.com"]
        subprocess.run(["git", *identity, *args], cwd=tmp_path, check=True, capture_output=True)

    git("init", "-q")
    (tmp_path / "app.py").write_text("def total(xs):\n    return sum(xs)\n", encoding="utf-8")
    git("add", ".")
    git("commit", "-q", "-m", "init")
    (tmp_path / "app.py").write_text("def total(xs):\n    return sum(xs) / 0\n", encoding="utf-8")

    finding = {"path": "app.py", "line": 2, "severity": "bug", "message": "Division by zero."}
    provider = ScriptedProvider(
        [
            call("changed_files"),
            [
                call("file_diff", path="app.py"),
                call("read_file", path="app.py"),
                call("read_file", path="../outside.txt"),
            ],
            call(FINAL_ANSWER, summary="Broken.", verdict="request_changes", findings=[finding]),
        ]
    )
    repo = reviewer.Repo(root=tmp_path.resolve(), base="HEAD")

    result = await reviewer.build_agent(provider).run("Review the changes.", deps=repo)

    assert results(provider, 1)[0].content == '["app.py"]'
    diff, source, escaped = results(provider, 2)
    assert "+    return sum(xs) / 0" in diff.content
    assert source.content == "1: def total(xs):\n2:     return sum(xs) / 0"
    assert escaped.is_error and "No file" in escaped.content
    review = result.data
    assert isinstance(review, reviewer.Review) and review.verdict == "request_changes"
    assert review.findings == [reviewer.Finding("app.py", 2, "bug", "Division by zero.")]


async def test_docs_qa_ranks_sections_and_serves_them_to_the_agent(tmp_path):
    qa = load("08_docs_qa")
    (tmp_path / "guide.md").write_text(
        "# Guide\n\nIntro.\n\n## Refunds\n\nRefunds take five days.\n\n"
        "## Shipping\n\nShipping is free over fifty dollars.\n\n## 退款\n\n退款需要五个工作日。\n",
        encoding="utf-8",
    )
    index = qa.DocIndex(tmp_path)

    assert index.search("how long do refunds take")[0].title == "Refunds"
    assert index.search("退款多久")[0].title == "退款"
    assert index.search("kubernetes") == []

    provider = ScriptedProvider(
        [
            call("search_docs", query="free shipping"),
            [call("read_section", section_id="guide.md#2"), call("read_section", section_id="x")],
            "Free over $50 [guide.md#2].",
        ]
    )
    result = await qa.build_agent(provider).run("Is shipping free?", deps=index)

    assert result.ok and '"section_id": "guide.md#2"' in results(provider, 1)[0].content
    section, missing = results(provider, 2)
    assert section.content == "# Shipping\n\nShipping is free over fifty dollars."
    assert missing.is_error


async def test_mcp_example_talks_to_its_server_over_stdio():
    pytest.importorskip("mcp.server.mcpserver")
    example = load("05_mcp")
    provider = ScriptedProvider(
        [
            call("stock_level", item="widget"),
            [call("reserve", item="widget", quantity=5), call("stock_level", item="gizmo")],
            "Reserved.",
        ]
    )

    async def approve(tool_call, ctx):
        return True

    async with example.inventory_tools() as tools:
        assert [(t.name, t.requires_approval) for t in tools] == [
            ("list_items", False),
            ("stock_level", False),
            ("reserve", True),
        ]
        agent = example.build_agent(provider, tools, approval=approve)
        agent.hooks.clear()
        await agent.run("Reserve 5 widgets.")

    assert results(provider, 1)[0].content == "12"
    reserved, unknown = results(provider, 2)
    assert reserved.content == "Reserved 5 x widget. 7 left."
    assert unknown.is_error and "stock_level" in unknown.content
