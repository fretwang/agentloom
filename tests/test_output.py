"""Structured final output through the ``final_answer`` tool."""

import enum
from dataclasses import dataclass

import pytest
from pydantic import BaseModel, Field

from agentloom import (
    FINAL_ANSWER,
    Agent,
    ConfigurationError,
    InMemorySession,
    Message,
    ModelResponse,
    RunLimits,
    RunStatus,
    StopReason,
    ToolCallFinished,
    tool,
)
from agentloom.testing import ScriptedProvider, call


@dataclass
class Verdict:
    approved: bool
    reasons: list[str]
    score: float = 0.0


class Ticket(BaseModel):
    title: str
    priority: int = Field(ge=1, le=5)


class Mood(enum.Enum):
    GOOD = "good"
    BAD = "bad"


def answer(**arguments):
    return call(FINAL_ANSWER, **arguments)


def last_results(provider: ScriptedProvider):
    return provider.requests[-1].messages[-1].tool_results


async def test_dataclass_answer_is_validated_and_converted():
    provider = ScriptedProvider([answer(approved=True, reasons=["tests pass"])])
    session = InMemorySession()

    result = await Agent(provider, output_type=Verdict).run("Review this.", session=session)

    assert result.ok and result.steps == 1
    assert result.data == Verdict(approved=True, reasons=["tests pass"])
    assert result.output == '{"approved": true, "reasons": ["tests pass"]}'
    # The answer tool is offered to the model with the type's schema.
    spec = provider.requests[0].tools[-1]
    assert spec.name == FINAL_ANSWER
    assert spec.input_schema["required"] == ["approved", "reasons"]
    # The stored history stays sendable: the answer call has a result.
    stored = await session.load()
    assert stored[-1].tool_results[0].content == "Answer accepted."


async def test_invalid_answer_goes_back_to_the_model():
    provider = ScriptedProvider(
        [
            answer(approved="yes", reasons=[]),
            answer(approved=False, reasons=["no tests"], score=0.2),
        ]
    )

    result = await Agent(provider, output_type=Verdict).run("Review this.")

    assert result.data == Verdict(False, ["no tests"], 0.2) and result.steps == 2
    rejected = last_results(provider)[0]
    assert rejected.is_error and "$.approved: expected boolean" in rejected.content


async def test_pydantic_constraints_are_enforced():
    provider = ScriptedProvider(
        [answer(title="Crash", priority=9), answer(title="Crash", priority=2)]
    )

    result = await Agent(provider, output_type=Ticket).run("File it.")

    assert result.data == Ticket(title="Crash", priority=2)
    assert "less than or equal to 5" in last_results(provider)[0].content


@pytest.mark.parametrize(
    "output_type, value, expected",
    [
        (list[str], ["a", "b"], ["a", "b"]),
        (int, 42, 42),
        (Mood, "good", Mood.GOOD),
        ({"type": "array", "items": {"type": "integer"}}, [1, 2], [1, 2]),
    ],
)
async def test_non_object_answers_are_wrapped_in_value(output_type, value, expected):
    provider = ScriptedProvider([answer(value=value)])

    result = await Agent(provider, output_type=output_type).run("go")

    assert result.data == expected
    assert provider.requests[0].tools[0].input_schema["required"] == ["value"]


async def test_json_schema_answer_is_returned_as_a_dict():
    schema = {
        "type": "object",
        "properties": {"city": {"type": "string"}},
        "required": ["city"],
        "additionalProperties": False,
    }
    provider = ScriptedProvider([answer(city="Oslo")])

    result = await Agent(provider, output_type=schema).run("Where?")

    assert result.data == {"city": "Oslo"}


async def test_plain_text_reply_is_reminded_to_use_the_tool():
    provider = ScriptedProvider(["Looks fine to me.", answer(approved=True, reasons=[])])

    result = await Agent(provider, output_type=Verdict).run("Review this.")

    assert result.data == Verdict(True, []) and result.steps == 2
    assert FINAL_ANSWER in provider.requests[1].messages[-1].text


async def test_text_only_model_stops_at_the_step_limit():
    provider = ScriptedProvider(["no", "still no"])
    agent = Agent(provider, output_type=Verdict, limits=RunLimits(max_steps=2))

    result = await agent.run("Review this.")

    assert result.status is RunStatus.MAX_STEPS and result.data is None


async def test_answer_alongside_other_calls_is_deferred():
    ran = []

    @tool
    def lookup(key: str) -> str:
        """Look something up."""
        ran.append(key)
        return "tests pass"

    provider = ScriptedProvider(
        [
            [call("lookup", key="ci"), answer(approved=True, reasons=["guess"])],
            answer(approved=True, reasons=["tests pass"]),
        ]
    )

    events = [e async for e in Agent(provider, tools=[lookup], output_type=Verdict).stream("go")]

    result = events[-1].result
    assert ran == ["ci"] and result.steps == 2
    assert result.data == Verdict(True, ["tests pass"])
    first = provider.requests[1].messages[-1].tool_results
    assert [r.is_error for r in first] == [False, True]  # results stay in call order
    assert "on its own" in first[1].content
    finished = [e for e in events if isinstance(e, ToolCallFinished)]
    assert [e.call.name for e in finished] == ["lookup", FINAL_ANSWER, FINAL_ANSWER]


async def test_truncated_answer_is_not_accepted():
    truncated = ModelResponse(
        Message("assistant", [answer(approved=True, reasons=["cut"])]), StopReason.MAX_TOKENS
    )
    provider = ScriptedProvider([truncated, answer(approved=True, reasons=["whole"])])

    result = await Agent(provider, output_type=Verdict).run("go")

    assert result.data == Verdict(True, ["whole"])
    assert "output limit" in last_results(provider)[0].content


async def test_sub_agent_with_output_type_returns_json_to_the_caller():
    grader = Agent(
        ScriptedProvider([answer(approved=False, reasons=["typo"])]),
        name="grader",
        output_type=Verdict,
    )
    lead = ScriptedProvider([call("grader", task="Grade it."), "Rejected."])

    await Agent(lead, tools=[grader.as_tool(description="Grades work.")]).run("go")

    assert last_results(lead)[0].content == '{"approved": false, "reasons": ["typo"]}'


def test_configuration_errors():
    @tool(name=FINAL_ANSWER)
    def clash() -> str:
        """Clashes with the answer tool."""
        return ""

    with pytest.raises(ConfigurationError, match="cannot be combined"):
        Agent(ScriptedProvider([]), tools=[clash], output_type=Verdict)
    with pytest.raises(ConfigurationError, match="output_type: unsupported type"):
        Agent(ScriptedProvider([]), output_type=complex)


async def test_without_output_type_nothing_changes():
    provider = ScriptedProvider(["hi"])

    result = await Agent(provider).run("hello")

    assert result.data is None and result.output == "hi"
    assert provider.requests[0].tools == []


@dataclass
class Line:
    sku: str
    quantity: int


@dataclass
class Order:
    lines: list[Line]
    mood: Mood


async def test_nested_dataclasses_and_enums_are_converted():
    provider = ScriptedProvider([answer(lines=[{"sku": "w", "quantity": 2}], mood="bad")])

    result = await Agent(provider, output_type=Order).run("go")

    assert result.data == Order(lines=[Line("w", 2)], mood=Mood.BAD)
