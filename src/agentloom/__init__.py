"""agentloom: a small, dependable framework for building tool-using LLM agents."""

from .agent import Agent, Approval, ApprovalHandler, RunResult, RunStatus
from .context import ContextStrategy, ContextView, KeepAll, SlidingWindow, Summarizing
from .errors import (
    AgentLoomError,
    ConfigurationError,
    ProviderError,
    RunFailed,
    SessionError,
    ToolDenied,
    ToolError,
)
from .events import (
    Event,
    ModelResponded,
    RetryScheduled,
    RunFinished,
    RunStarted,
    StepStarted,
    TextDelta,
    ThinkingDelta,
    ToolCallFinished,
    ToolCallStarted,
)
from .hooks import Hooks
from .policies import RetryPolicy, RunLimits
from .providers.base import Completed, Provider, ProviderChunk, TextChunk, ThinkingChunk
from .session import FileSession, InMemorySession, Session
from .tools import Tool, ToolContext, Toolset, tool
from .tracing import ConsoleTracer, JsonlTracer
from .types import (
    Message,
    ModelRequest,
    ModelResponse,
    StopReason,
    TextBlock,
    ToolCall,
    ToolResult,
    ToolSpec,
    Usage,
)

__version__ = "0.1.0"

__all__ = [
    "Agent",
    "AgentLoomError",
    "Approval",
    "ApprovalHandler",
    "Completed",
    "ConfigurationError",
    "ConsoleTracer",
    "ContextStrategy",
    "ContextView",
    "Event",
    "FileSession",
    "Hooks",
    "InMemorySession",
    "JsonlTracer",
    "KeepAll",
    "Message",
    "ModelRequest",
    "ModelResponded",
    "ModelResponse",
    "Provider",
    "ProviderChunk",
    "ProviderError",
    "RetryPolicy",
    "RetryScheduled",
    "RunFailed",
    "RunFinished",
    "RunLimits",
    "RunResult",
    "RunStarted",
    "RunStatus",
    "Session",
    "SessionError",
    "SlidingWindow",
    "StepStarted",
    "StopReason",
    "Summarizing",
    "TextBlock",
    "TextChunk",
    "TextDelta",
    "ThinkingChunk",
    "ThinkingDelta",
    "Tool",
    "ToolCall",
    "ToolCallFinished",
    "ToolCallStarted",
    "ToolContext",
    "ToolDenied",
    "ToolError",
    "ToolResult",
    "ToolSpec",
    "Toolset",
    "Usage",
    "tool",
]
