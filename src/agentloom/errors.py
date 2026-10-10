"""Exception hierarchy.

Everything the framework raises derives from :class:`AgentLoomError`, so callers
can catch one type at the boundary. Expected terminal conditions of a run
(step limit, refusal, truncation) are *not* exceptions; they are reported on
``RunResult.status``.
"""

from __future__ import annotations


class AgentLoomError(Exception):
    """Base class for all framework errors."""


class ConfigurationError(AgentLoomError):
    """The agent, a tool, or a provider was set up incorrectly."""


class ProviderError(AgentLoomError):
    """A model call failed.

    Providers translate their SDK's exceptions into this type so the agent
    loop can decide whether to retry without knowing the SDK.
    """

    def __init__(
        self,
        message: str,
        *,
        retryable: bool = False,
        status_code: int | None = None,
        retry_after: float | None = None,
        request_id: str | None = None,
    ) -> None:
        super().__init__(message)
        self.retryable = retryable
        self.status_code = status_code
        self.retry_after = retry_after
        self.request_id = request_id


class ToolError(AgentLoomError):
    """Raise inside a tool to report an expected failure to the model.

    The message is sent back as an error result so the model can correct
    itself; write it for the model ("date must be YYYY-MM-DD").
    """


class ToolDenied(AgentLoomError):
    """Raise from ``Hooks.before_tool`` to block a tool call."""


class SessionError(AgentLoomError):
    """A session could not be loaded or saved."""


class RunFailed(AgentLoomError):
    """Raised by ``RunResult.raise_for_status`` when a run did not complete."""
