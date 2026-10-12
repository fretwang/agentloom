"""Template: a code-review agent that returns a typed result.

It reads the diff of the current git repository against a base ref and returns
a `Review` object (not prose), so the result can be posted as comments, gate a
merge, or be stored.

    python examples/07_code_review.py            # uncommitted changes
    python examples/07_code_review.py main       # everything since main
"""

import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

from _provider import provider_from_env

from agentloom import Agent, ConsoleTracer, Provider, RunLimits, ToolContext, ToolError, tool

MAX_FILE_LINES = 400


@dataclass
class Finding:
    path: str
    line: int
    severity: Literal["bug", "risk", "nit"]
    message: str


@dataclass
class Review:
    summary: str
    verdict: Literal["approve", "request_changes"]
    findings: list[Finding]


@dataclass
class Repo:
    root: Path
    base: str

    def git(self, *args: str) -> str:
        done = subprocess.run(
            ["git", *args], cwd=self.root, capture_output=True, text=True, encoding="utf-8"
        )
        if done.returncode != 0:
            raise ToolError(done.stderr.strip() or "git failed")
        return done.stdout

    def resolve(self, path: str) -> Path:
        # The path comes from the model: keep it inside the repository.
        target = (self.root / path).resolve()
        if not target.is_relative_to(self.root) or not target.is_file():
            raise ToolError(f"No file {path!r} in this repository.")
        return target


@tool
def changed_files(ctx: ToolContext) -> list[str]:
    """List the files that differ from the base ref."""
    return ctx.deps.git("diff", "--name-only", ctx.deps.base, "--").splitlines()


@tool
def file_diff(path: str, ctx: ToolContext) -> str:
    """Show the diff of one changed file against the base ref.

    Args:
        path: Repository-relative path, as returned by changed_files.
    """
    return ctx.deps.git("diff", ctx.deps.base, "--", path) or "No changes in this file."


@tool
def read_file(path: str, ctx: ToolContext, start_line: int = 1) -> str:
    """Read a file from the working tree, with line numbers, to see context around a change.

    Args:
        path: Repository-relative path.
        start_line: First line to return. At most 400 lines are returned per call.
    """
    lines = ctx.deps.resolve(path).read_text(encoding="utf-8", errors="replace").splitlines()
    chunk = lines[start_line - 1 : start_line - 1 + MAX_FILE_LINES]
    return "\n".join(f"{start_line + i}: {line}" for i, line in enumerate(chunk))


def build_agent(provider: Provider, hooks=()) -> Agent:
    return Agent(
        provider,
        name="reviewer",
        instructions=(
            "You review code changes. List the changed files, read each diff, and read "
            "surrounding code when a change cannot be judged from the diff alone. Report "
            "only problems you can point to on a specific line; do not pad the review. "
            "Request changes only for bugs or real risks."
        ),
        tools=[changed_files, file_diff, read_file],
        output_type=Review,
        limits=RunLimits(max_steps=30),
        hooks=hooks,
    )


def main() -> None:
    root = Path(
        subprocess.run(
            ["git", "rev-parse", "--show-toplevel"], capture_output=True, text=True, check=True
        ).stdout.strip()
    )
    repo = Repo(root=root.resolve(), base=sys.argv[1] if len(sys.argv) > 1 else "HEAD")
    agent = build_agent(provider_from_env(), hooks=[ConsoleTracer()])

    result = agent.run_sync("Review the changes.", deps=repo).raise_for_status()

    review: Review = result.data
    print(f"\n{review.verdict.upper()}: {review.summary}")
    for finding in review.findings:
        print(f"  [{finding.severity}] {finding.path}:{finding.line}  {finding.message}")


if __name__ == "__main__":
    main()
