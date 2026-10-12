"""Template: question answering over your own documents (RAG), with no extra dependencies.

Markdown files are split into sections and ranked with BM25. The agent searches,
reads the sections it needs, and answers with citations. Swap `DocIndex.search`
for your vector store when keyword ranking is not enough.

    python examples/08_docs_qa.py                      # asks about this project's docs
    python examples/08_docs_qa.py path/to/your/docs
"""

import math
import re
import sys
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path

from _provider import provider_from_env

from agentloom import Agent, ConsoleTracer, Provider, RunLimits, ToolContext, ToolError, tool

# Latin words, and CJK characters one by one.
_TOKEN = re.compile(r"[a-z0-9_]+|[一-鿿]")


def tokens(text: str) -> list[str]:
    return _TOKEN.findall(text.lower())


@dataclass
class Section:
    id: str
    title: str
    text: str
    counts: Counter = field(default_factory=Counter)


class DocIndex:
    """Sections of every Markdown file under a directory, searchable with BM25."""

    def __init__(self, root: Path) -> None:
        self.sections: dict[str, Section] = {}
        for path in sorted(root.rglob("*.md")):
            name = path.relative_to(root).as_posix()
            title, body = name, []
            parts: list[tuple[str, str]] = []
            for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
                if line.startswith("#"):
                    parts.append((title, "\n".join(body)))
                    title, body = line.lstrip("# ").strip(), []
                else:
                    body.append(line)
            parts.append((title, "\n".join(body)))
            for number, (heading, text) in enumerate(p for p in parts if p[1].strip()):
                section = Section(f"{name}#{number}", heading, text.strip())
                section.counts = Counter(tokens(f"{heading} {text}"))
                self.sections[section.id] = section
        lengths = [sum(s.counts.values()) for s in self.sections.values()]
        self.average_length = sum(lengths) / len(lengths) if lengths else 1.0
        self.document_frequency = Counter(t for s in self.sections.values() for t in s.counts)

    def search(self, query: str, top_k: int = 5) -> list[Section]:
        total = len(self.sections)
        scores: dict[str, float] = {}
        for term in set(tokens(query)):
            frequency = self.document_frequency.get(term)
            if not frequency:
                continue
            idf = math.log(1 + (total - frequency + 0.5) / (frequency + 0.5))
            for section in self.sections.values():
                count = section.counts.get(term)
                if count:
                    length = sum(section.counts.values()) / self.average_length
                    weight = count * 2.5 / (count + 1.5 * (0.25 + 0.75 * length))
                    scores[section.id] = scores.get(section.id, 0.0) + idf * weight
        ranked = sorted(scores, key=lambda key: scores[key], reverse=True)
        return [self.sections[key] for key in ranked[:top_k]]


@tool
def search_docs(query: str, ctx: ToolContext) -> list[dict]:
    """Search the documentation. Returns the best-matching sections with a short preview.

    Args:
        query: Keywords to look for. Use the terms the documents would use.
    """
    return [
        {"section_id": s.id, "title": s.title, "preview": s.text[:200]}
        for s in ctx.deps.search(query)
    ]


@tool
def read_section(section_id: str, ctx: ToolContext) -> str:
    """Read one section in full.

    Args:
        section_id: An id returned by search_docs, for example "guide.md#3".
    """
    section = ctx.deps.sections.get(section_id)
    if section is None:
        raise ToolError(f"No section {section_id!r}. Use an id returned by search_docs.")
    return f"# {section.title}\n\n{section.text}"


def build_agent(provider: Provider, hooks=()) -> Agent:
    return Agent(
        provider,
        name="docs",
        instructions=(
            "Answer questions using only the documentation you can search. Search first, "
            "read the sections that look relevant, then answer and cite the section ids "
            "you used in square brackets. If the documentation does not cover the "
            "question, say so instead of guessing."
        ),
        tools=[search_docs, read_section],
        limits=RunLimits(max_steps=10),
        hooks=hooks,
    )


def main() -> None:
    default = Path(__file__).resolve().parent.parent / "docs"
    index = DocIndex(Path(sys.argv[1]) if len(sys.argv) > 1 else default)
    print(f"Indexed {len(index.sections)} sections. Empty line to quit.")
    agent = build_agent(provider_from_env(), hooks=[ConsoleTracer()])
    while question := input("\nquestion> ").strip():
        print(agent.run_sync(question, deps=index).output)


if __name__ == "__main__":
    main()
