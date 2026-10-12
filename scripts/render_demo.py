"""Render the offline demo's real output as docs/assets/demo.svg for the README.

python scripts/render_demo.py
"""

import html
import re
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
COMMAND = "python examples/03_offline_demo.py"
COLORS = [
    (r"^  -> ", "#7dcfff"),  # tool call
    (r"^  ok ", "#9ece6a"),  # tool result
    (r"^  !! ", "#f7768e"),  # error result sent back to the model
    (r"^(> |= |  step )", "#787c99"),  # run and step markers
]
LINE_HEIGHT, CHAR_WIDTH, PAD, TITLE_BAR = 20, 8.4, 18, 34


def main() -> None:
    done = subprocess.run(
        [sys.executable, "examples/03_offline_demo.py"],
        cwd=ROOT,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        encoding="utf-8",
        check=True,
    )
    # Run ids and timings change on every run; keep the picture stable.
    output = re.sub(r"> run [0-9a-f]+", "> run 4d94d17f", done.stdout)
    lines = [f"$ {COMMAND}", *output.rstrip().splitlines()]

    width = int(max(len(line) for line in lines) * CHAR_WIDTH + 2 * PAD)
    height = len(lines) * LINE_HEIGHT + TITLE_BAR + 2 * PAD - 6
    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {width} {height}" '
        f'width="{width}" height="{height}" role="img" '
        f'aria-label="Output of {COMMAND}">',
        f'<rect width="{width}" height="{height}" rx="10" fill="#1a1b26"/>',
        *(
            f'<circle cx="{22 + 20 * i}" cy="17" r="6" fill="{color}"/>'
            for i, color in enumerate(("#f7768e", "#e0af68", "#9ece6a"))
        ),
        '<g font-family="ui-monospace,SFMono-Regular,Menlo,Consolas,monospace" '
        'font-size="14" xml:space="preserve">',
    ]
    for index, line in enumerate(lines):
        color = "#c0caf5" if index else "#e0af68"
        color = next((c for pattern, c in COLORS if re.match(pattern, line)), color)
        y = TITLE_BAR + PAD + index * LINE_HEIGHT
        parts.append(f'<text x="{PAD}" y="{y}" fill="{color}">{html.escape(line)}</text>')
    parts += ["</g>", "</svg>"]

    target = ROOT / "docs" / "assets" / "demo.svg"
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text("\n".join(parts) + "\n", encoding="utf-8", newline="\n")
    print(f"wrote {target.relative_to(ROOT)} ({width}x{height})")


if __name__ == "__main__":
    main()
