"""Show an answer with its receipts: in the terminal, or as HTML that highlights each quote."""

from __future__ import annotations

import html

from .answer import Answer
from .verify import fold


def to_terminal(answer: Answer) -> str:
    lines = [answer.text, ""]
    if answer.mode != "checked":
        return "\n".join(lines).rstrip()
    sources = {f"S{i + 1}": hit for i, hit in enumerate(answer.hits)}
    for n, claim in enumerate(answer.kept, 1):
        hit = sources[claim.source]
        lines.append(f"[{n}] {hit.title} ({hit.chunk.chunk_id})")
        lines.append(f'    "{claim.quote.strip()}"')
    if answer.set_aside:
        names = ", ".join(answer.off_topic)
        lines.append(f"Set aside {len(answer.set_aside)} sentence(s): their quotes are real, but")
        lines.append(f"none of the cited passages mention {names}.")
        for claim in answer.set_aside:
            lines.append(f"  - {claim.sentence.strip()}")
    if answer.cut:
        lines.append("")
        lines.append(f"Cut {len(answer.cut)} sentence(s) that failed the checks:")
        for claim in answer.cut:
            lines.append(f"  - {claim.sentence.strip()}")
            lines.append(f"    ({'; '.join(claim.problems)})")
    return "\n".join(lines).rstrip()


def to_html(answer: Answer) -> str:
    sources = {f"S{i + 1}": hit for i, hit in enumerate(answer.hits)}
    kept_rows = []
    for n, claim in enumerate(answer.kept, 1):
        hit = sources[claim.source]
        kept_rows.append(
            f'<li><p class="claim">{html.escape(claim.sentence)} <sup>[{n}]</sup></p>'
            f'<p class="src">{html.escape(hit.title)} · <code>{html.escape(hit.chunk.chunk_id)}'
            f"</code></p><blockquote>{_highlight(hit.chunk.text, claim.quote)}</blockquote></li>"
        )
    aside_rows = [
        f'<li><p class="claim cut">{html.escape(c.sentence)}</p></li>' for c in answer.set_aside
    ]
    cut_rows = [
        f'<li><p class="claim cut">{html.escape(c.sentence)}</p>'
        f'<p class="why">{html.escape("; ".join(c.problems))}</p></li>'
        for c in answer.cut
    ]
    body = (
        f"<h1>{html.escape(answer.question)}</h1>"
        f'<p class="answer">{html.escape(answer.text)}</p>'
        f'<p class="meta">{html.escape(answer.model)} · {len(answer.kept)} kept · '
        f"{len(answer.cut)} cut</p>"
        + (f"<h2>Receipts</h2><ol>{''.join(kept_rows)}</ol>" if kept_rows else "")
        + (
            f"<h2>Set aside: no cited passage mentions {html.escape(', '.join(answer.off_topic))}"
            f"</h2><ul>{''.join(aside_rows)}</ul>"
            if aside_rows
            else ""
        )
        + (f"<h2>Cut by the checks</h2><ul>{''.join(cut_rows)}</ul>" if cut_rows else "")
    )
    return PAGE.replace("{{title}}", html.escape(answer.question)).replace("{{body}}", body)


def _highlight(source: str, quote: str) -> str:
    """Escape `source` and wrap the stretch that matches `quote` in <mark>."""
    target = fold(quote)
    # Map each folded character back to its position in the original text.
    folded_chars: list[str] = []
    positions: list[int] = []
    for i, ch in enumerate(source):
        piece = fold(ch) if not ch.isspace() else " "
        for p in piece:
            if p == " " and folded_chars and folded_chars[-1] == " ":
                continue
            folded_chars.append(p)
            positions.append(i)
    folded = "".join(folded_chars)
    at = folded.find(target)
    if at < 0 or not target:
        return html.escape(source)
    start, end = positions[at], positions[at + len(target) - 1] + 1
    return (
        html.escape(source[:start])
        + f"<mark>{html.escape(source[start:end])}</mark>"
        + html.escape(source[end:])
    )


PAGE = """<!doctype html>
<html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>{{title}}</title>
<style>
:root { --bg:#fff; --fg:#1b1f24; --muted:#5b6470; --mark:#fff1a8; --line:#e3e6ea;
        --cut:#b42318; }
@media (prefers-color-scheme: dark) {
  :root { --bg:#111418; --fg:#e6e9ed; --muted:#9aa4b0; --mark:#5c4b00; --line:#2a3038;
          --cut:#f97066; }
}
body { background:var(--bg); color:var(--fg); font:16px/1.55 system-ui, sans-serif;
       max-width:46rem; margin:2rem auto; padding:0 1rem; }
h1 { font-size:1.35rem; } h2 { font-size:1.05rem; margin-top:2rem; }
.answer { font-size:1.1rem; } .meta, .src, .why { color:var(--muted); font-size:.9rem; }
blockquote { white-space:pre-wrap; border-left:3px solid var(--line); margin:.4rem 0 1.2rem;
             padding:.2rem .8rem; font-size:.9rem; max-height:14rem; overflow:auto; }
mark { background:var(--mark); color:inherit; }
.cut { color:var(--cut); text-decoration:line-through; }
li { margin-bottom:.6rem; }
</style></head><body>{{body}}</body></html>
"""
