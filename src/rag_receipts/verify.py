"""Deterministic checks on a model's claims. No second model is asked whether the first was right.

A claim is one answer sentence plus the source it cites and a quote the model says it copied
from that source. A claim is kept only if:

1. it cites one of the sources it was actually shown,
2. the quote really occurs in that source (after folding whitespace, case, typographic
   punctuation and markup), and is long enough to mean something,
3. every number and every `code name` in the sentence also occurs in the cited source.

Check 2 proves provenance: the words exist. It does not prove the sentence follows from them.
Check 3 catches the commonest way a fluent sentence drifts from its quote: a version, year,
count or identifier that the source never mentions.
"""

from __future__ import annotations

import re
import unicodedata
from dataclasses import dataclass, field

MIN_QUOTE_WORDS = 4

_FOLD = str.maketrans(
    {
        "\u2013": "-",
        "\u2014": "-",
        "\u00a0": " ",
        # Markup and quote marks: ``strict``, 'strict' and strict are the same words.
        "`": None,
        "*": None,
        "_": None,
        "'": None,
        '"': None,
        "\u2018": None,
        "\u2019": None,
        "\u201c": None,
        "\u201d": None,
    }
)
_NUMBER = re.compile(r"(?<![\w.])\d+(?:\.\d+)*(?![\w])")
_CODE = re.compile(r"`([^`]+)`")


def fold(text: str) -> str:
    """Normalise text so a faithful copy matches despite cosmetic differences."""
    text = unicodedata.normalize("NFKC", text).translate(_FOLD).lower()
    return re.sub(r"\s+", " ", text).strip()


@dataclass
class Claim:
    sentence: str
    source: str
    quote: str
    problems: list[str] = field(default_factory=list)
    reattributed_from: str | None = None
    repaired: bool = False

    @property
    def verified(self) -> bool:
        return not self.problems


def check_claim(claim: Claim, sources: dict[str, str], question: str = "") -> Claim:
    """Fill `claim.problems`; an empty list means the claim passed every check.

    If the quote is not in the cited source but is in another source the model was shown,
    the citation is corrected rather than the claim cut: the words still exist, and the
    reader is pointed at where they really are. Numbers that the question itself contains
    (such as a PEP number) are not new information, so check 3 ignores them.
    """
    claim.problems = []
    if claim.source not in sources:
        claim.problems.append(f"cites {claim.source!r}, which it was not shown")
        return claim
    folded_quote = fold(claim.quote)
    if len(folded_quote.split()) < MIN_QUOTE_WORDS:
        claim.problems.append(f"quote is shorter than {MIN_QUOTE_WORDS} words")
    elif folded_quote not in fold(sources[claim.source]):
        found_in = [sid for sid, text in sources.items() if folded_quote in fold(text)]
        if found_in:
            claim.reattributed_from, claim.source = claim.source, found_in[0]
        else:
            claim.problems.append("quote does not occur in the cited source")
    folded_source = fold(sources[claim.source])
    allowed = set(_NUMBER.findall(fold(question))) | set(_NUMBER.findall(folded_source))
    for number in _NUMBER.findall(claim.sentence):
        if number not in allowed:
            claim.problems.append(f"number {number} is not in the cited source")
    for code in _CODE.findall(claim.sentence):
        if fold(code) not in folded_source:
            claim.problems.append(f"`{code}` is not in the cited source")
    return claim


def unsupported_specifics(sentence: str, cited_texts: list[str]) -> list[str]:
    """Numbers and code names in `sentence` that none of the cited texts contain.

    Used to score answers that carry no quotes (the naive baseline) by the same rule as check 3.
    """
    folded = [fold(t) for t in cited_texts]
    numbers_in_sources = {n for t in folded for n in _NUMBER.findall(t)}
    missing = [n for n in _NUMBER.findall(sentence) if n not in numbers_in_sources]
    missing += [c for c in _CODE.findall(sentence) if not any(fold(c) in t for t in folded)]
    return missing
