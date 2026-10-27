from rag_receipts.verify import Claim, check_claim, fold, unsupported_specifics

SOURCE = (
    "This PEP proposes adding an optional ``strict`` boolean keyword\n"
    "parameter to the built-in ``zip``. When enabled, a ``ValueError`` is raised\n"
    "if one of the arguments is exhausted before the others. Python-Version: 3.10"
)


def check(sentence: str, quote: str, source: str = "S1", question: str = "") -> Claim:
    sources = {"S1": SOURCE, "S2": "Other text."}
    return check_claim(Claim(sentence, source, quote), sources, question)


def test_a_faithful_quote_passes_despite_markup_case_and_line_breaks() -> None:
    claim = check(
        "zip() gains a strict flag.",
        "adding an optional 'strict' boolean keyword parameter to the built-in zip",
    )
    assert claim.verified


def test_a_paraphrased_quote_is_cut() -> None:
    claim = check("zip() gains a strict flag.", "adds a strict option to the zip builtin function")
    assert claim.problems == ["quote does not occur in the cited source"]


def test_a_quote_that_is_too_short_is_cut() -> None:
    assert check("zip() gains a strict flag.", "the built-in").problems == [
        "quote is shorter than 4 words"
    ]


def test_citing_a_source_it_was_not_shown_is_cut() -> None:
    claim = check("zip() gains a strict flag.", "boolean keyword parameter to", source="S9")
    assert claim.problems == ["cites 'S9', which it was not shown"]


def test_a_number_the_source_never_mentions_is_cut() -> None:
    claim = check("This arrived in Python 3.9.", "boolean keyword parameter to the built-in zip")
    assert claim.problems == ["number 3.9 is not in the cited source"]


def test_a_number_from_the_question_is_not_new_information() -> None:
    claim = check(
        "PEP 618 adds a strict flag to zip.",
        "boolean keyword parameter to the built-in zip",
        question="What does PEP 618 add to zip()?",
    )
    assert claim.verified


def test_a_code_name_the_source_never_mentions_is_cut() -> None:
    claim = check("It raises `TypeError`.", "boolean keyword parameter to the built-in zip")
    assert claim.problems == ["`TypeError` is not in the cited source"]


def test_a_real_quote_cited_to_the_wrong_source_is_moved_not_cut() -> None:
    claim = check_claim(
        Claim("zip() gains a strict flag.", "S2", "boolean keyword parameter to the built-in zip"),
        {"S1": "Unrelated.", "S2": "Also unrelated.", "S3": SOURCE},
    )
    assert claim.verified
    assert (claim.reattributed_from, claim.source) == ("S2", "S3")


def test_fold_ignores_cosmetic_differences_only() -> None:
    curly_apostrophe = "\u2019"
    assert fold(f"It{curly_apostrophe}s  ``strict``\n") == fold("its strict")
    assert fold("strict") != fold("strictly")


def test_unsupported_specifics_flags_numbers_and_code_names() -> None:
    missing = unsupported_specifics("Added in 3.9, raises `TypeError`.", [SOURCE])
    assert missing == ["3.9", "TypeError"]


def test_unsupported_specifics_allows_question_numbers_and_skips_code_fences() -> None:
    answer = "PEP 8 says so.\n```\ndef f[T](x: T) -> T: ...\n```"
    assert unsupported_specifics(answer, [SOURCE], question="What does PEP 8 say?") == []
