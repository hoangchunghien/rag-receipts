# rag-receipts

RAG answers you can check. Every sentence of an answer carries a quote copied from the
source it cites. The quote is verified by plain string matching, and a sentence whose quote
is not really there is cut. If nothing survives, the answer says the documents do not
answer the question.

```
$ receipts ask "What does PEP 618 add to the zip() built-in?"
PEP 618 adds an optional 'strict' boolean keyword parameter to the built-in 'zip'.

[1] Add Optional Length-Checking To zip (pep-0618#0)
    "This PEP proposes adding an optional 'strict' boolean keyword parameter to the built-in 'zip'."
```

`--html receipt.html` writes a page that shows each quote highlighted inside its source
passage ([example](docs/example-receipt.html)).

It runs locally by default: Ollama with a 3B model and a small embedding model, about
2.3 GB of weights, on a modest GPU or split between GPU and CPU. It can also use Claude or
any OpenAI-compatible server with your own key.

## Why

A RAG answer with a citation looks checkable, but a reader rarely opens the source, and a
model can cite a passage that does not say what the sentence claims. Asking a second model
"is this supported?" moves the trust problem rather than solving it.

This project takes the narrow, mechanical route. The model must show its working, and the
code checks the working without another model:

1. **The citation is real.** The sentence cites one of the passages it was actually shown.
2. **The quote is real.** The quote occurs in that passage, ignoring only whitespace, case,
   quote marks and markup. A quote shorter than four words does not count. If the quote is
   real but sits in a different passage than the one cited, the citation is corrected
   instead of the sentence being cut.
3. **The specifics are real.** Every number and every `code name` in the sentence also
   occurs in the cited passage (or in the question). This catches the most common way a
   fluent sentence drifts from its quote: a version, year, count or identifier that the
   source never mentions.

Claims that fail get one repair round: the model is told which quotes were not found and
may copy the exact words or drop the sentence. A repaired claim passes the same string
match as any other.

## Results

40 Python Enhancement Proposals (public domain), 22 questions they answer, and 14 "traps":
plausible Python questions the corpus does not answer, such as `typing.override` or the
`zoneinfo` module. Same model (`qwen2.5:3b` on Ollama), same retrieval, temperature 0.

| | plain RAG | checked |
|---|---|---|
| Answerable: answered | 86% (19/22) | 82% (18/22) |
| Answerable: answered with a citation you can follow | 9% (2/22) | 82% (18/22) |
| Answerable: ... and it points at the right PEP | 9% (2/22) | 68% (15/22) |
| Traps: answered anyway (lower is better) | 0% (0/14) | 0% (0/14) |
| Answers with a number or code name not in their sources | 5% (1/19) | 0 (check 3 cuts them) |

The plain baseline gets the same sources and the same "say NOT_FOUND if the sources do not
answer" instruction. It answers about as often, but a 3B model rarely writes the `[S1]`
markers it was asked for, so 17 of its 19 answers cannot be traced to anything. Checked mode
costs one answer and makes every remaining sentence traceable to an exact passage.

In checked mode the model proposed 22 claims: 17 passed first time, 1 more after the repair
round, and 2 kept claims had their citation moved to the passage that really holds the
quote. On one trap ("which PEP introduced asyncio?") the model answered "PEP 492"; check 3
cut it because the passage it cited never mentions 492.

Full per-question output: [`docs/results/`](docs/results/). Reproduce with
`receipts eval examples/questions.jsonl`.

## What it does not prove

A verified quote proves the words exist in the source. It does not prove the sentence
follows from them. In the run above, a sentence about `async with` cited a real quote from
the async generators PEP: every check passed and the sentence was still wrong. The receipt
makes that visible in one click, which is the point, but it is not a truth guarantee.
Retrieval can also miss the passage that answers the question, in which case the honest
output is "the documents I was given do not answer this".

## Quick start

```bash
# A local model server (or point OLLAMA_HOST at one you already run)
docker run -d --gpus all -p 127.0.0.1:11434:11434 -v ollama:/root/.ollama --name ollama ollama/ollama
docker exec ollama ollama pull qwen2.5:3b
docker exec ollama ollama pull nomic-embed-text

uv tool install git+https://github.com/hoangchunghien/rag-receipts   # or: pip install .
receipts ingest examples/peps                  # or your own .md / .rst / .txt / .pdf files
receipts ask "What does the walrus operator do?" --html receipt.html
receipts ask "..." --plain                     # the baseline, for comparison
receipts ask "..." --model anthropic/claude-sonnet-5   # needs ANTHROPIC_API_KEY
receipts eval examples/questions.jsonl         # checked vs plain on your question set
```

PDF support: `pip install 'rag-receipts[pdf]'`.

## How it is built

- `documents.py`: loading, and paragraph-packed chunks that keep their character offsets,
  so a quote can be highlighted in the original text.
- `index.py`: one SQLite file. FTS5 (BM25) for exact names like `TypeVarTuple`, embeddings
  for paraphrases, merged with reciprocal rank fusion.
- `answer.py`: the claim format, the repair round, and the plain baseline.
- `verify.py`: the three checks. No model involved.
- `evaluate.py`: the comparison above.

## Related work

[verbatim-rag](https://github.com/KRLabsOrg/verbatim-rag) takes the extractive route: it
returns source spans instead of generated sentences, with a trained span extractor. This
project keeps generated answers and checks them afterwards, which reads more naturally and
gives weaker guarantees. Both are reasonable; pick by how much free text you can accept.

## License

Apache-2.0. The example PEPs are in the public domain or CC0-1.0, as each states.
