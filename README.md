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
`zoneinfo` module. Plain RAG and checked mode get the same retrieved passages and the same
instruction to say so when the passages do not answer. Temperature 0, local models on Ollama.

| | 3B, nomic embeddings | 7B, nomic embeddings | 3B, Qwen3-Embedding-8B |
|---|---|---|---|
| Answered (of 22): plain / checked | 19 / 18 | 22 / 22 | 21 / 18 |
| ... with a citation you can follow | 2 / 18 | 16 / 22 | 2 / 18 |
| ... that points at the right PEP | 2 / 15 | 14 / 19 | 2 / 17 |
| Traps answered anyway (of 14): plain / checked | 0 / 0 | 0 / 0 | 0 / 2 |
| Plain answers citing a passage that lacks one of their numbers or code names | 1 of 19 | 2 of 22 | 0 of 21 |

Chat models: `qwen2.5:3b` and `qwen2.5:7b`. Embeddings: `nomic-embed-text` on Ollama, or
`Qwen3-Embedding-8B` on llama.cpp's `llama-server` (any `/v1/embeddings` server works).

What the numbers say:

- **Checked answers are traceable; plain ones often are not.** A 3B model rarely writes
  the `[S1]` markers it is asked for, so 17 of its 19 plain answers point at nothing. The
  7B model cites more often, but its citations are right less often than checked mode's
  (14 vs 19 of 22).
- **The checks cost little with a stronger model.** The 7B model answered all 22 in both
  modes; checked mode added a verified quote to every sentence.
- **Re-attribution matters.** Across the runs, 8 kept claims quoted real text but cited
  the wrong passage. Instead of being cut, they were pointed at the passage that holds
  the quote.
- **The specifics check catches real slips.** Asked which PEP introduced asyncio (it is
  not in the corpus), the 3B model answered "PEP 492"; check 3 cut it because the passage
  it cited never mentions 492.

Per-question output and summaries: [`docs/results/`](docs/results/). Reproduce with
`receipts eval examples/questions.jsonl`.

## What it does not prove

A verified quote proves the words exist in the source. It does not prove the sentence
follows from them. The third run shows this plainly. With better retrieval, the 3B model
answered two traps with real quotes and wrong sentences:

- "How do you mark a TypedDict item as ReadOnly?" was answered with the PEP 591 passage on
  `@final`.
- "What file name does the standard lock file format use?" was answered with a rejected
  idea from PEP 723 (a `.reqs` file next to the script).

Every check passed. The receipt makes the mistake visible in one click, which is the
point, but it is not a truth guarantee. A natural next check is relevance: the cited passage
should mention what the question asks about (`ReadOnly`, "lock file"). Retrieval can also
miss the passage that answers the question, and then the honest output is "the documents I
was given do not answer this".

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
receipts --embedder openai/qwen3-embedding-8b --embedder-url http://localhost:8081/v1 ingest docs/
                                               # any /v1/embeddings server
receipts eval examples/questions.jsonl         # checked vs plain on your question set
```

PDF support: `pip install 'rag-receipts[pdf]'`.

## How it is built

- `documents.py`: loading, and paragraph-packed chunks that keep their character offsets,
  so a quote can be highlighted in the original text.
- `index.py`: one SQLite file. FTS5 (BM25) for exact names like `TypeVarTuple`, embeddings
  for paraphrases, merged with reciprocal rank fusion.
- `models.py`: Ollama, Anthropic and OpenAI-compatible clients. Embedders know whether
  they embed a question or a document, for instruction-tuned models like Qwen3-Embedding.
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
