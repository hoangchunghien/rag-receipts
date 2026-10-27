Model: ollama/qwen2.5:7b

| | plain RAG | checked |
|---|---|---|
| Answerable: answered | 100% (22/22) | 100% (22/22) |
| Answerable: answered with a citation you can follow | 73% (16/22) | 100% (22/22) |
| Answerable: ... and it points at the right PEP | 64% (14/22) | 86% (19/22) |
| Traps: answered anyway (lower is better) | 0% (0/14) | 0% (0/14) |
| Answers with a number or code name not in their sources | 9% (2/22) | 0 (check 3 cuts them) |

Checked mode: the model proposed 24 claims; 21 passed first time, 2 more passed after one repair round, 2 kept claims had their citation corrected to the source that really holds the quote.
Still cut after repair:
- quote does not occur in the cited source: 2
- quote is shorter than 4 words: 1
