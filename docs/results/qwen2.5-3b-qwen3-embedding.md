Model: ollama/qwen2.5:3b

| | plain RAG | checked |
|---|---|---|
| Answerable: answered | 95% (21/22) | 82% (18/22) |
| Answerable: answered with a citation you can follow | 9% (2/22) | 82% (18/22) |
| Answerable: ... and it points at the right PEP | 9% (2/22) | 77% (17/22) |
| Traps: answered anyway (lower is better) | 0% (0/14) | 14% (2/14) |
| Answers with a number or code name not in their sources | 0% (0/21) | 0 (check 3 cuts them) |

Checked mode: the model proposed 25 claims; 23 passed first time, 0 more passed after one repair round, 4 kept claims had their citation corrected to the source that really holds the quote.
Still cut after repair:
- quote does not occur in the cited source: 2
