Model: ollama/qwen2.5:3b

| | plain RAG | checked |
|---|---|---|
| Answerable: answered | 86% (19/22) | 82% (18/22) |
| Answerable: answered with a citation you can follow | 9% (2/22) | 82% (18/22) |
| Answerable: ... and it points at the right PEP | 9% (2/22) | 68% (15/22) |
| Traps: answered anyway (lower is better) | 0% (0/14) | 0% (0/14) |
| Answers with a number or code name not in their sources | 5% (1/19) | 0 (check 3 cuts them) |

Checked mode: the model proposed 22 claims; 17 passed first time, 1 more passed after one repair round, 2 kept claims had their citation corrected to the source that really holds the quote.
Check 4 set aside 0 answerable and 0 trap answers whose cited passages never mention the names the question asks about.
Still cut after repair:
- quote does not occur in the cited source: 2
- a specific is not in the cited source: 2
