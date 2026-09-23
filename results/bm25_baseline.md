# Retrieval results — XQuAD, chunk size 800 chars

- **en**: 1190 questions, 337 chunks, 6 answers split across a chunk boundary
- **tr**: 1190 questions, 342 chunks, 4 answers split across a chunk boundary

| lang | embedder | retriever | hit@1 | hit@3 | hit@5 | hit@10 | MRR@10 |
|---|---|---|---|---|---|---|---|
| en | - | bm25 | 0.891 | 0.964 | 0.974 | 0.986 | 0.927 |
| en | - | bm25-p5 | 0.886 | 0.962 | 0.974 | 0.988 | 0.926 |
| tr | - | bm25 | 0.786 | 0.890 | 0.916 | 0.939 | 0.841 |
| tr | - | bm25-p5 | 0.862 | 0.944 | 0.962 | 0.982 | 0.907 |

hit@3 is the app's setting (TOP_K = 3).
