# Retrieval results — XQuAD, chunk size 800 chars

- **en**: 1190 questions, 337 chunks, 6 answers split across a chunk boundary
- **tr**: 1190 questions, 342 chunks, 4 answers split across a chunk boundary

| lang | embedder | retriever | hit@1 | hit@3 | hit@5 | hit@10 | MRR@10 |
|---|---|---|---|---|---|---|---|
| en | - | bm25 | 0.891 | 0.964 | 0.974 | 0.986 | 0.927 |
| en | - | bm25-p5 | 0.886 | 0.962 | 0.974 | 0.988 | 0.926 |
| en | qwen3-0.6b | dense | 0.860 | 0.961 | 0.981 | 0.989 | 0.912 |
| en | qwen3-0.6b | hybrid(bm25) | 0.912 | 0.984 | 0.987 | 0.991 | 0.947 |
| en | qwen3-0.6b | hybrid(bm25-p5) | 0.911 | 0.984 | 0.989 | 0.994 | 0.947 |
| en | e5-small | dense | 0.906 | 0.977 | 0.986 | 0.996 | 0.944 |
| en | e5-small | hybrid(bm25) | 0.919 | 0.982 | 0.988 | 0.991 | 0.951 |
| en | e5-small | hybrid(bm25-p5) | 0.924 | 0.985 | 0.987 | 0.995 | 0.955 |
| tr | - | bm25 | 0.786 | 0.890 | 0.916 | 0.939 | 0.841 |
| tr | - | bm25-p5 | 0.862 | 0.944 | 0.962 | 0.982 | 0.907 |
| tr | qwen3-0.6b | dense | 0.745 | 0.881 | 0.908 | 0.942 | 0.819 |
| tr | qwen3-0.6b | hybrid(bm25) | 0.814 | 0.929 | 0.946 | 0.966 | 0.874 |
| tr | qwen3-0.6b | hybrid(bm25-p5) | 0.845 | 0.946 | 0.968 | 0.985 | 0.898 |
| tr | e5-small | dense | 0.869 | 0.955 | 0.972 | 0.987 | 0.915 |
| tr | e5-small | hybrid(bm25) | 0.861 | 0.948 | 0.961 | 0.966 | 0.904 |
| tr | e5-small | hybrid(bm25-p5) | 0.895 | 0.971 | 0.982 | 0.990 | 0.933 |

hit@3 is the app's setting (TOP_K = 3).
