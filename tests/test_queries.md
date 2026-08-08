# Test queries (Week 5 — functional testing)

Use these against the sample `documents/` knowledge base to sanity-check the
pipeline before swapping in real content. Fill in the "Result" column after
running each query through the app.

| # | Question | Expected behavior | Result |
|---|----------|--------------------|--------|
| 1 | What platforms does Foundry Local support? | Answers "Windows, macOS (Apple silicon), and Linux", citing foundry_local.txt | ✅ Pass — correct answer, cited foundry_local.txt |
| 2 | What are the three steps of RAG? | Answers "Retrieve, Augment, Generate", citing rag_concepts.txt | ✅ Pass — correct, matches rag_concepts.txt |
| 3 | What table does this project use in SQLite? | Answers "chunks", citing sqlite_basics.txt | ✅ Pass — correct, matches sqlite_basics.txt |
| 4 | What is the capital of France? | Says it doesn't know / not in the documents (no relevant context) | ✅ Pass — after adding `MIN_RELEVANCE_SCORE` threshold in config.py (first attempt answered "Paris" from general knowledge — small model ignored the system prompt; fixed by skipping the LLM call when the best match score is too low) |
| 5 | (empty input) | App should not crash; ignores empty submissions | Not run — low risk, `st.chat_input` ignores empty submissions by default |
| 6 | How is cosine similarity used in this project? | Answers using embeddings_vector_search.txt | ✅ Pass — correct, matches embeddings_vector_search.txt |
| 7 | Why does the assistant cite its sources? | Answers using prompt_engineering.txt | ⚠️ Partial — retrieval was correct (top match: prompt_engineering.txt, score 0.67, exact relevant passage), but the qwen2.5-0.5b chat model paraphrased loosely instead of tightly reflecting the source ("credible/reputable sources" instead of "so a user can verify the claim"). Known limitation of this small model; decided to keep it for speed rather than switch to phi-3.5-mini. |
| 8 | What are the four layers of this project's architecture? | Answers using project_architecture.txt | ✅ Pass — correctly listed all four layers, cited project_architecture.txt |

## Findings & decisions (Week 5 evaluation)

- Retrieval (finding the right source chunks) has been reliable across all tests.
- The chat model (qwen2.5-0.5b) sometimes ignores the "answer only from context" instruction for out-of-scope questions — fixed with a hard `MIN_RELEVANCE_SCORE` cutoff in `config.py` so the app never calls the model when nothing relevant was retrieved.
- The same small model occasionally paraphrases loosely even when given the right context (test #7). Accepted as a known trade-off for speed/size; `phi-3.5-mini` remains available as a drop-in upgrade in `config.py` if answer quality becomes more important than speed later.

## How to log a new test

1. Ask the question in the app.
2. Expand "Sources used" and confirm the retrieved chunks are actually relevant.
3. Record whether the answer was correct, partially correct, or wrong in a new row.
