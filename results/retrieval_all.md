# Retrieval results — XQuAD, chunk size 800 chars

- **en**: 1190 questions, 337 chunks, 6 answers split across a chunk boundary
- **tr**: 1190 questions, 342 chunks, 4 answers split across a chunk boundary

All numbers in %. Brackets: 95% bootstrap confidence interval.

| lang | embedder | retriever | hit@1 | hit@3 | hit@10 | MRR@10 |
|---|---|---|---|---|---|---|
| en | - | bm25 | 89.1 [87.1–90.8] | 96.4 [95.4–97.4] | 98.6 | 92.7 |
| en | - | bm25-p5 | 88.6 [86.7–90.3] | 96.2 [95.0–97.2] | 98.8 | 92.6 |
| en | qwen3-0.6b | dense | 86.0 [84.0–88.0] | 96.1 [94.9–97.1] | 98.9 | 91.2 |
| en | qwen3-0.6b | hybrid(bm25) | 91.2 [89.6–92.8] | 98.4 [97.7–99.1] | 99.1 | 94.7 |
| en | qwen3-0.6b | hybrid(bm25-p5) | 91.1 [89.5–92.7] | 98.4 [97.7–99.1] | 99.4 | 94.7 |
| en | qwen3-0.6b-instruct | dense | 90.0 [88.2–91.7] | 97.8 [97.0–98.7] | 99.7 | 94.1 |
| en | qwen3-0.6b-instruct | hybrid(bm25) | 91.8 [90.3–93.4] | 98.3 [97.6–99.0] | 99.1 | 95.1 |
| en | qwen3-0.6b-instruct | hybrid(bm25-p5) | 91.5 [89.9–93.1] | 98.3 [97.6–99.0] | 99.5 | 95.0 |
| en | e5-small | dense | 90.6 [88.8–92.2] | 97.7 [96.8–98.6] | 99.6 | 94.4 |
| en | e5-small | hybrid(bm25) | 91.9 [90.3–93.5] | 98.2 [97.5–98.9] | 99.1 | 95.1 |
| en | e5-small | hybrid(bm25-p5) | 92.4 [90.8–93.9] | 98.5 [97.7–99.2] | 99.5 | 95.5 |
| tr | - | bm25 | 78.6 [76.1–80.9] | 89.0 [87.1–90.7] | 93.9 | 84.1 |
| tr | - | bm25-p5 | 86.2 [84.2–88.1] | 94.4 [92.9–95.6] | 98.2 | 90.7 |
| tr | qwen3-0.6b | dense | 74.5 [72.1–76.9] | 88.1 [86.1–89.8] | 94.2 | 81.9 |
| tr | qwen3-0.6b | hybrid(bm25) | 81.4 [79.2–83.5] | 92.9 [91.3–94.3] | 96.6 | 87.4 |
| tr | qwen3-0.6b | hybrid(bm25-p5) | 84.5 [82.4–86.6] | 94.6 [93.2–95.8] | 98.5 | 89.8 |
| tr | qwen3-0.6b-instruct | dense | 79.5 [77.1–81.8] | 91.7 [90.0–93.1] | 96.8 | 86.0 |
| tr | qwen3-0.6b-instruct | hybrid(bm25) | 82.9 [80.8–85.0] | 93.7 [92.3–95.0] | 96.9 | 88.4 |
| tr | qwen3-0.6b-instruct | hybrid(bm25-p5) | 85.3 [83.2–87.3] | 95.1 [93.9–96.3] | 98.9 | 90.5 |
| tr | e5-small | dense | 86.9 [85.0–88.8] | 95.5 [94.4–96.6] | 98.7 | 91.5 |
| tr | e5-small | hybrid(bm25) | 86.1 [84.1–88.0] | 94.8 [93.5–96.0] | 96.6 | 90.4 |
| tr | e5-small | hybrid(bm25-p5) | 89.5 [87.7–91.2] | 97.1 [96.1–98.1] | 99.0 | 93.3 |

## Paired comparison vs baseline (`qwen3-0.6b/dense`)

Difference in points, 95% paired-bootstrap CI; ✓ = CI excludes 0.

| lang | config | Δ hit@1 | Δ hit@3 |
|---|---|---|---|
| en | -/bm25 | +3.1 [+0.8, +5.4] ✓ | +0.3 [-1.0, +1.8] |
| en | -/bm25-p5 | +2.6 [+0.3, +4.8] ✓ | +0.2 [-1.2, +1.6] |
| en | qwen3-0.6b/hybrid(bm25) | +5.2 [+3.5, +6.8] ✓ | +2.4 [+1.3, +3.5] ✓ |
| en | qwen3-0.6b/hybrid(bm25-p5) | +5.1 [+3.5, +6.7] ✓ | +2.4 [+1.3, +3.5] ✓ |
| en | qwen3-0.6b-instruct/dense | +4.0 [+2.6, +5.5] ✓ | +1.8 [+0.8, +2.7] ✓ |
| en | qwen3-0.6b-instruct/hybrid(bm25) | +5.9 [+4.1, +7.6] ✓ | +2.3 [+1.2, +3.5] ✓ |
| en | qwen3-0.6b-instruct/hybrid(bm25-p5) | +5.5 [+3.9, +7.2] ✓ | +2.3 [+1.2, +3.4] ✓ |
| en | e5-small/dense | +4.6 [+2.8, +6.4] ✓ | +1.7 [+0.6, +2.9] ✓ |
| en | e5-small/hybrid(bm25) | +6.0 [+3.9, +7.9] ✓ | +2.2 [+1.1, +3.4] ✓ |
| en | e5-small/hybrid(bm25-p5) | +6.5 [+4.5, +8.4] ✓ | +2.4 [+1.3, +3.6] ✓ |
| tr | -/bm25 | +4.0 [+1.2, +6.7] ✓ | +0.9 [-1.1, +3.1] |
| tr | -/bm25-p5 | +11.7 [+9.2, +14.3] ✓ | +6.3 [+4.5, +8.1] ✓ |
| tr | qwen3-0.6b/hybrid(bm25) | +6.9 [+4.8, +9.0] ✓ | +4.8 [+3.1, +6.5] ✓ |
| tr | qwen3-0.6b/hybrid(bm25-p5) | +10.0 [+8.2, +11.8] ✓ | +6.6 [+5.0, +8.1] ✓ |
| tr | qwen3-0.6b-instruct/dense | +5.0 [+3.3, +6.7] ✓ | +3.6 [+2.4, +4.9] ✓ |
| tr | qwen3-0.6b-instruct/hybrid(bm25) | +8.4 [+6.1, +10.6] ✓ | +5.6 [+3.9, +7.4] ✓ |
| tr | qwen3-0.6b-instruct/hybrid(bm25-p5) | +10.8 [+8.6, +12.7] ✓ | +7.1 [+5.5, +8.7] ✓ |
| tr | e5-small/dense | +12.4 [+10.1, +14.7] ✓ | +7.5 [+5.8, +9.2] ✓ |
| tr | e5-small/hybrid(bm25) | +11.6 [+9.0, +14.1] ✓ | +6.7 [+5.0, +8.6] ✓ |
| tr | e5-small/hybrid(bm25-p5) | +15.0 [+12.6, +17.3] ✓ | +9.1 [+7.4, +10.8] ✓ |

hit@3 is the app's setting (TOP_K = 3).
