# Answer quality — XQuAD, `foundry:qwen2.5-7b`, prompt `short`

300 questions per language (seed 0, same ids in EN and TR), top-3 chunks of 800 chars. All numbers in %; brackets: 95% bootstrap CI. Primary metric: **f1**.

| lang | condition | EM | F1 | contains | refused | retrieval hit@k | median s |
|---|---|---|---|---|---|---|---|
| en | oracle | 72.3 [67.3–77.7] | 81.4 [77.4–85.4] | 77.3 [72.3–82.0] | 8.7 | – | 2.05 |
| en | e5-small/hybrid(bm25-p5) | 69.0 [63.7–74.0] | 78.3 [74.0–82.5] | 75.3 [70.3–80.0] | 10.0 | 98.3 | 3.83 |
| tr | oracle | 59.7 [54.0–65.3] | 71.8 [67.2–76.2] | 69.3 [64.0–74.3] | 8.3 | – | 2.96 |
| tr | e5-small/hybrid(bm25-p5) | 55.0 [49.3–60.7] | 68.7 [63.9–73.3] | 65.3 [59.7–70.7] | 10.0 | 98.3 | 6.57 |

## Paired differences (f1, points)

✓ = 95% paired-bootstrap CI excludes 0.

| comparison | Δ |
|---|---|
| en: e5-small/hybrid(bm25-p5) − oracle | -3.0 [-6.5, +0.3] |
| tr: e5-small/hybrid(bm25-p5) − oracle | -3.1 [-6.3, -0.1] ✓ |
| TR − EN: oracle | -9.6 [-14.3, -4.9] ✓ |
| TR − EN: e5-small/hybrid(bm25-p5) | -9.7 [-14.9, -4.7] ✓ |

## Where wrong answers come from

Share of all questions; correct = gold answer contained in the model's answer.

| lang | retriever | wrong | …retrieval missed | …retrieved, model still wrong | missed but right anyway |
|---|---|---|---|---|---|
| en | e5-small/hybrid(bm25-p5) | 24.7 | 1.3 | 23.3 | 0.3 |
| tr | e5-small/hybrid(bm25-p5) | 34.7 | 1.7 | 33.0 | 0.0 |

## Examples: gold chunk was retrieved, answer still wrong

| lang | question | gold | model |
|---|---|---|---|
| en | Who won Super Bowl XLIX? | New England Patriots | unanswerable |
| en | What team was the winner of Super Bowl XXXIII? | Broncos | unanswerable |
| en | What did Marlee Matlin translate? | the national anthem | American Sign Language (ASL) |
| en | Who was the Normans' main enemy in Italy, the Byzantine Empire and Armenia? | Seljuk Turks | unanswerable |
| tr | Luke Kuechly kaç tane top çalma kaydetmiştir? | 118 | unanswerable |
| tr | Josh Norman kaç tane top çalmıştır? | dört | 4 |
| tr | Panthers'lere sack etmede kim önderlik etmektedir? | Kawann Short | Jared Allen |
| tr | Lady Gaga neyi söylemiştir? | ulusal marşı | unalterable |
