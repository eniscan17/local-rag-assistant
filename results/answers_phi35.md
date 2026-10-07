# Answer quality — XQuAD, `foundry:phi-3.5-mini`, prompt `short`

300 questions per language (seed 0, same ids in EN and TR), top-3 chunks of 800 chars. All numbers in %; brackets: 95% bootstrap CI. Primary metric: **f1**.

| lang | condition | EM | F1 | contains | refused | retrieval hit@k | median s |
|---|---|---|---|---|---|---|---|
| en | closed | 5.0 [2.7–7.7] | 16.8 [13.7–20.0] | 11.0 [7.7–14.3] | 3.0 | – | 0.67 |
| en | oracle | 56.7 [51.3–62.3] | 77.2 [73.5–80.9] | 82.0 [77.7–86.3] | 1.0 | – | 1.42 |
| en | e5-small/hybrid(bm25-p5) | 55.3 [49.7–61.0] | 75.4 [71.4–79.1] | 80.7 [76.3–85.0] | 1.3 | 98.3 | 2.83 |
| en | qwen3-0.6b/dense | 54.0 [48.0–59.7] | 72.3 [68.2–76.4] | 76.3 [71.7–81.0] | 1.7 | 96.0 | 2.59 |
| tr | closed | 0.3 [0.0–1.0] | 3.3 [2.1–4.7] | 2.3 [0.7–4.3] | 3.0 | – | 1.09 |
| tr | oracle | 32.3 [26.7–37.7] | 51.3 [46.4–56.4] | 51.0 [45.0–57.0] | 0.0 | – | 2.38 |
| tr | e5-small/hybrid(bm25-p5) | 27.3 [22.3–32.3] | 43.6 [38.6–48.5] | 40.3 [34.7–46.0] | 0.0 | 98.3 | 5.37 |
| tr | qwen3-0.6b/dense | 29.0 [23.7–34.0] | 44.9 [39.6–49.7] | 44.0 [38.3–49.7] | 0.7 | 91.0 | 4.67 |

## Paired differences (f1, points)

✓ = 95% paired-bootstrap CI excludes 0.

| comparison | Δ |
|---|---|
| en: e5-small/hybrid(bm25-p5) − oracle | -1.8 [-4.6, +0.9] |
| en: e5-small/hybrid(bm25-p5) − closed | +58.5 [+53.8, +63.0] ✓ |
| en: qwen3-0.6b/dense − oracle | -4.9 [-7.9, -1.9] ✓ |
| en: qwen3-0.6b/dense − closed | +55.5 [+50.4, +60.4] ✓ |
| en: e5-small/hybrid(bm25-p5) − qwen3-0.6b/dense | +3.1 [+0.3, +6.0] ✓ |
| tr: e5-small/hybrid(bm25-p5) − oracle | -7.6 [-11.8, -3.6] ✓ |
| tr: e5-small/hybrid(bm25-p5) − closed | +40.3 [+35.5, +45.2] ✓ |
| tr: qwen3-0.6b/dense − oracle | -6.4 [-10.0, -2.6] ✓ |
| tr: qwen3-0.6b/dense − closed | +41.6 [+36.6, +46.4] ✓ |
| tr: e5-small/hybrid(bm25-p5) − qwen3-0.6b/dense | -1.3 [-5.1, +2.4] |
| TR − EN: closed | -13.6 [-16.7, -10.4] ✓ |
| TR − EN: oracle | -26.0 [-31.2, -21.2] ✓ |
| TR − EN: e5-small/hybrid(bm25-p5) | -31.8 [-37.0, -26.5] ✓ |
| TR − EN: qwen3-0.6b/dense | -27.5 [-32.8, -22.3] ✓ |

## Where wrong answers come from

Share of all questions; correct = gold answer contained in the model's answer.

| lang | retriever | wrong | …retrieval missed | …retrieved, model still wrong | missed but right anyway |
|---|---|---|---|---|---|
| en | e5-small/hybrid(bm25-p5) | 19.3 | 1.3 | 18.0 | 0.3 |
| en | qwen3-0.6b/dense | 23.7 | 3.7 | 20.0 | 0.3 |
| tr | e5-small/hybrid(bm25-p5) | 59.7 | 1.7 | 58.0 | 0.0 |
| tr | qwen3-0.6b/dense | 56.0 | 9.0 | 47.0 | 0.0 |

## Examples: gold chunk was retrieved, answer still wrong

| lang | question | gold | model |
|---|---|---|---|
| en | What did Marlee Matlin translate? | the national anthem | American Sign Language (ASL)  (Note: The context provided does not contain a full sentence about Marlee Matlin's transla |
| en | Who was the leader when the Franks entered the Euphrates valley? | Oursel | Raimbaud |
| en | What did the General Conference on Weights and Measures name after Tesla in 1960? | SI unit of magnetic flux density | The tesla |
| en | Other than his scientific achievements what was Tesla famous for? | showmanship | "mad scientist" |
| tr | Luke Kuechly kaç tane top çalma kaydetmiştir? | 118 | 88 top çalma |
| tr | Josh Norman kaç tane top çalmıştır? | dört | 4 |
| tr | Bu sezon takımdaki en fazla sack etmeyi kim kaydetmiştir? | Kawann Short | Kony Ealy |
| tr | Broncos ve Steelers arasındaki bölge turu kazananı hangi takımdı? | Broncos | Pittsburgh Steelers |
