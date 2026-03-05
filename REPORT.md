# CSL7110 – Assignment 2
## MinHashing and Locality Sensitive Hashing (LSH)

**Name:** Ruby Mythili  
**Roll No:** M25DE1006  
**Course:** CSL7110 – Machine Learning with Big Data  
**Tools:** Python, Google Colab

---

# 1. Introduction

This assignment studies similarity detection using Jaccard similarity, MinHashing, and Locality Sensitive Hashing (LSH). The goal is to:
1) compute exact similarity using k-grams and Jaccard,
2) estimate similarity efficiently using MinHash signatures,
3) use LSH banding to find candidate similar pairs without comparing every pair.

---

# 2. Datasets

## 2.1 Text Documents (D1–D4)
Four text files were used for document similarity:
- D1, D2, D3: news articles related to Tim Cook / Apple
- D4: political news article

Files are in `data/`.

## 2.2 MovieLens 100k
MovieLens 100k dataset (943 users, 1682 movies).  
Each user is treated as a set of movie IDs they rated.

---

# 3. k-Grams Construction

All documents were converted to lowercase. k-grams were stored as sets (duplicates removed).

Constructed k-grams:
- Character 2-grams
- Character 3-grams
- Word 2-grams

k-gram counts:

| Document | char-2 | char-3 | word-2 |
|---|---:|---:|---:|
| D1 | 263 | 765 | 279 |
| D2 | 262 | 762 | 278 |
| D3 | 269 | 828 | 337 |
| D4 | 255 | 698 | 232 |

---

# 4. Exact Jaccard Similarity (18 values)

Jaccard similarity:

\[
J(A,B) = \frac{|A \cap B|}{|A \cup B|}
\]

## 4.1 Results

### Character 2-grams
| Pair | Jaccard |
|---|---:|
| D1–D2 | 0.9811 |
| D1–D3 | 0.8157 |
| D1–D4 | 0.6444 |
| D2–D3 | 0.8000 |
| D2–D4 | 0.6413 |
| D3–D4 | 0.6530 |

### Character 3-grams
| Pair | Jaccard |
|---|---:|
| D1–D2 | 0.9780 |
| D1–D3 | 0.5804 |
| D1–D4 | 0.3051 |
| D2–D3 | 0.5680 |
| D2–D4 | 0.3059 |
| D3–D4 | 0.3121 |

### Word 2-grams
| Pair | Jaccard |
|---|---:|
| D1–D2 | 0.9408 |
| D1–D3 | 0.1823 |
| D1–D4 | 0.0302 |
| D2–D3 | 0.1737 |
| D2–D4 | 0.0303 |
| D3–D4 | 0.0161 |

CSV saved at: `results/tables/jaccard_results.csv`

## 4.2 Observation
- D1 and D2 are near-duplicates → very high similarity in all k-gram types.
- Word 2-grams are stricter, so similarity drops sharply when content differs.
- D4 is about a different topic, so its similarity with the others is low.

---

# 5. MinHash for D1 vs D2 (3-grams)

MinHash signatures were built using character 3-grams for D1 and D2.
Estimated Jaccard similarity was computed for different signature lengths (t).

## 5.1 MinHash Estimates

(True Jaccard for D1–D2 using char-3 = 0.9780)

| t (hash functions) | Estimated Jaccard |
|---:|---:|
| 20  | 0.9500 |
| 60  | 1.0000 |
| 150 | 0.9800 |
| 300 | 0.9700 |
| 600 | 0.9750 |

CSV saved at: `results/tables/minhash_estimates.csv`

## 5.2 Accuracy vs Runtime (Task 2B)

| t | Approx | True | Abs Error | Runtime (sec) |
|---:|---:|---:|---:|---:|
| 20  | 1.0000 | 0.9780 | 0.0220 | 0.0942 |
| 60  | 0.9833 | 0.9780 | 0.0053 | 0.0428 |
| 150 | 0.9800 | 0.9780 | 0.0020 | 0.1771 |
| 300 | 0.9767 | 0.9780 | 0.0013 | 0.5460 |
| 600 | 0.9817 | 0.9780 | 0.0037 | 0.7297 |

CSV saved at: `results/tables/minhash_runtime_analysis.csv`

### Chosen t (justification)
A practical choice is **t = 150 or t = 300**:
- errors are very small,
- runtime is still reasonable compared to very large t.

---

# 6. LSH Theory (t = 160, τ = 0.7)

Using S-curve probability:

\[
P(\text{candidate}) = 1 - (1 - s^{r})^{b}
\]

where:
- t = 160 = r × b
- τ (threshold) ≈ 0.7

Candidate (r, b) options produced thresholds near τ.  
A suitable choice close to 0.7:

- **r = 8**, **b = 20** (threshold ≈ 0.688)

## 6.1 Candidate Pair Probability (using char-3 similarities)

| Pair | s (Jaccard char-3) | P(candidate) |
|---|---:|---:|
| D1–D2 | 0.9780 | 1.0000 |
| D1–D3 | 0.5804 | 0.2283 |
| D1–D4 | 0.3051 | 0.0015 |
| D2–D3 | 0.5680 | 0.1958 |
| D2–D4 | 0.3059 | 0.0015 |
| D3–D4 | 0.3121 | 0.0018 |

## 6.2 Observation
- Very similar pairs (D1–D2) are almost always selected as candidates.
- Medium similarity pairs have moderate probability.
- Low similarity pairs are almost never selected → reduces unnecessary comparisons.

---

# 7. MovieLens — MinHash Similar Users (similarity ≥ 0.5)

Ground truth:
- total users: **943**
- exact similar user pairs (≥ 0.5): **10**

MinHash was used to approximate similarity for t ∈ {50, 100, 200}.

| t (hash functions) | False Positives | False Negatives |
|---:|---:|---:|
| 50  | 206 | 1 |
| 100 | 40  | 1 |
| 200 | 9   | 1 |

## Observation
Increasing signature length reduces false positives significantly.  
At t = 200, results become much cleaner while still being efficient.

---

# 8. MovieLens — LSH Candidate Detection (similarity ≥ 0.6)

Configurations used (as required):
- (t=50, r=5, b=10)
- (t=100, r=5, b=20)
- (t=200, r=5, b=40)
- (t=200, r=10, b=20)

## 8.1 Results for similarity ≥ 0.6

| t | r | b | False Positives | False Negatives |
|---:|---:|---:|---:|---:|
| 50  | 5  | 10 | 5 | 7 |
| 100 | 5  | 20 | 0 | 7 |
| 200 | 5  | 40 | 0 | 7 |
| 200 | 10 | 20 | 0 | 8 |

CSV saved at: `results/tables/lsh_movielens_results_tau0p6.csv`

## 8.2 Results for similarity ≥ 0.8

| t | r | b | False Positives | False Negatives | Exact Pairs | Predicted Pairs |
|---:|---:|---:|---:|---:|---:|---:|
| 50  | 5  | 10 | 0 | 0 | 1 | 1 |
| 100 | 5  | 20 | 0 | 0 | 1 | 1 |
| 200 | 5  | 40 | 0 | 0 | 1 | 1 |
| 200 | 10 | 20 | 0 | 0 | 1 | 1 |

CSV saved at: `results/tables/lsh_movielens_results_tau0p8.csv`

## Observation (τ = 0.8 vs 0.6)
- At τ = 0.8, very few pairs qualify as “similar”, so both exact and predicted pairs are very small.
- With stricter threshold, LSH becomes more selective and results become easier to match exactly.

---

# 9. Repository Contents

- Implementation Notebook: `notebooks/CSL7110_Assignment2_M25DE1006.ipynb`
- Results CSVs: `results/tables/`
- Input docs: `data/`

---
# 10. Conclusion

This assignment demonstrated:
- how k-grams and Jaccard similarity measure document similarity,
- how MinHash approximates Jaccard efficiently,
- how LSH selects candidate pairs with high probability for similar items,
- and how these methods scale to larger datasets like MovieLens.