# CSL7110 – Assignment 2
## MinHashing and Locality Sensitive Hashing

**Name:** Ruby Mythili  
**Roll No:** M25DE1006  
**Course:** CSL7110 – Machine Learning with Big Data  

---

# 1. Introduction

This assignment focuses on measuring document similarity using MinHashing and Locality Sensitive Hashing (LSH). The objective is to efficiently estimate the Jaccard similarity between documents and identify candidate pairs of similar items without comparing every pair directly.

The implementation was carried out in Python using Google Colab. The assignment explores different k-gram representations, MinHash approximation, and the use of LSH for efficient similarity search.

---

# 2. Dataset

Four text documents were used for similarity analysis:

| Document | Description |
|--------|-------------|
| D1 | News article related to Apple CEO Tim Cook |
| D2 | Slight variation of the first article |
| D3 | Interview-based article |
| D4 | Political news article |

These documents were stored in the `data/` folder and used for computing similarity using different k-gram representations.