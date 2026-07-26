# Hybrid-RAG Evaluation Report

**Generated:** 2026-07-26T07:11:03Z
**Dataset:** LlamaIndex core (50 queries)
**Ground-truth coverage:** 50/50 (valid)
**Hardware:** Apple Mac Studio (M2 Max, 64GB RAM)
**Config:** `top_k=20, rrf_k=60, structural_weight=3.0, hybrid_weight=1.5`

---

## 1. Hit Rate @5 Comparison

| Hop Category | Vector-only RAG | Hybrid-RAG | Delta (Δ) |
|---|---|---|---|
| 1-hop | 0.579 (11/19) | 1.000 (19/19) | +0.421 |
| 2-hop | 0.368 (7/19) | 1.000 (19/19) | +0.632 |
| 3-hop | 0.667 (8/12) | 1.000 (12/12) | +0.333 |
| **Overall (micro-avg)** | **0.520** (26/50) | **1.000** (50/50) | **+0.480** |

### Per-Query Detail

| Query | Hops | Type | Vector Hit | Hybrid Hit | Hybrid Latency (ms) |
|---|---|---|---|---|---|
| Q1 | 1 | structural | ❌ | ✅ | 3 |
| Q2 | 1 | structural | ❌ | ✅ | 10 |
| Q3 | 1 | structural | ❌ | ✅ | 7 |
| Q4 | 1 | structural | ✅ | ✅ | 2 |
| Q5 | 1 | structural | ✅ | ✅ | 7 |
| Q6 | 2 | structural | ❌ | ✅ | 3 |
| Q7 | 2 | structural | ❌ | ✅ | 8 |
| Q8 | 2 | structural | ✅ | ✅ | 6 |
| Q9 | 2 | structural | ❌ | ✅ | 15 |
| Q10 | 2 | structural | ✅ | ✅ | 2 |
| Q11 | 3 | structural | ❌ | ✅ | 5 |
| Q12 | 3 | structural | ❌ | ✅ | 2 |
| Q13 | 3 | structural | ✅ | ✅ | 345 |
| Q14 | 3 | structural | ✅ | ✅ | 53 |
| Q15 | 3 | structural | ✅ | ✅ | 46 |
| Q16 | 2 | hybrid | ❌ | ✅ | 52 |
| Q17 | 2 | hybrid | ✅ | ✅ | 51 |
| Q18 | 2 | hybrid | ❌ | ✅ | 49 |
| Q19 | 3 | hybrid | ✅ | ✅ | 45 |
| Q20 | 3 | hybrid | ✅ | ✅ | 51 |
| Q21 | 1 | structural | ✅ | ✅ | 8 |
| Q22 | 1 | structural | ✅ | ✅ | 6 |
| Q23 | 1 | structural | ❌ | ✅ | 7 |
| Q24 | 1 | structural | ✅ | ✅ | 19 |
| Q25 | 1 | structural | ❌ | ✅ | 4 |
| Q26 | 2 | structural | ✅ | ✅ | 32 |
| Q27 | 2 | structural | ❌ | ✅ | 21 |
| Q28 | 2 | structural | ❌ | ✅ | 4 |
| Q29 | 2 | structural | ❌ | ✅ | 32 |
| Q30 | 2 | structural | ❌ | ✅ | 11 |
| Q31 | 2 | structural | ✅ | ✅ | 4 |
| Q32 | 2 | structural | ✅ | ✅ | 3 |
| Q33 | 2 | structural | ✅ | ✅ | 10 |
| Q34 | 2 | structural | ❌ | ✅ | 4 |
| Q35 | 2 | structural | ❌ | ✅ | 2 |
| Q36 | 3 | structural | ✅ | ✅ | 9 |
| Q37 | 3 | structural | ✅ | ✅ | 4 |
| Q38 | 3 | structural | ❌ | ✅ | 2 |
| Q39 | 3 | structural | ✅ | ✅ | 5 |
| Q40 | 3 | structural | ❌ | ✅ | 8 |
| Q41 | 1 | semantic | ✅ | ✅ | 52 |
| Q42 | 1 | hybrid | ✅ | ✅ | 49 |
| Q43 | 1 | semantic | ✅ | ✅ | 51 |
| Q44 | 1 | semantic | ✅ | ✅ | 57 |
| Q45 | 1 | semantic | ✅ | ✅ | 46 |
| Q46 | 1 | structural | ❌ | ✅ | 8 |
| Q47 | 2 | structural | ❌ | ✅ | 50 |
| Q48 | 1 | structural | ❌ | ✅ | 3 |
| Q49 | 1 | structural | ❌ | ✅ | 13 |
| Q50 | 1 | structural | ✅ | ✅ | 4 |

---

## 2. Latency Breakdown

Average retrieval latency per query type:

| Query Type | Avg Latency (ms) | Min (ms) | Max (ms) |
|---|---|---|---|
| hybrid | 50 | 45 | 52 |
| semantic | 52 | 46 | 57 |
| structural | 20 | 2 | 345 |
| **Overall** | **26** | **2** | **345** |

---

## 4. Summary & DOD Compliance

| Criterion | Target | Actual | Status |
|---|---|---|---|
| Hit Rate @5 (overall) | ≥ 0.60 | 1.000 | ✅ PASS |
| Retrieval Latency | < 1000 ms | 26 ms | ✅ PASS |
