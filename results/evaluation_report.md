# Hybrid-RAG Evaluation Report

**Generated:** 2026-07-26T06:29:44Z
**Dataset:** LlamaIndex core (50 queries)
**Ground-truth coverage:** 50/50 (valid)
**Hardware:** Apple Mac Studio (M2 Max, 64GB RAM)
**Config:** `top_k=20, rrf_k=60, structural_weight=3.0, hybrid_weight=1.5`

---

## 1. Hit Rate @5 Comparison

| Hop Category | Vector-only RAG | Hybrid-RAG | Delta (Δ) |
|---|---|---|---|
| 1-hop | 0.579 (11/19) | 0.947 (18/19) | +0.368 |
| 2-hop | 0.368 (7/19) | 0.895 (17/19) | +0.526 |
| 3-hop | 0.667 (8/12) | 1.000 (12/12) | +0.333 |
| **Overall (micro-avg)** | **0.520** (26/50) | **0.940** (47/50) | **+0.420** |

### Per-Query Detail

| Query | Hops | Type | Vector Hit | Hybrid Hit | Hybrid Latency (ms) |
|---|---|---|---|---|---|
| Q1 | 1 | structural | ❌ | ✅ | 3 |
| Q2 | 1 | structural | ❌ | ✅ | 14 |
| Q3 | 1 | structural | ❌ | ✅ | 8 |
| Q4 | 1 | structural | ✅ | ✅ | 2 |
| Q5 | 1 | structural | ✅ | ✅ | 7 |
| Q6 | 2 | structural | ❌ | ✅ | 4 |
| Q7 | 2 | structural | ❌ | ✅ | 8 |
| Q8 | 2 | structural | ✅ | ✅ | 7 |
| Q9 | 2 | structural | ❌ | ✅ | 16 |
| Q10 | 2 | structural | ✅ | ✅ | 2 |
| Q11 | 3 | structural | ❌ | ✅ | 6 |
| Q12 | 3 | structural | ❌ | ✅ | 2 |
| Q13 | 3 | structural | ✅ | ✅ | 342 |
| Q14 | 3 | structural | ✅ | ✅ | 47 |
| Q15 | 3 | structural | ✅ | ✅ | 49 |
| Q16 | 2 | hybrid | ❌ | ❌ | 39 |
| Q17 | 2 | hybrid | ✅ | ✅ | 43 |
| Q18 | 2 | hybrid | ❌ | ❌ | 41 |
| Q19 | 3 | hybrid | ✅ | ✅ | 39 |
| Q20 | 3 | hybrid | ✅ | ✅ | 43 |
| Q21 | 1 | structural | ✅ | ✅ | 8 |
| Q22 | 1 | structural | ✅ | ✅ | 7 |
| Q23 | 1 | structural | ❌ | ✅ | 7 |
| Q24 | 1 | structural | ✅ | ✅ | 19 |
| Q25 | 1 | structural | ❌ | ✅ | 3 |
| Q26 | 2 | structural | ✅ | ✅ | 31 |
| Q27 | 2 | structural | ❌ | ✅ | 20 |
| Q28 | 2 | structural | ❌ | ✅ | 5 |
| Q29 | 2 | structural | ❌ | ✅ | 30 |
| Q30 | 2 | structural | ❌ | ✅ | 10 |
| Q31 | 2 | structural | ✅ | ✅ | 3 |
| Q32 | 2 | structural | ✅ | ✅ | 3 |
| Q33 | 2 | structural | ✅ | ✅ | 10 |
| Q34 | 2 | structural | ❌ | ✅ | 4 |
| Q35 | 2 | structural | ❌ | ✅ | 2 |
| Q36 | 3 | structural | ✅ | ✅ | 9 |
| Q37 | 3 | structural | ✅ | ✅ | 4 |
| Q38 | 3 | structural | ❌ | ✅ | 2 |
| Q39 | 3 | structural | ✅ | ✅ | 5 |
| Q40 | 3 | structural | ❌ | ✅ | 7 |
| Q41 | 1 | semantic | ✅ | ✅ | 46 |
| Q42 | 1 | hybrid | ✅ | ✅ | 36 |
| Q43 | 1 | semantic | ✅ | ✅ | 39 |
| Q44 | 1 | semantic | ✅ | ❌ | 35 |
| Q45 | 1 | semantic | ✅ | ✅ | 31 |
| Q46 | 1 | structural | ❌ | ✅ | 6 |
| Q47 | 2 | structural | ❌ | ✅ | 56 |
| Q48 | 1 | structural | ❌ | ✅ | 4 |
| Q49 | 1 | structural | ❌ | ✅ | 13 |
| Q50 | 1 | structural | ✅ | ✅ | 4 |

---

## 2. Latency Breakdown

Average retrieval latency per query type:

| Query Type | Avg Latency (ms) | Min (ms) | Max (ms) |
|---|---|---|---|
| hybrid | 40 | 36 | 43 |
| semantic | 38 | 31 | 46 |
| structural | 20 | 2 | 342 |
| **Overall** | **24** | **2** | **342** |

---

## 4. Summary & DOD Compliance

| Criterion | Target | Actual | Status |
|---|---|---|---|
| Hit Rate @5 (overall) | ≥ 0.60 | 0.940 | ✅ PASS |
| Retrieval Latency | < 1000 ms | 24 ms | ✅ PASS |
