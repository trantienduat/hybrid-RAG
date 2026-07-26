# Hybrid-RAG Evaluation Report

**Generated:** 2026-07-26T06:13:55Z
**Dataset:** LlamaIndex core (50 queries)
**Ground-truth coverage:** 50/50 (valid)
**Hardware:** Apple Mac Studio (M2 Max, 64GB RAM)
**Config:** `top_k=20, rrf_k=60, structural_weight=3.0, hybrid_weight=1.5`

---

## 1. Hit Rate @5 Comparison

| Hop Category | Vector-only RAG | Hybrid-RAG | Delta (Δ) |
|---|---|---|---|
| 1-hop | 0.579 (11/19) | 0.947 (18/19) | +0.368 |
| 2-hop | 0.368 (7/19) | 0.684 (13/19) | +0.316 |
| 3-hop | 0.667 (8/12) | 1.000 (12/12) | +0.333 |
| **Overall (micro-avg)** | **0.520** (26/50) | **0.860** (43/50) | **+0.340** |

### Per-Query Detail

| Query | Hops | Type | Vector Hit | Hybrid Hit | Hybrid Latency (ms) |
|---|---|---|---|---|---|
| Q1 | 1 | structural | ❌ | ✅ | 2 |
| Q2 | 1 | structural | ❌ | ✅ | 8 |
| Q3 | 1 | structural | ❌ | ✅ | 6 |
| Q4 | 1 | structural | ✅ | ✅ | 1 |
| Q5 | 1 | structural | ✅ | ✅ | 5 |
| Q6 | 2 | structural | ❌ | ✅ | 3 |
| Q7 | 2 | structural | ❌ | ✅ | 7 |
| Q8 | 2 | structural | ✅ | ✅ | 6 |
| Q9 | 2 | structural | ❌ | ✅ | 1 |
| Q10 | 2 | structural | ✅ | ✅ | 2 |
| Q11 | 3 | structural | ❌ | ✅ | 5 |
| Q12 | 3 | structural | ❌ | ✅ | 2 |
| Q13 | 3 | structural | ✅ | ✅ | 322 |
| Q14 | 3 | structural | ✅ | ✅ | 45 |
| Q15 | 3 | structural | ✅ | ✅ | 39 |
| Q16 | 2 | hybrid | ❌ | ❌ | 44 |
| Q17 | 2 | hybrid | ✅ | ✅ | 45 |
| Q18 | 2 | hybrid | ❌ | ❌ | 44 |
| Q19 | 3 | hybrid | ✅ | ✅ | 38 |
| Q20 | 3 | hybrid | ✅ | ✅ | 39 |
| Q21 | 1 | structural | ✅ | ✅ | 7 |
| Q22 | 1 | structural | ✅ | ✅ | 6 |
| Q23 | 1 | structural | ❌ | ✅ | 7 |
| Q24 | 1 | structural | ✅ | ✅ | 17 |
| Q25 | 1 | structural | ❌ | ✅ | 3 |
| Q26 | 2 | structural | ✅ | ❌ | 3 |
| Q27 | 2 | structural | ❌ | ❌ | 69 |
| Q28 | 2 | structural | ❌ | ✅ | 17 |
| Q29 | 2 | structural | ❌ | ❌ | 73 |
| Q30 | 2 | structural | ❌ | ❌ | 90 |
| Q31 | 2 | structural | ✅ | ✅ | 8 |
| Q32 | 2 | structural | ✅ | ✅ | 4 |
| Q33 | 2 | structural | ✅ | ✅ | 17 |
| Q34 | 2 | structural | ❌ | ✅ | 6 |
| Q35 | 2 | structural | ❌ | ✅ | 3 |
| Q36 | 3 | structural | ✅ | ✅ | 13 |
| Q37 | 3 | structural | ✅ | ✅ | 5 |
| Q38 | 3 | structural | ❌ | ✅ | 3 |
| Q39 | 3 | structural | ✅ | ✅ | 6 |
| Q40 | 3 | structural | ❌ | ✅ | 8 |
| Q41 | 1 | semantic | ✅ | ✅ | 47 |
| Q42 | 1 | hybrid | ✅ | ✅ | 44 |
| Q43 | 1 | semantic | ✅ | ✅ | 41 |
| Q44 | 1 | semantic | ✅ | ❌ | 45 |
| Q45 | 1 | semantic | ✅ | ✅ | 46 |
| Q46 | 1 | structural | ❌ | ✅ | 8 |
| Q47 | 2 | structural | ❌ | ✅ | 7 |
| Q48 | 1 | structural | ❌ | ✅ | 5 |
| Q49 | 1 | structural | ❌ | ✅ | 17 |
| Q50 | 1 | structural | ✅ | ✅ | 5 |

---

## 2. Latency Breakdown

Average retrieval latency per query type:

| Query Type | Avg Latency (ms) | Min (ms) | Max (ms) |
|---|---|---|---|
| hybrid | 42 | 38 | 45 |
| semantic | 45 | 41 | 47 |
| structural | 22 | 1 | 322 |
| **Overall** | **26** | **1** | **322** |

---

## 4. Summary & DOD Compliance

| Criterion | Target | Actual | Status |
|---|---|---|---|
| Hit Rate @5 (overall) | ≥ 0.60 | 0.860 | ✅ PASS |
| Retrieval Latency | < 1000 ms | 26 ms | ✅ PASS |
