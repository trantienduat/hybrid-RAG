# Hybrid-RAG Evaluation Report

**Generated:** 2026-07-26T05:20:07Z
**Dataset:** LlamaIndex core (20 queries)
**Ground-truth coverage:** 20/20 (valid)
**Hardware:** Apple Mac Studio (M2 Max, 64GB RAM)
**Config:** `top_k=20, rrf_k=60, structural_weight=3.0, hybrid_weight=1.5`

---

## 1. Hit Rate @5 Comparison

| Hop Category | Vector-only RAG | Hybrid-RAG | Delta (Δ) |
|---|---|---|---|
| 1-hop | 0.400 (2/5) | 0.800 (4/5) | +0.400 |
| 2-hop | 0.625 (5/8) | 0.375 (3/8) | -0.250 |
| 3-hop | 0.429 (3/7) | 0.571 (4/7) | +0.143 |
| **Overall (micro-avg)** | **0.500** (10/20) | **0.550** (11/20) | **+0.050** |

### Per-Query Detail

| Query | Hops | Type | Vector Hit | Hybrid Hit | Hybrid Latency (ms) |
|---|---|---|---|---|---|
| Q1 | 1 | structural | ❌ | ✅ | 2 |
| Q2 | 1 | structural | ❌ | ✅ | 7 |
| Q3 | 1 | structural | ❌ | ❌ | 618 |
| Q4 | 1 | structural | ✅ | ✅ | 3 |
| Q5 | 1 | structural | ✅ | ✅ | 54 |
| Q6 | 2 | structural | ✅ | ❌ | 4 |
| Q7 | 2 | structural | ❌ | ✅ | 10 |
| Q8 | 2 | structural | ✅ | ❌ | 8 |
| Q9 | 2 | structural | ❌ | ✅ | 2 |
| Q10 | 2 | structural | ✅ | ❌ | 51 |
| Q11 | 3 | structural | ❌ | ❌ | 44 |
| Q12 | 3 | structural | ❌ | ✅ | 8 |
| Q13 | 3 | structural | ✅ | ✅ | 40 |
| Q14 | 3 | structural | ❌ | ❌ | 39 |
| Q15 | 3 | structural | ✅ | ✅ | 41 |
| Q16 | 2 | hybrid | ✅ | ❌ | 36 |
| Q17 | 2 | hybrid | ✅ | ✅ | 41 |
| Q18 | 2 | hybrid | ❌ | ❌ | 35 |
| Q19 | 3 | hybrid | ❌ | ❌ | 36 |
| Q20 | 3 | hybrid | ✅ | ✅ | 37 |

---

## 2. Latency Breakdown

Average retrieval latency per query type:

| Query Type | Avg Latency (ms) | Min (ms) | Max (ms) |
|---|---|---|---|
| hybrid | 37 | 35 | 41 |
| structural | 62 | 2 | 618 |
| **Overall** | **56** | **2** | **618** |

---

## 4. Summary & DOD Compliance

| Criterion | Target | Actual | Status |
|---|---|---|---|
| Hit Rate @5 (overall) | ≥ 0.60 | 0.550 | ❌ FAIL |
| Retrieval Latency | < 1000 ms | 56 ms | ✅ PASS |
