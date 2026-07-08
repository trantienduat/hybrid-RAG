# Hybrid-RAG Evaluation Report

**Generated:** 2026-07-06T13:48:15Z  
**Dataset:** LlamaIndex core (20 queries)  
**Hardware:** Apple Mac Studio (M2 Max, 64GB RAM)  
**Config:** `top_k=20, rrf_k=60, structural_weight=3.0, hybrid_weight=1.5`

---

## 1. Hit Rate @5 Comparison

| Hop Category | Vector-only RAG | Hybrid-RAG | Delta (Δ) |
|---|---|---|---|
| 1-hop | 0.000 (0/5) | 0.400 (2/5) | +0.400 |
| 2-hop | 0.000 (0/8) | 0.000 (0/8) | +0.000 |
| 3-hop | 0.000 (0/7) | 0.143 (1/7) | +0.143 |
| **Overall (micro-avg)** | **0.000** (0/20) | **0.150** (3/20) | **+0.150** |

### Per-Query Detail

| Query | Hops | Type | Vector Hit | Hybrid Hit | Hybrid Latency (ms) |
|---|---|---|---|---|---|
| Q1 | 1 | structural | ❌ | ❌ | 907 |
| Q2 | 1 | structural | ❌ | ✅ | 319 |
| Q3 | 1 | structural | ❌ | ❌ | 0 |
| Q4 | 1 | structural | ❌ | ❌ | 219 |
| Q5 | 1 | structural | ❌ | ✅ | 379 |
| Q6 | 2 | structural | ❌ | ❌ | 364 |
| Q7 | 2 | structural | ❌ | ❌ | 0 |
| Q8 | 2 | structural | ❌ | ❌ | 334 |
| Q9 | 2 | structural | ❌ | ❌ | 305 |
| Q10 | 2 | structural | ❌ | ❌ | 0 |
| Q11 | 3 | structural | ❌ | ❌ | 0 |
| Q12 | 3 | structural | ❌ | ❌ | 369 |
| Q13 | 3 | structural | ❌ | ❌ | 0 |
| Q14 | 3 | structural | ❌ | ✅ | 349 |
| Q15 | 3 | structural | ❌ | ❌ | 212 |
| Q16 | 2 | hybrid | ❌ | ❌ | 35 |
| Q17 | 2 | hybrid | ❌ | ❌ | 33 |
| Q18 | 2 | hybrid | ❌ | ❌ | 31 |
| Q19 | 3 | hybrid | ❌ | ❌ | 28 |
| Q20 | 3 | hybrid | ❌ | ❌ | 326 |

---

## 2. Latency Breakdown

Average retrieval latency per query type:

| Query Type | Avg Latency (ms) | Min (ms) | Max (ms) |
|---|---|---|---|
| hybrid | 90 | 28 | 326 |
| structural | 376 | 212 | 907 |
| **Overall** | **281** | **28** | **907** |

---

## 4. Summary & DOD Compliance

| Criterion | Target | Actual | Status |
|---|---|---|---|
| Hit Rate @5 (overall) | ≥ 0.60 | 0.150 | ❌ FAIL |
| Retrieval Latency | < 1000 ms | 281 ms | ✅ PASS |
