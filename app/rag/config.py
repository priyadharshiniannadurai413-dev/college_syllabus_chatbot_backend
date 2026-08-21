"""
Hybrid Search Configuration
----------------------------
Configurable parameters for the hybrid retrieval system
(vector search + keyword search + RRF fusion).
"""

# ── Retrieval counts ─────────────────────────────────────────────────────────
# How many results to fetch from each search method before fusion.
HYBRID_VECTOR_TOP_K = 8
HYBRID_KEYWORD_TOP_K = 8

# How many results to return after RRF fusion.
HYBRID_FINAL_TOP_K = 4

# ── RRF constant ─────────────────────────────────────────────────────────────
# Standard Reciprocal Rank Fusion constant. Higher k = less weight to top ranks.
HYBRID_RRF_K = 60
