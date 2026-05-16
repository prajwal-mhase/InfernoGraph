# Round 2 Scale Plan

Target: scale beyond 100M tokens while preserving low-token inference quality.

Plan:

1. Move retrieval and path expansion to TigerGraph-native traversals.
2. Add asynchronous evidence compression workers.
3. Add caching tiers for hot question patterns and community summaries.
4. Introduce model routing and budget guardrails by query class.
5. Add continuous benchmark gates for token/cost/accuracy regression control.
