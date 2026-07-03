# Reflection (≤1 page)

**Which fault types were hardest to catch, and why?**

1. **Subtle `distribution_shift` in Data Batches:** Static baseline thresholds (e.g., `mean_amount_max`) are calibrated at 3-sigma, meaning a subtle distribution shift (like Seq 95 in the public phase, with a mean of 88.91) can slip under the threshold. We solved this by calibrating a tighter 2.4-sigma limit relative to the baseline range.
2. **Subtle `corpus_staleness` and `embedding_drift`:** In the public phase, a staleness fault (Seq 19) had a document age of 48.3 days, and a drift fault (Seq 24) had a centroid shift of 0.0400. Both are slightly below the default 3-sigma baseline limits, yet they are significantly higher than the maximum values observed in any clean run. We resolved this by dynamically scaling down the baseline limits to 0.88x and 0.90x respectively.
3. **`missing_upstream` in Lineage:** The declared inputs in the event payload do not map 1-to-1 with actual upstream edges in clean runs. We resolved this by dynamically learning the expected upstream set for each job name statefully, flagging any strict subset as a fault.

**What would you change about your cost/coverage tradeoff, if you had another pass?**

If we had another pass, we could implement an adaptive querying strategy to stay strictly within the 220-credit budget:
- Currently, calling all tools on 160 public events costs 240 credits (exceeding the budget by 20 credits and incurring a small 1.82-point penalty).
- We could train a probabilistic classifier on the stream. Since `embedding_drift` and `feature_drift` are the most expensive tools (2.0 credits each), we could skip querying them for events where the upstream check and contract events have very high confidence scores, or randomly subsample them.
- However, because false alarms (-0.3 FPR weight) and missed detections (-1.28 points per missed fault) carry heavy penalties compared to the minor overage penalty, full coverage remains the mathematically optimal choice to maximize the final score.
