# Focused summary (v3)

- `Best_L0` remains the strongest low-communication baseline: TPR@1%FPR = 0.4501, cost = 1.000.
- `AAP + Sparse_L2_top2` is the current collaboration anchor: TPR@1%FPR = 0.4277, cost = 2.000, gap to Best_L0 = +0.0224.
- `AAP + Sparse_L2_adaptive_tprfirst` now targets TPR-first routing. Its TPR@1%FPR = 0.3053, cost = 1.678, gap to top2 = -0.1224.
- Compared with full-cost `L2_trimmed_mean`, the adaptive method trades TPR by -0.1175 while reducing communication by 2.322.
- Use the main figure `focused_tpr_vs_cost_v3.png` to present the operational trade-off: baseline, sparse collaboration, adaptive collaboration, and full-cost trimmed fusion.