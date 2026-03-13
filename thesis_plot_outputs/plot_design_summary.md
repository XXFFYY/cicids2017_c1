# Thesis plot plan based on current results

## Recommended mainline
1. **Stage 1 — communication is not monotonic with performance**
   - Use `fig01_stage1_non_monotonic_tpr_vs_comm.png`
   - Optional companion: `fig02_stage1_non_monotonic_pr_vs_comm.png`

2. **Stage 2 — identify promising low-communication baselines**
   - Use `fig03_stage2_partition_tpr_bars.png`
   - Use `fig04_stage2_partition_tradeoff.png`
   - Current strongest single-agent deployment point: **SUPERVISED + Best_L0**
     - TPR@1%FPR = **0.4537** at communication cost **1**

3. **Stage 3 — optimize the collaborative baseline**
   - Use `fig05_stage3_pair_gain.png`
   - Use `fig06_stage3_aap_optimization_path.png`
   - Largest pair gain over single appears under **AAP**
     - Pair gain vs single = **+0.1086**
     - Pair gain vs best full-L2 = **+0.0639**
   - Current strongest pair point: **SUPERVISED + Best_Pair_L2**
     - TPR@1%FPR = **0.4518** at communication cost **2**

## Interpretation aligned with your current results
- If you want the cleanest **deployment baseline**, write the story around **supervised + Best_L0**.
- If you want the clearest **collaborative optimization story**, write the story around **AAP + Best_Pair_L2**, because it improves over the best full-L2 family while using lower communication.
- Therefore, your paper can explicitly separate:
  - **single-agent optimum**
  - **pair-collaboration optimum**

## Appendix suggestion
- If you keep the robustness section, use `figA1_appendix_robustness_delta_tpr.png` as a compact appendix figure.
