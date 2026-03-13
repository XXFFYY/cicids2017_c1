# Agent Failure Response Summary

## AAP
### Global_Best_L0
- score_flip
  - fallback_best_healthy_single: avg_tpr=0.3947, delta_tpr=+0.0555, avg_monitor_fpr=0.0117, cost=1.0
- score_noise_003
  - fallback_best_healthy_single: avg_tpr=0.3947, delta_tpr=+0.0555, avg_monitor_fpr=0.0117, cost=1.0
- score_zero
  - fallback_best_healthy_single: avg_tpr=0.3947, delta_tpr=+0.0555, avg_monitor_fpr=0.0117, cost=1.0

### Global_Best_Pair_L2
- score_flip
  - switch_backup_pair: avg_tpr=0.4605, delta_tpr=+0.1375, avg_monitor_fpr=0.0104, cost=2.0
  - keep_compromised_pair: avg_tpr=0.2280, delta_tpr=-0.0950, avg_monitor_fpr=0.0100, cost=2.0
  - isolate_to_healthy_partner_single: avg_tpr=0.2194, delta_tpr=-0.1035, avg_monitor_fpr=0.0103, cost=1.0
- score_noise_003
  - switch_backup_pair: avg_tpr=0.4605, delta_tpr=+0.1375, avg_monitor_fpr=0.0104, cost=2.0
  - isolate_to_healthy_partner_single: avg_tpr=0.2194, delta_tpr=-0.1035, avg_monitor_fpr=0.0103, cost=1.0
  - keep_compromised_pair: avg_tpr=0.1050, delta_tpr=-0.2180, avg_monitor_fpr=0.0103, cost=2.0
- score_zero
  - switch_backup_pair: avg_tpr=0.4605, delta_tpr=+0.1375, avg_monitor_fpr=0.0104, cost=2.0
  - isolate_to_healthy_partner_single: avg_tpr=0.2194, delta_tpr=-0.1035, avg_monitor_fpr=0.0103, cost=1.0
  - keep_compromised_pair: avg_tpr=0.2194, delta_tpr=-0.1035, avg_monitor_fpr=0.0103, cost=2.0

## SUPERVISED
### Global_Best_L0
- score_flip
  - fallback_best_healthy_single: avg_tpr=0.3769, delta_tpr=+0.0768, avg_monitor_fpr=0.0099, cost=1.0
- score_noise_003
  - fallback_best_healthy_single: avg_tpr=0.3769, delta_tpr=+0.0768, avg_monitor_fpr=0.0099, cost=1.0
- score_zero
  - fallback_best_healthy_single: avg_tpr=0.3769, delta_tpr=+0.0768, avg_monitor_fpr=0.0099, cost=1.0

### Global_Best_Pair_L2
- score_flip
  - switch_backup_pair: avg_tpr=0.4128, delta_tpr=-0.0401, avg_monitor_fpr=0.0100, cost=2.0
  - isolate_to_healthy_partner_single: avg_tpr=0.3769, delta_tpr=-0.0761, avg_monitor_fpr=0.0099, cost=1.0
  - keep_compromised_pair: avg_tpr=0.2226, delta_tpr=-0.2303, avg_monitor_fpr=0.0098, cost=2.0
- score_noise_003
  - switch_backup_pair: avg_tpr=0.4128, delta_tpr=-0.0401, avg_monitor_fpr=0.0100, cost=2.0
  - isolate_to_healthy_partner_single: avg_tpr=0.3769, delta_tpr=-0.0761, avg_monitor_fpr=0.0099, cost=1.0
  - keep_compromised_pair: avg_tpr=0.2292, delta_tpr=-0.2237, avg_monitor_fpr=0.0101, cost=2.0
- score_zero
  - switch_backup_pair: avg_tpr=0.4128, delta_tpr=-0.0401, avg_monitor_fpr=0.0100, cost=2.0
  - isolate_to_healthy_partner_single: avg_tpr=0.3769, delta_tpr=-0.0761, avg_monitor_fpr=0.0099, cost=1.0
  - keep_compromised_pair: avg_tpr=0.3769, delta_tpr=-0.0761, avg_monitor_fpr=0.0099, cost=2.0
