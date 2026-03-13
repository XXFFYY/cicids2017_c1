# Health-aware routing (option B)

This experiment treats agent health as a **routing signal** rather than directly rewriting the classifier score.

- If the primary pair is healthy enough, keep the primary pair.

- Else, switch to a healthy backup pair.

- If no healthy pair exists, fall back to the best healthy single agent.


## Route summary

| partition_mode | share_primary_pair | share_backup_pair | share_single_fallback | avg_comm_cost_test || --- | --- | --- | --- | --- || AAP | 0.390071 | 0.148936 | 0.460993 | 1.539007 || SUPERVISED | 0.375887 | 0.248227 | 0.375887 | 1.624113 |


## Main results

| method | partition_mode | avg_comm_cost | test_roc_auc | test_pr_auc | test_tpr_at_fpr1 | monitor_fpr | decision_threshold | val_proxy_tpr | selection_note || --- | --- | --- | --- | --- | --- | --- | --- | --- | --- || Global_Best_L0 | AAP | 1.000000 | 0.765421 | 0.740090 | 0.339211 | 0.009730 | 0.000079 | 0.999981 | Agent_0 || Global_Best_Pair_L2 | AAP | 2.000000 | 0.739799 | 0.730741 | 0.322965 | 0.010420 | 0.000397 | 0.999981 | Agent_0+Agent_3|w=0.504,0.496 || HealthAware_PairSwitch_B | AAP | 1.539007 | 0.702441 | 0.679497 | 0.098268 | 0.010020 | 0.000470 | 0.999981 | primary=('Agent_0', 'Agent_3'); backups=[('Agent_2', 'Agent_3'), ('Agent_0', 'Agent_2'), ('Agent_0', 'Agent_1')]; tau_pair=0.6; tau_single=0.5 || Global_Best_L0 | SUPERVISED | 1.000000 | 0.760839 | 0.731698 | 0.300149 | 0.009964 | 0.000583 | 0.999981 | Agent_1 || Global_Best_Pair_L2 | SUPERVISED | 2.000000 | 0.756302 | 0.736405 | 0.452968 | 0.009907 | 0.000326 | 0.999981 | Agent_1+Agent_3|w=0.500,0.500 || HealthAware_PairSwitch_B | SUPERVISED | 1.624113 | 0.747517 | 0.724534 | 0.413204 | 0.010224 | 0.000335 | 0.999981 | primary=('Agent_1', 'Agent_3'); backups=[('Agent_0', 'Agent_1'), ('Agent_1', 'Agent_2'), ('Agent_2', 'Agent_3')]; tau_pair=0.6; tau_single=0.5 |