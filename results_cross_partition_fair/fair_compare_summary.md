# Cross-partition fair comparison summary

## aap
- Best L0: Agent_0 | TPR@1%FPR=0.340482 | PR-AUC=0.740090
- Best Pair (weighted): Agent_0+Agent_2 | weights=[0.5012476786940062, 0.4987523213059939] | TPR@1%FPR=0.449102 | PR-AUC=0.741780
- Best Pair gain vs L0: +0.108621
- Full robust gain vs L0: -0.013675

## random
- Best L0: Agent_0 | TPR@1%FPR=0.405402 | PR-AUC=0.728259
- Best Pair (weighted): Agent_0+Agent_2 | weights=[0.5006972881423294, 0.49930271185767056] | TPR@1%FPR=0.357552 | PR-AUC=0.710875
- Best Pair gain vs L0: -0.047850
- Full robust gain vs L0: -0.082520

## supervised
- Best L0: Agent_3 | TPR@1%FPR=0.453737 | PR-AUC=0.751760
- Best Pair (weighted): Agent_0+Agent_3 | weights=[0.49995589575800353, 0.5000441042419965] | TPR@1%FPR=0.451837 | PR-AUC=0.745489
- Best Pair gain vs L0: -0.001900
- Full robust gain vs L0: -0.020853
