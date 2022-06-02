import numpy as np
from math import ceil

nPosAbs = 876
nNegAbs = 13254-876

all_evals = []

precision = [0.97]
evals = []
for metric_val in precision:
    # calculate confidence interval
    SE = 1.96 * np.sqrt(metric_val * (1 - metric_val) / (nPosAbs + nNegAbs))

    # three decimal points
    metric_val = ceil(metric_val * 1000) / 1000
    SE = ceil(SE * 1000) / 1000
    evals.append('$' + str(metric_val) + ' \pm ' + str(SE) + '$')
all_evals.append(evals)

recall = [0.72]
evals = []
for metric_val in recall:
    # calculate confidence interval
    SE = 1.96 * np.sqrt(metric_val * (1 - metric_val) / (nPosAbs + nNegAbs))

    # three decimal points
    metric_val = ceil(metric_val * 1000) / 1000
    SE = ceil(SE * 1000) / 1000
    evals.append('$' + str(metric_val) + ' \pm ' + str(SE) + '$')
all_evals.append(evals)

accuracy = [0.63]
evals = []
for metric_val in accuracy:
    # calculate confidence interval
    SE = 1.96 * np.sqrt(metric_val * (1 - metric_val) / (nPosAbs + nNegAbs))

    # three decimal points
    metric_val = ceil(metric_val * 1000) / 1000
    SE = ceil(SE * 1000) / 1000
    evals.append('$' + str(metric_val) + ' \pm ' + str(SE) + '$')
all_evals.append(evals)

auc = [0.66]
evals = []
for metric_val in auc:
    # calculate confidence interval
    pxxy = metric_val / (2 - metric_val)
    pxyy = (2 * metric_val ** 2) / (1 + metric_val)
    SE = 1.96 * np.sqrt((metric_val*(1-metric_val) + (nPosAbs-1)*(pxxy-metric_val**2)
                  + (nNegAbs-1)*(pxyy-metric_val**2)) / (nPosAbs*nNegAbs))

    # three decimal points
    metric_val = ceil(metric_val * 1000) / 1000
    SE = ceil(SE * 1000) / 1000
    evals.append('$'+str(metric_val) + ' \pm ' + str(SE) + '$')
all_evals.append(evals)

print(np.array(all_evals).T)