"""
Source: Jared Dunnmon
"""
# Run stats tests for xmodal paper
import rpy2
import pickle
import logging
import numpy as np
import os
import pandas as pd

# Setting up conversion from numpy to R
import rpy2.robjects.numpy2ri
rpy2.robjects.numpy2ri.activate()

# Setting up logger
logging.getLogger().setLevel(logging.INFO)
# Getting R packages

from rpy2.robjects.packages import importr
proc = importr('pROC')

# Setting up function to get delong statistics
def delong_p(scores_1, scores_2, labels_1, labels_2=None):
    if labels_2 is None:
        labels_2 = labels_1
    # uses two-sided delong test
    roc1 = proc.roc(labels_1, scores_1)
    roc2 = proc.roc(labels_2, scores_2)
    test = proc.roc_test(roc1, roc2, method='delong')
    test_dict = dict(zip(test.names, list(test)))
    auc1, auc2 = [float(a) for a in test_dict['estimate']]
    p = [float(a) for a in (test_dict['p.value'])][0]
    return p, auc1, auc2