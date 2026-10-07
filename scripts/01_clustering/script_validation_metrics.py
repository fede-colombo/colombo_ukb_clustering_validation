# -*- coding: utf-8 -*-
"""
Script to compute additional agreement and validation metrics (accuracy, balanced accuracy, kappa, ARI, NMI, AMI)

author: @ Federica Colombo
"""

import os
import pickle as pkl
import numpy as np
import pandas as pd
from sklearn.metrics import (zero_one_loss, accuracy_score, balanced_accuracy_score,
                             cohen_kappa_score, confusion_matrix, adjusted_rand_score,
                             normalized_mutual_info_score, adjusted_mutual_info_score)
from neureval.utils import kuhn_munkres_algorithm
import re

# Set paths
main_path = "/path/to/project/results"                                          # main analysis folder (HSR)
path_data_all = "/path/to/project/data/data_all.csv"                             # UKB training + HSR (same order)
path_cov_all = "/path/to/project/data/covariates_all.csv"
findbestclust_file = "/path/to/project/data/findbestclust_model.pkl"
path_cohend_ref = "/path/to/project/data/cohend_reference.csv"                   # Cohen's d map (region order and names)
training_dim = 842                               # number of UKB training subjects
tag = 'UKB_test_validation'
# -------------------------------------------------------------------------------

data_all = pd.read_csv(os.path.join(main_path, path_data_all), sep=';').drop(['train_test'], axis=1)
cov_all = pd.read_csv(os.path.join(main_path, path_cov_all), sep=';').drop(['train_test'], axis=1)
tr_idx = list(range(0, training_dim))
val_idx = list(range(training_dim, len(data_all)))
modalities_val = {'gm_thickness': data_all.iloc[:, 1:]}
covariates_val = {'gm_thickness': cov_all.iloc[:, 1:-1]}
print(f"UKB training: {len(tr_idx)}, UKB validation: {len(val_idx)}")

with open(os.path.join(main_path, findbestclust_file), "rb") as f:
    fbc = pkl.load(f)
fbc.clust_method.n_clusters = 2

# --- Same steps as evaluate_confounds() ---
train_cor_dic, test_cor_dic = fbc.GLMcorrection_confounds(modalities_val, covariates_val, tr_idx, val_idx)
tr_misc, modelfit, labels_tr, X_train_cor = fbc.train(train_cor_dic)

X_test_cor = np.concatenate([np.array(test_cor_dic[m]) for m in test_cor_dic], axis=1)
if fbc.preproc_method is not None:
    X_test_cor = fbc.preproc_method.transform(X_test_cor)

clustlab_ts = fbc.clust_method.fit_predict(X_test_cor)           # KMeans on HSR
classlab_ts = modelfit.predict(X_test_cor)                        # SVM trained on UKB
bestperm = kuhn_munkres_algorithm(np.int32(classlab_ts), np.int32(clustlab_ts))

acc_neureval = 1 - zero_one_loss(classlab_ts, bestperm)
print(f"Training ACC: {1 - tr_misc:.3f} | Test ACC: {acc_neureval:.3f}")

# --- All metrics from the same pair of label vectors ---
y_ref, y_pred = bestperm, classlab_ts      # reference = KMeans HSR, prediction = SVM UKB
cm = confusion_matrix(y_ref, y_pred)
print("Confusion matrix (rows = KMeans UKB, columns = SVM UKB):\n", cm)

res = {
    'accuracy': accuracy_score(y_ref, y_pred),
    'balanced_accuracy': balanced_accuracy_score(y_ref, y_pred),
    'cohen_kappa': cohen_kappa_score(y_ref, y_pred),
    'ARI': adjusted_rand_score(y_ref, y_pred),
    'NMI': normalized_mutual_info_score(y_ref, y_pred),
    'AMI': adjusted_mutual_info_score(y_ref, y_pred),
    'majority_class_baseline': np.bincount(y_ref).max() / len(y_ref),
    'n_KMeans_cluster0': int(np.sum(y_ref == 0)), 'n_KMeans_cluster1': int(np.sum(y_ref == 1)),
    'n_SVM_cluster0': int(np.sum(y_pred == 0)), 'n_SVM_cluster1': int(np.sum(y_pred == 1)),
}
for k, v in res.items():
    print(f"{k:>24} = {v:.3f}" if isinstance(v, float) else f"{k:>24} = {v}")

assert abs(res['accuracy'] - acc_neureval) < 1e-9
assert res['cohen_kappa'] <= res['accuracy'] + 1e-9

pd.DataFrame.from_dict(res, orient='index', columns=['value']).to_csv(
    os.path.join(main_path, f"summary_metrics_{tag}.csv"), sep=';')
pd.DataFrame(cm, index=['KMeans_0', 'KMeans_1'], columns=['SVM_0', 'SVM_1']).to_csv(
    os.path.join(main_path, f"confusion_matrix_{tag}.csv"), sep=';')
ids = data_all.iloc[val_idx, 0].reset_index(drop=True)
pd.DataFrame({'ID': ids, 'KMeans_UKB': y_ref, 'SVM_pred_UKB': y_pred}).to_csv(
    os.path.join(main_path, f"labels_both_{tag}.csv"), sep=';', index=False)

# --- Computation of the shift in values from UKB to HSR --- #

raw = modalities_val['gm_thickness']
diff = (raw.iloc[val_idx].mean() - raw.iloc[tr_idx].mean()) / raw.iloc[tr_idx].std()

def norm(s):
    return re.sub(r'gyrus|lobule|cortex|\s', '', s.lower())

def key_ukb(col):   # "Mean thickness of X (left hemisphere) | Instance 2" -> ("L", "x")
    region, hemi = re.search(r'of (\w+) \((left|right) hemisphere\)', col).groups()
    return ('L' if hemi == 'left' else 'R', norm(region))

shift = {key_ukb(c): v for c, v in diff.items()}

# Order and names from the Cohen's d map
ref = pd.read_csv(path_cohend_ref, sep=';')
keys = [(v.split(' ', 1)[0], norm(v.split(' ', 1)[1])) for v in ref['variable']]
missing = [k for k in keys if k not in shift]
assert not missing, f"Regions not found: {missing}"

out = pd.DataFrame({'variable': ref['variable'], 'shift': [shift[k] for k in keys]})
print(out.head(10))
out.to_csv(os.path.join(main_path, f"shift_UKB_HSR_{tag}.csv"), sep=';', index=False)
