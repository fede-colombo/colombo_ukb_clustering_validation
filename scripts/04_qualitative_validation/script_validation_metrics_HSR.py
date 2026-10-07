# -*- coding: utf-8 -*-
"""
QUALITATIVE VALIDATION IN HSR - metrics consistent with the actual design of the analysis.

The qualitative validation does NOT use a fixed training/test split: best_nclust_confounds()
runs a 2-fold CV repeated 100 times (RepeatedKFold, random_state=42) on the whole HSR sample.
The value reported in the paper (71%) is 1 - mean normalized stability, NOT an accuracy:
normalized stability = classification error / error with random labels.

Part A - re-runs the same 200 folds with k = 2 and, for each one, computes:
         true accuracy, balanced accuracy, kappa, ARI, AMI, majority-cluster
         baseline and normalized stability (to check that the 71% is reproduced).
Part B - per-patient comparison between the labels of the qualitative validation
         (tr_lab saved in Labels_HSR_qualitative_validation.csv) and those of the
         external validation.

@author: Federica Colombo

"""

import os
import pickle as pkl
import numpy as np
import pandas as pd
from scipy import stats
from scipy.optimize import linear_sum_assignment
from sklearn.model_selection import RepeatedKFold
from sklearn.metrics import (zero_one_loss, accuracy_score, balanced_accuracy_score,
                             cohen_kappa_score, confusion_matrix, adjusted_rand_score,
                             adjusted_mutual_info_score)
from neureval.utils import kuhn_munkres_algorithm

# ---------------- TO ADAPT ----------------
data_dir = "/path/to/project/data"
out_dir =  "/path/to/project/results/qualitative_validation"
database = 'data_HSR.csv'
covariates_file = 'covariates_HSR.csv'
findbestclust_file = os.path.join(out_dir, "findbestclust_qualitative_validation.pkl")
qual_labels_file = os.path.join(out_dir, "Labels_HSR_qualitative_validation.csv")
# HSR KMeans labels from the EXTERNAL validation (columns ID, KMeans_HSR) saved by the
# previous script; None to skip Part B.
external_labels_file = "/path/to/project/results/labels_both_HSR_external_validation.csv"
nclust, nfold, iter_cv = 2, 2, 100        # as in the original script
# ------------------------------------------

data = pd.read_csv(os.path.join(data_dir, database), sep=';')
cov = pd.read_csv(os.path.join(data_dir, covariates_file), sep=';')
modalities = {'gm_thickness': data.iloc[:, 1:]}
covariates = {'gm_thickness': cov.iloc[:, 1:]}
print(f"N HSR = {len(data)}")

with open(findbestclust_file, "rb") as f:
    fbc = pkl.load(f)
fbc.clust_method.n_clusters = nclust
print("Clustering:", fbc.clust_method, "| Classifier:", fbc.class_method)

# ===================== PART A: same folds as NeuReval =====================
kfold = RepeatedKFold(n_splits=nfold, n_repeats=iter_cv, random_state=42)
rows = []
for i, (tr_idx, val_idx) in enumerate(kfold.split(data)):
    train_cor, test_cor = fbc.GLMcorrection_confounds(modalities, covariates, tr_idx, val_idx)
    tr_misc, modelfit, lab_tr, X_tr = fbc.train(train_cor)
    X_ts = np.concatenate([np.array(test_cor[m]) for m in test_cor], axis=1)
    if fbc.preproc_method is not None:
        X_ts = fbc.preproc_method.transform(X_ts)

    clust_ts = fbc.clust_method.fit_predict(X_ts)                  # KMeans on the validation fold
    class_ts = modelfit.predict(X_ts)                              # SVM trained on the other fold
    perm = kuhn_munkres_algorithm(np.int32(class_ts), np.int32(clust_ts))
    miscl = zero_one_loss(class_ts, perm)

    # normalized stability as in NeuReval (after saving the predictions:
    # rndlabels_traineval retrains the same classifier object)
    rnd = fbc.rndlabels_traineval(X_tr, X_ts, lab_tr, perm)
    ns = miscl / rnd if rnd > 0 else 1.0

    rows.append({
        'fold': i, 'n_val': len(val_idx),
        'accuracy': accuracy_score(perm, class_ts),
        'balanced_accuracy': balanced_accuracy_score(perm, class_ts),
        'cohen_kappa': cohen_kappa_score(perm, class_ts),
        'ARI': adjusted_rand_score(perm, class_ts),
        'AMI': adjusted_mutual_info_score(perm, class_ts),
        'majority_class_baseline': np.bincount(perm).max() / len(perm),
        'n_small_cluster': np.bincount(perm, minlength=2).min(),
        'SVM_single_class': len(np.unique(class_ts)) == 1,
        'normalized_stability': ns,
        '1_minus_NS': 1 - ns,
    })

res = pd.DataFrame(rows)
res.to_csv(os.path.join(out_dir, "metrics_per_fold_HSR_qualitative_validation.csv"), sep=';', index=False)

def mean_ci(x):
    x = np.asarray(x, dtype=float)
    h = stats.t.ppf(0.975, len(x) - 1) * x.std() / np.sqrt(len(x))
    return x.mean(), h

summary = {}
for col in ['accuracy', 'balanced_accuracy', 'cohen_kappa', 'ARI', 'AMI',
            'majority_class_baseline', 'n_small_cluster', 'normalized_stability', '1_minus_NS']:
    m, h = mean_ci(res[col])
    summary[col] = {'mean': m, 'ci95_half_width': h, 'min': res[col].min(), 'max': res[col].max()}
summary = pd.DataFrame(summary).T
summary.loc['prop_folds_accuracy_above_baseline'] = [
    np.mean(res['accuracy'] > res['majority_class_baseline']), np.nan, np.nan, np.nan]
summary.loc['prop_folds_SVM_single_class'] = [res['SVM_single_class'].mean(), np.nan, np.nan, np.nan]
print("\n", summary.round(3))
print("\nCheck: mean 1 - NS should reproduce the value reported in the paper (~0.71).")
summary.to_csv(os.path.join(out_dir, "summary_metrics_HSR_qualitative_validation.csv"), sep=';')

# ============ PART B: same patients in the same clusters? ============
if external_labels_file is not None:
    qual = pd.read_csv(qual_labels_file, sep=None, engine='python')
    id_col = qual.columns[1] if qual.columns[0].startswith('Unnamed') else qual.columns[0]
    qual = qual.rename(columns={id_col: 'ID'})[['ID', 'Labels']]
    print(qual.head())
    ext = pd.read_csv(external_labels_file, sep=';')[['ID', 'KMeans_HSR']]
    merged = qual.merge(ext, on='ID', how='inner')
    print(f"\nPatients in common: {len(merged)} (out of {len(qual)} and {len(ext)})")

    a, b = merged['KMeans_HSR'].to_numpy(), merged['Labels'].to_numpy()
    cm_raw = confusion_matrix(a, b)
    r, c = linear_sum_assignment(-cm_raw)
    mapping = {cc: rr for rr, cc in zip(r, c)}
    b_al = np.array([mapping[x] for x in b])
    print("Matrix (rows = external, columns = qualitative):\n", confusion_matrix(a, b_al))
    res_B = {'N': len(a), 'agreement_%': accuracy_score(a, b_al),
             'cohen_kappa': cohen_kappa_score(a, b_al),
             'ARI': adjusted_rand_score(a, b), 'AMI': adjusted_mutual_info_score(a, b)}
    for k, v in res_B.items():
        print(f"{k:>14} = {v:.3f}" if isinstance(v, float) else f"{k:>14} = {v}")
    pd.DataFrame.from_dict(res_B, orient='index', columns=['value']).to_csv(
        os.path.join(out_dir, "summary_external_vs_qualitative.csv"), sep=';')
