# -*- coding: utf-8 -*-
"""
Created on Sat Nov  8 12:11:06 2025

@author: Admin
"""
############################################################
### MULTILINEAR REGRESSION MODEL AND DOMINANCE ANALYSIS ###
############################################################

import numpy as np
import abagen
from neuromaps.parcellate import Parcellater
from enigmatoolbox.permutation_testing import rotate_parcellation
from nilearn import image
import nibabel as nib
import pandas as pd
from neuromaps import nulls
from scipy.stats import zscore, pearsonr, ttest_ind, spearmanr
import pyls
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.gridspec import GridSpec
from netneurotools.stats import get_dominance_stats
from sklearn.linear_model import LinearRegression
from scipy.spatial.distance import squareform, pdist
from matplotlib.colors import ListedColormap

############################################################################

## FUNCTIONS

def get_reg_r_sq(X, y):
    lin_reg = LinearRegression()
    lin_reg.fit(X, y)
    yhat = lin_reg.predict(X)
    SS_Residual = sum((y - yhat) ** 2)
    SS_Total = sum((y - np.mean(y)) ** 2)
    r_squared = 1 - (float(SS_Residual)) / SS_Total
    adjusted_r_squared = 1 - (1 - r_squared) * \
        (len(y) - 1) / (len(y) - X.shape[1] - 1)
    return adjusted_r_squared


def cv_slr_distance_dependent(X, y, coords, train_pct=.75, metric='rsq'):
    '''
    cross validates linear regression model using distance-dependent method.
    X = n x p matrix of input variables
    y = n x 1 matrix of output variable
    coords = n x 3 coordinates of each observation
    train_pct (between 0 and 1), percent of observations in training set
    metric = {'rsq', 'corr'}
    '''

    P = squareform(pdist(coords, metric="euclidean"))
    train_metric = []
    test_metric = []

    for i in range(len(y)):
        distances = P[i, :]  # for every node
        idx = np.argsort(distances)

        train_idx = idx[:int(np.floor(train_pct * len(coords)))]
        test_idx = idx[int(np.floor(train_pct * len(coords))):]

        mdl = LinearRegression()
        mdl.fit(X[train_idx, :], y[train_idx])
        if metric == 'rsq':
            # get r^2 of train set
            train_metric.append(get_reg_r_sq(X[train_idx, :], y[train_idx]))

        elif metric == 'corr':
            rho, _ = pearsonr(mdl.predict(X[train_idx, :]), y[train_idx])
            train_metric.append(rho)

        yhat = mdl.predict(X[test_idx, :])
        if metric == 'rsq':
            # get r^2 of test set
            SS_Residual = sum((y[test_idx] - yhat) ** 2)
            SS_Total = sum((y[test_idx] - np.mean(y[test_idx])) ** 2)
            r_squared = 1 - (float(SS_Residual)) / SS_Total
            adjusted_r_squared = 1-(1-r_squared)*((len(y[test_idx]) - 1) /
                                                  (len(y[test_idx]) -
                                                   X.shape[1]-1))
            test_metric.append(adjusted_r_squared)

        elif metric == 'corr':
            rho, _ = pearsonr(yhat, y[test_idx])
            test_metric.append(rho)

    return train_metric, test_metric


def get_perm_p(emp, null):
    return (1 + sum(abs(null - np.mean(null))
                    > abs(emp - np.mean(null)))) / (len(null) + 1)


def get_reg_r_pval(X, y, spins, nspins):
    emp = get_reg_r_sq(X, y)
    null = np.zeros((nspins, ))
    for s in range(nspins):
        null[s] = get_reg_r_sq(X[spins[:, s], :], y)
    return (1 + sum(null > emp))/(nspins + 1)


###############################################################################

## SET UP

# Define output path
out_path = 'path/to/receptors/results'

# Set up parcellations per spatial autocorrelation
coords = pd.read_csv("path/to/atlases/dkt_coord.csv", sep=';')
coords_l = np.array(coords[coords['hemi']=='L'][['x.mni', 'y.mni', 'z.mni']])
coords_r = np.array(coords[coords['hemi']=='R'][['x.mni', 'y.mni', 'z.mni']])
coords = np.array(coords[['x.mni', 'y.mni', 'z.mni']])

# Set up number of permutations (spins) for spatial autocorrelation and number of regions (nnodes)
nspins = 10000
nnodes = 62

# Genereate spin permutations
print("Spin pemrutations for spatial autocorrelation...")
spins = rotate_parcellation(coords_l, coords_r, nrot=nspins)
spins = spins.astype(int)

# Load receptor data
receptor_data = pd.read_csv("path/to/receptor_data_matrix.csv",
                              delimiter=';')
receptor_names = receptor_data.iloc[:,1:].columns.values
receptor_data_raw = np.array(receptor_data.iloc[:,1:])

# Load effect size data
cohend_data = pd.read_csv("path/to/CT_cohend_UKB_55_training.csv",
                         delimiter=';')
region_names = list(cohend_data['variable'])
cohend_data_raw = np.array(cohend_data.drop(columns=['index','variable'])).flatten()


# Initialize data stuctures
model_metrics = {}
train_metric = np.zeros(nnodes)
test_metric = np.zeros(nnodes)

# Dominance analysis
print("Dominance analysis...")
m, _ = get_dominance_stats(zscore(receptor_data_raw),
                           zscore(cohend_data_raw), n_jobs=-1)
model_metrics['cohend'] = m

# Cross validate the model
train_metric, test_metric = \
    cv_slr_distance_dependent(zscore(receptor_data_raw), zscore(cohend_data_raw),
                              coords, .75, metric='corr')

# Get p-value of model
model_pval = get_reg_r_pval(zscore(receptor_data_raw),
                            zscore(cohend_data_raw), 
                            spins, nspins)

# Get total dominance
dominance = model_metrics['cohend']["total_dominance"]

# Save results
np.save(out_path + '/dominance_receptors_ct_55_training.npy', dominance)
np.save(out_path + '/receptors_ct_cv_train_55_training.npy', train_metric)
np.save(out_path + '/receptors_ct_cv_test_55_training.npy', test_metric)
np.save(out_path + '/receptors_ct_model_pval_55_training.npy', model_pval)


## FIGURES

# Total dominance bar plot
plt.ion()
plt.figure(figsize=(12, 6))
plt.bar(np.arange(len(receptor_names)), dominance,
        tick_label=receptor_names)
plt.xticks(rotation=90)
plt.xlabel('Receptors')
plt.ylabel('Total Dominance')
plt.title('Receptor Dominance Analysis')
plt.tight_layout()
plt.savefig(out_path + '/figures/bar_total_dominance_receptors_55_training.png', dpi=300, bbox_inches='tight')
plt.show()

# Relative dominance bar plot
if model_pval >= 0.05:
    dominance[:] = 0

plt.figure(figsize=(12, 6))
dominance_normalized = dominance / np.sum(dominance) if np.sum(dominance) > 0 else dominance

plt.bar(np.arange(len(receptor_names)), dominance_normalized, color='steelblue')
plt.xticks(np.arange(len(receptor_names)), receptor_names, rotation=90)
plt.ylabel('Relative Dominance')
plt.xlabel('Receptors')
plt.title(f'Receptor Dominance (p = {model_pval:.4f})')
plt.tight_layout()
plt.savefig(out_path + '/figures/bar_relative_dominance_receptors_55_training.png', dpi=300, bbox_inches='tight')


# Plot cross validation
data_to_plot = [train_metric, test_metric]
labels = ['Train', 'Test']

fig, ax = plt.subplots(figsize=(8, 6))

# Horizontal boxplot
bp = ax.boxplot(data_to_plot, 
                vert=False,  # Horizontal
                labels=labels,
                patch_artist=True,
                widths=0.6,
                showfliers=True,
                boxprops=dict(facecolor='white', edgecolor='black', linewidth=1.5),
                whiskerprops=dict(color='black', linewidth=1.5),
                capprops=dict(color='black', linewidth=1.5),
                medianprops=dict(color='black', linewidth=2),
                flierprops=dict(marker='D', markerfacecolor='gray', markeredgecolor='black', 
                               markersize=6, alpha=0.6))

ax.axvline(x=0, color='gray', linestyle='--', linewidth=1, alpha=0.5)

ax.set_xlabel("Score correlation (Pearson's r)", fontsize=13, fontweight='normal')
ax.set_xlim(-0.5, 1.0)
ax.tick_params(axis='both', labelsize=12)
ax.grid(axis='x', alpha=0.3, linestyle=':', linewidth=0.8)

ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)

plt.tight_layout()
plt.savefig(out_path + '/figures/boxplot_receptors_cv_55_training.png', dpi=300, bbox_inches='tight')
plt.show()

# Print statistics
print(f"\n{'='*50}")
print(f"Train - Mean: {np.mean(train_metric):.3f}, Median: {np.median(train_metric):.3f}")
print(f"Test  - Mean: {np.mean(test_metric):.3f}, Median: {np.median(test_metric):.3f}")
print(f"{'='*50}\n")
