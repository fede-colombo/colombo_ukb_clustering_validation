# -*- coding: utf-8 -*-
"""
Created on Sat Nov  8 12:11:06 2025

Multilinear regression analysis with dominance analyses
Updated to add: p_spin(CV) significance for the distance-dependent
cross-validation (following Hansen et al., Nat Neurosci 2022, PLS
out-of-sample significance procedure) and the mean cross-validated
(out-of-sample) R2 as headline fit statistic.

@author: Federica Colombo
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
#import pyls
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.gridspec import GridSpec
import scipy.stats._stats_py as _stats_py
if not hasattr(_stats_py, '_chk2_asarray'):
    def _chk2_asarray(a, b, axis):
        if axis is None:
            a = np.ravel(a)
            b = np.ravel(b)
            outaxis = 0
        else:
            a = np.asarray(a)
            b = np.asarray(b)
            outaxis = axis
        if a.ndim == 0:
            a = np.atleast_1d(a)
        if b.ndim == 0:
            b = np.atleast_1d(b)
        return a, b, outaxis
    _stats_py._chk2_asarray = _chk2_asarray

import sys, types
_stats_mod = types.ModuleType('scipy.stats.stats')
_stats_mod._chk2_asarray = _stats_py._chk2_asarray
sys.modules['scipy.stats.stats'] = _stats_mod
from netneurotools.stats import get_dominance_stats
from sklearn.linear_model import LinearRegression
from scipy.spatial.distance import squareform, pdist
from matplotlib.colors import ListedColormap

############################################################################

## FUNCTIONS

def get_reg_r_sq_adjusted(X, y):
    lin_reg = LinearRegression()
    lin_reg.fit(X, y)
    yhat = lin_reg.predict(X)
    SS_Residual = sum((y - yhat) ** 2)
    SS_Total = sum((y - np.mean(y)) ** 2)
    r_squared = 1 - (float(SS_Residual)) / SS_Total
    adjusted_r_squared = 1 - (1 - r_squared) * \
        (len(y) - 1) / (len(y) - X.shape[1] - 1)
    return adjusted_r_squared

def get_reg_r_sq(X, y):
    lin_reg = LinearRegression()
    lin_reg.fit(X, y)
    yhat = lin_reg.predict(X)
    SS_Residual = sum((y - yhat) ** 2)
    SS_Total = sum((y - np.mean(y)) ** 2)
    r_squared = 1 - (float(SS_Residual)) / SS_Total
    return r_squared

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
        
        if metric == 'rsq_adj':
            # get r^2 adjusted of train set
            train_metric.append(get_reg_r_sq_adjusted(X[train_idx, :], y[train_idx]))

        elif metric == 'corr':
            rho, _ = pearsonr(mdl.predict(X[train_idx, :]), y[train_idx])
            train_metric.append(rho)

        yhat = mdl.predict(X[test_idx, :])
        
        if metric == 'rsq':
            # get r^2 of test set
            SS_Residual = sum((y[test_idx] - yhat) ** 2)
            SS_Total = sum((y[test_idx] - np.mean(y[test_idx])) ** 2)
            r_squared = 1 - (float(SS_Residual)) / SS_Total
            test_metric.append(r_squared)
            
        if metric == 'rsq_adj':
            # get r^2 adjusted of test set
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


def get_cv_r_pval(X, y, coords, spins, nspins_cv=1000, train_pct=.75,
                   metric='rsq'):
    '''
    Spin-test significance of the distance-dependent cross-validation
    (out-of-sample) performance, following the procedure used by
    Hansen et al. (Nat Neurosci, 2022) for the PLS out-of-sample
    correlation (Fig. 5d): the *entire* distance-dependent CV procedure
    is repeated on spatial-autocorrelation-preserving permutations
    (spins) of the predictor matrix X, and the empirical mean
    out-of-sample metric (R2 or correlation) is compared against the
    resulting null distribution of mean out-of-sample metrics.

    Note this is a distinct test from get_reg_r_pval: get_reg_r_pval
    tests the significance of the *full model fit* (in-sample, using
    all regions at once), whereas get_cv_r_pval tests the significance
    of the model's *out-of-sample generalizability* as assessed by the
    distance-dependent cross-validation.

    X = n x p matrix of input variables
    y = n x 1 matrix of output variable
    coords = n x 3 coordinates of each observation
    spins = n x nspins array of spin-permutation indices, as generated
            by rotate_parcellation (only the first nspins_cv columns
            are used)
    nspins_cv = number of permutations for this test. Hansen et al.
                used 1,000 repetitions for the analogous PLS test
                (fewer than the 10,000 typically used for the main
                spin test), since each permutation here requires
                refitting the full distance-dependent CV (one
                regression per region).
    metric = {'rsq', 'corr'}

    Returns
    -------
    emp_mean : empirical mean out-of-sample metric
    null_mean : nspins_cv x 1 null distribution of mean out-of-sample
                metrics
    pval : p_spin(CV)
    '''
    if nspins_cv > spins.shape[1]:
        raise ValueError('nspins_cv cannot exceed the number of spin '
                          'permutations available in `spins`.')

    _, emp_test_metric = cv_slr_distance_dependent(X, y, coords,
                                                     train_pct, metric)
    emp_mean = np.mean(emp_test_metric)

    null_mean = np.zeros((nspins_cv, ))
    for s in range(nspins_cv):
        X_null = X[spins[:, s], :]  # spatially-permuted predictor maps
        _, null_test_metric = cv_slr_distance_dependent(X_null, y, coords,
                                                          train_pct, metric)
        null_mean[s] = np.mean(null_test_metric)

    pval = (1 + np.sum(null_mean >= emp_mean)) / (nspins_cv + 1)
    return emp_mean, null_mean, pval


###############################################################################

## SET UP

# Define paths
out_path          = '/path/to/project/results/receptors'           # output folder
path_coords       = '/path/to/atlases/dkt_coord.csv'                # DKT region centroid coordinates (MNI)
path_receptors    = '/path/to/project/data/receptor_data_matrix.csv'
path_cohend       = '/path/to/project/results/receptors/CT_cohend.csv'
path_colourmap    = '/path/to/visualization/colourmap.csv'

# Set up parcellations for spatial autocorrelation
coords = pd.read_csv(path_coords, sep=';')
coords_l = np.array(coords[coords['hemi']=='L'][['x.mni', 'y.mni', 'z.mni']])
coords_r = np.array(coords[coords['hemi']=='R'][['x.mni', 'y.mni', 'z.mni']])
coords = np.array(coords[['x.mni', 'y.mni', 'z.mni']])

# Set up number of permutations (spins) for spatial autocorrelation and number of regions (nnodes)
nspins = 10000
nspins_cv = 1000  # number of permutations for the CV significance test
                   # (fewer than nspins, since each one refits the full CV)
nnodes = 62

# Generate spin permutations
print("Spin permutations for spatial autocorrelation...")
spins = rotate_parcellation(coords_l, coords_r, nrot=nspins)
spins = spins.astype(int)

# Load receptor data
receptor_data = pd.read_csv(path_receptors, delimiter=';')
receptor_names = receptor_data.iloc[:,1:].columns.values
receptor_data_raw = np.array(receptor_data.iloc[:,1:])

# Load effect size data
cohend_data = pd.read_csv(path_cohend, delimiter=';')
region_names = list(cohend_data['variable'])
cohend_data_raw = np.array(cohend_data.drop(columns=['index','variable'])).flatten()

# colourmaps
cmap = np.genfromtxt(path_colourmap, delimiter=';')
cmap_div = ListedColormap(cmap)
cmap_seq = ListedColormap(cmap[128:, :])

# Initialize data structures
model_metrics = {}
train_metric = np.zeros(nnodes)  # 1D vector
test_metric = np.zeros(nnodes)   # 1D vector

# Dominance analysis
print("Dominance analysis...")
m, _ = get_dominance_stats(zscore(receptor_data_raw),
                           zscore(cohend_data_raw), n_jobs=-1)
model_metrics['cohend'] = m

# Cross validate the model (out-of-sample correlation, for the boxplot)
train_metric, test_metric = \
    cv_slr_distance_dependent(zscore(receptor_data_raw), zscore(cohend_data_raw),
                              coords, .75, metric='corr')

# Cross validate the model (out-of-sample R2, headline fit statistic)
train_rsq, test_rsq = \
    cv_slr_distance_dependent(zscore(receptor_data_raw), zscore(cohend_data_raw),
                              coords, .75, metric='rsq')
    
# Cross validate the model (out-of-sample R2 adjusted, headline fit statistic)
train_rsq_adj, test_rsq_adj = \
    cv_slr_distance_dependent(zscore(receptor_data_raw), zscore(cohend_data_raw),
                              coords, .75, metric='rsq_adj')
    
mean_cv_rsq = np.mean(test_rsq)
median_cv_rsq = np.median(test_rsq)
mean_cv_rsq_adj = np.mean(test_rsq_adj)
median_cv_rsq_adj = np.median(test_rsq_adj)
insample_rsq_adj = get_reg_r_sq_adjusted(zscore(receptor_data_raw), zscore(cohend_data_raw))

# Get p-value of the full model fit (in-sample), via spin test
print("Spin-test significance of the full model fit...")
model_pval = get_reg_r_pval(zscore(receptor_data_raw),
                            zscore(cohend_data_raw),
                            spins, nspins)

# Get p-value of the distance-dependent CV (out-of-sample performance),
# via spin test on the entire CV procedure (Hansen et al. PLS approach)
print("Spin-test significance of the distance-dependent cross-validation...")
cv_emp_mean_corr, cv_null_mean_corr, cv_pval = get_cv_r_pval(
    zscore(receptor_data_raw), zscore(cohend_data_raw), coords, spins,
    nspins_cv=nspins_cv, train_pct=.75, metric='corr')

# Get total dominance
dominance = model_metrics['cohend']["total_dominance"]

# Save results
np.save(out_path + '/dominance_receptors_ct_UKB_test.npy', dominance)
np.save(out_path + '/receptors_ct_cv_train_UKB_test.npy', train_metric)
np.save(out_path + '/receptors_ct_cv_test_UKB_test.npy', test_metric)
np.save(out_path + '/receptors_ct_model_pval_UKB_test.npy', model_pval)
np.save(out_path + '/receptors_ct_cv_pval_UKB_test.npy', cv_pval)
np.save(out_path + '/receptors_ct_cv_test_rsq_UKB_test.npy', test_rsq)
np.save(out_path + '/receptors_ct_cv_mean_rsq_UKB_test.npy', mean_cv_rsq)
np.save(out_path + '/receptors_ct_cv_test_rsq_adj_UKB_test.npy', test_rsq_adj)
np.save(out_path + '/receptors_ct_cv_mean_rsq_adj_UKB_test.npy', mean_cv_rsq_adj)

# Print statistics
print(f"\n{'='*60}")
print(f"In-sample adjusted R2 (full model)             = {insample_rsq_adj:.3f}")
print(f"p_spin(model)  [significance of full model fit] = {model_pval:.4f}")
print(f"{'-'*60}")
print(f"Mean CV out-of-sample R2                = {mean_cv_rsq:.3f} "
      f"(median = {median_cv_rsq:.3f})")
print(f"Mean CV out-of-sample adjusted R2                = {mean_cv_rsq_adj:.3f} "
      f"(median = {median_cv_rsq_adj:.3f})")
print(f"Mean CV out-of-sample correlation                = {cv_emp_mean_corr:.3f}")
print(f"p_spin(CV)     [significance of CV out-of-sample] = {cv_pval:.4f}")
print(f"{'='*60}\n")


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
plt.savefig(out_path + '/bar_total_dominance_receptors_UKB_test.png', dpi=300, bbox_inches='tight')
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
plt.savefig(out_path + '/bar_relative_dominance_receptors_UKB_test.png', dpi=300, bbox_inches='tight')


# Plot cross validation
data_to_plot = [train_metric, test_metric]
labels = ['Train', 'Test']

fig, ax = plt.subplots(figsize=(8, 6))

# Horizontal boxplot
bp = ax.boxplot(data_to_plot,
                vert=False,  # Horizontal
                tick_labels=labels,
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

# # --- Significance annotation for the distance-dependent CV (p_spin(CV)) ---
if cv_pval < 0.001:
    p_label = 'p$_{spin}$(CV) < 0.001'
else:
    p_label = f'p$_{{spin}}$(CV) = {cv_pval:.3f}'

# sig_symbol = ('***' if cv_pval < 0.001 else
#               '**' if cv_pval < 0.01 else
#               '*' if cv_pval < 0.05 else 'n.s.')

# # With data_to_plot = [train_metric, test_metric], the "Test" box is the second
# # one created: its fliers are bp['fliers'][1], its whiskers are
# # bp['whiskers'][2] and bp['whiskers'][3] (2 whiskers per box, in order).
# train_flier_x = bp['fliers'][0].get_xdata()
# train_whisker_x = np.concatenate([bp['whiskers'][0].get_xdata(),
#                                     bp['whiskers'][1].get_xdata()])
# test_flier_x = bp['fliers'][1].get_xdata()
# test_whisker_x = np.concatenate([bp['whiskers'][2].get_xdata(),
#                                    bp['whiskers'][3].get_xdata()])

# all_x = np.concatenate([train_flier_x, train_whisker_x,
#                          test_flier_x, test_whisker_x])
# x_min, x_max = all_x.min(), all_x.max()

# bracket_y = 2.22
# tick_height = 0.04
# label_y = bracket_y + 0.03

# ax.plot([x_min, x_max], [bracket_y, bracket_y], color='black',
#         linewidth=1.2, clip_on=False)
# ax.plot([x_min, x_min], [bracket_y, bracket_y - tick_height],
#         color='black', linewidth=1.2, clip_on=False)
# ax.plot([x_max, x_max], [bracket_y, bracket_y - tick_height],
#         color='black', linewidth=1.2, clip_on=False)

# ax.text((x_min + x_max) / 2, label_y, sig_symbol, ha='center', va='bottom',
#         fontsize=15, fontweight='bold')

ax.set_title(f'Distance-dependent cross-validation ({p_label})',
             fontsize=13)

plt.tight_layout()
plt.savefig(out_path + '/boxplot_receptors_cv_UKB_test.png', dpi=300, bbox_inches='tight')
plt.show()

# Print statistics
print(f"\n{'='*50}")
print(f"Train - Mean: {np.mean(train_metric):.3f}, Median: {np.median(train_metric):.3f}")
print(f"Test  - Mean: {np.mean(test_metric):.3f}, Median: {np.median(test_metric):.3f}")
print(f"p_spin(CV) = {cv_pval:.4f}")
print(f"{'='*50}\n")

################################################################################################

# Compare dominance across receptor classes
exc = ['5HT2a', '5HT4', '5HT6', 'D1', 'mGluR5', 'A4B2', 'M1', 'NMDA']
inh = ['5HT1a', '5HT1b', 'CB1', 'D2', 'GABAa', 'H3', 'MOR']
mami = ['5HT1a', '5HT1b', '5HT2a', '5HT4', '5HT6', '5HTT', 'D1',
        'D2', 'DAT', 'H3', 'NET']
nmami = list(set(receptor_names) - set(mami))
metab = ['5HT1a', '5HT1b', '5HT2a', '5HT4', '5HT6', 'CB1', 'D1',
         'D2', 'H3', 'M1', 'mGluR5', 'MOR']
iono = ['A4B2', 'GABAa', 'NMDA']
gspath = ['5HT4', '5HT6', 'D1']
gipath = ['CB1', 'D2', 'H3', '5HT1a', '5HT1b', 'MOR']
gqpath = ['5HT2a', 'mGluR5', 'M1']

i_exc = np.array([list(receptor_names).index(i) for i in exc])
i_inh = np.array([list(receptor_names).index(i) for i in inh])
i_mami = np.array([list(receptor_names).index(i) for i in mami])
i_nmami = np.array([list(receptor_names).index(i) for i in nmami])
i_metab = np.array([list(receptor_names).index(i) for i in metab])
i_iono = np.array([list(receptor_names).index(i) for i in iono])
i_gs = np.array([list(receptor_names).index(i) for i in gspath])
i_gi = np.array([list(receptor_names).index(i) for i in gipath])
i_gq = np.array([list(receptor_names).index(i) for i in gqpath])

classes = [[i_exc, i_inh], [i_mami, i_nmami],
           [i_metab, i_iono], [i_gs, i_gi, i_gq]]
class_names = [['excitatory', 'inhibitory'], ['monoamine', 'non-monoamine'],
               ['metabotropic', 'ionotropic'], ['Gs', 'Gi', 'Gq']]

# Set colors for each receptor class
colors = [
    ['#6BAED6', '#74C476'],  # excitatory (blue), inhibitory (green)
    ['#E74C3C', '#F39C12'],  # monoamine (red), non-monoamine (orange)
    ['#9B59B6', '#E59866'],  # metabotropic (purple), ionotropic (peach)
    ['#5DADE2', '#C0392B', '#F39C12']  # Gs (light blue), Gi (dark red), Gq (orange)
]

plt.ion()
fig, axs = plt.subplots(1, 4, figsize=(16, 4))
axs = axs.ravel()

for i in range(len(classes)):
    print(class_names[i])
    d = [dominance[classes[i][j]].flatten() for j in range(len(classes[i]))]

    # Statistical tests
    print(ttest_ind(d[0], d[1]))
    if len(d) > 2:
        print(ttest_ind(d[0], d[2]))
        print(ttest_ind(d[1], d[2]))

    # Violin plots
    parts = axs[i].violinplot(d, positions=range(len(d)), widths=0.7,
                               showmeans=False, showmedians=False, showextrema=False)

    for j, pc in enumerate(parts['bodies']):
        pc.set_facecolor(colors[i][j])
        pc.set_edgecolor('none')  # Remove outlines
        pc.set_alpha(0.6)

    for j in range(len(d)):
        x = np.random.normal(j, 0.04, size=len(d[j]))  # Jitter
        axs[i].scatter(x, d[j], alpha=0.8, s=25, color=colors[i][j],
                      edgecolors='none')  # Filled dots without outline

    axs[i].set_xticks(range(len(class_names[i])))
    axs[i].set_xticklabels(class_names[i], fontsize=11)
    axs[i].set_ylabel('dominance', fontsize=11)
    axs[i].spines['top'].set_visible(False)
    axs[i].spines['right'].set_visible(False)
    axs[i].grid(axis='y', alpha=0.3, linestyle=':', linewidth=0.8)

plt.tight_layout()
plt.savefig(out_path + '/stripplot_receptors_classes_UKB_test.png', dpi=300, bbox_inches='tight')
plt.show()
