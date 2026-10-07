# -*- coding: utf-8 -*-
"""
Created on Mon Jul 21 10:39:00 2025

@author: Federica Colombo

Behavioural PLS analysis with distance-dependent cross-validation
Analysis of cognition terms vs Cohen's d cortical thickness
"""

import numpy as np
import pandas as pd
import pyls
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns
from netneurotools import datasets, stats, plotting
from scipy.stats import zscore, pearsonr, ttest_ind
from scipy.spatial.distance import squareform, pdist, cdist
from nilearn.datasets import fetch_atlas_schaefer_2018
from enigmatoolbox.permutation_testing import rotate_parcellation

# Set the plotting style
plt.rcParams['font.family'] = 'Arial'
plt.rcParams['font.size'] = 20


# ======================================================================= #
#  DIVERGING BAR PLOT FUNCTION (fixed: adaptive bar heights)
# ======================================================================= #

def plot_diverging_loadings(
    neg_loadings, neg_errors, neg_names,
    pos_loadings, pos_errors, pos_names,
    title="Significant Cognitive Processes",
    out_path=None
):
    """
    Plot a diverging horizontal bar chart where negative and positive
    loadings are shown side by side.  Bar heights are automatically
    scaled so both columns span the same total figure height,
    regardless of how many terms each side contains.

    Parameters
    ----------
    neg_loadings : array-like  (sorted most-negative first)
    neg_errors   : array-like
    neg_names    : array-like
    pos_loadings : array-like  (sorted most-positive first)
    pos_errors   : array-like
    pos_names    : array-like
    title        : str
    out_path     : str or None  – if given, figure is saved here
    """
    n_neg   = len(neg_loadings)
    n_pos   = len(pos_loadings)
    n_total = max(n_neg, n_pos)          # height reference (in "units")

    # Each bar occupies 1 unit in the reference (larger) group.
    # The smaller group gets proportionally taller bars so both columns
    # cover exactly n_total units.
    h_neg = n_total / n_neg if n_neg > 0 else 1.0
    h_pos = n_total / n_pos if n_pos > 0 else 1.0

    def bar_centres(n, h):
        return np.array([i * h + h / 2 for i in range(n)])

    neg_y = bar_centres(n_neg, h_neg)
    pos_y = bar_centres(n_pos, h_pos)

    # Figure height scales with total number of rows
    bar_unit_in = 0.38
    fig_h = max(5, n_total * bar_unit_in + 0.5)
    fig, ax = plt.subplots(figsize=(11, fig_h))

    neg_color = '#4f88ba'
    pos_color = '#c0395c'
    capsize   = 3

    if n_neg > 0:
        ax.barh(neg_y, neg_loadings, height=h_neg * 0.82,
                xerr=neg_errors, color=neg_color, capsize=capsize,
                error_kw=dict(elinewidth=1))

    if n_pos > 0:
        ax.barh(pos_y, pos_loadings, height=h_pos * 0.82,
                xerr=pos_errors, color=pos_color, capsize=capsize,
                error_kw=dict(elinewidth=1))

    # --- Labels -----------------------------------------------------------
    max_abs = np.nanmax(np.abs(
        np.concatenate([neg_loadings if n_neg else [0],
                        pos_loadings if n_pos else [0]])
    ))
    x_lim = max_abs * 1.15

    ax.set_xlim(-x_lim * 1.55, x_lim * 1.55)

    for i, (y, name) in enumerate(zip(neg_y, neg_names)):
        ax.text(-x_lim * 1.58, y, name,
                va='center', ha='right', fontsize=14, color='#222222')

    for i, (y, name) in enumerate(zip(pos_y, pos_names)):
        ax.text(x_lim * 1.58, y, name,
                va='center', ha='left', fontsize=14, color='#222222')

    # --- Cosmetics --------------------------------------------------------
    ax.axvline(0, color='black', linewidth=0.9)
    ax.set_ylim(0, n_total)
    ax.set_yticks([])
    ax.set_xlabel("Loadings", fontsize=14)
    ax.set_title(
        f"{title} (n={n_neg + n_pos})",
        fontsize=14, fontweight='bold', pad=10
    )

    #neg_patch = mpatches.Patch(color=neg_color, label='Negative loadings')
    #pos_patch = mpatches.Patch(color=pos_color, label='Positive loadings')
    #ax.legend(handles=[neg_patch, pos_patch],
             # loc='lower center', bbox_to_anchor=(0.5, -0.06),
              #ncol=2, frameon=False, fontsize=11)

    for spine in ['top', 'right', 'left']:
        ax.spines[spine].set_visible(False)

    plt.tight_layout()

    if out_path:
        plt.savefig(out_path, dpi=300, bbox_inches='tight')
        print(f"  Figure saved to {out_path}")

    return fig, ax


# ======================================================================= #
#  DISTANCE-DEPENDENT CROSS-VALIDATION FUNCTIONS
# ======================================================================= #

def distance_dependent_cv_pls(X, Y, coords, n_components=1):
    """
    Perform distance-dependent cross-validation for PLS.

    For each brain region as source node:
    - Training set: 75% closest regions
    - Test set: remaining 25% regions
    """
    n_regions = X.shape[0]
    n_train   = int(np.ceil(n_regions * 0.75))

    train_corrs = np.zeros(n_regions)
    test_corrs  = np.zeros(n_regions)

    dist_matrix = cdist(coords, coords, metric='euclidean')

    for source_node in range(n_regions):
        distances     = dist_matrix[source_node, :]
        sorted_indices = np.argsort(distances)

        train_idx = sorted_indices[:n_train]
        test_idx  = sorted_indices[n_train:]

        X_train = zscore(X[train_idx, :], axis=0)
        Y_train = zscore(Y[train_idx, :], axis=0)
        X_test  = zscore(X[test_idx, :],  axis=0)
        Y_test  = zscore(Y[test_idx, :],  axis=0)

        pls_train = pyls.behavioral_pls(X_train, Y_train,
                                        n_boot=0, n_perm=0,
                                        test_split=0)

        train_x_scores = pls_train['x_scores'][:, n_components - 1]
        train_y_scores = pls_train['y_scores'][:, n_components - 1]
        train_corrs[source_node] = pearsonr(train_x_scores, train_y_scores)[0]

        test_x_scores = X_test @ pls_train['x_weights'][:, n_components - 1]
        test_y_scores = Y_test @ pls_train['y_weights'][:, n_components - 1]
        test_corrs[source_node] = pearsonr(test_x_scores, test_y_scores)[0]

    return train_corrs, test_corrs


def permutation_test_distance_cv(X, Y, coords, n_perm=1000, n_components=1, seed=None):
    """
    Assess significance of distance-dependent CV using random permutations.
    """
    if seed is not None:
        np.random.seed(seed)

    print("Computing empirical cross-validation...")
    train_corrs, test_corrs = distance_dependent_cv_pls(X, Y, coords, n_components)
    emp_mean_test = np.mean(test_corrs)

    null_test_corrs = np.zeros((n_perm, len(test_corrs)))

    print(f"Computing null distribution with {n_perm} permutations...")
    for i in range(n_perm):
        perm_idx  = np.random.permutation(Y.shape[0])
        Y_perm    = Y[perm_idx, :]
        _, test_corrs_perm = distance_dependent_cv_pls(X, Y_perm, coords, n_components)
        null_test_corrs[i, :] = test_corrs_perm

        if (i + 1) % 100 == 0:
            print(f"  Permutation {i+1}/{n_perm}")

    null_mean_test = np.mean(null_test_corrs, axis=1)
    p_value = (1 + np.sum(null_mean_test >= emp_mean_test)) / (1 + n_perm)

    print(f"\nEmpirical mean test correlation: {emp_mean_test:.4f}")
    print(f"P-value: {p_value:.4f}")

    return train_corrs, test_corrs, null_test_corrs, p_value


# ======================================================================= #
#  PATHS & PARAMETERS
# ======================================================================= #

path_cognition = '/path/to/project/data'                    # folder with parcellated Neurosynth maps
path_cohend    = '/path/to/project/results/neurosynth'      # folder with the Cohen's d map
out_path       = '/path/to/project/results/neurosynth'      # output folder
path_coords    = '/path/to/atlases/dkt_coord.csv'           # DKT region centroid coordinates (MNI)

nnodes = 62

coords = pd.read_csv(path_coords, sep=';')
coords_l     = np.array(coords[coords['hemi'] == 'L'][['x.mni', 'y.mni', 'z.mni']])
coords_r     = np.array(coords[coords['hemi'] == 'R'][['x.mni', 'y.mni', 'z.mni']])
coords_array = np.array(coords[['x.mni', 'y.mni', 'z.mni']])
nspins = 10000

print("Spin permutations for spatial autocorrelation...")
spins = rotate_parcellation(coords_l, coords_r, nrot=nspins)
spins = spins.astype(int)

# Load data
cognition_data  = pd.read_csv(path_cognition + '/dkt_parcellated_neurosynth.csv', sep=';')
cognition_names = np.array(cognition_data.iloc[:, 0])
cognition_data  = cognition_data.iloc[:, 1:].transpose()

cohend_data  = pd.read_csv(path_cohend + '/CT_cohend.csv', sep=';')
cohend_names = np.array(cohend_data.iloc[:, 0])
cohend_data  = cohend_data.iloc[:, 2]


# ======================================================================= #
#  PREPARE DATA
# ======================================================================= #

X = zscore(np.array(cognition_data))
Y = np.array(cohend_data).reshape(-1, 1)


# ======================================================================= #
#  FULL PLS MODEL
# ======================================================================= #

print("\n" + "=" * 60)
print("FULL PLS MODEL")
print("=" * 60)

pls_result = pyls.behavioral_pls(
    X, Y,
    n_boot=nspins, n_perm=nspins, permsamples=spins,
    test_split=0, seed=42
)
pyls.save_results(out_path + '/pls_cognition_results_CT_UKB_test_global_CT.hdf5', pls_result)

lv = 0
cv = pls_result["singvals"] ** 2 / np.sum(pls_result["singvals"] ** 2)
p  = pls_result['permres']['pvals']

print(f"\nPLS Results:")
print(f"  Covariance explained (LV{lv}): {cv[lv] * 100:.2f}%")
print(f"  P-value: {p[lv]:.6f}")

# Scatter plot of scores
plt.ion()
plt.figure(figsize=(6, 5))
sns.regplot(x=pls_result['x_scores'][:, lv], y=pls_result['y_scores'][:, lv],
            scatter=False)
plt.scatter(pls_result['x_scores'][:, lv], pls_result['y_scores'][:, lv])
plt.xlabel('Cognition scores')
plt.ylabel("Cohen's d cortical thickness")
corr = pearsonr(pls_result['x_scores'][:, lv], pls_result['y_scores'][:, lv])[0]
plt.title(f'r = {corr:.3f}, p = {p[lv]:.4f}')
plt.tight_layout()
plt.savefig(out_path + '/scatter_scores_cognition_CT_UKB_test_global_CT.png', dpi=300)


# ======================================================================= #
#  COGNITION LOADINGS – DIVERGING BAR PLOT (fixed)
# ======================================================================= #

xload = pyls.behavioral_pls(Y, X, n_boot=nspins, n_perm=0, test_split=0)

ci_lower = xload["bootres"]["y_loadings_ci"][:, lv, 0]
ci_upper = xload["bootres"]["y_loadings_ci"][:, lv, 1]
err      = (ci_upper - ci_lower) / 2

# Select only significant loadings (CI does not include 0)
significant_mask = (ci_lower > 0) | (ci_upper < 0)
significant_idx  = np.where(significant_mask)[0]

print(f"\nSignificant cognitive processes: {len(significant_idx)} out of {len(cognition_names)}")

if len(significant_idx) == 0:
    print("No significant loadings found!")
else:
    # Sort by loading value (most negative first → most positive last)
    sorted_sig_idx = sorted(significant_idx,
                            key=lambda i: xload["y_loadings"][i, lv])

    # Keep only top 10% (keep all by default; adjust multiplier as needed)
    n_top      = max(1, int(np.ceil(1.0 * len(sorted_sig_idx))))
    top_sig_idx = sorted_sig_idx[:n_top]

    selected_loadings = np.array(xload["y_loadings"][top_sig_idx, lv])
    selected_errors   = np.array(err[top_sig_idx])
    selected_names    = np.array(
        [cognition_names[i].replace('_', ' ') for i in top_sig_idx]
    )

    # Split into negative / positive
    neg_mask = selected_loadings < 0
    pos_mask = selected_loadings > 0

    neg_loadings = selected_loadings[neg_mask]
    neg_errors   = selected_errors[neg_mask]
    neg_names    = selected_names[neg_mask]

    pos_loadings = selected_loadings[pos_mask]
    pos_errors   = selected_errors[pos_mask]
    pos_names    = selected_names[pos_mask]

    # Sort: negatives most-negative first; positives most-positive first
    neg_order = np.argsort(neg_loadings)           # ascending (most neg first)
    pos_order = np.argsort(pos_loadings)[::-1]     # descending (most pos first)

    neg_loadings = neg_loadings[neg_order]
    neg_errors   = neg_errors[neg_order]
    neg_names    = neg_names[neg_order]

    pos_loadings = pos_loadings[pos_order]
    pos_errors   = pos_errors[pos_order]
    pos_names    = pos_names[pos_order]

    # --- Draw figure with adaptive bar heights ---
    plot_diverging_loadings(
        neg_loadings, neg_errors, neg_names,
        pos_loadings, pos_errors, pos_names,
        title="Significant Cognitive Processes",
        out_path=out_path + '/bar_pls_cognition_diverging_CT_UKB_test_global_CT.png'
    )

    # Print summary
    print(f"\nTop significant cognitive processes (n={len(top_sig_idx)}):")
    for idx in top_sig_idx:
        print(f"  {cognition_names[idx]}: {xload['y_loadings'][idx, lv]:.4f} "
              f"[{ci_lower[idx]:.4f}, {ci_upper[idx]:.4f}]")


# ======================================================================= #
#  DISTANCE-DEPENDENT CROSS-VALIDATION
# ======================================================================= #

print("\n" + "=" * 60)
print("DISTANCE-DEPENDENT CROSS-VALIDATION")
print("=" * 60)

train_corrs, test_corrs, null_test_corrs, p_value = \
    permutation_test_distance_cv(X, Y, coords_array,
                                 n_perm=1000, n_components=1, seed=42)

np.savetxt(out_path + '/pls_train_UKB_test_global_CT.csv',    train_corrs,    delimiter=',')
np.savetxt(out_path + '/pls_test_UKB_test_global_CT.csv',     test_corrs,     delimiter=',')
np.savetxt(out_path + '/pls_testnull_UKB_test_global_CT.csv', null_test_corrs, delimiter=',')

# Boxplot
# fig, ax = plt.subplots(figsize=(5, 5))
# sns.boxplot(data=[train_corrs, test_corrs, np.mean(null_test_corrs, axis=1)], ax=ax)
# ax.set_xticklabels(['Train', 'Test', 'Null'])
# ax.set_ylabel('Score correlation')
# ax.set_title(f'Distance-dependent CV (p = {p_value:.6f})')
# ax.axhline(y=0, color='gray', linestyle='--', alpha=0.5)
# plt.tight_layout()
# plt.savefig(out_path + '/boxplot_pls_cv.png', dpi=300)

fig, ax = plt.subplots(figsize=(6.4, 6.4))  # proportions closer to the reference figure
sns.boxplot(data=[train_corrs, test_corrs, np.mean(null_test_corrs, axis=1)], ax=ax)
ax.set_xticklabels(['Train', 'Test', 'Null'], fontsize=14)
ax.set_ylabel('Score correlation', fontsize=14)
ax.set_title(f'Distance-dependent CV (p = {p_value:.6f})', fontsize=14)
ax.axhline(y=0, color='gray', linestyle='--', alpha=0.5)
ax.set_ylim(-0.5, 0.75)  # same Y range as the reference figure
ax.tick_params(axis='y', labelsize=12)
plt.tight_layout()
plt.savefig(out_path + '/boxplot_pls_cv_UKB_test_global_CT.png', dpi=300, bbox_inches='tight')

# Distribution diagnostic plot
fig, axes = plt.subplots(1, 2, figsize=(12, 5))

axes[0].hist(train_corrs, bins=20, alpha=0.6, label='Train', color='blue')
axes[0].hist(test_corrs,  bins=20, alpha=0.6, label='Test',  color='orange')
axes[0].axvline(np.mean(train_corrs), color='blue',   linestyle='--', linewidth=2)
axes[0].axvline(np.mean(test_corrs),  color='orange', linestyle='--', linewidth=2)
axes[0].set_xlabel('Correlation')
axes[0].set_ylabel('Frequency')
axes[0].set_title('Distribution of CV correlations')
axes[0].legend()

axes[1].hist(np.mean(null_test_corrs, axis=1), bins=30, alpha=0.6,
             label='Null', color='gray')
axes[1].axvline(np.mean(test_corrs), color='red', linestyle='--',
                linewidth=2, label='Empirical')
axes[1].set_xlabel('Mean correlation')
axes[1].set_ylabel('Frequency')
axes[1].set_title(f'Test vs Null (p = {p_value:.4f})')
axes[1].legend()

plt.tight_layout()
plt.savefig(out_path + '/distributions_UKB_test_global_CT.png', dpi=300)

# Summary
print(f"\nCross-validation summary:")
print(f"  Mean train correlation:  {np.mean(train_corrs):.4f} ± {np.std(train_corrs):.4f}")
print(f"  Mean test correlation:   {np.mean(test_corrs):.4f} ± {np.std(test_corrs):.4f}")
print(f"  Median test correlation: {np.median(test_corrs):.4f}")
print(f"  Full model correlation:  {corr:.4f}")
print(f"  P-value (test vs null):  {p_value:.6f}")

print("\n" + "=" * 60)
print("ANALYSIS COMPLETE")
print("=" * 60)
print(f"\nResults saved to: {out_path}")
