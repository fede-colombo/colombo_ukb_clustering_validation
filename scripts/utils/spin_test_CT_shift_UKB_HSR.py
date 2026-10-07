# -*- coding: utf-8 -*-
"""
SPIN TEST BETWEEN CORTICAL THICKNESS PROFILES (Cohen's d between clusters)
UKB training, UKB test, HSR + UKB-HSR shift map

Same set-up as the dominance analysis script (DKT coordinates, ENIGMA
rotate_parcellation, 10,000 spins). For each pair of maps: Pearson and Spearman,
parametric p and two-tailed p_spin; partial correlation UKB vs HSR controlling for
the shift (with p_spin); results tables and scatter plots.

@Author: Federica Colombo
"""

import itertools
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import pearsonr, spearmanr
from enigmatoolbox.permutation_testing import rotate_parcellation

# ---------------- TO ADAPT ----------------
base = '/path/to/project/results/spin_test'
out_path = '/path/to/project/results/spin_test'
coords_file = '/path/to/atlases/dkt_coord.csv'

maps_files = {
    'UKB_training': '/path/to/project/results/spin_CT_profiles/CT_cohend_training.csv',
    'UKB_test':     '/path/to/project/results/spin_CT_profiles/CT_cohend_test.csv',
    'HSR':          '/path/to/project/results/spin_CT_profiles/CT_cohend_HSR.csv',
}
# UKB-HSR shift map (columns: variable; shift). None to exclude it
#shift_file = '/path/to/project/results/shift_UKB_HSR.csv'   # adapt to the folder where you saved it
shift_file = None

nspins = 10000
seed = 42
# ------------------------------------------

np.random.seed(seed)   # rotate_parcellation uses np.random: makes the spins reproducible

# --- Coordinates (same region order as in the maps: all L first, then all R) ---
coords = pd.read_csv(coords_file, sep=';')
print(coords.head())   # check that the region order matches the one in the maps
coords_l = np.array(coords[coords['hemi'] == 'L'][['x.mni', 'y.mni', 'z.mni']])
coords_r = np.array(coords[coords['hemi'] == 'R'][['x.mni', 'y.mni', 'z.mni']])
nnodes = len(coords_l) + len(coords_r)

# --- Maps ---
maps = {}
region_names = None
for name, f in maps_files.items():
    df = pd.read_csv(f, sep=';')
    if region_names is None:
        region_names = list(df['variable'])
    else:
        assert list(df['variable']) == region_names, f"Different region order in {name}"
    maps[name] = df['CohenD'].to_numpy(dtype=float)

if shift_file is not None:
    sh = pd.read_csv(shift_file, sep=';')
    assert list(sh['variable']) == region_names, "Different region order in the shift map"
    maps['shift_UKB_HSR'] = sh['shift'].to_numpy(dtype=float)

assert all(len(v) == nnodes for v in maps.values()), "Number of regions differs from the coordinates"

# --- Spins ---
print("Spin permutations...")
spins = rotate_parcellation(coords_l, coords_r, nrot=nspins).astype(int)   # (nnodes, nspins)

def spin_p(x, y, spins, method='pearson'):
    f = pearsonr if method == 'pearson' else spearmanr
    emp = f(x, y)[0]
    null = np.array([f(x[spins[:, s]], y)[0] for s in range(spins.shape[1])])
    p = (1 + np.sum(np.abs(null) >= np.abs(emp))) / (len(null) + 1)   # two-tailed
    return emp, p, null

rows, nulls = [], {}
for a, b in itertools.combinations(maps.keys(), 2):
    x, y = maps[a], maps[b]
    r, p_spin_r, null_r = spin_p(x, y, spins, 'pearson')
    rho, p_spin_rho, _ = spin_p(x, y, spins, 'spearman')
    rows.append({'map_1': a, 'map_2': b,
                 'pearson_r': r, 'p_param_pearson': pearsonr(x, y)[1], 'p_spin_pearson': p_spin_r,
                 'spearman_rho': rho, 'p_param_spearman': spearmanr(x, y)[1], 'p_spin_spearman': p_spin_rho})
    nulls[(a, b)] = null_r

res = pd.DataFrame(rows)
pd.set_option('display.width', 200)
print("\n", res.round(4).to_string(index=False))
res.to_csv(out_path + '/spin_test_CT_shift.csv', sep=';', index=False)

# --- Partial correlation UKB vs HSR controlling for the shift ---
# p_spin: the UKB map is rotated and the partial correlation recomputed (shift and HSR fixed)
def resid(v, z):
    A = np.c_[np.ones_like(z), z]
    return v - A @ np.linalg.lstsq(A, v, rcond=None)[0]

def partial_r(x, y, z):
    return np.corrcoef(resid(x, z), resid(y, z))[0, 1]

if 'shift_UKB_HSR' in maps:
    z = maps['shift_UKB_HSR']
    prow = []
    for a in ['UKB_training', 'UKB_test']:
        x, y = maps[a], maps['HSR']
        emp = partial_r(x, y, z)
        null = np.array([partial_r(x[spins[:, s]], y, z) for s in range(nspins)])
        p = (1 + np.sum(np.abs(null) >= np.abs(emp))) / (nspins + 1)
        prow.append({'map_1': a, 'map_2': 'HSR', 'controlling_for': 'shift_UKB_HSR',
                     'pearson_r_zero_order': pearsonr(x, y)[0],
                     'partial_r': emp, 'p_spin_partial': p})
    pres = pd.DataFrame(prow)
    print("\n", pres.round(4).to_string(index=False))
    pres.to_csv(out_path + '/partial_correlation_UKB_HSR_given_shift.csv', sep=';', index=False)

# --- Map descriptives (useful for the sign and magnitude of the effects) ---
desc = pd.DataFrame(maps).describe().T[['mean', 'std', 'min', 'max']]
desc['n_regions_negative'] = [(v < 0).sum() for v in maps.values()]
print("\n", desc.round(2))
desc.to_csv(out_path + '/descriptives_CT_shift.csv', sep=';')

# --- Figures: scatter plot for each pair with r and p_spin (3-column grid) ---
pairs = list(nulls.keys())
ncol = 3
nrow = int(np.ceil(len(pairs) / ncol))
fig, axs = plt.subplots(nrow, ncol, figsize=(4.5 * ncol, 4.2 * nrow))
axs = np.atleast_1d(axs).ravel()
for ax, (a, b) in zip(axs, pairs):
    x, y = maps[a], maps[b]
    row = res[(res.map_1 == a) & (res.map_2 == b)].iloc[0]
    ax.scatter(x, y, s=25, alpha=0.8, color='steelblue', edgecolors='none')
    m, q = np.polyfit(x, y, 1)
    xs = np.linspace(x.min(), x.max(), 50)
    ax.plot(xs, m * xs + q, color='black', linewidth=1.2)
    p_lab = 'p$_{spin}$ < 0.001' if row.p_spin_pearson < 0.001 else f'p$_{{spin}}$ = {row.p_spin_pearson:.3f}'
    ax.set_title(f'r = {row.pearson_r:.2f}, {p_lab}', fontsize=11)
    xl = 'shift (SD)' if a == 'shift_UKB_HSR' else "Cohen's d"
    yl = 'shift (SD)' if b == 'shift_UKB_HSR' else "Cohen's d"
    ax.set_xlabel(f"{a} ({xl})")
    ax.set_ylabel(f"{b} ({yl})")
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
for ax in axs[len(pairs):]:
    ax.axis('off')
plt.tight_layout()
plt.savefig(out_path + '/scatter_spin_test_CT_shift.png', dpi=300, bbox_inches='tight')
plt.show()

np.save(out_path + '/spins_CT_shift.npy', spins)
