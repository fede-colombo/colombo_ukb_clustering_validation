# -*- coding: utf-8 -*-
"""
Greedy matching of controls to cases within a single database
Matching variables: BMI, Age, Sex
Cases are kept fixed, controls are reduced until matched

@author: Admin
"""

import numpy as np
import pandas as pd
from scipy.stats import ttest_ind, chi2_contingency, fisher_exact

# ----------------------------
# USER SETTINGS
# ----------------------------
DATA_EXCEL_PATH = "path/to/data_to_match.csv"
DATA_SHEET = 0

GROUP_COL = "CASE"
CASE_VALUE = 1
CONTROL_VALUE = 0

AGE_COL = "AGE"
SEX_COL = "SEX"

ALPHA = 0.05
N_ATTEMPTS = 30
MAX_SUBSETS = 15
MAX_REMOVALS = 2000
RANDOM_SEED = 42

OUTPUT_EXCEL_PATH = "path/to/matched_data_all_subsets_age_sex.xlsx"

DB_FILTER_QUERY = None  # es: "Age >= 18"

CANDIDATE_REMOVALS_PER_STEP = 60
START_FRAC_RANGE = (0.70, 1.00)
MIN_SUBSET_N = 10


# ----------------------------
# HELPERS
# ----------------------------
def welch_t_pvalue(x, y):
    res = ttest_ind(x, y, equal_var=False, nan_policy="omit")
    return float(res.pvalue) if res is not None else np.nan


def sex_pvalue(subset_sex, target_sex):
    s = pd.Series(subset_sex).dropna().astype(str)
    t = pd.Series(target_sex).dropna().astype(str)

    if len(s) < 2 or len(t) < 2:
        return np.nan

    cats = sorted(set(s.unique()).union(set(t.unique())))
    if len(cats) < 2:
        return 1.0

    s_counts = s.value_counts().reindex(cats, fill_value=0).to_numpy()
    t_counts = t.value_counts().reindex(cats, fill_value=0).to_numpy()
    table = np.vstack([s_counts, t_counts])

    if table.shape == (2, 2):
        chi2, p_chi, _, expected = chi2_contingency(table, correction=False)
        if (expected < 5).any():
            _, p_fisher = fisher_exact(table)
            return float(p_fisher)
        return float(p_chi)

    chi2, p, _, _ = chi2_contingency(table, correction=False)
    return float(p)


def compute_all_pvalues(controls_df, cases_df, idxs):
    subset = controls_df.loc[idxs]
    
    p_age = welch_t_pvalue(
        subset[AGE_COL].to_numpy(dtype=float),
        cases_df[AGE_COL].to_numpy(dtype=float),
    )
    p_sex = sex_pvalue(subset[SEX_COL], cases_df[SEX_COL])

    return {"Age": p_age, "Sex": p_sex}


def passes(pvals, alpha):
    for v in pvals.values():
        if np.isnan(v) or v < alpha:
            return False
    return True


def combined_score(pvals):
    vals = [v for v in pvals.values() if not np.isnan(v)]
    return min(vals) if vals else -np.inf


def greedy_reduce_until_pass(rng, controls_df, cases_df, start_idxs):
    idxs = np.array(start_idxs, dtype=int)

    if len(idxs) < MIN_SUBSET_N:
        return None

    pvals = compute_all_pvalues(controls_df, cases_df, idxs)

    if passes(pvals, ALPHA):
        return idxs, pvals

    removals = 0
    while removals < MAX_REMOVALS:
        if len(idxs) <= MIN_SUBSET_N:
            return None

        if passes(pvals, ALPHA):
            return idxs, pvals

        current_score = combined_score(pvals)
        k = min(CANDIDATE_REMOVALS_PER_STEP, len(idxs))
        candidates = rng.choice(idxs, size=k, replace=False)

        best_remove = None
        best_pvals = None
        best_score = current_score

        for rid in candidates:
            trial_idxs = idxs[idxs != rid]
            if len(trial_idxs) < MIN_SUBSET_N:
                continue

            trial_pvals = compute_all_pvalues(
                controls_df, cases_df, trial_idxs
            )
            trial_score = combined_score(trial_pvals)

            if trial_score > best_score + 1e-12:
                best_score = trial_score
                best_remove = rid
                best_pvals = trial_pvals
            elif abs(trial_score - best_score) <= 1e-12 and best_pvals is not None:
                if np.nansum(list(trial_pvals.values())) > np.nansum(list(best_pvals.values())):
                    best_remove = rid
                    best_pvals = trial_pvals

        if best_remove is None:
            return None

        idxs = idxs[idxs != best_remove]
        pvals = best_pvals
        removals += 1

    return None


# ----------------------------
# MAIN
# ----------------------------
def main():
    rng = np.random.default_rng(RANDOM_SEED)
    
    print("Loading dataset...")
    data = pd.read_csv(DATA_EXCEL_PATH, sep=';')

    if DB_FILTER_QUERY:
        data = data.query(DB_FILTER_QUERY).copy()

    cases = data[data[GROUP_COL] == CASE_VALUE].copy()
    controls = data[data[GROUP_COL] == CONTROL_VALUE].copy()

    cases = cases.dropna(subset=[AGE_COL, SEX_COL])
    controls = controls.dropna(subset=[AGE_COL, SEX_COL])

    print(f"Cases: {len(cases)}")
    print(f"Controls (initial): {len(controls)}")

    valid = []
    control_index = controls.index.to_numpy()

    for attempt in range(1, N_ATTEMPTS + 1):
        if len(valid) >= MAX_SUBSETS:
            break

        frac = rng.uniform(*START_FRAC_RANGE)
        start_size = max(MIN_SUBSET_N, int(frac * len(controls)))
        start_idxs = rng.choice(control_index, size=start_size, replace=False)

        print(f"Attempt {attempt}/{N_ATTEMPTS} | start_size={start_size}")

        res = greedy_reduce_until_pass(
            rng, controls, cases, start_idxs
        )
        if res is None:
            continue

        idxs, pvals = res
        valid.append((len(idxs), combined_score(pvals), pvals, idxs))
        valid.sort(key=lambda x: (-x[0], -x[1]))
        valid = valid[:MAX_SUBSETS]

    if not valid:
        print("No valid subsets found.")
        return

    unique = {}
    for n, score, pvals, idxs in valid:
        key = tuple(sorted(idxs.tolist()))
        if key not in unique or n > unique[key][0]:
            unique[key] = (n, score, pvals, idxs)

    valid = sorted(unique.values(), key=lambda x: (-x[0], -x[1]))

    print(f"Saving {len(valid)} matched control subsets...")
    with pd.ExcelWriter(OUTPUT_EXCEL_PATH, engine="openpyxl") as writer:
        summary_rows = []
        for i, (n, score, pvals, idxs) in enumerate(valid, start=1):
            summary_rows.append({
                "subset_id": i,
                "n_controls": n,
                "worst_pvalue": score,
                "p_Age": pvals["Age"],
                "p_Sex": pvals["Sex"],
            })
            controls.loc[idxs].to_excel(
                writer, sheet_name=f"controls_subset_{i}", index=False
            )

        pd.DataFrame(summary_rows).to_excel(
            writer, sheet_name="summary", index=False
        )
        cases.to_excel(writer, sheet_name="cases_fixed", index=False)

    print("Done.")
    print("Best subset:")
    print("  N controls =", valid[0][0])
    print("  p-values =", valid[0][2])


if __name__ == "__main__":
    main()
