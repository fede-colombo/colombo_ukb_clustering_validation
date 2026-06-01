# -*- coding: utf-8 -*-
"""
Flexible Multi-Group Post-Hoc Analysis Script with FDR Correction
- Supports BOTH weighted and unweighted analysis
- Uses full matching weights from MatchIt (R) when available
- Handles 2+ groups
- Computes descriptive statistics for each group
- Performs comparisons: ANOVA/t-test, Chi-square/Fisher
- Post-hoc pairwise comparisons with multiple correction methods
- Computes effect sizes: Cohen's d, Cramer's V, eta-squared
- Applies FDR correction for multiple comparisons
- Generates publication-ready Excel tables
"""

import pandas as pd
import numpy as np
from scipy.stats import chi2_contingency, fisher_exact, f_oneway, ttest_ind, kruskal
from statsmodels.stats.weightstats import DescrStatsW
from statsmodels.stats.multitest import multipletests
import statsmodels.api as sm
from itertools import combinations
import warnings
warnings.filterwarnings('ignore')

# ================= CONFIGURATION =================
file_path = "path/to/database_clinical_outcomes.csv"
output_path = "path/to/output/file/comparisons_UKB_clinical_outcomes.xlsx"

cluster_col = 'mdd_hc'  # Group variable (can have 2+ groups)
WEIGHT_COL = None  # Set to column name (e.g., 'weights') for weighted analysis, or None for unweighted

# Post-hoc comparison settings
PERFORM_POSTHOC = True  # Set to True for pairwise comparisons after significant omnibus test
POSTHOC_CORRECTION = 'fdr'  # 'bonferroni', 'holm', or 'fdr'

# ================= LOAD DATA =================
df = pd.read_csv(file_path, sep=';')

# Drop unnecessary columns
for col in ['train_test']:
    if col in df.columns:
        df = df.drop([col], axis=1)

df = df.iloc[:,1:]  # remove first column if it's just an index

# Handle weights
USE_WEIGHTS = WEIGHT_COL is not None and WEIGHT_COL in df.columns
if USE_WEIGHTS:
    df[WEIGHT_COL] = pd.to_numeric(df[WEIGHT_COL], errors='coerce')
    print(f"\n✓ Using WEIGHTED analysis with column: '{WEIGHT_COL}'")
else:
    print(f"\n✓ Using UNWEIGHTED analysis")
    WEIGHT_COL = '__dummy_weight__'
    df[WEIGHT_COL] = 1.0  # Equal weights for unweighted analysis

# Check number of groups
n_groups = df[cluster_col].nunique()
print(f"✓ Detected {n_groups} groups in '{cluster_col}'")
print(f"  Groups: {sorted(df[cluster_col].unique())}")

# ================= HELPER FUNCTIONS =================

def weighted_mean_std(values, weights):
    """Calculate weighted mean and std"""
    if USE_WEIGHTS:
        wstats = DescrStatsW(values, weights=weights)
        return wstats.mean, np.sqrt(wstats.var)
    else:
        return values.mean(), values.std()

def weighted_median_iqr(values, weights):
    """Calculate weighted median and IQR"""
    if USE_WEIGHTS:
        sorted_idx = np.argsort(values)
        sorted_vals = values.iloc[sorted_idx]
        sorted_wts = weights.iloc[sorted_idx]
        cum_wts = np.cumsum(sorted_wts)
        total_wt = cum_wts.iloc[-1]
        def wq(q):
            target = q * total_wt
            idx = np.searchsorted(cum_wts, target)
            return sorted_vals.iloc[min(idx, len(sorted_vals)-1)]
        median = wq(0.5)
        q1 = wq(0.25)
        q3 = wq(0.75)
        return median, q1, q3
    else:
        return values.median(), values.quantile(0.25), values.quantile(0.75)

def cohens_d_weighted(g1, g2, w1=None, w2=None):
    """Cohen's d for pairwise comparison"""
    m1, s1 = weighted_mean_std(g1, w1)
    m2, s2 = weighted_mean_std(g2, w2)
    
    if USE_WEIGHTS:
        n1 = w1.sum()
        n2 = w2.sum()
    else:
        n1 = len(g1)
        n2 = len(g2)
    
    pooled_std = np.sqrt(((n1-1)*s1**2 + (n2-1)*s2**2)/(n1+n2-2))
    return (m1 - m2)/pooled_std if pooled_std>0 else 0

def eta_squared(groups_data, groups_weights):
    """Calculate eta-squared (effect size for ANOVA)"""
    all_vals = np.concatenate([g.values for g in groups_data])
    all_wts = np.concatenate([w.values for w in groups_weights])
    
    if USE_WEIGHTS:
        grand_mean = np.average(all_vals, weights=all_wts)
    else:
        grand_mean = all_vals.mean()
    
    # Between-group sum of squares
    ss_between = 0
    for vals, wts in zip(groups_data, groups_weights):
        if USE_WEIGHTS:
            group_mean = np.average(vals, weights=wts)
            n_eff = wts.sum()
        else:
            group_mean = vals.mean()
            n_eff = len(vals)
        ss_between += n_eff * (group_mean - grand_mean)**2
    
    # Total sum of squares
    if USE_WEIGHTS:
        ss_total = np.sum(all_wts * (all_vals - grand_mean)**2)
    else:
        ss_total = np.sum((all_vals - grand_mean)**2)
    
    return ss_between / ss_total if ss_total > 0 else 0

def weighted_contingency_table(group, outcome, weights=None):
    """Create contingency table (weighted or unweighted)"""
    if USE_WEIGHTS and weights is not None:
        df_temp = pd.DataFrame({'group': group, 'outcome': outcome, 'weight': weights})
        return df_temp.groupby(['group','outcome'])['weight'].sum().unstack(fill_value=0)
    else:
        return pd.crosstab(group, outcome)

def cramers_v(table):
    """Calculate Cramer's V effect size"""
    chi2, _, _, _ = chi2_contingency(table)
    n = table.sum().sum()
    k = min(table.shape)-1
    return np.sqrt(chi2/(n*k)) if k>0 else 0

def perform_anova(groups_data, groups_weights):
    """Perform ANOVA (weighted or unweighted)"""
    if USE_WEIGHTS:
        # Weighted ANOVA using WLS regression
        y_all = []
        group_indicators = []
        weights_all = []
        
        for i, (vals, wts) in enumerate(zip(groups_data, groups_weights)):
            y_all.extend(vals.values)
            group_indicators.extend([i] * len(vals))
            weights_all.extend(wts.values)
        
        y_all = np.array(y_all, dtype=float)
        weights_all = np.array(weights_all, dtype=float)
        group_indicators = np.array(group_indicators, dtype=int)
        
        # Create dummy variables
        k = len(groups_data)
        X = np.zeros((len(y_all), k))
        for i in range(len(y_all)):
            X[i, group_indicators[i]] = 1
        X = np.column_stack([np.ones(len(y_all)), X])
        
        model = sm.WLS(y_all, X, weights=weights_all).fit()
        hypothesis = np.eye(k+1)[1:]
        
        try:
            f_test = model.f_test(hypothesis)
            return f_test.fvalue[0][0], f_test.pvalue
        except:
            return model.fvalue, model.f_pvalue
    else:
        # Standard unweighted ANOVA
        f_stat, p_val = f_oneway(*[g.values for g in groups_data])
        return f_stat, p_val

def perform_ttest(g1, g2, w1, w2):
    """Perform t-test (weighted or unweighted)"""
    if USE_WEIGHTS:
        # Weighted t-test
        y = np.concatenate([g1, g2])
        x = np.concatenate([np.zeros(len(g1)), np.ones(len(g2))])
        w = np.concatenate([w1, w2])
        X = sm.add_constant(x)
        model = sm.WLS(y, X, weights=w).fit()
        return model.tvalues[1], model.pvalues[1]
    else:
        # Standard unweighted t-test
        t_stat, p_val = ttest_ind(g1, g2)
        return t_stat, p_val

def determine_variable_type(series):
    """Determine if variable is binary, ordinal, or continuous"""
    n_unique = series.dropna().nunique()
    if n_unique==2:
        return 'binary'
    elif 3<=n_unique<6:
        return 'ordinal'
    elif pd.api.types.is_numeric_dtype(series):
        return 'continuous'
    return None

def choose_categorical_test(tbl):
    """Choose between Chi-square and Fisher's exact test"""
    expected = chi2_contingency(tbl)[3]
    if (expected < 5).any() or tbl.sum().sum()<20:
        return 'fisher'
    return 'chi_square'

def format_p_value(p):
    """Format p-value for publication"""
    if pd.isna(p):
        return ""
    elif p < 0.001:
        return "<0.001"
    else:
        return f"{p:.3f}"

# ================= DETERMINE VARIABLE TYPES =================
exclude_cols = [cluster_col, WEIGHT_COL]
var_types = {col: determine_variable_type(df[col]) for col in df.columns if col not in exclude_cols}

print("\nDetected variable types:")
for k,v in var_types.items():
    if v is not None:
        print(f"  {k}: {v}")

# ================= OMNIBUS TEST ANALYSIS =================
omnibus_results = []
posthoc_results = []

for col, vtype in var_types.items():
    if vtype is None:
        continue
        
    temp = df[[cluster_col, col, WEIGHT_COL]].dropna()
    if temp.empty:
        continue
    
    # Get unique groups
    groups = sorted(temp[cluster_col].unique())
    n_grps = len(groups)
    
    result = {'Variable': col, 'N_groups': n_grps}
    
    # Prepare group data
    groups_data = [temp[temp[cluster_col]==g][col] for g in groups]
    groups_weights = [temp[temp[cluster_col]==g][WEIGHT_COL] for g in groups]
    
    # Calculate descriptive statistics for each group
    for i, g in enumerate(groups):
        gdata = groups_data[i]
        gwts = groups_weights[i]
        
        if vtype in ['continuous', 'ordinal']:
            mean, std = weighted_mean_std(gdata, gwts)
            median, q1, q3 = weighted_median_iqr(gdata, gwts)
            result[f'Group{g}_mean'] = mean
            result[f'Group{g}_std'] = std
            result[f'Group{g}_median'] = median
            result[f'Group{g}_Q1'] = q1
            result[f'Group{g}_Q3'] = q3
            result[f'Group{g}_n_eff'] = gwts.sum() if USE_WEIGHTS else len(gdata)
        elif vtype == 'binary':
            n_eff = gwts.sum() if USE_WEIGHTS else len(gdata)
            result[f'Group{g}_n_eff'] = n_eff
    
    # ===== OMNIBUS TESTS =====
    if vtype == 'binary':
        # Chi-square/Fisher test for categorical data
        tbl = weighted_contingency_table(temp[cluster_col], temp[col], temp[WEIGHT_COL] if USE_WEIGHTS else None)
        
        if n_grps == 2 and tbl.shape == (2, 2):
            test_type = choose_categorical_test(tbl)
            if test_type == 'fisher':
                or_val, p_val = fisher_exact(tbl)
                test_name = "Fisher's exact"
                result.update({
                    'Test': test_name + (" (weighted)" if USE_WEIGHTS else ""),
                    'Test_stat': or_val,
                    'p_value': p_val,
                    'Effect_size': cramers_v(tbl),
                    'Effect_size_type': "Cramer's V"
                })
            else:
                chi2, p_val, _, _ = chi2_contingency(tbl)
                result.update({
                    'Test': "Chi-square" + (" (weighted)" if USE_WEIGHTS else ""),
                    'Test_stat': chi2,
                    'p_value': p_val,
                    'Effect_size': cramers_v(tbl),
                    'Effect_size_type': "Cramer's V"
                })
        else:
            # Multi-group chi-square
            chi2, p_val, _, _ = chi2_contingency(tbl)
            result.update({
                'Test': f"Chi-square ({n_grps} groups)" + (" (weighted)" if USE_WEIGHTS else ""),
                'Test_stat': chi2,
                'p_value': p_val,
                'Effect_size': cramers_v(tbl),
                'Effect_size_type': "Cramer's V"
            })
        
        # Add contingency table info
        for i, g in enumerate(groups):
            if tbl.shape[1] == 2:  # Binary outcome
                n_yes = tbl.loc[g, 1] if 1 in tbl.columns else 0
                n_total = tbl.loc[g].sum()
                pct = 100 * n_yes / n_total if n_total > 0 else 0
                result[f'Group{g}_binary'] = f"{n_yes:.1f}/{n_total:.1f} ({pct:.1f}%)"
    
    elif vtype in ['continuous', 'ordinal']:
        if n_grps == 2:
            # Two-group t-test
            t_stat, p_val = perform_ttest(
                groups_data[0].values, groups_data[1].values,
                groups_weights[0].values, groups_weights[1].values
            )
            effect_size = cohens_d_weighted(groups_data[0], groups_data[1], 
                                           groups_weights[0], groups_weights[1])
            result.update({
                'Test': 't-test' + (" (weighted)" if USE_WEIGHTS else ""),
                'Test_stat': t_stat,
                'p_value': p_val,
                'Effect_size': effect_size,
                'Effect_size_type': "Cohen's d"
            })
        else:
            # Multi-group ANOVA
            try:
                f_stat, p_val = perform_anova(groups_data, groups_weights)
                eta2 = eta_squared(groups_data, groups_weights)
                result.update({
                    'Test': f'ANOVA ({n_grps} groups)' + (" (weighted)" if USE_WEIGHTS else ""),
                    'Test_stat': f_stat,
                    'p_value': p_val,
                    'Effect_size': eta2,
                    'Effect_size_type': 'Eta-squared'
                })
            except Exception as e:
                print(f"  Warning: ANOVA failed for {col}: {str(e)}")
                result.update({
                    'Test': f'ANOVA ({n_grps} groups)' + (" (weighted)" if USE_WEIGHTS else ""),
                    'Test_stat': np.nan,
                    'p_value': np.nan,
                    'Effect_size': np.nan,
                    'Effect_size_type': 'Eta-squared',
                    'Note': f'ANOVA failed: {str(e)}'
                })
    
    omnibus_results.append(result)
    
    # ===== POST-HOC PAIRWISE COMPARISONS =====
    if PERFORM_POSTHOC and n_grps > 2 and vtype in ['continuous', 'ordinal'] and result.get('p_value', 1) < 0.05:
        print(f"\n  Performing post-hoc tests for {col} (p={result['p_value']:.4f})...")
        
        # All pairwise combinations
        for (g1, g2) in combinations(groups, 2):
            idx1 = groups.index(g1)
            idx2 = groups.index(g2)
            
            gdata1 = groups_data[idx1]
            gdata2 = groups_data[idx2]
            gwts1 = groups_weights[idx1]
            gwts2 = groups_weights[idx2]
            
            # Pairwise t-test
            t_stat, p_val = perform_ttest(
                gdata1.values, gdata2.values,
                gwts1.values, gwts2.values
            )
            
            # Effect size
            cohens_d = cohens_d_weighted(gdata1, gdata2, gwts1, gwts2)
            
            # Descriptive stats
            mean1, std1 = weighted_mean_std(gdata1, gwts1)
            mean2, std2 = weighted_mean_std(gdata2, gwts2)
            median1, q1_1, q3_1 = weighted_median_iqr(gdata1, gwts1)
            median2, q1_2, q3_2 = weighted_median_iqr(gdata2, gwts2)
            
            posthoc_results.append({
                'Variable': col,
                'Comparison': f'Group {g1} vs Group {g2}',
                'Group1': g1,
                'Group2': g2,
                'Group1_mean': mean1,
                'Group1_std': std1,
                'Group1_median': median1,
                'Group1_Q1': q1_1,
                'Group1_Q3': q3_1,
                'Group2_mean': mean2,
                'Group2_std': std2,
                'Group2_median': median2,
                'Group2_Q1': q1_2,
                'Group2_Q3': q3_2,
                'Test': 't-test' + (" (weighted)" if USE_WEIGHTS else ""),
                't_statistic': t_stat,
                'p_value_raw': p_val,
                'Cohens_d': cohens_d
            })

# ================= CONVERT TO DATAFRAMES =================
omnibus_df = pd.DataFrame(omnibus_results)
posthoc_df = pd.DataFrame(posthoc_results) if posthoc_results else None

# ================= APPLY FDR CORRECTION =================
# FDR for omnibus tests
if 'p_value' in omnibus_df.columns:
    pvals = omnibus_df['p_value'].dropna().values
    if len(pvals) > 0:
        fdr_corrected = multipletests(pvals, alpha=0.05, method='fdr_bh')
        omnibus_df.loc[omnibus_df['p_value'].notna(), 'p_value_FDR'] = fdr_corrected[1]
        omnibus_df['Significant_FDR'] = omnibus_df['p_value_FDR'] < 0.05

# Correction for post-hoc tests
if posthoc_df is not None and len(posthoc_df) > 0:
    pvals_posthoc = posthoc_df['p_value_raw'].values
    
    if POSTHOC_CORRECTION == 'bonferroni':
        # Bonferroni correction within each variable
        for var in posthoc_df['Variable'].unique():
            mask = posthoc_df['Variable'] == var
            n_comparisons = mask.sum()
            posthoc_df.loc[mask, 'p_value_adjusted'] = posthoc_df.loc[mask, 'p_value_raw'] * n_comparisons
            posthoc_df.loc[mask, 'p_value_adjusted'] = posthoc_df.loc[mask, 'p_value_adjusted'].clip(upper=1.0)
        posthoc_df['Correction'] = 'Bonferroni'
        
    elif POSTHOC_CORRECTION == 'holm':
        # Holm-Bonferroni within each variable
        for var in posthoc_df['Variable'].unique():
            mask = posthoc_df['Variable'] == var
            pvals_var = posthoc_df.loc[mask, 'p_value_raw'].values
            _, pvals_adj, _, _ = multipletests(pvals_var, method='holm')
            posthoc_df.loc[mask, 'p_value_adjusted'] = pvals_adj
        posthoc_df['Correction'] = 'Holm-Bonferroni'
        
    else:  # 'fdr'
        fdr_posthoc = multipletests(pvals_posthoc, alpha=0.05, method='fdr_bh')
        posthoc_df['p_value_adjusted'] = fdr_posthoc[1]
        posthoc_df['Correction'] = 'FDR'
    
    posthoc_df['Significant'] = posthoc_df['p_value_adjusted'] < 0.05

# ================= CREATE PUBLICATION TABLES =================

# Omnibus publication table
pub_omnibus = []
for _, row in omnibus_df.iterrows():
    var_name = row['Variable']
    vtype = var_types.get(var_name)
    test = row.get('Test', '')
    p_val = format_p_value(row.get('p_value'))
    fdr_p = format_p_value(row.get('p_value_FDR'))
    effect = f"{row.get('Effect_size', np.nan):.3f}" if pd.notna(row.get('Effect_size')) else ""
    effect_type = row.get('Effect_size_type', '')
    
    pub_row = {
        'Variable': var_name,
        'Test': test,
        'Test Statistic': f"{row.get('Test_stat', np.nan):.3f}" if pd.notna(row.get('Test_stat')) else "",
        'p-value': p_val,
        'FDR p-value': fdr_p,
        f'{effect_type}': effect if effect_type else ""
    }
    
    # Add group descriptives
    for g in sorted(df[cluster_col].unique()):
        if vtype == 'binary' and f'Group{g}_binary' in row:
            # For binary variables, show frequencies
            pub_row[f'Group {g}'] = row[f'Group{g}_binary']
        elif f'Group{g}_mean' in row:
            # For continuous variables, show mean ± SD
            pub_row[f'Group {g}'] = f"{row[f'Group{g}_mean']:.2f} ± {row[f'Group{g}_std']:.2f}"
    
    pub_omnibus.append(pub_row)

pub_omnibus_df = pd.DataFrame(pub_omnibus)

# Post-hoc publication table
if posthoc_df is not None:
    pub_posthoc = []
    for _, row in posthoc_df.iterrows():
        pub_posthoc.append({
            'Variable': row['Variable'],
            'Comparison': row['Comparison'],
            f"Group {row['Group1']}": f"{row['Group1_mean']:.2f} ± {row['Group1_std']:.2f}",
            f"Group {row['Group2']}": f"{row['Group2_mean']:.2f} ± {row['Group2_std']:.2f}",
            'p-value': format_p_value(row['p_value_raw']),
            'p-value (adj)': format_p_value(row['p_value_adjusted']),
            "Cohen's d": f"{row['Cohens_d']:.3f}",
            'Significant': '***' if row.get('p_value_adjusted', 1) < 0.001 else 
                          '**' if row.get('p_value_adjusted', 1) < 0.01 else
                          '*' if row.get('p_value_adjusted', 1) < 0.05 else 'ns'
        })
    pub_posthoc_df = pd.DataFrame(pub_posthoc)
else:
    pub_posthoc_df = None

# ================= SAVE RESULTS =================
with pd.ExcelWriter(output_path, engine='openpyxl') as writer:
    pub_omnibus_df.to_excel(writer, sheet_name='Omnibus Tests (Publication)', index=False)
    omnibus_df.to_excel(writer, sheet_name='Omnibus Tests (Detailed)', index=False)
    if pub_posthoc_df is not None:
        pub_posthoc_df.to_excel(writer, sheet_name='Post-hoc (Publication)', index=False)
        posthoc_df.to_excel(writer, sheet_name='Post-hoc (Detailed)', index=False)

print("\n✓ Multi-group analysis with FDR correction complete")
print(f"✓ Analysis type: {'WEIGHTED' if USE_WEIGHTS else 'UNWEIGHTED'}")
print(f"✓ Results saved to: {output_path}")
print(f"✓ Number of groups: {n_groups}")
print(f"✓ Post-hoc correction method: {POSTHOC_CORRECTION if PERFORM_POSTHOC else 'None (disabled)'}")
if posthoc_df is not None:
    n_sig = (posthoc_df['Significant'] == True).sum()
    print(f"✓ Significant post-hoc comparisons: {n_sig}/{len(posthoc_df)}")