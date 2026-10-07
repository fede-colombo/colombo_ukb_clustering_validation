# -*- coding: utf-8 -*-
"""
Created on Fri Jan  3 15:39:13 2025

Qualitative validation of clustering analysis in HSR

@author: Federica Colombo
"""

# Make the required imports
import pandas as pd
import numpy as np
from neureval.param_selection_confounds import ParamSelectionConfounds,FindBestClustCVConfounds
from sklearn.svm import SVC
from sklearn.cluster import KMeans
import umap
import os
import pickle as pkl
import sys
from neureval.utils import kuhn_munkres_algorithm
import logging
from neureval.visualization import plot_metrics
from sklearn.metrics import adjusted_mutual_info_score, silhouette_score, davies_bouldin_score
from neureval.internal_baselines_confounds import select_best, select_best_bic_aic
from neureval.utils import kuhn_munkres_algorithm


## Define working directories and create a new folder called 'models'
main_path = "path/to/HSR_qualitative_validation"
#data_path = os.path.join(main_path,'models')

# Define output folder called 'results'. 
# Within 'results, create a subfolder corresponding to the input features (e.g., 'GM + FA').
# Within this subfolder, create another folder corresponding to the set of covariates used (e.g., 'age_sex_TIV')
out_dir = "path/to/output/directory"
os.makedirs(out_dir, exist_ok=True)
os.chdir(out_dir)


# Import data and covariates files
database = 'data_HSR_DKT.csv'
covariates_file = 'covariates_HSR_DKT.csv'
# If there are multiple sheets, specify the name of the current sheet 
data = pd.read_csv(os.path.join(main_path,database), sep=';')
cov= pd.read_csv(os.path.join(main_path, covariates_file), sep=';')

# Define two dictionaries for modalities (i.e., views in multiview clustering) and covariates variables.
# For each kind of modality, specify the indexes of the columuns related to the features to be used for clustering
# You can also specify different covaiates for each kind of modality
modalities = {'gm_thickness': data.iloc[:,1:]}
covariates = {'gm_thickness': cov.iloc[:,1:]}


# Define multiview clustering and classifier parameters to be optimized.
# params should be a dictionary of the form {‘s’: {classifier parameter grid}, ‘c’: {clustering parameter grid}} 
# including the lists of classifiers and clustering methods to fit to the data.
params = {'s': {'C': [0.01, 0.1, 1, 10, 100],
                'kernel':['linear']},
          'c': {'init':['k-means++']}}

# Specify clustering (c), classifier (s), and preprocessing (preproc) algorithms
c = KMeans(random_state=42)
s = SVC(random_state=42)

# Parameters selection
best_model = ParamSelectionConfounds(params, 
                                      cv=2, 
                                      s=s, 
                                      c=c,
                                      preprocessing=None,
                                      nrand=10,
                                      n_jobs=-1,
                                      iter_cv=100,
                                      clust_range=list(range(2,5)),
                                      strat=None)

best_model.fit(data,modalities,covariates)

# Save model's parameters in the output directory. Change file name to match the model you performed
best_results = best_model.best_param_
pkl.dump(best_results, open('./best_results_HSR_qualitative_validation.pkl', 'wb'))

# Extract best parameters (clustering and classifier) from grid search
best_init = best_results[0][0]
best_C = best_results[0][1]
best_kernel = best_results[0][2]

# Specify clustering (c), classifier (s), and preprocessing (preproc) algorithms with optimized parameters
c_optimal = KMeans(init=best_init, random_state=42)
s_optimal = SVC(C=best_C,kernel=best_kernel, random_state=42)

# Initialize FindBestClustCVMultiview class. It performs (repeated) k-folds cross-validation to select the best number of clusters.
# Parameters to be specified:
# nfold: cross-validation folds
# nrand: number of random labelling iterations, default 10
# n_jobs: number of jobs to run in parallel, default (number of cpus - 1)
# clust_range: list with number of clusters (e.g., list(range(2,3))), default None
findbestclust = FindBestClustCVConfounds(c_optimal,s_optimal, preprocessing=None, nfold=2, nrand=10,  n_jobs=-1, nclust_range=list(range(2,5)))

# Run FindBestClustCVMultiview. It returns normlaized stability (metrics), best number of clusters (bestncl), and clusters' labels (tr_lab).
# Parameters to be specified:
# iter_cv: number of repeated cross-validation, default 1
# strat: stratification vector for cross-validation splits, default None
metrics, bestncl, tr_lab = findbestclust.best_nclust_confounds(data, modalities, covariates, iter_cv=100, strat_vect=None)
val_results = list(metrics['val'].values())
val_results = np.array(val_results, dtype=object)

print(f"Best number of clusters: {bestncl}")
print(f"Validation set normalized stability (misclassification): {metrics['val'][bestncl]}")
print(f"Result accuracy (on test set): "
      f"{1-val_results[0,0]}")

# Make sure that n_clusters is bestncl
findbestclust.clust_method.n_clusters=bestncl

# Normalized stability plot. For each number of clusters, normalized stabilities are represented for both training (dashed line) and validation sets (continuous line).
# Colors for training and validation sets can be changed (default: ('black', 'black')).
# To save the plot, specify the file name for saving figure in png format.
plot = plot_metrics(metrics, color=('black', 'black'), save_fig='plot_HSR_qualitative_validation.png')

# Save database with cluster labels for post-hoc analyses
labels_tr = pd.DataFrame(tr_lab, columns=['Labels'])
data_all_tr = pd.concat([data, labels_tr], axis=1)
data_all_tr.to_csv('Labels_HSR_qualitative_validation.csv', index=True)

# Save the object to a file
findbestclust_filepath = os.path.join(out_dir, "HSR_findbestclust_qualitative_validation.pkl")
with open(findbestclust_filepath, "wb") as f:
    pkl.dump(findbestclust, f)
 
    
## INTERNAL MEASURES
# 1) Silhouette score
# Use select_best to calcluate silhouette score. Specify silhouette_score as int_measure and 'max' as select parameter.
# It returns silhouette score (sil_score), number of clusters selected (sil_best), and silhouette labels (sil_labels)
logging.info("Silhouette score based selection")
sil_score, sil_best, sil_label = select_best(data, modalities, covariates, c_optimal, silhouette_score, preprocessing=None, select='max', nclust_range=list(range(2,5)))

logging.info(f"Best number of clusters (and scores): "
             f"{{{sil_best}({sil_score})}}")
logging.info(f'AMI (true labels vs clustering labels) training = '
             f'{adjusted_mutual_info_score(tr_lab, kuhn_munkres_algorithm(np.int32(tr_lab), np.int32(sil_label)))}')
logging.info('\n\n')

# Save results obtained with silhouette score.
# Silhouette score, number of clusters, and clusters' labels are organized as a dictionary and saved as a pickle object
sil_results = {'sil_score': sil_score,'sil_best': sil_best,'sil_label': sil_label}
out_sil_name =  os.path.join(out_dir,'sil_results_HSR_qualiative_validation.pkl')
pkl.dump(sil_results, open(out_sil_name, 'wb'))

print(f"Saved Silhouette score results to: {out_sil_name}")

# 2) Davies-Bouldin score
# Use select_best to calcluate Davies-Bouldin score. Specify davies_bouldin_score as int_measure and 'min' as select parameter.
# It returns davies-bouldin score (db_score), number of clusters selected (db_best), and davies-bouldin labels (db_labels)
logging.info("Davies-Bouldin score based selection")
db_score, db_best, db_label = select_best(data,modalities, covariates, c_optimal, davies_bouldin_score,preprocessing=None,
                                                   select='min', nclust_range=list(range(2,5)))

logging.info(f"Best number of clusters (and scores): "
             f"{{{db_best}({db_score})}}")
logging.info(f'AMI (true labels vs clustering labels) training = '
             f'{adjusted_mutual_info_score(tr_lab, kuhn_munkres_algorithm(np.int32(tr_lab), np.int32(db_label)))}')
logging.info('\n\n')

# Save results obtained with Davies-Bouldin score.
# Davies-Bouldin score, number of clusters, and clusters' labels are organized as a dictionary and saved as a pickle object
db_results = {'db_score': db_score,'db_best': db_best,'db_label': db_label}
out_db_name =  os.path.join(out_dir,f'db_results_HSR_qualiative_validation.pkl')
pkl.dump(db_results, open(out_db_name, 'wb'))

print(f"Saved Davies-Bouldin score results to: {out_db_name}")


