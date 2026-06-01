# -*- coding: utf-8 -*-
"""
Created on Fri Jan  3 15:39:13 2025

External validation of clustering solution in HSR

@author: Federica Colombo
"""

# Make the required imports
import pandas as pd
import numpy as np
from neureval.best_nclust_cv_confounds import FindBestClustCVConfounds
from sklearn.cluster import KMeans
from sklearn.svm import SVC
from sklearn.metrics import zero_one_loss, adjusted_mutual_info_score, silhouette_score, davies_bouldin_score
from sklearn.preprocessing import StandardScaler, MinMaxScaler
import os
import pickle as pkl
import logging
import sys
from neureval.visualization import plot_metrics
from neureval.internal_baselines_confounds import select_best
from neureval.utils import kuhn_munkres_algorithm


## Define working directories and create a new folder called 'models'
main_path = "path/to/results/HSR_validation"

# Define output folder called 'results'. 
out_dir = main_path

# Import data and covariates files for the test set
path_data_all = 'data_all_UKB_HSR_DKT.csv'
path_cov_all = 'covariates_all_UKB_HSR_DKT.csv'

# If there are multiple sheets, specify the name of the current sheet 
data_all = pd.read_csv(os.path.join(main_path,path_data_all), sep=';')
cov_all = pd.read_csv(os.path.join(main_path, path_cov_all), sep=';')
data_all = data_all.drop(['train_test'],axis=1)
cov_all = cov_all.drop(['train_test'],axis=1)

# Specify the number of subjects belonging to the training set
training_dim = 842
tr_idx = list(range(0,training_dim))
val_idx = list(range(training_dim,len(data_all)))

print(f"Training set sample size: {len(tr_idx)}")
print(f"Validation set sample size: {len(val_idx)}")

# Define two dictionaries for modalities and covariates variables.
# For each kind of modality, specify the indeces of the columuns related to the features to be used for clustering
# You can also specify different covaiates for each kind of modality
modalities_val = {'gm_thickness': data_all.iloc[:,1:]}
covariates_val = {'gm_thickness': cov_all.iloc[:,1:]}

# Load the findbestclust object
findbestclust_path = os.path.join(main_path,'findbestclust_55_training_45_test_DKT.pkl')
with open(findbestclust_path, "rb") as f:
    findbestclust = pkl.load(f)

# Validate the identified clusters
# Parameters to be specified:
# data: dataset
# modalities: dictionary specifying the features for each modality
# covariates: dictionary specifying the covariates for each modality
# tr_idx: number of training samples
# val_idx: number of validation samples
# nclust: specify the best number of clusters idenfied in the training dataset (i.e., bestncl)
out = findbestclust.evaluate_confounds(data_all, modalities_val, covariates_val, tr_idx, val_idx, nclust=2, tr_lab=None)
print(f"Training ACC: {out.train_acc}, Test ACC: {out.test_acc}")

# Save cluster labels in the validation set
labels_val = pd.DataFrame(out.test_cllab, columns=['Labels'])
df_val_idnum = data_all.iloc[:,0][val_idx].reset_index()
df_idnum_labels_val = pd.concat([df_val_idnum, labels_val], axis=1)
df_idnum_labels_val.to_csv(os.path.join(out_dir,'Labels_HSR_validation.csv'), index=False)
