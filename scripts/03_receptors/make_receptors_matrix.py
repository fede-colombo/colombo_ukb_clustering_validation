# -*- coding: utf-8 -*-

"""
Concatenate parcellated PET images into region x receptor matrix of densities.
"""

import numpy as np
import pandas as pd
import abagen
from netneurotools import datasets, plotting
from matplotlib.colors import ListedColormap
from scipy.stats import zscore
from nibabel.freesurfer.io import read_annot
import os

# Define path to parcellated data
path = 'path/to/PET_parcellated/'
path_annot = 'path/to/atlases'
out_path = 'path/to/results/receptors/'

# Define number of ROIs (e.g,., for DKT = 62, for DK = 83 (cortical + subcortical))
nnodes = 62

# Get atlas info
lh_annot, _, lh_names = read_annot(os.path.join(path_annot,'lh.aparc.DKTatlas.annot'))
rh_annot, _, rh_names = read_annot(os.path.join(path_annot,'rh.aparc.DKTatlas.annot'))

regions = []

for label in np.unique(lh_annot):
    if label == -1:
        continue  # Skip unknown
    name = lh_names[label].decode('utf-8')
    regions.append(f'lh_{name}')

for label in np.unique(rh_annot):
    if label == -1:
        continue  # Skip unknown
    name = rh_names[label].decode('utf-8')
    regions.append(f'rh_{name}')

# concatenate the receptors
# The list of selected studies is based on Hansen et al., 2022 (https://doi.org/10.1038/s41593-022-01186-3)
# See also the original script on GitHub (https://github.com/netneurolab/hansen_receptors/blob/main/code/make_receptor_matrix.py)

receptors_csv = [path+'5HT1a_way_hc36_savli.csv',
path+ '5HT1b_p943_hc22_savli.csv',
path+ '5HT1b_p943_hc65_gallezot.csv',
path+ '5HT2a_cimbi_hc29_beliveau.csv',
path+ '5HT4_sb20_hc59_beliveau.csv',
path+ '5HT6_gsk_hc30_radhakrishnan.csv',
path+ '5HTT_dasb_hc100_beliveau.csv',
path+ 'A4B2_flubatine_hc30_hillmer.csv',
path+ 'CB1_omar_hc77_normandin.csv',
path+ 'D1_SCH23390_hc13_kaller.csv',
path+ 'D2_flb457_hc37_smith.csv',
path+ 'D2_flb457_hc55_sandiego.csv',
path+ 'DAT_fpcit_hc174_dukart_spect.csv',
path+ 'GABAa-bz_flumazenil_hc16_norgaard.csv',
path+ 'H3_cban_hc8_gallezot.csv',
path+ 'M1_lsn_hc24_naganawa.csv',
path+ 'mGluR5_abp_hc22_rosaneto.csv',
path+ 'mGluR5_abp_hc28_dubois.csv',
path+ 'mGluR5_abp_hc73_smart.csv',
path+ 'MU_carfentanil_hc204_kantonen.csv',
path+ 'NAT_MRB_hc77_ding.csv',
path+ 'NMDA_ge179_hc29_galovic.csv',
path+ 'VAChT_feobv_hc4_tuominen.csv',
path+ 'VAChT_feobv_hc5_bedard_sum.csv',
path+ 'VAChT_feobv_hc18_aghourian_sum.csv']

# combine all the receptors (including repeats)
r = np.zeros([nnodes, len(receptors_csv)])
for i in range(len(receptors_csv)):
    r[:, i] = np.genfromtxt(receptors_csv[i], delimiter=';')

receptor_names = np.array(["5HT1a", "5HT1b", "5HT2a", "5HT4", "5HT6", "5HTT", "A4B2",
                           "CB1", "D1", "D2", "DAT", "GABAa", "H3", "M1", "mGluR5",
                           "MOR", "NET", "NMDA", "VAChT"])
np.save(out_path+'receptor_names_pet.npy', receptor_names)

# make final region x receptor matrix

# Assign receptors that have only one single map
receptor_data = np.zeros([nnodes, len(receptor_names)])
receptor_data[:, 0] = r[:, 0] 
receptor_data[:, 2:9] = r[:, 3:10] 
receptor_data[:, 10:14] = r[:, 12:16]
receptor_data[:, 15:18] = r[:, 19:22]

# weighted average of 5HT1B p943
# weighted average = z-score of each map multiplied by its sample size
receptor_data[:, 1] = (zscore(r[:, 1])*22 + zscore(r[:, 2])*65) / (22+65)

# weighted average of D2 flb457
receptor_data[:, 9] = (zscore(r[:, 10])*37 + zscore(r[:, 11])*55) / (37+55)

# weighted average of mGluR5 ABP688
receptor_data[:, 14] = (zscore(r[:, 16])*22 + zscore(r[:, 17])*28 + zscore(r[:, 18])*73) / (22+28+73)

# weighted average of VAChT FEOBV (note: spreng2022 is missing)
receptor_data[:, 18] = (zscore(r[:, 22])*4 + zscore(r[:, 23])*5 + zscore(r[:, 24])*18) / \
                       (4+5+18)

receptor_data = pd.DataFrame(receptor_data, columns=receptor_names)
receptor_data = pd.concat([pd.DataFrame(regions), receptor_data], axis=1)
receptor_data.to_csv(out_path+'receptor_data_matrix.csv', sep=';')
