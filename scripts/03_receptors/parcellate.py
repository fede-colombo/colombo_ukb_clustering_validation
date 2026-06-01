
# -*- coding: utf-8 -*-
"""
Parcellate volumetric PET images
"""
import numpy as np
import pandas as pd
import nibabel as nib
from neuromaps.datasets import available_annotations, fetch_annotation
from nilearn.datasets import fetch_atlas_schaefer_2018
from neuromaps.parcellate import Parcellater
from neuromaps import images, transforms
from nibabel.freesurfer.io import read_annot
import os
from nibabel import gifti

# Define atlas and output paths
path_annot = 'path/to/atlases'
path_output = 'path/to/PET_parcellated/'
path = 'path/to/PET_nifti_images/'

# Load annot files
DKT_gii_lh =images.annot_to_gifti(os.path.join(path_annot,'lh.aparc.DKTatlas.annot'))
DKT_gii_rh=images.annot_to_gifti(os.path.join(path_annot,'rh.aparc.DKTatlas.annot'))

# Save annot files into gifti
nib.save(DKT_gii_lh[0], os.path.join(path_annot,'lh.DKT_new.gii'))
nib.save(DKT_gii_rh[0],os.path.join(path_annot,'rh.DKT_new.gii'))

# Create parcellater for lefth and right hemispheres
parcL = Parcellater((DKT_gii_lh[0]), 'fsaverage', hemi='L')  # DKT parcellater
parcR = Parcellater((DKT_gii_rh[0]), 'fsaverage', hemi='R')  # DKT parcellater
parc = [parcL, parcR]

# Get annot labels
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

# define PET images
receptors_nii = [path+'5HT1a_way_hc36_savli.nii',
                 path+'5HT1a_cumi_hc8_beliveau.nii',
                 path+'5HT1b_az_hc36_beliveau.nii',
                 path+'5HT1b_p943_hc22_savli.nii',
                 path+'5HT1b_p943_hc65_gallezot.nii.gz',
                 path+'5HT2a_cimbi_hc29_beliveau.nii',
                 path+'5HT2a_alt_hc19_savli.nii',
                 path+'5HT2a_mdl_hc3_talbot.nii.gz',
                 path+'5HT4_sb20_hc59_beliveau.nii',
                 path+'5HT6_gsk_hc30_radhakrishnan.nii.gz',
                 path+'5HTT_dasb_hc100_beliveau.nii',
                 path+'5HTT_dasb_hc30_savli.nii',
                 path+'A4B2_flubatine_hc30_hillmer.nii.gz',
                 path+'CB1_omar_hc77_normandin.nii.gz',
                 path+'CB1_FMPEPd2_hc22_laurikainen.nii',
                 path+'D1_SCH23390_hc13_kaller.nii',
                 path+'D2_fallypride_hc49_jaworska.nii',
                 path+'D2_flb457_hc37_smith.nii.gz',
                 path+'D2_flb457_hc55_sandiego.nii.gz',
                 path+'D2_raclopride_hc7_alakurtti.nii',
                 path+'DAT_fpcit_hc174_dukart_spect.nii',
                 path+'DAT_fepe2i_hc6_sasaki.nii.gz',
                 path+'GABAa-bz_flumazenil_hc16_norgaard.nii',
                 path+'GABAa_flumazenil_hc6_dukart.nii',
                 path+'H3_cban_hc8_gallezot.nii.gz',
                 path+'M1_lsn_hc24_naganawa.nii.gz',
                 path+'mGluR5_abp_hc22_rosaneto.nii',
                 path+'mGluR5_abp_hc28_dubois.nii',
                 path+'mGluR5_abp_hc73_smart.nii',
                 path+'MU_carfentanil_hc204_kantonen.nii',
                 path+'MU_carfentanil_hc39_turtonen.nii',
                 path+'NAT_MRB_hc77_ding.nii.gz',
                 path+'NAT_MRB_hc10_hesse.nii',
                 path+'NMDA_ge179_hc29_galovic.nii.gz',
                 #path+'VAChT_feobv_hc3_spreng.nii',
                 path+'VAChT_feobv_hc4_tuominen.nii',
                 path+'VAChT_feobv_hc5_bedard_sum.nii',
                 path+'VAChT_feobv_hc18_aghourian_sum.nii']

parcellated = {}
for receptor in receptors_nii:
    name = receptor.split('/')[-1]  # get nifti file name
    name = name.split('.')[0]  # remove .nii
    receptor = transforms.mni152_to_fsaverage(receptor)
    # Parcellate maps
    NT_parc_lh = parcL.fit_transform(receptor[0], 'fsaverage', hemi='L')
    NT_parc_rh = parcR.fit_transform(receptor[1], 'fsaverage', hemi='R')
    NT_parc = np.concatenate((NT_parc_lh, NT_parc_rh))
    parcellated[receptor] = NT_parc
    np.savetxt(path_output+name+'.csv', parcellated[receptor], delimiter=';')