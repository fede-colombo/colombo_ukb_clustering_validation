

# -*- coding: utf-8 -*-
"""
Integrated script: fetches NeuroSynth maps, runs meta-analyses for CogAtlas terms,
and parcellates with the FreeSurfer DKT atlas using surface-based processing.
"""

import contextlib
import json
import os
from pathlib import Path
import warnings

import numpy as np
import pandas as pd
import requests
import nibabel as nib
from nibabel.freesurfer.io import read_annot
from nilearn import surface
from nilearn._utils import check_niimg

import neurosynth as ns

# Suppress warnings
warnings.filterwarnings('ignore', category=FutureWarning)
warnings.filterwarnings('ignore', category=RuntimeWarning)


# ------------------ CONFIG ------------------ #
# NeuroSynth data paths
NSDIR = Path('path/to/data/raw/neurosynth').resolve()
PARDIR = Path('path/to/data/derivatives/neurosynth').resolve()


# Annotation and surface files
LH_ANNOT = 'path/to/atlases/lh.aparc.DKTatlas.annot'
RH_ANNOT = 'path/to/atlases/rh.aparc.DKTatlas.annot'
LH_SURF = 'path/to/atlases/lh.pial'
RH_SURF = 'path/to/atlases/rh.pial'

# NeuroSynth map to parcellate
IMAGES = ['association-test_z']
OUTPUT_CSV = PARDIR / 'dkt_parcellated_neurosynth_123.csv'
# -------------------------------------------- #


def fetch_ns_data(directory):
    directory = Path(directory)
    database, features = directory / 'database.txt', directory / 'features.txt'
    if not database.exists() or not features.exists():
        with open(os.devnull, 'w') as f, contextlib.redirect_stdout(f):
            ns.dataset.download(path=directory, unpack=True)
        try:
            (directory / 'current_data.tar.gz').unlink()
        except FileNotFoundError:
            pass
    return database, features


def get_cogatlas_concepts(url=None):
    if url is None:
        url = 'https://cognitiveatlas.org/api/v-alpha/concept'
    req = requests.get(url)
    req.raise_for_status()
    concepts = set([f.get('name') for f in json.loads(req.content)])
    return concepts


def run_meta_analyses(database, features, use_features=None, outdir=None):
    if outdir is None:
        outdir = NSDIR
    outdir = Path(outdir)

    dataset = ns.Dataset(str(database))
    dataset.add_features(str(features))
    features = set(dataset.get_feature_names())

    if use_features is not None:
        features = set(features) & set(use_features)
    pad = max([len(f) for f in features])

    generated = []
    for word in sorted(features):
        msg = f'Running meta-analysis for term: {word:<{pad}}'
        print(msg, end='\r', flush=True)

        path = outdir / word.replace(' ', '_')
        path.mkdir(exist_ok=True)
        if not all((path / f'{f}.nii.gz').exists() for f in IMAGES):
            ma = ns.MetaAnalysis(dataset, dataset.get_studies(features=word))
            ma.save_results(path, image_list=IMAGES)

        generated.append(path)

    print(' ' * len(msg) + '\b' * len(msg), end='', flush=True)
    return generated


def average_within_labels(data, annot, names, hemi):
    """Average values within each parcel label (prefixing hemisphere)"""
    result = {}
    for label in np.unique(annot):
        if label == -1:
            continue  # Skip unknown
        name = names[label].decode('utf-8')
        region = f'{hemi}_{name}'
        result[region] = np.mean(data[annot == label])
    return result


def parcellate_meta_surface(outputs, lh_annot_path, rh_annot_path,
                             lh_surf_path, rh_surf_path, out_csv,
                             map_name='association-test_z.nii.gz'):

    # Load annotations and surfaces
    lh_annot, _, lh_names = read_annot(lh_annot_path)
    rh_annot, _, rh_names = read_annot(rh_annot_path)
    surf_mesh_lh = nib.freesurfer.read_geometry(lh_surf_path)
    surf_mesh_rh = nib.freesurfer.read_geometry(rh_surf_path)

    results = []

    for outdir in sorted(outputs):
        nii_file = outdir / map_name
        if not nii_file.exists():
            print(f"Skipping {outdir.name}, map not found.")
            continue

        print(f"Parcellating: {outdir.name}")

        texture_lh = surface.vol_to_surf(str(nii_file), surf_mesh_lh)
        texture_rh = surface.vol_to_surf(str(nii_file), surf_mesh_rh)

        avg_lh = average_within_labels(texture_lh, lh_annot, lh_names, 'lh')
        avg_rh = average_within_labels(texture_rh, rh_annot, rh_names, 'rh')

        combined = {**avg_lh, **avg_rh}
        df = pd.DataFrame(combined, index=[outdir.name])
        results.append(df)

    final_df = pd.concat(results)
    final_df.to_csv(out_csv)
    print(f"✅ Saved parcellated data to: {out_csv}")
    return out_csv


if __name__ == '__main__':
    NSDIR.mkdir(parents=True, exist_ok=True)
    PARDIR.mkdir(parents=True, exist_ok=True)

    # Step 1: Download NeuroSynth + Cognitive Atlas
    database, features = fetch_ns_data(NSDIR)
    cog_terms = get_cogatlas_concepts()
    generated = run_meta_analyses(database, features, cog_terms, outdir=NSDIR)

    # Step 2: Parcellate using surface-based DKT atlas
    parcellate_meta_surface(
        outputs=generated,
        lh_annot_path=LH_ANNOT,
        rh_annot_path=RH_ANNOT,
        lh_surf_path=LH_SURF,
        rh_surf_path=RH_SURF,
        out_csv=OUTPUT_CSV
    )