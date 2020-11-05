# -*- coding: utf-8 -*-
r'''
NeuroLang Example Reverse Inference in NeuroSynth
=================================================

'''

import logging
from typing import Iterable
import sys
import warnings

from neurolang import frontend as fe
import nibabel as nib
from nilearn import datasets, image
import numpy as np
import pandas as pd

logger = logging.getLogger('neurolang.probabilistic')
logger.setLevel(logging.INFO)
logger.addHandler(logging.StreamHandler(sys.stderr))
warnings.filterwarnings("ignore")


###############################################################################
# Data preparation
# ----------------

###############################################################################
# Load the MNI template and resample it to 4mm voxels

mni_t1 = nib.load(datasets.fetch_icbm152_2009()['t1'])
mni_t1_4mm = image.resample_img(mni_t1, np.eye(3) * 4)


###############################################################################
# Load Destrieux's atlas
destrieux_dataset = datasets.fetch_atlas_destrieux_2009()
destrieux = nib.load(destrieux_dataset['maps'])
destrieux_resampled = image.resample_img(
    destrieux, mni_t1_4mm.affine, interpolation='nearest'
)
destrieux_resampled_data = np.asanyarray(
    destrieux_resampled.dataobj, dtype=np.int32
)
destrieux_voxels_ijk = destrieux_resampled_data.nonzero()
destrieux_voxels_value = destrieux_resampled_data[destrieux_voxels_ijk]
destrieux_table = pd.DataFrame(
    np.transpose(destrieux_voxels_ijk), columns=['i', 'j', 'k']
)
destrieux_table['label'] = destrieux_voxels_value

destrieux_label_names = []
for label_number, name in destrieux_dataset['labels']:
    if label_number == 0:
        continue
    name = name.decode()
    name = name.replace('-', '_').replace(' ', '_')
    destrieux_label_names.append((name.lower(), label_number))


###############################################################################
# Load the NeuroSynth database

ns_database_fn, ns_features_fn = datasets.utils._fetch_files(
    datasets.utils._get_dataset_dir('neurosynth'),
    [
        (
            'database.txt',
            'https://github.com/neurosynth/neurosynth-data/raw/master/current_data.tar.gz',
            {'uncompress': True}
        ),
        (
            'features.txt',
            'https://github.com/neurosynth/neurosynth-data/raw/master/current_data.tar.gz',
            {'uncompress': True}
        ),
    ]
)

ns_database = pd.read_csv(ns_database_fn, sep=f'\t')
ijk_positions = (
    np.round(nib.affines.apply_affine(
        np.linalg.inv(mni_t1_4mm.affine),
        ns_database[['x', 'y', 'z']].values.astype(float)
    )).astype(int)
)
ns_database['i'] = ijk_positions[:, 0]
ns_database['j'] = ijk_positions[:, 1]
ns_database['k'] = ijk_positions[:, 2]

ns_features = pd.read_csv(ns_features_fn, sep=f'\t')
ns_docs = ns_features[['pmid']].drop_duplicates()
ns_terms = (
    pd.melt(
            ns_features,
            var_name='term', id_vars='pmid', value_name='TfIdf'
       )
    .query('TfIdf > 1e-3')[['pmid', 'term']]
)


###############################################################################
# Probabilistic Logic Programming in NeuroLang
# --------------------------------------------

nl = fe.NeurolangPDL()


###############################################################################
# Loading the database

activations = nl.add_tuple_set(ns_database.values, name='activations')
terms = nl.add_tuple_set(ns_terms.values, name='terms')
docs = nl.add_uniform_probabilistic_choice_over_set(
        ns_docs.values, name='docs'
)
destrieux_image = nl.add_tuple_set(
    destrieux_table.values,
    name='destrieux_image'
)
destrieux_labels = nl.add_tuple_set(
    destrieux_label_names, name='destrieux_labels'
)

for set_symbol in (
    'activations', 'terms', 'docs', 'destrieux_image', 'destrieux_labels'
):
    print(f"#{set_symbol}: {len(nl.symbols[set_symbol].value)}")

###############################################################################
# Adding new aggregation function to build a region overlay
@nl.add_symbol
def agg_sum(i: Iterable) -> float:
    return np.sum(i)

###############################################################################
# Probabilistic program and querying


with nl.scope as e:
    e.destrieux_voxel_names[e.i, e.j, e.k, e.region_name] = (
        e.destrieux_labels(e.region_name, e.region_label) &
        e.destrieux_image(e.i, e.j, e.k, e.region_label)
    )

    e.vox_term_prob[e.i, e.j, e.k, e.t, e.PROB[e.i, e.j, e.k, e.t]] = (
        e.activations[
            e.d, ..., ..., ..., ..., 'MNI', ..., ..., ..., ...,
            ..., ..., ..., e.i, e.j, e.k
        ] &
        e.destrieux_voxel_names(e.i, e.j, e.k, 'l_g_front_inf_opercular') &
        e.terms[e.d, e.t] &
        e.docs[e.d]
    )

    e.vox_prob[e.i, e.j, e.k, e.PROB[e.i, e.j, e.k]] = (
        e.activations[
            e.d, ..., ..., ..., ..., 'MNI', ..., ..., ..., ...,
            ..., ..., ..., e.i, e.j, e.k
        ] &
        e.destrieux_voxel_names(e.i, e.j, e.k, 'l_g_front_inf_opercular') &
        e.terms[e.d, e.t] &
        e.docs[e.d]
    )

    e.term_cond_act_prob[e.t, agg_sum(e.p)] = (
        e.vox_term_prob[e.i, e.j, e.k, e.t, e.num_prob] &
        e.vox_prob[e.i, e.j, e.k, e.denom_prob]
        & (e.p == (e.num_prob / e.denom_prob))
    )

    res = nl.query((e.t, e.p), e.term_cond_act_prob(e.t, e.p))

###############################################################################
# Results
# --------------------------------------------

###############################################################################
# Probability of a term appearing on a study conditioned if a voxel in the left
# frontal opercular gyrus is active in the study.
print(
    res
    .as_pandas_dataframe()
    .sort_values(res.columns[-1], ascending=False)
    .head(50)
)
