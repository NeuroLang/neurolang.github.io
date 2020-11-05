# -*- coding: utf-8 -*-
r'''
NeuroLang Datalog Example based on the Destrieux Atlas and Neurosynth
=====================================================================


Uploading the Destrieux left sulci into NeuroLang and
executing some simple queries.
'''
import logging
from operator import contains as contains_
from typing import Iterable
import sys

import nibabel as nib
from nilearn import datasets
from nilearn import plotting
import numpy as np
import pandas as pd

from neurolang import frontend as fe

logger = logging.getLogger('neurolang.datalog.chase')
logger.setLevel(logging.INFO)
logger.addHandler(logging.StreamHandler(sys.stderr))


###############################################################################
# Load the Destrieux example from nilearn
# ---------------------------------------

destrieux_dataset = datasets.fetch_atlas_destrieux_2009()
destrieux_map = nib.load(destrieux_dataset['maps'])


###############################################################################
# Initialize the NeuroLang instance and load Destrieux's cortical parcellation
# -----------------------------------------------------------------------------


nl = fe.NeurolangDL()
destrieux_tuples = []
for label_number, name in destrieux_dataset['labels']:
    if label_number == 0:
        continue
    name = name.decode()
    region = nl.create_region(destrieux_map, label=label_number)
    if region is None:
        continue
    name = name.replace('-', '_').replace(' ', '_')
    destrieux_tuples.append((name.lower(), region))

destrieux = nl.add_tuple_set(destrieux_tuples, name='destrieux')


###############################################################################
# Add a function to measure a region's volume
# -----------------------------------------------------------------------------

@nl.add_symbol
def region_volume(region: fe.ExplicitVBR) -> float:
    volume = (
        len(region.voxels) *
        float(np.product(np.abs(np.linalg.eigvals(region.affine[:-1, :-1]))))
    )
    return volume

contains = nl.add_symbol(contains_, name='contains')

########################################################################
# Query all Destrieux regions having volume larger than 2500mm3
# ----------------------------------------------------------------------


with nl.scope as e:

    e.anterior_to_precentral[e.name, e.region] = (
        e.destrieux(e.name, e.region) &
        e.destrieux('l_g_precentral', e.region_) &
        contains(e.region, (..., e.j, ...)) &
        contains(e.region_, (..., e.j_, ...)) &
        (e.j_ > e.j)
    )

    res = nl.query(
            (e.name, e.region),
            e.anterior_to_precentral(e.name, e.region)
            # & (region_volume(e.region) > 2500)
    )
