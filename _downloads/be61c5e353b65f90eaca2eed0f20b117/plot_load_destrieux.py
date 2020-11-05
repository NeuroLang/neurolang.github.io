# coding: utf-8
r'''
Loading and Querying the Destrieux et al. Atlas
========================================================================


Uploading the Destrieux regions NeuroLang and
executing a simple query.
'''

import nilearn
import numpy as np
from matplotlib import pyplot as plt
from nilearn import plotting

import nibabel as nib
from neurolang.frontend import NeurolangPDL, ExplicitVBR


nl = NeurolangPDL


###############################################################################
# Load the Destrieux example from nilearn as a fact list
# ------------------------------------------------------
atlas_destrieux = nilearn.datasets.fetch_atlas_destrieux_2009()

image = nib.load(atlas_destrieux['maps'])
image_data = image.get_data()


##################################################
# Load the regions into Voxel-style regions
region_table = []
for label, name in atlas_destrieux['labels']:
    if label == 0:
        continue

    voxels = np.transpose((image_data == label).nonzero())
    if voxels.shape[0] == 0:
        continue

    r = ExplicitVBR(
            voxels,
            image.affine, image_dim=image.shape
    )
    region_table.append((str(name.decode('utf8')), r))

##################################################
# Initialise the atlas to the Neurolang Engine
# add two symbols to split left and right structures
# in the Destrieux atlas and add the atlas.

nl = NeurolangPDL()


@nl.add_symbol
def lh(x: str) -> bool:
    return x.startswith('L ')


@nl.add_symbol
def rh(x: str) -> bool:
    return x.startswith('R ')


destrieux = nl.add_tuple_set(region_table, name='destrieux')


with nl.environment as e:
    e.superior_sts_l[e.name, e.r] = (
        e.destrieux('L S_temporal_sup', e.superior_sts_l) &
        e.destrieux('L S_central', e.central_l) &
        e.anatomical_superior_of(e.r, e.superior_sts_l) &
        e.anatomical_anterior_of(e.r, e.central_l) &
        e.lh(e.name) &
        e.destrieux(e.name, e.r)
    )

    result = nl.query((e.name, e.r), e.superior_sts_l(e.name, e.r))

for name, region in result:
    print(name)
    plt.figure()
    plotting.plot_roi(region.spatial_image(), title=name)

