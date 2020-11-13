"""
Sulcal Identification Queries in Neurolang
==============================================

"""

from matplotlib import pyplot as plt
import nibabel as nib
from nilearn import datasets, plotting
from neurolang.frontend import NeurolangDL


##################################################
# Initialise the NeuroLang probabilistic engine.

nl = NeurolangDL()


###############################################################################
# Load the Destrieux example from nilearn as a fact list


atlas_destrieux = datasets.fetch_atlas_destrieux_2009()
atlas_labels = {
    label: str(name.decode('utf8'))
    for label, name in atlas_destrieux['labels']
}


nl.add_atlas_set('destrieux_atlas', atlas_labels, nib.load(atlas_destrieux['maps']))

###############################################################################
# Add utility function


@nl.add_symbol
def startswith(prefix: str, s: str) -> bool:
    return s.startswith(prefix)


#############################################################
# Define all left sulci and the primary ones

with nl.environment as e:
    e.left_sulcus[e.name, e.region] = (
        e.destrieux_atlas(e.name, e.region) &
        startswith('L S', e.name)
    )

    e.left_primary_sulcus[e.name, e.region] = (
        e.destrieux_atlas(e.name, e.region) & (
            (e.name == "L S_central") |
            (e.name == "L Lat_Fis-post") |
            (e.name == "L S_pericallosal") |
            (e.name == "L S_parieto_occipital") |
            (e.name == "L S_calcarine") |
            (e.name == "L Lat_Fis-ant-Vertical")
        )
    )
