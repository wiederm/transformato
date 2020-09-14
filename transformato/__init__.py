"""
transformato
Workflow to set up a relative free energy calculation of ligands with a common core scaffold
"""

# Add imports here
from .utils import load_config_yaml, get_bin_dir, get_toppar_dir
from .transformato import canvas
from .system import SystemStructure

from .state import IntermediateStateFactory
from .mutate import ProposeMutationRoute
from .analysis import FreeEnergyCalculator

# Handle versioneer
from ._version import get_versions
versions = get_versions()
__version__ = versions['version']
__git_revision__ = versions['full-revisionid']
del get_versions, versions

 
import logging
# format logging message
FORMAT = "[%(filename)s:%(lineno)s - %(funcName)1s()] %(message)s"
# set logging level
logging.basicConfig(format=FORMAT,
    datefmt='%d-%m-%Y:%H:%M',
    level=logging.INFO)
