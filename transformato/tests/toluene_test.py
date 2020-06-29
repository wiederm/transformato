import transformato
import sys, os
import logging
import shutil
import pathlib
from transformato import load_config_yaml, SystemStructure, ProposeMutationRoute, IntermediateStateFactory, FreeEnergyCalculator
import parmed as pm
import copy
import numpy as np
import subprocess

conf = '/home/master/transformato/config/toluene-methane-solvation-free-energy.yaml'
configuration = load_config_yaml(config=conf,
                                 input_dir='/home/master/transformato-systems', 
                                 output_dir='/home/master/debug') 
                                 
# load systems
s1 = SystemStructure(configuration, 'structure1')
s2 = SystemStructure(configuration, 'structure2')
a = ProposeMutationRoute(s1, s2)

# generate mutation route
mutation_list = a.generate_mutations_to_common_core_for_mol1(
    nr_of_steps_for_el=2, nr_of_steps_for_bonded_parameters=2)

# write intermediate states for systems
i = IntermediateStateFactory(system=s1, mutation_list=mutation_list, configuration=configuration)
i.generate_intermediate_states()