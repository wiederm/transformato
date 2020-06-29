import transformato
import sys, os
import logging
import shutil
import pathlib
from transformato import load_config_yaml, SystemStructure, ProposeMutationRoute, IntermediateStateFactory, FreeEnergyCalculator
import parmed as pm
import copy
import numpy as np
# read in specific topology with parameters
import subprocess

# list of tuples for values nstep and nstdcd
from numpy import arange 

def createList(r1,r2,d):
    return arange(r1,r2+1,d)


nsteps_list = createList(10000, 100000, 5000)
nstdcd_list = createList(1000, 10000, 1000)

steps_map = [(x,y) for x in nsteps_list for y in nstdcd_list]

#user_input = input("Enter the path to destination folder: ")
#assert os.path.exists(user_input), "There is no folder "+str(user_input)

from transformato import FreeEnergyCalculator
conf = '/home/master/transformato/config/ethane-methanol-solvation-free-energy.yaml'
configuration = load_config_yaml(config=conf,
                                 input_dir='/home/master/transformato-systems', 
                                 output_dir='/home/master/Master/test') #user_input 

for i in steps_map:
    configuration['simulation']['parameters']['nstep'] = i[0]
    configuration['simulation']['parameters']['nstdcd'] = i[1]

    # load systems
    s1 = SystemStructure(configuration, 'structure1')
    s2 = SystemStructure(configuration, 'structure2')
    a = ProposeMutationRoute(s1, s2)
    
    #generate mutation_map
    guess = [2,4,6]
    mutation_list1 = []
    mutation_list2 = []
    for i in guess:
        tmp1 = a.generate_mutations_to_common_core_for_mol1(nr_of_steps_for_el=i, nr_of_steps_for_bonded_parameters=i)
        mutation_list1.append(tmp1)#list of lists
        tmp2 = a.generate_mutations_to_common_core_for_mol2(nr_of_steps_for_el=i)
        mutation_list2.append(tmp2) #list of lists
        
    
    mutation_map = [(x,y) for x in mutation_list1 for y in mutation_list2] #tuple of two lists
    
    
    for j in mutation_map:
        # write intermediate states for systems
        i = IntermediateStateFactory(system=s1, mutation_list=j[0], configuration=configuration)
        i.generate_intermediate_states()
        paths = pathlib.Path(i.path).glob('**/*.sh')
        for path in sorted(paths):
            run_dir = path.parent
            print(f"Start sampling for: {path}")
            print(f"In directory: {run_dir}")
            try:
                exe = subprocess.run(['bash', str(path), str(run_dir)], check=True, capture_output=True, text=True)
            except TypeError:
                exe = subprocess.run(['bash', str(path), str(run_dir)], check=True,
                                     stdout=subprocess.PIPE, stderr=subprocess.PIPE)
            #print(exe.stdout)
            print('Capture stderr')
            print(exe.stderr)
            
        
        # write intermediate states
        i = IntermediateStateFactory(system=s2, mutation_list=j[1], configuration=configuration)
        i.generate_intermediate_states()

        paths = pathlib.Path(i.path).glob('**/*.sh')
        for path in sorted(paths):
            run_dir = path.parent
            print(f"Start sampling for: {path}")
            print(f"In directory: {run_dir}")
            try:
                exe = subprocess.run(['bash', str(path), str(run_dir)], check=True, capture_output=True, text=True)
            except TypeError:
                exe = subprocess.run(['bash', str(path), str(run_dir)], check=True,
                                     stdout=subprocess.PIPE, stderr=subprocess.PIPE)
            #print(exe.stdout)
            print('Capture stderr')
            print(exe.stderr)
            
    
    f = FreeEnergyCalculator(configuration, 'ethane')
    f.load_trajs(thinning=1)
    f.calculate_dG_to_common_core()
    ddG, dddG = f.end_state_free_energy_difference
    print(f"Free energy difference: {ddG}")
    print(f"Uncertanty: {dddG}")

    f.show_summary()
    
    f = FreeEnergyCalculator(configuration, 'methanol')
    f.load_trajs(thinning=1)
    f.calculate_dG_to_common_core()
    print(f"Free energy difference: {ddG}")
    print(f"Uncertanty: {dddG}")

    f.show_summary()
    