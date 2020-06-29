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
from datetime import datetime
import pickle



def benchmark_simulation(steps_map,input_path,system):
    """This function runs the simulation over all parameter pairs given in steps_map for the system given in system.

    Args:
        steps_map (list of tuples): [description]
        input_path (string): [description]
        system (string): given benchmark system

    Returns:
        dictionary: returns the ddG, dddG, for every nstep and nstdcd and guess combination
    """
    conf = input_path + 'output/' + system + '/input.yaml'
    output_dir = input_path + 'output/' + system
    configuration = load_config_yaml(config=conf,
                                    input_dir=input_path, 
                                    output_dir=output_dir) 

    startTime = datetime.now()

    for i in steps_map:
        steps = i
        configuration['simulation']['parameters']['nstep'] = i[0]
        configuration['simulation']['parameters']['nstdcd'] = i[1]

        # load systems
        s1 = SystemStructure(configuration, 'structure1')
        s2 = SystemStructure(configuration, 'structure2')
        a = ProposeMutationRoute(s1, s2)
        #a_rev = ProposeMutationRoute(s2, s1)
    
        #generate mutation_map
        guess = [2,4,6]
        mutation_list1 = []
        mutation_list2 = []

        for i in guess:
            guess_step = i
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
                #print(f"Start sampling for: {path}")
                #print(f"In directory: {run_dir}")
                try:
                    exe = subprocess.run(['bash', str(path), str(run_dir)], check=True, capture_output=True, text=True)
                except TypeError:
                    exe = subprocess.run(['bash', str(path), str(run_dir)], check=True,
                                         stdout=subprocess.PIPE, stderr=subprocess.PIPE)
                #print(exe.stdout)
                #print('Capture stderr')
                print(exe.stderr)
            
        
            # write intermediate states
            i = IntermediateStateFactory(system=s2, mutation_list=j[1], configuration=configuration)
            i.generate_intermediate_states()

            paths = pathlib.Path(i.path).glob('**/*.sh')
            for path in sorted(paths):
                run_dir = path.parent
                #print(f"Start sampling for: {path}")
                #print(f"In directory: {run_dir}")
                try:
                    exe = subprocess.run(['bash', str(path), str(run_dir)], check=True, capture_output=True, text=True)
                except TypeError:
                    exe = subprocess.run(['bash', str(path), str(run_dir)], check=True,
                                         stdout=subprocess.PIPE, stderr=subprocess.PIPE)
                #print(exe.stdout)
                #print('Capture stderr')
                print(exe.stderr)
            
    
            f = FreeEnergyCalculator(configuration, configuration['system']['structure1']['name'])
            f.load_trajs(thinning=1)
            f.calculate_dG_to_common_core()
            ddG1, dddG1 = f.end_state_free_energy_difference
            #print(f"Free energy difference: {ddG}")
            #print(f"Uncertanty: {dddG}")

            pickle.dump(f, open(output_dir + '/' + configuration['system']['structure1']['name'] + '_output.p', 'wb'))
    
            f = FreeEnergyCalculator(configuration, configuration['system']['structure2']['name'])
            f.load_trajs(thinning=1)
            f.calculate_dG_to_common_core()
            ddG2, dddG2 = f.end_state_free_energy_difference
            #print(f"Free energy difference: {ddG}")
            #print(f"Uncertanty: {dddG}")

            pickle.dump(f, open(output_dir + '/' + configuration['system']['structure2']['name'] + '_output.p', 'wb'))

            ddG = ddG1-ddG2
            dddG = dddG1 + dddG2

            runTime = datetime.now() - startTime

            simulation_dict = {'nsteps/nstdcd ' + str(steps[0]) + '/' + str(steps[1]) : {'guess' : guess_step, 'runtime': runTime, 'ddG': ddG, 'dddG':dddG}}

###### reverse run #######
        """for i in guess:
            guess_step = i
            tmp1 = a_rev .generate_mutations_to_common_core_for_mol1(nr_of_steps_for_el=i, nr_of_steps_for_bonded_parameters=i)
            mutation_list1.append(tmp1)#list of lists
            tmp2 = a_rev .generate_mutations_to_common_core_for_mol2(nr_of_steps_for_el=i)
            mutation_list2.append(tmp2) #list of lists
        
    
        mutation_map = [(x,y) for x in mutation_list1 for y in mutation_list2] #tuple of two lists
    
    
        for j in mutation_map:
            # write intermediate states for systems
            i = IntermediateStateFactory(system=s2, mutation_list=j[0], configuration=configuration)
            i.generate_intermediate_states()
            paths = pathlib.Path(i.path).glob('**/*.sh')
            
            for path in sorted(paths):
                run_dir = path.parent
                #print(f"Start sampling for: {path}")
                #print(f"In directory: {run_dir}")
                try:
                    exe = subprocess.run(['bash', str(path), str(run_dir)], check=True, capture_output=True, text=True)
                except TypeError:
                    exe = subprocess.run(['bash', str(path), str(run_dir)], check=True,
                                         stdout=subprocess.PIPE, stderr=subprocess.PIPE)
                #print(exe.stdout)
                #print('Capture stderr')
                #print(exe.stderr)
            
        
            # write intermediate states
            i = IntermediateStateFactory(system=s1, mutation_list=j[1], configuration=configuration)
            i.generate_intermediate_states()

            paths = pathlib.Path(i.path).glob('**/*.sh')
            for path in sorted(paths):
                run_dir = path.parent
                #print(f"Start sampling for: {path}")
                #print(f"In directory: {run_dir}")
                try:
                    exe = subprocess.run(['bash', str(path), str(run_dir)], check=True, capture_output=True, text=True)
                except TypeError:
                    exe = subprocess.run(['bash', str(path), str(run_dir)], check=True,
                                         stdout=subprocess.PIPE, stderr=subprocess.PIPE)
                #print(exe.stdout)
                #print('Capture stderr')
                #print(exe.stderr)
            
    
            f = FreeEnergyCalculator(configuration, configuration['system']['structure2']['name'])
            f.load_trajs(thinning=1)
            f.calculate_dG_to_common_core()
            ddG1, dddG1 = f.end_state_free_energy_difference
            #print(f"Free energy difference: {ddG}")
            #print(f"Uncertanty: {dddG}")

            pickle.dump(f, open(output_dir + '/' + configuration['system']['structure2']['name'] + '_output.p', 'wb'))
    
            f = FreeEnergyCalculator(configuration, configuration['system']['structure1']['name'])
            f.load_trajs(thinning=1)
            f.calculate_dG_to_common_core()
            ddG2, dddG2 = f.end_state_free_energy_difference
            #print(f"Free energy difference: {ddG}")
            #print(f"Uncertanty: {dddG}")

            pickle.dump(f, open(output_dir + '/' + configuration['system']['structure1']['name'] + '_output.p', 'wb'))

            ddG = ddG1-ddG2
            dddG = dddG1 + dddG2

            simulation_dict_rev = {'nsteps/nstdcd ' + str(steps[0]) + '/' + str(steps[1]) : {'guess' : guess_step, 'runtime': runTime, 'ddG': ddG, 'dddG':dddG}}"""
        
    return simulation_dict
