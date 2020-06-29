from utils import loader,load_param_yaml, createList, createMap, createStepsMap, targetList, create_input_yaml, check_folder, output_yaml
from simulation import benchmark_simulation
import os


#user_input = input("Enter the path to destination folder: ")
user_input = '/home/bene/test'

loader(user_input)
path = user_input + '/transformato-systems-master/'
parameter = load_param_yaml(parameter=path + 'parameter.yaml') 
#print (parameter)
steps_map = createStepsMap(parameter['simulation']['parameters']['nstep'],parameter['simulation']['parameters']['nstdcd'])
molecule_pairs = targetList(path)
create_input_yaml(path,molecule_pairs)
systems = next(os.walk(path + 'output'))[1]

output_dict = {}
for i in systems:
    simulation_dict = benchmark_simulation(steps_map,path,i)
    output_dict = {i : simulation_dict}

output_yaml(path,output_dict)








