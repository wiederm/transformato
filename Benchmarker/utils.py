from numpy import arange 
import yaml
from urllib.request import urlretrieve
import os
from zipfile import ZipFile


def loader(input_location):
    """ Downloads benchmark systems from https://github.com/wiederm/transformato-systems to input_location

    Args:
        input_location (string): path to the working and saving directory

    Returns:
        [list]: benchmark system names
    """
    try:
        os.makedirs(input_location)
        print ('Directory ' , input_location ,  ' Created')
    except FileExistsError:
        print('Directory ', input_location , ' already exists')
        pass
    
    print ("Downloading benchmark molecules...")
    url = 'https://github.com/bbraunsfeld/transformato-systems/archive/master.zip' #'https://github.com/wiederm/transformato-systems/archive/master.zip'
    dest_file = os.path.join(input_location, 'transformato-systems-master.zip')
    urlretrieve(url, dest_file)
    
    print ("Extracting...")
    with ZipFile(dest_file, 'r') as zipObj:
        zipObj.extractall(input_location)

    targets = next(os.walk(input_location + '/transformato-systems-master'))[1]

    try:
        os.makedirs(input_location + '/transformato-systems-master/output')
    except FileExistsError:
        pass
    
    print ("Cleaning...")
    os.remove(dest_file)
    print ("Finished!")
    return targets

def createList(r1,r2,d):
    """creates a list in range r1 to r2 with distance d between every entry

    Args:
        r1 (int): startpoint
        r2 (int): endpoint
        d (int): steps size

    Returns:
        [list]: list of integers
    """
    return arange(r1,r2+1,d)

def createMap(nsteps_list,nstdcd_list):
    """Creates a list of tuples

    Args:
        nsteps_list (list): values for nsteps
        nstdcd_list (list): values for nstdcd

    Returns:
        [list]: all possible pairs as tuples for nsteps_list and nstdcd_list
    """
    print (nsteps_list)

    print (nstdcd_list)

    map = [(x,y) for x in nsteps_list for y in nstdcd_list]
    #print (map)
    for pair in map:
        #print (pair)
        if type(pair[0]) == int and type(pair[1]) == int:
            if pair[0]/pair[1] < 20:    
                map.remove(pair)
        else:
            break

    return map

def createStepsMap(param1,param2):
    """Function to create a parameter map with all possible pairs for the range given in the parameter.yaml

    Args:
        param1 (list): startpoint, endpoint, steps size for nsteps
        param2 (list): startpoint, endpoint, steps size for nstdcd

    Returns:
        [list]: list of tuples
    """
    nsteps_list = createList(param1[0],param1[1],param1[2])
    #print (nsteps_list)
    nstdcd_list = createList(param2[0],param2[1],param2[2])
    #print (nstdcd_list)
    steps_map = createMap(nsteps_list,nstdcd_list)
    return steps_map

def load_param_yaml(parameter):
    """[summary]

    Args:
        parameter (string): path to parameter.yaml

    Raises:
        KeyError: Yaml Error

    Returns:
        [dictionary]: Benchmarking parameters
    """

    with open(f"{parameter}", 'r') as stream:
        try:
            settingsMap = yaml.load(stream)
        except yaml.YAMLError as exc:
            print(exc)

    if settingsMap['simulation']['parameters'].get('nstep') == None or settingsMap['simulation']['parameters'].get('nstdcd') == None:
        raise KeyError('nsteps or nstdcd is not defined in parameter file')

    return settingsMap

def check_folder(path):
    """checks for folder Lig, Unk, Unl

    Args:
        path (string): input path

    Returns:
        [string]: existing folder name
    """
    dir_list =  [x[0] for x in os.walk(path)]

    for dir in dir_list:
        if '/lig' in dir:
            tlc = 'LIG'
        elif '/unk' in dir:
            tlc = 'UNK'
        elif '/unl' in dir:
            tlc = 'UNL'

    return tlc


def create_input_yaml(path,pairs): 
    """creates input.yaml for given system to a given path

    Args:
        path (string): location path
        pairs (string): benchmarking systems
    """

    for pair in pairs:

        new_path = path + 'output/' + pair[0] + '-' + pair[1] + '-solvation-free-energy'

        try:
            os.makedirs(new_path)       
        except FileExistsError:       
            pass

        tlc1 = check_folder(path + pair[0])
        tlc2 = check_folder(path + pair[1])

        input_dict = {'system': 
        {'structure1': 
        {'name': pair[0], 
        'tlc': tlc1, 
        'vacuum': 
        {'dirname': 'vacuum', 
        'psf_file_name': 'step3_charmm2omm', 
        'crd_file_name': 'step3_charmm2omm', 
        'rst_file_name': 'step4_equilibration',
        'simulation_parameter': 'step5_production.inp', 
        'intermediate-filename': 'lig_in_vacuum'}, 
        'waterbox':
        {'dirname': 'waterbox', 
        'psf_file_name': 'step3_charmm2omm', 
        'crd_file_name': 'step3_charmm2omm', 
        'rst_file_name': 'step4_equilibration', 
        'simulation_parameter': 'step5_production.inp', 
        'intermediate-filename': 'lig_in_waterbox'}}, 
        'structure2': 
        {'name': pair[1], 
        'tlc': tlc2, 
        'vacuum': 
        {'dirname': 'vacuum', 
        'psf_file_name': 'step3_charmm2omm', 
        'crd_file_name': 'step3_charmm2omm', 
        'rst_file_name': 'step4_equilibration', 
        'simulation_parameter': 'step5_production.inp', 
        'intermediate-filename': 'lig_in_vacuum'}, 
        'waterbox': 
        {'dirname': 'waterbox', 
        'psf_file_name': 'step3_charmm2omm', 
        'crd_file_name': 'step3_charmm2omm', 
        'rst_file_name': 'step4_equilibration', 
        'simulation_parameter': 'step5_production.inp', 
        'intermediate-filename': 'lig_in_waterbox'}}}, 
        'simulation': 
        {'parameters': 
        {'nstep': 500000, 
        'nstdcd': 10000, 
        'nstout': 10000, 
        'cons': 'None', 
        'dt': 0.001}, 
        'free-energy-type': 'solvation-free-energy'}, 
        'solvation': 
        {'steps_for_equilibration': 1000}}

        with open(f"{new_path + '/input.yaml'}", 'w') as file:
            try:
                yaml.dump(input_dict, file)
            except yaml.YAMLError as exc:
                print(exc)

def output_yaml(path,input_dict):
    """Creates an output yaml file 

    Args:
        path (string): path to save the file to
        input_dict (dictionary): otput dictionary
    """
    with open(f"{path + '/output/output.yaml'}", 'w') as file:
            try:
                yaml.dump(input_dict, file)
            except yaml.YAMLError as exc:
                print(exc)


def reverse(tuples): 
    """reverses a tuple

    Args:
        tuples (tuple): input tuple (2)

    Returns:
        [tuple]: output tuple (2)
    """
    new_tup = tuples[::-1] 
    return new_tup 


def targetList(input_location):
    """Creating a list of targets and removing double or reverse entries

    Args:
        input_location (string): path

    Returns:
        [list]: list of strings
    """
    targets = next(os.walk(input_location))[1]

    try:
        targets.remove('output')
    except ValueError:
        pass 
    except AttributeError:
        print("targets is not a list!")

    targets = createMap(targets,targets)

    for pair in targets:
        if pair[0] == pair[1]:    
            targets.remove(pair)

    for pair in targets:
        for other_pair in targets:
            if pair == reverse(other_pair):
                targets.remove(other_pair)

    return targets

