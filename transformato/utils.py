import os

import parmed as pm
import yaml

from io import StringIO
import io 

def get_bin_dir():
    """Returns the bin directory of this package"""
    return os.path.abspath(os.path.join(os.path.dirname(__file__), "bin"))


def get_toppar_dir():
    """Returns the bin directory of this package"""
    return os.path.abspath(os.path.join(os.path.dirname(__file__), "toppar"))


def map_lj_mutations_to_atom_idx(lj_mutations: list) -> dict:

    d = {}
    for lj in lj_mutations:
        key = lj.atoms_to_be_mutated
        d[tuple(key)] = lj

    return d


def print_mutations(mutation: dict):

    if "charge" in mutation.keys():
        for charge_mutation in mutation["charge"]:
            print("Charge mutation")
            print(f"Atoms to be mutated: {charge_mutation.atoms_to_be_mutated}")
    if "hydrogen-lj" in mutation.keys():
        print("Hydrogen LJ mutation")
        print(f"Atoms to be mutated: {charge_mutation.atoms_to_be_mutated}")
        print(f"LJ mutation to default:")

    if "transform" in mutation.keys():
        pass


def load_config_yaml(config, input_dir, output_dir):

    with open(f"{config}", "r") as stream:
        try:
            settingsMap = yaml.load(stream)
        except yaml.YAMLError as exc:
            print(exc)

    if (
        settingsMap["simulation"]["parameters"].get("nstep") == None
        or settingsMap["simulation"]["parameters"].get("nstdcd") == None
    ):
        raise KeyError("nsteps or nstdcd is not defined in config file")
    else:
        if (
            settingsMap["simulation"]["parameters"]["nstep"]
            / settingsMap["simulation"]["parameters"]["nstdcd"]
            < 20
        ):
            raise RuntimeError(
                "nsteps size and nstdcd size in config file does not match"
            )

    # set the bin, data and analysis dir
    settingsMap["bin_dir"] = get_bin_dir()
    settingsMap["analysis_dir_base"] = os.path.abspath(f"{output_dir}")
    settingsMap["data_dir_base"] = os.path.abspath(f"{input_dir}")
    system_name = f"{settingsMap['system']['structure1']['name']}-{settingsMap['system']['structure2']['name']}-{settingsMap['simulation']['free-energy-type']}"
    settingsMap["system_dir"] = f"{settingsMap['analysis_dir_base']}/{system_name}"
    settingsMap["cluster_dir"] = f"/data/local/{system_name}"

    settingsMap["system"]["structure1"][
        "charmm_gui_dir"
    ] = f"{settingsMap['data_dir_base']}/{settingsMap['system']['structure1']['name']}/"
    settingsMap["system"]["structure2"][
        "charmm_gui_dir"
    ] = f"{settingsMap['data_dir_base']}/{settingsMap['system']['structure2']['name']}/"
    settingsMap["system"]["name"] = system_name
    return settingsMap

def psf_correction(StringIo_object):
    """Correcting the issue with 2 missing spaces in the waterbox psf files"""
    file = StringIo_object.readlines()
    new_file = f"" 
    flag = False
    for line in file:
        if flag == False:
            new_file = f"{new_file}{line}"
            flag = "!NATOM" in line
        else:
            if "!NBOND" in line:
                new_file = f"{new_file}{line}"
                flag = False
            else:
                line = line.split()
                if len(line) != 11:
                    new_file = f"{new_file}\n"
                else:
                    space = " "
                    a = 10 - len(line[0])
                    b = 1
                    c = 9 - len(line[1]) 
                    d = 9 - len(line[2])
                    e = 9 - len(line[3])
                    f = 9 - len(line[4])
                    g = 17 - len(line[6]) - len(line[5])
                    h = 14 - len(line[7])
                    i = 12 - len(line[8])
                    j = 10 - len(line[9])
                    k = 18 - len(line[10])
                    new_line = f"{space*a}{line[0]}{space*b}{line[1]}{space*c}{line[2]}{space*d}{line[3]}{space*e}{line[4]}{space*f}{line[5]}{space*g}{line[6]}{space*h}{line[7]}{space*i}{line[8]}{space*j}{line[9]}{space*k}{line[10]}\n"
                    if len(new_line) == 119:
                        new_file = f"{new_file}{new_line}"
                    else:
                        print("Error")
                        
    return new_file