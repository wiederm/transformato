import os

#import parmed as pm //called but not used in this file
import yaml


def get_bin_dir():
    """Returns the bin directory of this package"""
    return os.path.abspath(os.path.join(os.path.dirname(__file__), 'bin'))


def get_toppar_dir():
    """Returns the bin directory of this package"""
    return os.path.abspath(os.path.join(os.path.dirname(__file__), 'toppar'))


def load_config_yaml(config, input_dir, output_dir):

    with open(f"{config}", 'r') as stream:
        try:
            settingsMap = yaml.load(stream)
        except yaml.YAMLError as exc:
            print(exc)

    if settingsMap['simulation']['parameters'].get('nstep') == None or settingsMap['simulation']['parameters'].get('nstdcd') == None:
        raise KeyError('nsteps or nstdcd is not defined in config file')
    else:
        if settingsMap['simulation']['parameters']['nstep']/settingsMap['simulation']['parameters']['nstdcd'] < 20:
            raise RuntimeError('nsteps size and nstdcd size in config file does not match')

    # set the bin, data and analysis dir
    settingsMap['bin_dir'] = get_bin_dir()
    settingsMap['analysis_dir_base'] = os.path.abspath(f"{output_dir}")
    settingsMap['data_dir_base'] = os.path.abspath(f"{input_dir}")
    system_name = f"{settingsMap['system']['structure1']['name']}-{settingsMap['system']['structure2']['name']}-{settingsMap['simulation']['free-energy-type']}"
    settingsMap['system_dir'] = f"{settingsMap['analysis_dir_base']}/{system_name}"
    settingsMap['cluster_dir'] = f"/data/local/{system_name}"

    settingsMap['system']['structure1']['charmm_gui_dir'] = f"{settingsMap['data_dir_base']}/{settingsMap['system']['structure1']['name']}/"
    settingsMap['system']['structure2']['charmm_gui_dir'] = f"{settingsMap['data_dir_base']}/{settingsMap['system']['structure2']['name']}/"
    settingsMap['system']['name'] = system_name
    return settingsMap
