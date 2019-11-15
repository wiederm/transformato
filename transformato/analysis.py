import logging
from simtk import unit
import parmed as pm
from simtk.openmm import XmlSerializer, System
from simtk.openmm.app import Simulation
import mdtraj
import numpy as np
from pymbar import mbar
from simtk.openmm.vec3 import Vec3
import json
from collections import defaultdict, namedtuple

logger = logging.getLogger(__name__)

def return_reduced_potential(potential_energy:unit.Quantity, volume:unit.Quantity, temperature:unit.Quantity):
    """Retrieve the reduced potential for a given context.
    The reduced potential is defined as in Ref. [1]
    u = \beta [U(x) + p V(x)]
    where the thermodynamic parameters are
    \beta = 1/(kB T) is the inverse temperature
    p is the pressure
    and the configurational properties are
    x the atomic positions
    U(x) is the potential energy
    V(x) is the instantaneous box volume
    References
    ----------
    [1] Shirts MR and Chodera JD. Statistically optimal analysis of
    equilibrium states. J Chem Phys 129:124105, 2008.


    Parameters
    ----------
    potential_energy : simtk.unit of float
    context
    ensamble: NVT or NPT
    """

    assert(type(temperature) == unit.Quantity)
    assert(type(volume) == unit.Quantity)

    pressure = 1.0 * unit.atmosphere # atm      

    beta = 1.0 / (unit.BOLTZMANN_CONSTANT_kB * temperature)
    reduced_potential = potential_energy / unit.AVOGADRO_CONSTANT_NA
    if pressure is not None:
        reduced_potential += pressure * volume
    return beta * reduced_potential



def calculate_energies_with_potential_on_conf(env:str, potential:int, conformations:int, structure_name:str, configuration:dict)->list:

    """
    Uses the potential defined with the topology and parameters to evaluate 
    conformations. Returns a list of unitless energies.
    Parameters
    ----------
    env : str
        either 'complex' or 'waterbox
    potential : int
        the intermediate state that defines the potential
    conformations : int
        the intermediate state from which the conformations are evaluated
    structure_name : str
        the name of the structure that is evaluated as indicated in configuration['system']['structure1']['name']
    configuration : dict
        the configuration dict
    """
    assert(env == 'waterbox' or env == 'complex')
    assert(type(conformations) == int)

    def _energy_at_ts(simulation:Simulation, coordinates, bxl:unit.Quantity):
        """
        Calculates the potential energy with the correct periodic boundary conditions.
        """
        a = Vec3(bxl.value_in_unit(unit.nanometer), 0.0, 0.0)
        b = Vec3(0.0, bxl.value_in_unit(unit.nanometer), 0.0)
        c = Vec3(0.0, 0.0, bxl.value_in_unit(unit.nanometer))
        simulation.context.setPeriodicBoxVectors(a,b,c)
        simulation.context.setPositions((coordinates))
        state = simulation.context.getState(getEnergy=True)
        return state.getPotentialEnergy()

    def _setup_calculation(psf_file_path:str, traj_file_path:str, simulation:Simulation):
        """
        Loops over the conformations in the trajectory and evaluates every frame using the 
        potential energy function defined in the simulation object
        """
        list_e = []
        # load traj
        traj  = mdtraj.load(traj_file_path, top=psf_file_path)
        for ts in range(traj.n_frames):
            # extract the box size at the given ts
            bxl = traj.unitcell_lengths[ts][0] * (unit.nanometer)
            # calculate the potential energy 
            e = _energy_at_ts(simulation, traj.openmm_positions(ts), bxl)
            # obtain the reduced potential (for NpT)
            volumn = bxl ** 3
            red_e = return_reduced_potential(e, volumn, 300 * unit.kelvin)
            list_e.append(red_e)
        return list_e
    
    # decide if the name of the system corresponds to structure1 or structure2
    if configuration['system']['structure1']['name'] == structure_name:
        structure = 'structure1'
    elif configuration['system']['structure2']['name'] == structure_name:
        structure = 'structure2'
    else:
        raise RuntimeError(f"Could not finde structure entry for : {structure_name}")


    #############
    # set all file paths for potential
    conf_sub = configuration['system'][structure][env]
    base = f"{configuration['analysis_dir_base']}/{structure_name}/"

    file_name = f"{base}/intst{potential}/{conf_sub['intermediate-filename']}_system.xml"
    system  = XmlSerializer.deserialize(open(file_name).read())

    file_name = f"{base}/intst{potential}/{conf_sub['intermediate-filename']}_integrator.xml"
    integrator  = XmlSerializer.deserialize(open(file_name).read())

    psf_file_path = f"{base}/intst{potential}/{conf_sub['intermediate-filename']}.psf"
    psf = pm.charmm.CharmmPsfFile(psf_file_path)

    # generate simulations object and set states
    simulation = Simulation(psf.topology, system, integrator)
    simulation.context.setState(XmlSerializer.deserialize(open(f"{base}/intst{potential}/{conf_sub['intermediate-filename']}.rst", 'r').read()))
    
    # set path to conformations
    logger.info('#############')
    logger.info('- Energy evaluation with potential from lambda: {}'.format(str(potential)))
    logger.info('  - Looking at conformations from lambda: {}'.format(str(conformations)))
    traj_file_path = f"{base}/intst{conformations}/{conf_sub['intermediate-filename']}.dcd"

    # calculate pot energy using the potential on the conformations
    energy = _setup_calculation(psf_file_path, traj_file_path, simulation)

    return energy


    
class FreeEnergyCalculator(object):
    
    def __init__(self, configuration:dict, nr_of_states:int, structure:str):
        self.configuration = configuration
        self.nr_of_states = nr_of_states
        self.structure = structure
        
        self.waterbox_mbar = None
        self.complex_mbar = None

        self._parse_files()
        self._calculate_dG_to_common_core()

    def _calculate_dG_to_common_core(self):

        def _analyse_results_using_mbar(results_dict:dict, nr_of_states:int):

            nr_of_conformations_per_state = int(len(results_dict[1][1])) # => there is always a [0][0] entry
            test = np.full(shape=nr_of_states, fill_value=nr_of_conformations_per_state)
            u_kln = []
            for u_for_traj in sorted(results_dict):
                u_kn = []
                for u_x in results_dict[u_for_traj]:
                    u_kn.extend((results_dict[u_for_traj][u_x]))
                u_kln.append(u_kn)       

            u_kln = np.asanyarray(u_kln)
            return mbar.MBAR(u_kln, test)


        self.waterbox_mbar = _analyse_results_using_mbar(self.r_waterbox_state, self.nr_of_states)
        self.complex_mbar =   _analyse_results_using_mbar(self.r_complex_state, self.nr_of_states)

    def _parse_files(self)->(dict,dict):

        r_waterbox_state = defaultdict(dict)
        r_complex_state = defaultdict(dict)
        for i in range(1, self.nr_of_states+1):
            for j in range(1, self.nr_of_states+1):
                file_path = f"{self.configuration['system_dir']}/results/energy_{self.structure}_{i}_{j}.json"
                f = open(file_path, 'r')
                r = json.load(f)
                r_waterbox_state[i][j] = r['waterbox']
                r_complex_state[i][j] = r['complex']
                f.close()
        
        self.r_waterbox_state = r_waterbox_state
        self.r_complex_state = r_complex_state 


    @property
    def complex_free_energy_differences(self):
        """matrix of free energy differences"""
        return self.complex_mbar.getFreeEnergyDifferences()[0]
    
    @property
    def complex_free_energy_overlap(self):
        """overlap of lambda states"""
        return self.complex_mbar.computeOverlap()

    @property
    def complex_free_energy_difference_uncertainties(self):
        """matrix of asymptotic uncertainty-estimates accompanying free energy differences"""
        return self.complex_mbar.getFreeEnergyDifferences()[1]
    
    @property
    def waterbox_free_energy_differences(self):
        """matrix of free energy differences"""
        return self.waterbox_mbar.getFreeEnergyDifferences()[0]

    @property
    def waterbox_free_energy_overlap(self):
        """overlap of lambda states"""
        return self.waterbox_mbar.computeOverlap()
    
    @property
    def waterbox_free_energy_difference_uncertainties(self):
        """matrix of asymptotic uncertainty-estimates accompanying free energy differences"""
        return self.waterbox_mbar.getFreeEnergyDifferences()[1]


    @property
    def end_state_free_energy_difference(self):
        """DeltaF[lambda=1 --> lambda=0]"""
        waterbox_DeltaF_ij, waterbox_dDeltaF_ij, _ = self.waterbox_mbar.getFreeEnergyDifferences()
        complex_DeltaF_ij, complex_dDeltaF_ij, _ = self.complex_mbar.getFreeEnergyDifferences()
        K = len(complex_DeltaF_ij)
        return complex_DeltaF_ij[0, K-1] - waterbox_DeltaF_ij[0, K-1], waterbox_dDeltaF_ij[0, K-1] + complex_dDeltaF_ij[0, K-1] 



