import logging
import os
import pickle
from collections import defaultdict, namedtuple

import matplotlib.pyplot as plt
import mdtraj
import numpy as np
import parmed as pm
from pymbar import mbar
from simtk import unit
from simtk.openmm import System, XmlSerializer
from simtk.openmm.app import Simulation
from simtk.openmm.vec3 import Vec3
from tqdm import tqdm
import seaborn as sns

logger = logging.getLogger(__name__)
kB = unit.BOLTZMANN_CONSTANT_kB * unit.AVOGADRO_CONSTANT_NA 

def return_reduced_potential(potential_energy: unit.Quantity, volume: unit.Quantity, temperature: unit.Quantity):
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
    pressure = 1.0 * unit.atmosphere  # atm

    beta = 1.0 / (unit.BOLTZMANN_CONSTANT_kB * temperature)
    reduced_potential = potential_energy / unit.AVOGADRO_CONSTANT_NA
    if volume is not None:
        reduced_potential += pressure * volume
    return beta * reduced_potential


class FreeEnergyCalculator(object):

    def __init__(self, configuration: dict, structure_name: str):
        self.configuration = configuration
        self.structure_name = structure_name
        self.envs: set = ()
        # decide if the name of the system corresponds to structure1 or structure2
        if configuration['system']['structure1']['name'] == self.structure_name:
            structure = 'structure1'
        elif configuration['system']['structure2']['name'] == self.structure_name:
            structure = 'structure2'
        else:
            raise RuntimeError(f"Could not finde structure entry for : {self.structure_name}")

        if configuration['simulation']['free-energy-type'] == 'solvation-free-energy':
            self.envs = ('vacuum', 'waterbox')
            self.mbar_results = {'waterbox': None, 'vacuum': None}

        elif configuration['simulation']['free-energy-type'] == 'binding-free-energy':
            self.envs = ('complex', 'waterbox')
            self.mbar_results = {'waterbox': None, 'complex': None}
        else:
            raise RuntimeError(f"Either binding or solvation free energy.")

        self.base_path = f"{self.configuration['system_dir']}/{self.structure_name}/"
        self.structure = structure
        self.mbar_results = {'waterbox': None, 'vacuum': None, 'complex': None}
        self.snapshost = []
        self.nr_of_states = -1
        self.N_k = []
        self.thinning = -1

    def load_trajs(self, thinning: int = 10):
        """
        load trajectories, thin trajs and merge themn.
        Also calculate N_k for mbar.
        """

        assert(type(thinning) == int)
        self.thinning = thinning
        self.snapshost, self.nr_of_states, self.N_k = self._merge_trajs()

    def _merge_trajs(self) -> (dict, int, list):
        """
        load trajectories, thin trajs and merge themn.
        Also calculate N_k for mbar.
        """

        #############
        # set all file paths for potential
        assert(os.path.isdir(f"{self.base_path}"))
        nr_of_states = len(next(os.walk(f"{self.base_path}"))[1])

        logger.info(f"Evaluating {nr_of_states} states.")
        snapshost = {}
        for env in self.envs:
            confs = []
            conf_sub = self.configuration['system'][self.structure][env]
            N_k = []
            for i in tqdm(range(1, nr_of_states+1)):
                traj = mdtraj.load(f"{self.base_path}/intst{i}/{conf_sub['intermediate-filename']}.dcd",
                                   top=f"{self.base_path}/intst{i}/{conf_sub['intermediate-filename']}.psf")

                # NOTE: removing the first 25% confs and thinning
                start = int(len(traj)/25)
                traj = traj[start::self.thinning]
                if len(traj) < 10:
                    raise RuntimeError(
                        f"Below 10 conformations per lambda ({len(traj)}) -- decrease the thinning factor (currently: {self.thinning}).")

                confs.append(traj)
                logger.info(f"Nr of snapshots: {len(traj)}")
                N_k.append(len(traj))

            joined_trajs = mdtraj.join(confs, check_topology=True)
            logger.info(f"Combined nr of snapshots: {len(joined_trajs)}")
            snapshost[env] = joined_trajs

        return snapshost, nr_of_states, N_k

    def _analyse_results_using_mbar(self, env: str, snapshots: mdtraj.Trajectory, nr_of_states: int, save_results: bool):

        def _energy_at_ts(simulation: Simulation, coordinates, bxl):
            """
            Calculates the potential energy with the correct periodic boundary conditions.
            """
            if bxl:
                a = Vec3(bxl.value_in_unit(unit.nanometer), 0.0, 0.0)
                b = Vec3(0.0, bxl.value_in_unit(unit.nanometer), 0.0)
                c = Vec3(0.0, 0.0, bxl.value_in_unit(unit.nanometer))
                simulation.context.setPeriodicBoxVectors(a, b, c)
            simulation.context.setPositions((coordinates))
            state = simulation.context.getState(getEnergy=True)
            return state.getPotentialEnergy()

        def _evaluated_e_on_all_snapshots(snapshots, i: int, env: str):

            # read in necessary files
            conf_sub = self.configuration['system'][self.structure][env]
            file_name = f"{self.base_path}/intst{i}/{conf_sub['intermediate-filename']}_system.xml"
            system = XmlSerializer.deserialize(open(file_name).read())
            file_name = f"{self.base_path}/intst{i}/{conf_sub['intermediate-filename']}_integrator.xml"
            integrator = XmlSerializer.deserialize(open(file_name).read())
            psf_file_path = f"{self.base_path}/intst{i}/{conf_sub['intermediate-filename']}.psf"
            psf = pm.charmm.CharmmPsfFile(psf_file_path)

            # generate simulations object and set states
            simulation = Simulation(psf.topology, system, integrator)
            simulation.context.setState(XmlSerializer.deserialize(
                open(f"{self.base_path}/intst{i}/{conf_sub['intermediate-filename']}.rst", 'r').read()))

            energies = []
            for ts in tqdm(range(snapshots.n_frames)):
                if env == 'vacuum':
                    bxl = None
                    volumn = None
                else:
                    # extract the box size at the given ts
                    bxl = snapshots.unitcell_lengths[ts][0] * (unit.nanometer)
                    volumn = bxl ** 3
                # calculate the potential energy
                e = _energy_at_ts(simulation, snapshots.openmm_positions(ts), bxl)
                # obtain the reduced potential (for NpT)
                red_e = return_reduced_potential(e, volumn, 303.15 * unit.kelvin)
                energies.append(red_e)
            return np.array(energies)


        ##### main
        u_kn = np.stack(
            [_evaluated_e_on_all_snapshots(snapshots, i, env) for i in range(1, self.nr_of_states+1)]
        )

        if save_results:
            file = f"{self.save_results_to_path}/mbar_data_for_{self.structure_name}_in_{env}.pickle"
            logger.info(f"Saving results: {file}")
            results = {'u_kn': u_kn, 'N_k': self.N_k}
            pickle.dump(results, open(file, 'wb+'))

        return mbar.MBAR(u_kn, self.N_k)

    def calculate_dG_to_common_core(self, save_results=True):
        """
        Calculate mbar results or load save results from a serialized mbar results.
        """
        if save_results:
            os.makedirs(f"{self.configuration['system_dir']}/results/", exist_ok=True)
            self.save_results_to_path = f"{self.configuration['system_dir']}/results/"
            logger.info(f"Saving results in {self.save_results_to_path}")

        for env in self.envs:
            logger.info(f"Generating results for {env}.")
            self.mbar_results[env] = self._analyse_results_using_mbar(
                env, self.snapshost[env], self.nr_of_states, save_results)

    def load_waterbox_results(self, file):
        self.mbar_results['waterbox'] = self._load_mbar_results(file)

    def load_complex_results(self, file):
        self.mbar_results['complex'] = self._load_mbar_results(file)

    def load_vacuum_results(self, file):
        self.mbar_results['vacuum'] = self._load_mbar_results(file)

    def _load_mbar_results(self, file):
        results = pickle.load(open(file, 'rb'))
        return mbar.MBAR(results['u_kn'], results['N_k'])

    def free_energy_differences(self, env='vacuum'):
        """matrix of free energy differences"""
        try:
            r = self.mbar_results[env].getFreeEnergyDifferences(return_dict=True)['Delta_f']
        except KeyError:
            raise KeyError(f"Free energy difference not obtained for : {env}")
        return r

    def free_energy_overlap(self, env='vacuum'):
        """overlap of lambda states"""
        try:
            r = self.mbar_results[env].computeOverlap(return_dict=True)['matrix']
        except KeyError:
            raise KeyError(f"Free energy overlap not obtained for : {env}")

        return r

    def free_energy_difference_uncertainties(self, env='vacuum'):
        """matrix of asymptotic uncertainty-estimates accompanying free energy differences"""
        try:
            r = self.mbar_results[env].getFreeEnergyDifferences(return_dict=True)['dDelta_f']
        except KeyError:
            raise KeyError(f"Free energy uncertanties not obtained for : {env}")
        return r

    @property
    def waterbox_free_energy_differences(self):
        """matrix of free energy differences"""
        return self.free_energy_differences(env='waterbox')

    @property
    def complex_free_energy_differences(self):
        """matrix of free energy differences"""
        return self.free_energy_differences(env='complex')

    @property
    def vacuum_free_energy_differences(self):
        """matrix of free energy differences"""
        return self.free_energy_differences(env='vacuum')

    @property
    def waterbox_free_energy_difference_uncertanties(self):
        """matrix of free energy differences"""
        return self.free_energy_difference_uncertainties(env='waterbox')

    @property
    def complex_free_energy_difference_uncertanties(self):
        """matrix of free energy differences"""
        return self.free_energy_difference_uncertainties(env='complex')

    @property
    def vacuum_free_energy_difference_uncertanties(self):
        """matrix of free energy differences"""
        return self.free_energy_difference_uncertainties(env='vacuum')

    @property
    def waterbox_free_energy_difference_overlap(self):
        """matrix of free energy differences"""
        return self.free_energy_overlap(env='waterbox')

    @property
    def complex_free_energy_difference_overlap(self):
        """matrix of free energy differences"""
        return self.free_energy_overlap(env='complex')

    @property
    def vacuum_free_energy_difference_overlap(self):
        """matrix of free energy differences"""
        return self.free_energy_overlap(env='vacuum')

    def plot_free_energy_overlap(self, env):
        plt.figure(figsize=[8, 8], dpi=300)
        if env == 'vacuum':
            ax = sns.heatmap(self.vacuum_free_energy_difference_overlap, cmap='Blues', linewidth=0.5)
        elif env == 'waterbox':
            ax = sns.heatmap(self.waterbox_free_energy_difference_overlap, cmap='Blues', linewidth=0.5)
        elif env == 'complex':
            ax = sns.heatmap(self.complex_free_energy_difference_overlap, cmap='Blues', linewidth=0.5)
        else:
            raise RuntimeError()
        plt.title(f"Overlap of lambda states for ligand in {env}", fontsize=15)
        plt.xlabel('lambda state (0 to 1)', fontsize=15)
        plt.ylabel('lambda state (0 to 1)', fontsize=15)
        plt.legend()
        plt.show()
        plt.close()

    def plot_free_energy(self, env):

        if env == 'vacuum':
            x = [a for a in np.linspace(0, 1, len(self.vacuum_free_energy_differences[0]))]
            y = self.vacuum_free_energy_differences[0]
            y_error = self.vacuum_free_energy_difference_uncertanties[0]
        elif env == 'waterbox':
            x = [a for a in np.linspace(0, 1, len(self.waterbox_free_energy_differences[0]))]
            y = self.waterbox_free_energy_differences[0]
            y_error = self.waterbox_free_energy_difference_uncertanties[0]
        elif env == 'complex':
            x = [a for a in np.linspace(0, 1, len(self.complex_free_energy_differences[0]))]
            y = self.complex_free_energy_differences[0]
            y_error = self.complex_free_energy_difference_uncertanties[0]
        else:
            raise RuntimeError()

        plt.errorbar(x, y, yerr=y_error, label='ddG +- stddev [kT]')
        plt.title(f"Free energy estimate for ligand in {env}", fontsize=15)
        plt.ylabel('Free energy estimate in kT', fontsize=15)
        plt.xlabel('lambda state (0 to 1)', fontsize=15)
        plt.show()
        plt.close()

    def plot_vacuum_free_energy_overlap(self):
        self.plot_free_energy_overlap('vacuum')

    def plot_complex_free_energy_overlap(self):
        self.plot_free_energy_overlap('complex')

    def plot_waterbox_free_energy_overlap(self):
        self.plot_free_energy_overlap('waterbox')

    def plot_vacuum_free_energy(self):
        self.plot_free_energy('vacuum')

    def plot_complex_free_energy(self):
        self.plot_free_energy('complex')

    def plot_waterbox_free_energy(self):
        self.plot_free_energy('waterbox')

    @property
    def end_state_free_energy_difference(self):
        """DeltaF[lambda=1 --> lambda=0]"""
        K = len(self.waterbox_free_energy_differences)
        if self.configuration['simulation']['free-energy-type'] == 'solvation-free-energy':
            return (self.waterbox_free_energy_differences[0, K-1] - self.vacuum_free_energy_differences[0, K-1],
                    self.waterbox_free_energy_difference_uncertanties[0, K-1] + self.vacuum_free_energy_difference_uncertanties[0, K-1])

        elif self.configuration['simulation']['free-energy-type'] == 'binding-free-energy':
            return (self.complex_free_energy_differences[0, K-1] - self.waterbox_free_energy_differences[0, K-1],
                    self.complex_free_energy_difference_uncertanties[0, K-1] + self.waterbox_free_energy_difference_uncertanties[0, K-1])
        else:
            raise RuntimeError()

    def show_summary(self):
        if self.configuration['simulation']['free-energy-type'] == 'solvation-free-energy':
            self.plot_vacuum_free_energy_overlap()
            self.plot_waterbox_free_energy_overlap()
            self.plot_vacuum_free_energy()
            self.plot_waterbox_free_energy()
        else:
            self.plot_complex_free_energy_overlap()
            self.plot_waterbox_free_energy_overlap()
            self.plot_complex_free_energy()
            self.plot_waterbox_free_energy()

        energy_estimate, uncertanty = self.end_state_free_energy_difference
        print(f"Free energy to common core: {energy_estimate} [kT] with uncertanty: {uncertanty} [kT].")
