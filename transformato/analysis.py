from collections import defaultdict
import logging
import os
import pickle
import multiprocessing as mp
import matplotlib.pyplot as plt
import mdtraj
import numpy as np
import seaborn as sns
from pymbar import mbar
from simtk import unit
from simtk.openmm import XmlSerializer, Platform
from simtk.openmm.app import CharmmPsfFile, Simulation
from tqdm import tqdm
import subprocess
from typing import Set, Tuple, List
from transformato.utils import get_structure_name
from transformato.constants import temperature
import copy

logger = logging.getLogger(__name__)


def return_reduced_potential(
    potential_energy: unit.Quantity, volume: unit.Quantity, temperature: unit.Quantity
) -> float:
    """Retrieve the reduced potential for a given context.
    The reduced potential is defined as in Ref. [1]
    u = \beta [U(x) + p V(x)]
    where the thermodynamic parameters are
    \beta = 1/(kB T) is the inverse temperature
    p is the pressure
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

    assert type(temperature) == unit.Quantity
    pressure = 1.0 * unit.atmosphere  # atm
    kB = unit.BOLTZMANN_CONSTANT_kB * unit.AVOGADRO_CONSTANT_NA
    beta = 1.0 / (kB * temperature)
    # potential_energy += pressure * volume
    return beta * potential_energy


class FreeEnergyCalculator(object):
    """
    FreeEnergyCalculator [summary]

    Parameters
    ----------
    object : [type]
        [description]
    """

    def __init__(self, configuration: dict, structure_name: str):
        self.configuration = configuration
        self.structure_name = structure_name
        self.envs: Tuple = ()
        # decide if the name of the system corresponds to structure1 or structure2
        structure = get_structure_name(configuration, structure_name)

        if configuration["simulation"]["free-energy-type"] == "rsfe":
            self.envs = ("vacuum", "waterbox")
            self.mbar_results = {"waterbox": None, "vacuum": None}

        elif configuration["simulation"]["free-energy-type"] == "rbfe":
            self.envs = ("complex", "waterbox")
            self.mbar_results = {"waterbox": None, "complex": None}
        else:
            raise RuntimeError(f"Either binding or solvation free energy.")

        self.base_path = f"{self.configuration['system_dir']}/{self.structure_name}/"
        self.structure = structure
        self.mbar_results: dict = {"waterbox": None, "vacuum": None, "complex": None}
        self.snapshots: dict = {}
        self.nr_of_states: int = 0
        self.N_k: dict = {}
        self.thinning: int = 0
        self.save_results_to_path: str = f"{self.configuration['system_dir']}/results/"

    def load_trajs(self, nr_of_max_snapshots: int = 300):
        """
        load trajectories, thin trajs and merge themn.
        Also calculate N_k for mbar.
        """

        assert type(nr_of_max_snapshots) == int
        self.nr_of_max_snapshots = nr_of_max_snapshots
        self.snapshots, self.nr_of_states, self.N_k = self._merge_trajs()

    def _generate_openMM_system(self, env: str, lambda_state: int) -> Simulation:
        # read in necessary files
        conf_sub = self.configuration["system"][self.structure][env]
        file_name = f"{self.base_path}/intst{lambda_state}/{conf_sub['intermediate-filename']}_system.xml"
        system = XmlSerializer.deserialize(open(file_name).read())
        file_name = f"{self.base_path}/intst{lambda_state}/{conf_sub['intermediate-filename']}_integrator.xml"
        integrator = XmlSerializer.deserialize(open(file_name).read())
        psf_file_path = f"{self.base_path}/intst{lambda_state}/{conf_sub['intermediate-filename']}.psf"
        psf = CharmmPsfFile(psf_file_path)

        # generate simulations object and set states
        if self.configuration["simulation"]["GPU"] == True:
            platform = Platform.getPlatformByName(
                "CUDA"
            )  # NOTE: FIXME: this needs to be set dynamically
            platformProperties = {"CudaPrecision": "mixed"}

            simulation = Simulation(
                psf.topology, system, integrator, platform, platformProperties
            )
        else:
            platform = Platform.getPlatformByName(
                "CPU"
            )  # NOTE: FIXME: this needs to be set dynamically
            simulation = Simulation(psf.topology, system, integrator, platform)

        simulation.context.setState(
            XmlSerializer.deserialize(
                open(
                    f"{self.base_path}/intst{lambda_state}/{conf_sub['intermediate-filename']}.rst",
                    "r",
                ).read()
            )
        )
        return simulation

    def _thinning_traj(self, traj):
        lenght = int(len(traj))
        start = int(lenght / 4)
        traj = traj[start:]  # remove the first 25% confs
        new_length = int(len(traj))
        further_thinning = max(
            int(new_length / self.nr_of_max_snapshots), 1
        )  # thinning
        return traj[::further_thinning][: self.nr_of_max_snapshots]

    def _merge_trajs(self) -> Tuple[dict, int, dict]:
        """
        load trajectories, thin trajs and merge themn.
        Also calculate N_k for mbar.
        """

        #############
        # set all file paths for potential
        if not os.path.isdir(f"{self.base_path}"):
            raise RuntimeError(f"{self.base_path} does not exist. Aborting.")

        nr_of_states = len(next(os.walk(f"{self.base_path}"))[1])

        logger.info(f"Evaluating {nr_of_states} states.")
        snapshots: dict = {}
        N_k: dict = defaultdict(list)
        for env in self.envs:
            confs = []
            conf_sub = self.configuration["system"][self.structure][env]
            for lambda_state in tqdm(range(1, nr_of_states + 1)):
                dcd_path = f"{self.base_path}/intst{lambda_state}/{conf_sub['intermediate-filename']}.dcd"
                if not os.path.isfile(dcd_path):
                    raise RuntimeError(f"{dcd_path} does not exist.")
                traj = mdtraj.load(
                    f"{dcd_path}",
                    top=f"{self.base_path}/intst{lambda_state}/{conf_sub['intermediate-filename']}.psf",
                )
                logger.info(f"Before: {len(traj)}")
                # NOTE: removing the first 25% confs and thinning
                traj = self._thinning_traj(traj)
                if len(traj) < 10:
                    raise RuntimeError(
                        f"Below 10 conformations per lambda ({len(traj)}) -- decrease the thinning factor (currently: {self.thinning})."
                    )

                confs.append(traj)
                logger.info(f"{dcd_path}")
                logger.info(f"Nr of snapshots: {len(traj)}")
                N_k[env].append(len(traj))

            joined_trajs = mdtraj.join(confs, check_topology=True)
            logger.info(f"Combined nr of snapshots: {len(joined_trajs)}")
            snapshots[env] = joined_trajs

        return (snapshots, nr_of_states, N_k)

    @staticmethod
    def _energy_at_ts(
        simulation: Simulation, traj: mdtraj.Trajectory, ts: int, env: str
    ):
        """
        Calculates the potential energy with the correct periodic boundary conditions.
        """
        if env != "vacuum":
            simulation.context.setPeriodicBoxVectors(
                traj.unitcell_vectors[ts][0],
                traj.unitcell_vectors[ts][1],
                traj.unitcell_vectors[ts][2],
            )
        simulation.context.setPositions(traj.openmm_positions(ts))
        state = simulation.context.getState(getEnergy=True)
        return state.getPotentialEnergy()

    @staticmethod
    def _get_V_for_ts(snapshots: mdtraj.Trajectory, env: str, ts: int):
        if env == "vacuum":
            volumn = (0.0 * unit.nanometer) ** 3
        else:
            # extract the box size at the given ts
            bxl_x = snapshots.unitcell_lengths[ts][0] * (unit.nanometer)
            bxl_y = snapshots.unitcell_lengths[ts][1] * (unit.nanometer)
            bxl_z = snapshots.unitcell_lengths[ts][2] * (unit.nanometer)

            volumn = bxl_x * bxl_y * bxl_z
        return volumn

    @staticmethod
    def _parse_CHARMM_energy_output(path: str, env: str) -> list:
        import math

        pot_energies = []
        file_name = ""
        if env == "waterbox":
            file_name = f"{path}/ener_solv.log"
        elif env == "vacuum":
            file_name = f"{path}/ener_vac.log"

        with open(file_name, "r") as f:
            for line in f.readlines():
                try:
                    v = float(line)
                except ValueError:
                    v = float(999999.0)

                if math.isinf(v) or math.isnan(v):
                    v = float(999999.0)

                v *= unit.kilocalorie_per_mole
                pot_energies.append(v)

        assert len(pot_energies) > 50
        return pot_energies

    def _evaluate_traj_with_CHARMM(
        self, path: str, env: str, volumn_list: list = []
    ) -> list:

        script_name = ""
        if env == "waterbox":
            script_name = "charmm_evaluate_energy_in_solv.inp"
            assert len(volumn_list) > 1
        elif env == "vacuum":
            script_name = "charmm_evaluate_energy_in_vac.inp"

        top = self.configuration["system"][self.structure][env]["intermediate-filename"]

        exe = subprocess.run(
            [
                "bash",
                f"{self.configuration['bin_dir']}/charmm_eval_energy.sh",
                str(path),
                str(top),
                str(script_name),
            ],
            check=True,
            capture_output=True,
            text=True,
            encoding="latin1",
        )
        # path=$1     # path in which the simulation will start
        # top=$2      # top file to use
        # script=$3   # which script is called

        with open(f"{path}/eval_charmm_{env}.log", "w+") as f:
            f.write("Capture stdout")
            f.write(exe.stdout)
            f.write("Capture stderr")
            f.write(exe.stderr)

        pot_energies = self._parse_CHARMM_energy_output(path, env)

        logger.info(f"Number of entries in pot_energies list: {len(pot_energies)}")
        logger.info(f"Number of entries in pot_energies list: {len(volumn_list)}")

        if volumn_list:
            assert len(volumn_list) == len(pot_energies)
            return [
                return_reduced_potential(e, volume=V, temperature=temperature)
                for e, V in zip(pot_energies, volumn_list)
            ]
        else:
            return [
                return_reduced_potential(
                    e, volume=(0.0 * unit.nanometer) ** 3, temperature=temperature
                )
                for e in pot_energies
            ]

    def _evaluated_e_on_all_snapshots_CHARMM(
        self, snapshots: mdtraj.Trajectory, lambda_state: int, env: str
    ):

        if env == "waterbox":
            volumn_list = [
                self._get_V_for_ts(snapshots, env, ts)
                for ts in range(snapshots.n_frames)
            ]

        elif env == "vacuum":
            volumn_list = []
        else:
            raise RuntimeError(f"{env}")

        return self._evaluate_traj_with_CHARMM(
            path=f"{self.base_path}/intst{lambda_state}/",
            env=env,
            volumn_list=volumn_list,
        )

    def _evaluated_e_on_all_snapshots_openMM(
        self, snapshots: mdtraj.Trajectory, lambda_state: int, env: str
    ) -> np.ndarray:

        simulation = self._generate_openMM_system(env=env, lambda_state=lambda_state)
        energies = []
        volumn_list = [
            self._get_V_for_ts(snapshots, env, ts) for ts in range(snapshots.n_frames)
        ]
        for ts in tqdm(range(snapshots.n_frames)):
            volumn = volumn_list[ts]
            # calculate the potential energy
            e = self._energy_at_ts(simulation, snapshots, ts, env)
            # obtain the reduced potential (for NpT)
            red_e = return_reduced_potential(e, volumn, temperature)
            energies.append(red_e)
        return np.array(energies)

    def _analyse_results_using_mbar(
        self,
        env: str,
        snapshots: mdtraj.Trajectory,
        save_results: bool,
        engine: str,
    ):
        from itertools import repeat
        from transformato.constants import NUM_PROC

        #########################################################
        #########################################################
        # main
        logger.debug(f"Evaluating with {engine}")
        logger.debug(f"using {NUM_PROC} processes for the analysis")
        ctx = mp.get_context("fork")
        pool = ctx.Pool(processes=NUM_PROC)

        if engine == "openMM":
            lambda_states = [
                lambda_state for lambda_state in range(1, self.nr_of_states + 1)
            ]
            r = pool.starmap(
                self._evaluated_e_on_all_snapshots_openMM,
                zip(repeat(snapshots), lambda_states, repeat(env)),
            )

            u_kn = np.stack([r_i for r_i in r])

        elif engine == "CHARMM":
            # write out traj in self.base_path
            snapshots.save_dcd(f"{self.base_path}/traj.dcd")
            u_kn = np.stack(
                [
                    self._evaluated_e_on_all_snapshots_CHARMM(
                        snapshots, lambda_state, env
                    )
                    for lambda_state in range(1, self.nr_of_states + 1)
                ]
            )
            # remove merged traj
            os.remove(f"{self.base_path}/traj.dcd")

        else:
            raise RuntimeError(f"Either openMM or CHARMM engine, not {engine}")

        if save_results:
            file = f"{self.save_results_to_path}/mbar_data_for_{self.structure_name}_in_{env}.pickle"
            logger.info(f"Saving results: {file}")
            results = {"u_kn": u_kn, "N_k": self.N_k}
            pickle.dump(results, open(file, "wb+"))

        logger.debug("#######################################")
        logger.debug("#######################################")
        logger.debug("Pairwise Free Energy Estimate")
        logger.debug("#######################################")
        logger.debug("#######################################")
        u_kn_ = copy.deepcopy(u_kn)
        start = 0
        for d in range(u_kn.shape[0] - 1):
            logger.debug(self.N_k)
            logger.debug(f"{d}->{d+1} [kT]")
            nr_of_snapshots = self.N_k[env][d] + self.N_k[env][d + 1]
            u_kn_ = u_kn[d : d + 2 :, start : start + nr_of_snapshots]

            m = mbar.MBAR(u_kn_, self.N_k[env][d : d + 2])
            logger.debug(m.getFreeEnergyDifferences(return_dict=True)["Delta_f"][0, 1])
            logger.debug(m.getFreeEnergyDifferences(return_dict=True)["dDelta_f"][0, 1])

            start += self.N_k[env][d]

        logger.debug("#######################################")
        return mbar.MBAR(u_kn, self.N_k[env], initialize="BAR", verbose=True)

    def calculate_dG_to_common_core(
        self, save_results: bool = True, engine: str = "openMM"
    ):
        """
        Calculate mbar results or load save results from a serialized mbar results.
        """
        if save_results:
            os.makedirs(f"{self.configuration['system_dir']}/results/", exist_ok=True)
            logger.info(f"Saving results in {self.save_results_to_path}")

        for env in self.envs:
            logger.info(f"Generating results for {env}.")
            self.mbar_results[env] = self._analyse_results_using_mbar(
                env, self.snapshots[env], save_results, engine
            )

    def load_waterbox_results(self, file: str):
        self.mbar_results["waterbox"] = self._load_mbar_results(file)

    def load_complex_results(self, file: str):
        self.mbar_results["complex"] = self._load_mbar_results(file)

    def load_vacuum_results(self, file: str):
        self.mbar_results["vacuum"] = self._load_mbar_results(file)

    @staticmethod
    def _load_mbar_results(file: str):
        results = pickle.load(open(file, "rb"))
        return mbar.MBAR(
            results["u_kn"], results["N_k"], initialize="BAR", verbose=True
        )

    def free_energy_differences(self, env="vacuum"):
        """matrix of free energy differences"""
        try:
            r = self.mbar_results[env].getFreeEnergyDifferences(return_dict=True)[
                "Delta_f"
            ]
        except KeyError:
            raise KeyError(f"Free energy difference not obtained for : {env}")
        return r

    def free_energy_overlap(self, env="vacuum"):
        """overlap of lambda states"""
        try:
            r = self.mbar_results[env].computeOverlap(return_dict=True)["matrix"]
        except KeyError:
            raise KeyError(f"Free energy overlap not obtained for : {env}")

        return r

    def free_energy_difference_uncertainties(self, env="vacuum"):
        """matrix of asymptotic uncertainty-estimates accompanying free energy differences"""
        try:
            r = self.mbar_results[env].getFreeEnergyDifferences(return_dict=True)[
                "dDelta_f"
            ]
        except KeyError:
            raise KeyError(f"Free energy uncertanties not obtained for : {env}")
        return r

    @property
    def waterbox_free_energy_differences(self):
        """matrix of free energy differences"""
        return self.free_energy_differences(env="waterbox")

    @property
    def complex_free_energy_differences(self):
        """matrix of free energy differences"""
        return self.free_energy_differences(env="complex")

    @property
    def vacuum_free_energy_differences(self):
        """matrix of free energy differences"""
        return self.free_energy_differences(env="vacuum")

    @property
    def waterbox_free_energy_difference_uncertanties(self):
        """matrix of free energy differences"""
        return self.free_energy_difference_uncertainties(env="waterbox")

    @property
    def complex_free_energy_difference_uncertanties(self):
        """matrix of free energy differences"""
        return self.free_energy_difference_uncertainties(env="complex")

    @property
    def vacuum_free_energy_difference_uncertanties(self):
        """matrix of free energy differences"""
        return self.free_energy_difference_uncertainties(env="vacuum")

    @property
    def waterbox_free_energy_difference_overlap(self):
        """matrix of free energy differences"""
        return self.free_energy_overlap(env="waterbox")

    @property
    def complex_free_energy_difference_overlap(self):
        """matrix of free energy differences"""
        return self.free_energy_overlap(env="complex")

    @property
    def vacuum_free_energy_difference_overlap(self):
        """matrix of free energy differences"""
        return self.free_energy_overlap(env="vacuum")

    def plot_free_energy_overlap(self, env: str):
        plt.figure(figsize=[8, 8], dpi=300)
        if env == "vacuum":
            ax = sns.heatmap(
                self.vacuum_free_energy_difference_overlap, cmap="Blues", linewidth=0.5
            )
        elif env == "waterbox":
            ax = sns.heatmap(
                self.waterbox_free_energy_difference_overlap,
                cmap="Blues",
                linewidth=0.5,
            )
        elif env == "complex":
            ax = sns.heatmap(
                self.complex_free_energy_difference_overlap, cmap="Blues", linewidth=0.5
            )
        else:
            raise RuntimeError()
        plt.title(f"Overlap of lambda states for ligand in {env}", fontsize=15)
        plt.xlabel("lambda state (0 to 1)", fontsize=15)
        plt.ylabel("lambda state (0 to 1)", fontsize=15)
        plt.legend()
        plt.savefig(f"{self.save_results_to_path}/ddG_to_common_core_overlap_{env}.png")

        plt.show()
        plt.close()

    def plot_free_energy(self, env: str):

        if env == "vacuum":
            x = [
                a
                for a in np.linspace(0, 1, len(self.vacuum_free_energy_differences[0]))
            ]
            y = self.vacuum_free_energy_differences[0]
            y_error = self.vacuum_free_energy_difference_uncertanties[0]
        elif env == "waterbox":
            x = [
                a
                for a in np.linspace(
                    0, 1, len(self.waterbox_free_energy_differences[0])
                )
            ]
            y = self.waterbox_free_energy_differences[0]
            y_error = self.waterbox_free_energy_difference_uncertanties[0]
        elif env == "complex":
            x = [
                a
                for a in np.linspace(0, 1, len(self.complex_free_energy_differences[0]))
            ]
            y = self.complex_free_energy_differences[0]
            y_error = self.complex_free_energy_difference_uncertanties[0]
        else:
            raise RuntimeError()

        plt.errorbar(x, y, yerr=y_error, label="ddG +- stddev [kT]")
        plt.title(f"Free energy estimate for ligand in {env}", fontsize=15)
        plt.ylabel("Free energy estimate in kT", fontsize=15)
        plt.xlabel("lambda state (0 to 1)", fontsize=15)
        plt.savefig(
            f"{self.save_results_to_path}/ddG_to_common_core_line_plot_{env}.png"
        )
        plt.show()
        plt.close()

    def plot_vacuum_free_energy_overlap(self):
        self.plot_free_energy_overlap("vacuum")

    def plot_complex_free_energy_overlap(self):
        self.plot_free_energy_overlap("complex")

    def plot_waterbox_free_energy_overlap(self):
        self.plot_free_energy_overlap("waterbox")

    def plot_vacuum_free_energy(self):
        self.plot_free_energy("vacuum")

    def plot_complex_free_energy(self):
        self.plot_free_energy("complex")

    def plot_waterbox_free_energy(self):
        self.plot_free_energy("waterbox")

    @property
    def end_state_free_energy_difference(self):
        """DeltaF[lambda=1 --> lambda=0]"""
        if self.configuration["simulation"]["free-energy-type"] == "rsfe":
            return (
                self.waterbox_free_energy_differences[0, -1]
                - self.vacuum_free_energy_differences[0, -1],
                self.waterbox_free_energy_difference_uncertanties[0, -1]
                + self.vacuum_free_energy_difference_uncertanties[0, -1],
            )

        elif self.configuration["simulation"]["free-energy-type"] == "rbfe":
            return (
                self.complex_free_energy_differences[0, -1]
                - self.waterbox_free_energy_differences[0, -1],
                self.complex_free_energy_difference_uncertanties[0, -1]
                + self.waterbox_free_energy_difference_uncertanties[0, -1],
            )
        else:
            raise RuntimeError()

    def show_summary(self):
        if self.configuration["simulation"]["free-energy-type"] == "rsfe":
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
        print(
            f"Free energy to common core: {energy_estimate} [kT] with uncertanty: {uncertanty} [kT]."
        )
