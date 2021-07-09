import logging
from collections import namedtuple
from copy import deepcopy
from dataclasses import dataclass, field
from typing import List, Tuple

import numpy as np
import parmed as pm
from IPython.core.display import display
from rdkit import Chem
from rdkit.Chem import AllChem, Draw, rdFMCS
from rdkit.Chem.Draw import rdMolDraw2D
from simtk.openmm.app.charmmpsffile import CharmmPsfFile

from transformato.system import SystemStructure

logger = logging.getLogger(__name__)


@dataclass
class DummyRegion:
    mol_name: str
    match_termin_real_and_dummy_atoms: dict
    connected_dummy_regions: list
    tlc: str
    lj_default: list

    def return_connecting_real_atom(self, dummy_atoms: list):

        for real_atom in self.match_termin_real_and_dummy_atoms:
            for dummy_atom in self.match_termin_real_and_dummy_atoms[real_atom]:
                if dummy_atom in dummy_atoms:
                    logger.debug(f"Connecting real atom: {real_atom}")
                    return real_atom

        logger.critical("No connecting real atom was found!")
        return None


@dataclass
class MutationDefinition:
    atoms_to_be_mutated: List[int]
    common_core: List[int]
    dummy_region: DummyRegion
    vdw_atom_idx: List[int] = field(default_factory=list)
    steric_mutation_to_default: bool = False

    def print_details(self):

        print("####################")
        print(f"Atoms to be mutated: {self.atoms_to_be_mutated}")
        print(f"Mutated on common core: {self.common_core}")
        if self.vdw_atom_idx:
            print(f"VDW atoms to be decoupled: {self.vdw_atom_idx}")


class ProposeMutationRoute(object):
    def __init__(
        self,
        s1: SystemStructure,
        s2: SystemStructure,
    ):
        """
        A class that proposes the mutation route between two molecules with a
        common core (same atom types) based on two mols and generates the mutation
        objects to perform the mutation on the psf objects.
        Parameters
        ----------
        mol1: Chem.Mol
        mol2: Chem.Mol
        """

        mol1_name: str = "m1"
        mol2_name: str = "m2"
        self.system: dict = {"system1": s1, "system2": s2}
        self.mols: dict = {mol1_name: s1.mol, mol2_name: s2.mol}
        self.graphs: dict = {mol1_name: s1.graph, mol2_name: s2.graph}
        # psfs for reference of only ligand
        self.psfs: dict = {
            mol1_name: s1.psfs["waterbox"][f":{s1.tlc}"],
            mol2_name: s2.psfs["waterbox"][f":{s2.tlc}"],
        }
        self._substructure_match: dict = {mol1_name: [], mol2_name: []}
        self.removed_indeces: dict = {mol1_name: [], mol2_name: []}
        self.added_indeces: dict = {mol1_name: [], mol2_name: []}
        self.s1_tlc = s1.tlc
        self.s2_tlc = s2.tlc

        self.terminal_real_atom_cc1: list = []
        self.terminal_real_atom_cc2: list = []
        self.terminal_dummy_atom_cc1: list = []
        self.terminal_dummy_atom_cc2: list = []

        self.bondCompare = rdFMCS.BondCompare.CompareAny
        self.atomCompare = rdFMCS.AtomCompare.CompareElements
        self.maximizeBonds: bool = True
        self.matchValences: bool = False
        self.completeRingsOnly: bool = False
        self.ringMatchesRingOnly: bool = True

        self.dummy_region_cc1: DummyRegion
        self.dummy_region_cc2: DummyRegion

        self._check_cgenff_versions()

    def _check_cgenff_versions(self):

        cgenff_sys1 = self.system["system1"].cgenff_version
        cgenff_sys2 = self.system["system2"].cgenff_version
        if cgenff_sys1 == cgenff_sys2:
            pass
        else:
            raise RuntimeError(
                f"CGenFF compatibility error. CGenFF: {cgenff_sys1} and CGenFF: {cgenff_sys2} are combined."
            )

    def _match_terminal_real_and_dummy_atoms_for_mol1(self):
        """
        Matches the terminal real and dummy atoms and returns a dict with real atom idx as key and a set of dummy atoms that connect
        to this real atom as a set
        """
        return self._match_terminal_real_and_dummy_atoms(
            self.mols["m1"], self.terminal_real_atom_cc1, self.terminal_dummy_atom_cc1
        )

    def _match_terminal_real_and_dummy_atoms_for_mol2(self) -> dict:
        """
        Matches the terminal real and dummy atoms and returns a dict with real atom idx as key and a set of dummy atoms that connect
        to this real atom as a set
        """
        return self._match_terminal_real_and_dummy_atoms(
            self.mols["m2"], self.terminal_real_atom_cc2, self.terminal_dummy_atom_cc2
        )

    @staticmethod
    def _match_terminal_real_and_dummy_atoms(
        mol, real_atoms_cc: list, dummy_atoms_cc: list
    ) -> dict:
        """
        Matches the terminal real and dummy atoms and returns a dict with real atom idx as key and a set of dummy atoms that connect
        to this real atom as a set

        Parameters
        ----------
        mol : [Chem.Mol]
            The mol object with the real and dummy atoms
        real_atoms_cc : list
            list of real atom idx
        dummy_atoms_cc : list
            list of dummy atom idx

        Returns
        -------
        [type]
            [description]
        """

        from collections import defaultdict

        real_atom_match_dummy_atom = defaultdict(set)
        for real_atom_idx in real_atoms_cc:
            real_atom = mol.GetAtomWithIdx(real_atom_idx)
            real_neighbors = [x.GetIdx() for x in real_atom.GetNeighbors()]
            for dummy_atoms_idx in dummy_atoms_cc:
                if dummy_atoms_idx in real_neighbors:
                    real_atom_match_dummy_atom[real_atom_idx].add(dummy_atoms_idx)

        return real_atom_match_dummy_atom

    def _set_common_core_parameters(self):
        # find terminal atoms
        (
            self.terminal_dummy_atom_cc1,
            self.terminal_real_atom_cc1,
        ) = self._find_terminal_atom(self.get_common_core_idx_mol1(), self.mols["m1"])
        (
            self.terminal_dummy_atom_cc2,
            self.terminal_real_atom_cc2,
        ) = self._find_terminal_atom(self.get_common_core_idx_mol2(), self.mols["m2"])

        # match terminal real atoms between cc1 and cc2 that connect dummy atoms
        cc_idx_mol1 = self.get_common_core_idx_mol1()
        cc_idx_mol2 = self.get_common_core_idx_mol2()
        matching_terminal_atoms_between_cc = list()

        for cc1_idx, cc2_idx in zip(cc_idx_mol1, cc_idx_mol2):
            if (
                cc1_idx in self.terminal_real_atom_cc1
                and cc2_idx in self.terminal_real_atom_cc2
            ):
                logger.info(
                    f"Dummy regions connect on the same terminal atoms. cc1: {cc1_idx} : cc2: {cc2_idx}"
                )
                matching_terminal_atoms_between_cc.append((cc1_idx, cc2_idx))
            elif (
                cc1_idx in self.terminal_real_atom_cc1
                and cc2_idx not in self.terminal_real_atom_cc2
            ) or (
                cc1_idx not in self.terminal_real_atom_cc1
                and cc2_idx in self.terminal_real_atom_cc2
            ):
                logger.info(
                    f"Single dummy region connects on terminal atom. cc1: {cc1_idx} : cc2: {cc2_idx}"
                )
                matching_terminal_atoms_between_cc.append((cc1_idx, cc2_idx))
            else:
                pass
        if not matching_terminal_atoms_between_cc:
            raise RuntimeError(
                "No terminal real atoms were matched between the common cores. Aborting."
            )

        self.matching_terminal_atoms_between_cc = matching_terminal_atoms_between_cc

    def _match_terminal_dummy_atoms_between_common_cores(
        self,
        match_terminal_atoms_cc1: dict,
        match_terminal_atoms_cc2: dict,
    ) -> Tuple[list, list]:

        cc1_idx = self._substructure_match["m1"]
        cc2_idx = self._substructure_match["m2"]

        lj_default_cc1 = []
        lj_default_cc2 = []

        # iterate through the common core substracter (the order represents the matched atoms)
        for idx1, idx2 in zip(cc1_idx, cc2_idx):

            # if both atoms are terminal atoms connected dummy regions can be identified
            if (
                idx1 in match_terminal_atoms_cc1.keys()
                and idx2 in match_terminal_atoms_cc2.keys()
            ):

                connected_dummy_cc1 = list(match_terminal_atoms_cc1[idx1])
                connected_dummy_cc2 = list(match_terminal_atoms_cc2[idx2])

                if len(connected_dummy_cc1) == 1 and len(connected_dummy_cc2) == 1:
                    pass
                # multiple, possible dummy regions
                elif len(connected_dummy_cc1) > 1 or len(connected_dummy_cc2) > 1:
                    logger.critical("There is a dual junction. Be careful.")
                    # NOTE: For now we are just taking the non hydrogen atom
                    for atom_idx in connected_dummy_cc1:
                        if self.mols["m1"].GetAtomWithIdx(atom_idx).GetSymbol() != "H":
                            connected_dummy_cc1 = [atom_idx]
                            break
                    for atom_idx in connected_dummy_cc2:
                        if self.mols["m2"].GetAtomWithIdx(atom_idx).GetSymbol() != "H":
                            connected_dummy_cc2 = [atom_idx]
                            break

                # hydrogen mutates to dummy atom (but not a LJ particle)
                elif len(connected_dummy_cc1) == 0 or len(connected_dummy_cc2) == 0:
                    logger.debug("Hydrogen to dummy mutation")
                    raise NotImplementedError()

                lj_default_cc1.append(connected_dummy_cc1[0])
                lj_default_cc2.append(connected_dummy_cc2[0])

        return (lj_default_cc1, lj_default_cc2)

    def propose_common_core(self):
        mcs = self._find_mcs("m1", "m2")
        return mcs

    def finish_common_core(
        self,
        connected_dummy_regions_cc1: list = [],
        connected_dummy_regions_cc2: list = [],
    ):
        # set the teriminal real/dummy atom indices
        self._set_common_core_parameters()
        # match the real/dummy atoms
        match_terminal_atoms_cc1 = self._match_terminal_real_and_dummy_atoms_for_mol1()
        match_terminal_atoms_cc2 = self._match_terminal_real_and_dummy_atoms_for_mol2()
        logger.info("Find connected dummy regions")
        # define connected dummy regions
        if not connected_dummy_regions_cc1:
            connected_dummy_regions_cc1 = self._find_connected_dummy_regions(
                mol_name="m1", match_terminal_atoms_cc=match_terminal_atoms_cc1
            )
        if not connected_dummy_regions_cc2:
            connected_dummy_regions_cc2 = self._find_connected_dummy_regions(
                mol_name="m2", match_terminal_atoms_cc=match_terminal_atoms_cc2
            )

        logger.info(f"connected dummy regions for mol1: {connected_dummy_regions_cc1}")
        logger.info(f"connected dummy regions for mol2: {connected_dummy_regions_cc2}")

        # find the atoms from dummy_region in s1 that needs to become lj default
        (
            lj_default_cc1,
            lj_default_cc2,
        ) = self._match_terminal_dummy_atoms_between_common_cores(
            match_terminal_atoms_cc1, match_terminal_atoms_cc2
        )

        self.dummy_region_cc1 = DummyRegion(
            mol_name="m1",
            tlc=self.s1_tlc,
            match_termin_real_and_dummy_atoms=match_terminal_atoms_cc1,
            connected_dummy_regions=connected_dummy_regions_cc1,
            lj_default=lj_default_cc1,
        )

        self.dummy_region_cc2 = DummyRegion(
            mol_name="m2",
            tlc=self.s2_tlc,
            match_termin_real_and_dummy_atoms=match_terminal_atoms_cc2,
            connected_dummy_regions=connected_dummy_regions_cc2,
            lj_default=lj_default_cc2,
        )

        # generate charge compmensated psfs
        psf1, psf2 = self._prepare_cc_for_charge_transfer()
        self.charge_compensated_ligand1_psf = psf1
        self.charge_compensated_ligand2_psf = psf2

    def calculate_common_core(self):

        self.propose_common_core()
        self.finish_common_core()

    def _prepare_cc_for_charge_transfer(self):
        # we have to run the same charge mutation that will be run on cc2 to get the
        # charge distribution AFTER the full mutation

        # make a copy of the full psf
        m2_psf = self.psfs["m2"][:, :, :]
        m1_psf = self.psfs["m1"][:, :, :]
        charge_transformed_psfs = []

        for psf, tlc, cc_idx, dummy_region in zip(
            [m1_psf, m2_psf],
            [self.s1_tlc, self.s2_tlc],
            [self.get_common_core_idx_mol1(), self.get_common_core_idx_mol2()],
            [self.dummy_region_cc1, self.dummy_region_cc2],
        ):

            # set `initial_charge` parameter for Mutation
            for atom in psf.view[f":{tlc}"].atoms:
                # charge, epsilon and rmin are directly modiefied
                atom.initial_charge = atom.charge

            offset = min([atom.idx for atom in psf.view[f":{tlc}"].atoms])

            # getting copy of the atoms
            atoms_to_be_mutated = []
            for atom in psf.view[f":{tlc}"].atoms:
                idx = atom.idx - offset
                if idx not in cc_idx:
                    atoms_to_be_mutated.append(idx)

            logger.debug("############################")
            logger.debug("Preparing cc2 for charge transfer")
            logger.debug(
                f"Atoms for which charge is set to zero: {atoms_to_be_mutated}"
            )
            logger.debug("############################")

            m = Mutation(
                atoms_to_be_mutated=atoms_to_be_mutated, dummy_region=dummy_region
            )
            m.mutate(psf, lambda_value_electrostatic=0.0)
            charge_transformed_psfs.append(psf)
        return charge_transformed_psfs[0], charge_transformed_psfs[1]

    def remove_idx_from_common_core_of_mol1(self, idx_list: list):
        for idx in idx_list:
            self._remove_idx_from_common_core("m1", idx)

    def remove_idx_from_common_core_of_mol2(self, idx_list: list):
        for idx in idx_list:
            self._remove_idx_from_common_core("m2", idx)

    def _remove_idx_from_common_core(self, name: str, idx: int):
        if idx in self.added_indeces[name] or idx in self._get_common_core(name):
            if idx in self.removed_indeces[name]:
                print(f"Idx: {idx} already removed from common core.")
                return
            self.removed_indeces[name].append(idx)
        else:
            print(f"Idx: {idx} not in common core.")

    def add_idx_to_common_core_of_mol1(self, idx_list: list):
        for idx in idx_list:
            self._add_common_core_atom("m1", idx)
        print(self.get_common_core_idx_mol1())

    def add_idx_to_common_core_of_mol2(self, idx_list: list):
        for idx in idx_list:
            self._add_common_core_atom("m2", idx)
        print(self.get_common_core_idx_mol2())

    def _add_common_core_atom(self, name: str, idx: int):
        if idx in self.added_indeces[name] or idx in self._get_common_core(name):
            print(f"Idx: {idx} already in common core.")
            return
        self.added_indeces[name].append(idx)

    def get_idx_not_in_common_core_for_mol1(self) -> list:
        return self._get_idx_not_in_common_core_for_mol("m1")

    def get_idx_not_in_common_core_for_mol2(self) -> list:
        return self._get_idx_not_in_common_core_for_mol("m2")

    def _get_idx_not_in_common_core_for_mol(self, mol_name: str) -> list:

        dummy_list_mol = [
            atom.GetIdx()
            for atom in self.mols[mol_name].GetAtoms()
            if atom.GetIdx() not in self._get_common_core(mol_name)
        ]
        return dummy_list_mol

    def get_common_core_idx_mol1(self) -> list:
        """
        Returns the common core of mol1.
        """
        return self._get_common_core("m1")

    def get_common_core_idx_mol2(self) -> list:
        """
        Returns the common core of mol2.
        """
        return self._get_common_core("m2")

    def _get_common_core(self, name: str) -> list:
        """
        Helper Function - should not be called directly.
        Returns the common core.
        """
        keep_idx = []
        # BEWARE: the ordering is important - don't cast set!
        for idx in self._substructure_match[name] + self.added_indeces[name]:
            if idx not in self.removed_indeces[name]:
                keep_idx.append(idx)
        return keep_idx

    def _find_mcs(
        self,
        mol1_name: str,
        mol2_name: str,
    ):
        """
        A class that proposes the mutation route between two molecules with a
        common core (same atom types) based on two mols and generates the mutation
        objects to perform the mutation on the psf objects.
        Parameters
        ----------
        mol1_name: str
        mol2_name: str
        """

        logger.info("MCS starting ...")
        logger.debug(f"bondCompare: {self.bondCompare}")
        logger.debug(f"atomCompare: {self.atomCompare}")
        logger.debug(f"maximizeBonds: {self.maximizeBonds}")
        logger.debug(f"matchValences: {self.matchValences} ")
        logger.debug(f"ringMatchesRingOnly: {self.ringMatchesRingOnly} ")
        logger.debug(f"completeRingsOnly: {self.completeRingsOnly} ")

        m1, m2 = [deepcopy(self.mols[mol1_name]), deepcopy(self.mols[mol2_name])]

        for m in [m1, m2]:
            logger.debug("Mol in SMILES format: {}.".format(Chem.MolToSmiles(m, True)))

        # make copy of mols
        changed_mols = [Chem.Mol(x) for x in [m1, m2]]

        # find substructure match (ignore bond order but enforce element matching)
        mcs = rdFMCS.FindMCS(
            changed_mols,
            bondCompare=self.bondCompare,
            timeout=120,
            atomCompare=self.atomCompare,
            maximizeBonds=self.maximizeBonds,
            matchValences=self.matchValences,
            completeRingsOnly=self.completeRingsOnly,
            ringMatchesRingOnly=self.ringMatchesRingOnly,
        )
        logger.debug("Substructure match: {}".format(mcs.smartsString))
        # convert from SMARTS
        mcsp = Chem.MolFromSmarts(mcs.smartsString, False)

        s1 = m1.GetSubstructMatch(mcsp)
        logger.debug("Substructere match idx: {}".format(s1))
        self._display_mol(m1)
        s2 = m2.GetSubstructMatch(mcsp)
        logger.debug("Substructere match idx: {}".format(s2))
        self._display_mol(m2)

        self._substructure_match[mol1_name] = list(s1)
        self._substructure_match[mol2_name] = list(s2)
        return mcs

    def _return_atom_idx_from_bond_idx(self, mol: Chem.Mol, bond_idx: int):
        return (
            mol.GetBondWithIdx(bond_idx).GetBeginAtomIdx(),
            mol.GetBondWithIdx(bond_idx).GetEndAtomIdx(),
        )

    def _find_connected_dummy_regions(
        self, mol_name: str, match_terminal_atoms_cc: dict
    ) -> List[set]:

        from itertools import chain

        sub = self._get_common_core(mol_name)

        #############################
        # start
        #############################
        mol = self.mols[mol_name]
        # find all dummy atoms
        dummy_list_mol = [
            atom.GetIdx() for atom in mol.GetAtoms() if atom.GetIdx() not in sub
        ]
        nr_of_dummy_atoms_mol = len(dummy_list_mol) + 1
        # add all unique subgraphs here
        unique_subgraphs = []

        # iterate over dummy regions
        for real_atom in match_terminal_atoms_cc:
            logger.debug(f"real atom: {real_atom}")
            set_of_terminal_dummy_atoms = match_terminal_atoms_cc[real_atom]
            for terminal_dummy_atom in set_of_terminal_dummy_atoms:
                logger.debug(f"terminal_dummy_atom: {terminal_dummy_atom}")
                # start with biggest possible subgraph at final dummy atom
                for i in range(nr_of_dummy_atoms_mol, -1, -1):
                    subgraphs_of_length_i = []
                    logger.debug(f"Length: {i}")
                    all_subgraphs = Chem.FindUniqueSubgraphsOfLengthN(
                        mol=mol,
                        length=i,
                        useHs=True,
                        useBO=False,
                        rootedAtAtom=terminal_dummy_atom,
                    )
                    for subgraphs in all_subgraphs:
                        subgraphs = set(
                            chain.from_iterable(
                                [
                                    self._return_atom_idx_from_bond_idx(
                                        mol=mol, bond_idx=e
                                    )
                                    for e in subgraphs
                                ]
                            )
                        )

                        # test that only dummy atoms are in subgraph
                        if any([real_atom in subgraphs for real_atom in sub]):
                            pass
                        else:
                            subgraphs_of_length_i.append(set(subgraphs))

                    # test that new subgraphs are not already contained in bigger subgraphs
                    for subgraphs in subgraphs_of_length_i:
                        if any(
                            [
                                subgraphs.issubset(old_subgraph)
                                for old_subgraph in unique_subgraphs
                            ]
                        ):
                            # not new set
                            pass
                        else:
                            unique_subgraphs.append(subgraphs)
                            logger.debug(subgraphs)

        # adding single dummy atoms that are not in a path
        for dummy_atom in dummy_list_mol:
            if dummy_atom not in list(chain.from_iterable(unique_subgraphs)):
                unique_subgraphs.append(set([dummy_atom]))

        logger.debug(unique_subgraphs)
        return unique_subgraphs

    def _display_mol(self, mol: Chem.Mol):
        """
        Gets mol as input and displays its 2D Structure using IPythonConsole.
        Parameters
        ----------
        mol: Chem.Mol
            a rdkit mol object
        """

        def mol_with_atom_index(mol):
            atoms = mol.GetNumAtoms()
            for idx in range(atoms):
                mol.GetAtomWithIdx(idx).SetProp(
                    "molAtomMapNumber", str(mol.GetAtomWithIdx(idx).GetIdx())
                )
            return mol

        mol = mol_with_atom_index(mol)
        AllChem.Compute2DCoords(mol)
        display(mol)

    def show_common_core_on_mol1(self):
        """
        Shows common core on mol1
        """
        return self._show_common_core(self.mols["m1"], self.get_common_core_idx_mol1())

    def show_common_core_on_mol2(self):
        """
        Shows common core on mol2
        """
        return self._show_common_core(self.mols["m2"], self.get_common_core_idx_mol2())

    def _show_common_core(self, mol, highlight):
        """
        Helper function - do not call directly.
        Show common core.
        """
        # https://rdkit.blogspot.com/2015/02/new-drawing-code.html

        mol = deepcopy(mol)
        AllChem.Compute2DCoords(mol)

        drawer = rdMolDraw2D.MolDraw2DSVG(800, 800)
        drawer.SetFontSize(0.3)

        opts = drawer.drawOptions()

        for i in mol.GetAtoms():
            opts.atomLabels[i.GetIdx()] = (
                str(i.GetProp("atom_index")) + ":" + i.GetProp("atom_type")
            )

        drawer.DrawMolecule(mol, highlightAtoms=highlight)
        Draw.DrawingOptions.includeAtomNumbers = False
        drawer.FinishDrawing()
        svg = drawer.GetDrawingText().replace("svg:", "")
        return svg

    def generate_mutations_to_common_core_for_mol1(self) -> dict:
        """
        Generates the mutation route to the common fore for mol1.
        Parameters
        ----------
        nr_of_steps_for_electrostatic : int
            nr of steps used for linearly scaling the charges to zero
        nr_of_steps_for_cc_transformation : int
        Returns
        ----------
        mutations: list
            list of mutations

        """
        if not self.terminal_real_atom_cc1:
            raise RuntimeError("First generate the MCS. Aborting.")

        m = self._mutate_to_common_core(
            self.dummy_region_cc1, self.get_common_core_idx_mol1()
        )
        m["transform"] = self._transform_common_core()
        return m

    def generate_mutations_to_common_core_for_mol2(self) -> dict:
        """
        Generates the mutation route to the common fore for mol2.
        Returns
        ----------
        mutations: list
            list of mutations
        """
        if not self.terminal_real_atom_cc1:
            raise RuntimeError("First generate the MCS")

        m = self._mutate_to_common_core(
            self.dummy_region_cc2, self.get_common_core_idx_mol2()
        )
        return m

    def _transform_common_core(self) -> list:
        """
        Common Core 1 is transformed to Common core 2. Bonded parameters and charges are adjusted.
        """

        transformations = []
        logger.warning("##############################")
        logger.warning("##############################")
        logger.warning("Transform common core")
        logger.warning("##############################")
        logger.warning("##############################")

        # test if bonded mutations are necessary
        bonded_terms_mutation = False
        charge_mutation = False

        for cc1, cc2 in zip(
            self.get_common_core_idx_mol1() + self.dummy_region_cc1.lj_default,
            self.get_common_core_idx_mol2() + self.dummy_region_cc2.lj_default,
        ):

            # did atom type change? if not don't add BondedMutations
            atom1 = self.psfs["m1"][cc1]
            atom2 = self.psfs["m2"][cc2]
            if atom1.type != atom2.type:
                logger.warning("##############################")
                logger.warning("Atom type transformation")
                logger.warning(f"Atom that needs to be transformed: {atom1}.")
                logger.warning(f"Atom type of atom in cc1: {atom1.type}.")
                logger.warning(f"Template atom: {atom2}.")
                logger.warning(f"Atom type of atom in cc2: {atom2.type}.")
                bonded_terms_mutation = True

        for cc1, cc2 in zip(
            self.get_common_core_idx_mol1(), self.get_common_core_idx_mol2()
        ):
            atom1 = self.charge_compensated_ligand1_psf[cc1]
            atom2 = self.charge_compensated_ligand2_psf[cc2]
            if atom1.charge != atom2.charge:
                logger.warning("##############################")
                logger.warning("Charge transformation")
                logger.warning("Charge needs to be transformed on common core")
                logger.warning(f"Atom that needs to be transformed: {atom1}.")
                logger.warning(f"Atom charge of atom in cc1: {atom1.charge}.")
                logger.warning(f"Template atom: {atom2}.")
                logger.warning(f"Atom charge of atom in cc2: {atom2.charge}.")
                charge_mutation = True

        # if necessary transform bonded parameters
        if bonded_terms_mutation or charge_mutation:
            logger.warning(f"Bonded parameters mutation: {bonded_terms_mutation}.")
            logger.warning(f"Charge parameters mutation: {charge_mutation}.")

            t = CommonCoreTransformation(
                self.get_common_core_idx_mol1() + self.dummy_region_cc1.lj_default,
                self.get_common_core_idx_mol2() + self.dummy_region_cc2.lj_default,
                self.psfs["m1"],
                self.psfs["m2"],
                self.s1_tlc,
                self.s2_tlc,
                self.charge_compensated_ligand2_psf,
                charge_mutation=charge_mutation,
                bonded_terms_mutation=bonded_terms_mutation,
            )
            transformations.append(t)
        else:
            logger.info("!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")
            logger.info("No transformations needed.")
            logger.info("!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")
            transformations = []

        return transformations

    @staticmethod
    def _find_terminal_atom(cc_idx: list, mol: Chem.Mol) -> Tuple[list, list]:
        """
        Find atoms that connect the molecule to the common core.

        Args:
            cc_idx (list): common core index atoms
            mol ([type]): rdkit mol object
        """
        terminal_dummy_atoms = []
        terminal_real_atoms = []

        for atom in mol.GetAtoms():
            idx = atom.GetIdx()
            if idx not in cc_idx:
                neighbors = [x.GetIdx() for x in atom.GetNeighbors()]
                if any([n in cc_idx for n in neighbors]):
                    terminal_dummy_atoms.append(idx)
            if idx in cc_idx:
                neighbors = [x.GetIdx() for x in atom.GetNeighbors()]
                if any([n not in cc_idx for n in neighbors]):
                    terminal_real_atoms.append(idx)

        logger.info(f"Terminal dummy atoms: {str(list(set(terminal_dummy_atoms)))}")
        logger.info(f"Terminal real atoms: {str(list(set(terminal_real_atoms)))}")

        return (list(set(terminal_dummy_atoms)), list(set(terminal_real_atoms)))

    def _mutate_to_common_core(self, dummy_region: DummyRegion, cc_idx: list) -> dict:
        """
        Helper function - do not call directly.
        Generates the mutation route to the common fore for mol.
        """

        from collections import defaultdict

        name = dummy_region.mol_name

        mol = self.mols[name]
        mutations = defaultdict(list)

        # TODO: this needs some work!
        # get the atom that connects the common core to the dummy regiom
        match_termin_real_and_dummy_atoms = (
            dummy_region.match_termin_real_and_dummy_atoms
        )
        # get the terminal dummy atoms
        list_termin_dummy_atoms = []
        for m in match_termin_real_and_dummy_atoms.values():
            list_termin_dummy_atoms.extend(list(m))
        logger.info(f"Terminal dummy atoms: {list_termin_dummy_atoms}")

        # iterate through atoms and select atoms that need to be mutated
        atoms_to_be_mutated = []
        hydrogens = []
        for atom in mol.GetAtoms():
            idx = atom.GetIdx()
            if idx not in cc_idx:
                # hydrogens are collected seperatly IF they are not terminal dummy atoms
                if atom.GetSymbol() == "H" and idx not in list_termin_dummy_atoms:
                    hydrogens.append(idx)
                atoms_to_be_mutated.append(idx)
                logger.info(
                    "Will be decoupled: Idx:{} Element:{}".format(idx, atom.GetSymbol())
                )

        if atoms_to_be_mutated:
            ############################################
            ############################################
            # charge mutation
            ############################################
            ############################################

            m = MutationDefinition(
                atoms_to_be_mutated=atoms_to_be_mutated,
                common_core=cc_idx,
                dummy_region=dummy_region,
                vdw_atom_idx=[],
                steric_mutation_to_default=False,
            )

            mutations["charge"].append(m)

            ############################################
            ############################################
            # LJ mutation
            ############################################
            ############################################

            # start with mutation of LJ of hydrogens
            # Only take hydrogens that are not terminal hydrogens
            if hydrogens:
                m = MutationDefinition(
                    atoms_to_be_mutated=atoms_to_be_mutated,
                    common_core=cc_idx,
                    dummy_region=dummy_region,
                    vdw_atom_idx=hydrogens,
                    steric_mutation_to_default=False,
                )

                mutations["hydrogen-lj"].append(m)

            for region in dummy_region.connected_dummy_regions:
                for atom_idx in region:
                    if (
                        atom_idx in list_termin_dummy_atoms
                        and atom_idx in dummy_region.lj_default
                    ):
                        # test if atom is a terminal atom and there is a corresponding atom on the other cc
                        # in this case the atom needs to become a default lj particle
                        m = MutationDefinition(
                            atoms_to_be_mutated=atoms_to_be_mutated,
                            common_core=cc_idx,
                            dummy_region=dummy_region,
                            vdw_atom_idx=[atom_idx],
                            steric_mutation_to_default=True,
                        )
                        mutations["default-lj"].append(m)
                    elif atom_idx in hydrogens:
                        # already mutated
                        continue
                    else:
                        # normal lj mutation
                        m = MutationDefinition(
                            atoms_to_be_mutated=atoms_to_be_mutated,
                            common_core=cc_idx,
                            dummy_region=dummy_region,
                            vdw_atom_idx=[atom_idx],
                            steric_mutation_to_default=False,
                        )
                        mutations["lj"].append(m)

        else:
            logger.critical("No atoms will be decoupled.")
            mutations = defaultdict()
        return mutations


class CommonCoreTransformation(object):
    def __init__(
        self,
        cc1_indicies: list,
        cc2_indicies: list,
        ligand1_psf: pm.charmm.CharmmPsfFile,
        ligand2_psf: pm.charmm.CharmmPsfFile,
        tlc_cc1: str,
        tlc_cc2: str,
        charge_compensated_ligand2_psf: pm.charmm.CharmmPsfFile,
        charge_mutation: bool,
        bonded_terms_mutation: bool,
    ):
        """
        Scale the bonded parameters inside the common core.
        Parameters
        ----------
        cc1_indicies : list
            indices of cc1
        cc2_indicies : list
            indices of cc2 (in the same order as cc1)
        ligand1_psf : pm.charmm.CharmmPsfFile (copy of only ligand)
        ligand2_psf : pm.charmm.CharmmPsfFile (copy of only ligand)
            the target psf that is used to generate the new bonded parmaeters
        tlc_cc1 : str
            three letter code of ligand in cc1
        tlc_cc2 : str
            three letter code of ligand in cc2
        """
        self.cc1_indicies: list = cc1_indicies
        self.cc2_indicies: list = cc2_indicies
        self.ligand2_psf: pm.charmm.CharmmPsfFile = ligand2_psf
        self.ligand1_psf: pm.charmm.CharmmPsfFile = ligand1_psf
        self.tlc_cc1: str = tlc_cc1
        self.tlc_cc2: str = tlc_cc2
        self.atom_names_mapping = self._get_atom_mapping()
        self.charge_mutation: bool = charge_mutation
        self.bonded_terms_mutation: bool = bonded_terms_mutation
        self.charge_compensated_ligand2_psf: pm.charmm.CharmmPsfFile = (
            charge_compensated_ligand2_psf
        )

        logger.info(f"Bonded terms mutation: {bonded_terms_mutation}")
        logger.info(f"Charge mutation: {charge_mutation}")

    def _get_atom_mapping(self) -> dict:
        """
        _get_atom_mapping -- match the atom names of the common cores

        Returns
        -------
        [dict]
            matched common core atom names
        """
        # match atomes in common cores
        match_atom_names_cc1_to_cc2 = {}
        for cc1_idx, cc2_idx in zip(self.cc1_indicies, self.cc2_indicies):
            ligand1_atom = self.ligand1_psf[cc1_idx]
            ligand2_atom = self.ligand2_psf[cc2_idx]
            match_atom_names_cc1_to_cc2[ligand1_atom.name] = ligand2_atom.name

        return match_atom_names_cc1_to_cc2

    def _mutate_charges(self, psf: pm.charmm.CharmmPsfFile, scale: float):

        # common core of psf 1 is transformed to psf 2
        for ligand1_atom in psf.view[f":{self.tlc_cc1}"]:
            if ligand1_atom.name not in self.atom_names_mapping:
                continue
            found = False

            # compare to charge compenstated psf 2
            for ligand2_atom in self.charge_compensated_ligand2_psf:
                if self.atom_names_mapping[ligand1_atom.name] == ligand2_atom.name:
                    found = True
                    # are the atoms different?
                    logger.debug(f"Modifying atom: {ligand1_atom}")
                    logger.debug(f"Template atom: {ligand2_atom}")

                    # scale epsilon
                    logger.debug(f"Original charge: {ligand1_atom.charge}")
                    modified_charge = (
                        scale * ligand1_atom.charge + (1 - scale) * ligand2_atom.charge
                    )
                    logger.debug(f"New charge: {modified_charge}")
                    ligand1_atom.charge = modified_charge

            if not found:
                raise RuntimeError("No corresponding atom in cc2 found")

    def _mutate_atoms(self, psf: pm.charmm.CharmmPsfFile, lambda_value: float):
        """
        mutate atom types.

        Raises
        ------
        RuntimeError
            if common core atoms can not be matched
        """
        # what will be changed
        mod_type = namedtuple("Atom", "epsilon, rmin")
        logger.debug("#######################")
        logger.debug("mutate_atoms")

        # iterate through the atoms of the ligand of system1
        for ligand1_atom in psf.view[f":{self.tlc_cc1}"]:
            # continue if not in atom_names_mapping
            if ligand1_atom.name not in self.atom_names_mapping:
                continue

            found = False
            # iterate through the atoms the ligand of system2
            for ligand2_atom in self.ligand2_psf:
                # is there a match up?
                if self.atom_names_mapping[ligand1_atom.name] == ligand2_atom.name:
                    found = True
                    # are the atoms different?
                    if ligand1_atom.type != ligand2_atom.type:
                        if "DDX" in ligand1_atom.type:
                            logger.warning(
                                "This is the terminal LJ atom. If everything went correct, this does not have to change atom types."
                            )
                        else:
                            self._modify_type_in_cc(ligand1_atom, psf)
                            logger.debug(f"Modifying atom: {ligand1_atom}")
                            logger.debug(f"Template atom: {ligand2_atom}")

                            # scale epsilon
                            logger.debug(f"Real epsilon: {ligand1_atom.epsilon}")
                            modified_epsilon = (
                                lambda_value * ligand1_atom.epsilon
                                + (1.0 - lambda_value) * ligand2_atom.epsilon
                            )
                            logger.debug(f"New epsilon: {modified_epsilon}")

                            # scale rmin
                            logger.debug(f"Real rmin: {ligand1_atom.rmin}")
                            modified_rmin = (
                                lambda_value * ligand1_atom.rmin
                                + (1.0 - lambda_value) * ligand2_atom.rmin
                            )
                            logger.debug(f"New rmin: {modified_rmin}")

                            ligand1_atom.mod_type = mod_type(
                                modified_epsilon, modified_rmin
                            )

            if not found:
                raise RuntimeError("No corresponding atom in cc2 found")

    def _mutate_bonds(self, psf: pm.charmm.CharmmPsfFile, lambda_value: float):

        logger.debug("#######################")
        logger.debug("mutate_bonds")

        mod_type = namedtuple("Bond", "k, req")
        for ligand1_bond in psf.view[f":{self.tlc_cc1}"].bonds:

            ligand1_atom1_name = ligand1_bond.atom1.name
            ligand1_atom2_name = ligand1_bond.atom2.name
            # all atoms of the bond must be in cc
            # everything outside the cc are bonded terms between dummies or
            # between real atoms and dummies and we can ignore them for now
            if not all(
                elem in self.atom_names_mapping
                for elem in [ligand1_atom1_name, ligand1_atom2_name]
            ):
                continue

            found = False
            for ligand2_bond in self.ligand2_psf.bonds:
                ligand2_atom1_name = ligand2_bond.atom1.name
                ligand2_atom2_name = ligand2_bond.atom2.name
                # all atoms of the bond must be in cc
                if not all(
                    elem in self.atom_names_mapping.values()
                    for elem in [ligand2_atom1_name, ligand2_atom2_name]
                ):
                    continue

                # match the two bonds
                if sorted(
                    [
                        self.atom_names_mapping[e]
                        for e in [ligand1_atom1_name, ligand1_atom2_name]
                    ]
                ) == sorted([ligand2_atom1_name, ligand2_atom2_name]):
                    found = True
                    # are the bonds different?
                    if sorted(
                        [ligand1_bond.atom1.type, ligand1_bond.atom2.type]
                    ) == sorted([ligand2_bond.atom1.type, ligand2_bond.atom2.type]):
                        continue
                    logger.debug(f"Modifying bond: {ligand1_bond}")

                    logger.debug(f"Template bond: {ligand2_bond}")
                    logger.debug(f"Original value for k: {ligand1_bond.type.k}")
                    logger.debug(f"Target k: {ligand2_bond.type.k}")
                    new_k = (lambda_value * ligand1_bond.type.k) + (
                        (1.0 - lambda_value) * ligand2_bond.type.k
                    )
                    logger.debug(new_k)

                    modified_k = new_k

                    logger.debug(f"New k: {modified_k}")

                    logger.debug(f"Old req: {ligand1_bond.type.req}")
                    # interpolating from ligand1 (original) to ligand2 (new) bond parameters
                    modified_req = (lambda_value * ligand1_bond.type.req) + (
                        (1.0 - lambda_value) * ligand2_bond.type.req
                    )
                    logger.debug(f"Modified bond: {ligand1_bond}")

                    ligand1_bond.mod_type = mod_type(modified_k, modified_req)
                    logger.debug(ligand1_bond.mod_type)

            if not found:
                logger.critical(ligand1_bond)
                raise RuntimeError(
                    "No corresponding bond in cc2 found: {}".format(ligand1_bond)
                )

    def _mutate_angles(self, psf: pm.charmm.CharmmPsfFile, lambda_value: float):

        mod_type = namedtuple("Angle", "k, theteq")
        for cc1_angle in psf.view[f":{self.tlc_cc1}"].angles:
            ligand1_atom1_name = cc1_angle.atom1.name
            ligand1_atom2_name = cc1_angle.atom2.name
            cc1_a3 = cc1_angle.atom3.name

            # only angles in cc
            if not all(
                elem in self.atom_names_mapping
                for elem in [ligand1_atom1_name, ligand1_atom2_name, cc1_a3]
            ):
                continue

            found = False
            for cc2_angle in self.ligand2_psf.angles:
                ligand2_atom1_name = cc2_angle.atom1.name
                ligand2_atom2_name = cc2_angle.atom2.name
                cc2_a3 = cc2_angle.atom3.name
                # only angles in cc
                if not all(
                    elem in self.atom_names_mapping.values()
                    for elem in [ligand2_atom1_name, ligand2_atom2_name, cc2_a3]
                ):
                    continue

                if sorted(
                    [
                        self.atom_names_mapping[e]
                        for e in [ligand1_atom1_name, ligand1_atom2_name, cc1_a3]
                    ]
                ) == sorted([ligand2_atom1_name, ligand2_atom2_name, cc2_a3]):
                    found = True
                    if sorted(
                        [
                            cc1_angle.atom1.type,
                            cc1_angle.atom2.type,
                            cc1_angle.atom3.type,
                        ]
                    ) == sorted(
                        [
                            cc2_angle.atom1.type,
                            cc2_angle.atom2.type,
                            cc2_angle.atom3.type,
                        ]
                    ):
                        continue

                    logger.debug(f"Modifying angle: {cc1_angle}")
                    logger.debug(f"Template bond: {cc2_angle}")
                    logger.debug("Scaling k and theteq")

                    logger.debug(f"Old k: {cc1_angle.type.k}")
                    modified_k = (
                        lambda_value * cc1_angle.type.k
                        + (1.0 - lambda_value) * cc2_angle.type.k
                    )
                    logger.debug(f"New k: {modified_k}")

                    logger.debug(f"Old k: {cc1_angle.type.theteq}")
                    modified_theteq = (
                        lambda_value * cc1_angle.type.theteq
                        + (1.0 - lambda_value) * cc2_angle.type.theteq
                    )
                    logging.debug(f"New k: {modified_theteq}")

                    cc1_angle.mod_type = mod_type(modified_k, modified_theteq)

            if not found:
                logger.critical(cc1_angle)
                raise RuntimeError("No corresponding angle in cc2 found")

    def _mutate_torsions(self, psf: pm.charmm.CharmmPsfFile, lambda_value: float):

        mod_type = namedtuple("Torsion", "phi_k, per, phase, scee, scnb")

        # get all torsions present in initial topology
        for original_torsion in psf.view[f":{self.tlc_cc1}"].dihedrals:

            found: bool = False
            original_atom1_name = original_torsion.atom1.name
            original_atom2_name = original_torsion.atom2.name
            original_atom3_name = original_torsion.atom3.name
            original_atom4_name = original_torsion.atom4.name
            # all atoms must be in the cc
            if not all(
                elem in self.atom_names_mapping
                for elem in [
                    original_atom1_name,
                    original_atom2_name,
                    original_atom3_name,
                    original_atom4_name,
                ]
            ):
                continue

            # get corresponding torsion types in the new topology
            for new_torsion in self.ligand2_psf.dihedrals:
                new_atom1_name = new_torsion.atom1.name
                new_atom2_name = new_torsion.atom2.name
                new_atom3_name = new_torsion.atom3.name
                new_atom4_name = new_torsion.atom4.name
                # only torsion in cc
                if not all(
                    elem in self.atom_names_mapping.values()
                    for elem in [
                        new_atom1_name,
                        new_atom2_name,
                        new_atom3_name,
                        new_atom4_name,
                    ]
                ):
                    continue

                if sorted(
                    [
                        self.atom_names_mapping[e]
                        for e in [
                            original_atom1_name,
                            original_atom2_name,
                            original_atom3_name,
                            original_atom4_name,
                        ]
                    ]
                ) == sorted(
                    [new_atom1_name, new_atom2_name, new_atom3_name, new_atom4_name]
                ):
                    found = True
                    if sorted(
                        [
                            original_torsion.atom1.type,
                            original_torsion.atom2.type,
                            original_torsion.atom3.type,
                            original_torsion.atom3.type,
                        ]
                    ) == sorted(
                        [
                            new_torsion.atom1.type,
                            new_torsion.atom2.type,
                            new_torsion.atom3.type,
                            new_torsion.atom4.type,
                        ]
                    ):
                        continue

                    mod_types = []
                    # torsion present at cc1 needs to be turned fully off starting at lambda_vlaue == 1.
                    f = max((1 - ((1 - lambda_value) * 2)), 0.0)

                    if f > 0.0 or lambda_value == 0.5:
                        for torsion_t in original_torsion.type:
                            modified_phi_k = torsion_t.phi_k * f
                            mod_types.append(
                                mod_type(
                                    modified_phi_k,
                                    torsion_t.per,
                                    torsion_t.phase,
                                    torsion_t.scee,
                                    torsion_t.scnb,
                                )
                            )

                    # torsion present at cc2 needs to be fully turned on at lambda_value == 0.0
                    f = 1 - min((lambda_value) * 2, 1.0)
                    if f > 0.0:
                        for torsion_t in new_torsion.type:
                            modified_phi_k = torsion_t.phi_k * f
                            if modified_phi_k >= 0.0:
                                mod_types.append(
                                    mod_type(
                                        modified_phi_k,
                                        torsion_t.per,
                                        torsion_t.phase,
                                        torsion_t.scee,
                                        torsion_t.scnb,
                                    )
                                )

                    original_torsion.mod_type = mod_types

            if not found:
                logger.critical(original_torsion)
                raise RuntimeError("No corresponding torsion in cc2 found")

    def mutate(self, psf: pm.charmm.CharmmPsfFile, lambda_value: float):
        """
        Mutates the bonded parameters of cc1 to cc2.
        Parameters
        ----------
        psf : pm.charmm.CharmmPsfFile
            psf that gets mutated
        lambda_value : float
            lambda_value
        """

        assert type(psf) == pm.charmm.CharmmPsfFile
        if self.charge_mutation:
            logger.info(f" -- Charge parameters from cc1 are transformed to cc2.")
            logger.info(f"Lambda value:{lambda_value}")
            # scale charge
            self._mutate_charges(psf, lambda_value)
        if self.bonded_terms_mutation:
            logger.info(
                f" -- Atom/Bond/Angle/Torsion parameters from cc1 are transformed to cc2."
            )
            logger.info(f"Lambda value:{lambda_value}")
            # scale atoms
            self._mutate_atoms(psf, lambda_value)
            # scale bonds
            self._mutate_bonds(psf, lambda_value)
            # scale angles
            self._mutate_angles(psf, lambda_value)
            # scale torsions
            self._mutate_torsions(psf, lambda_value)

    @staticmethod
    def _modify_type_in_cc(atom: pm.Atom, psf: pm.charmm.CharmmPsfFile):

        if hasattr(atom, "initial_type"):
            # only change parameters
            pass
        else:
            logger.info(f"Setting RRR atomtype for atom: {atom}.")
            atom.initial_type = atom.type
            psf.number_of_dummys += 1
            atom.type = f"RRR{psf.number_of_dummys}"


class Mutation(object):
    def __init__(self, atoms_to_be_mutated: list, dummy_region: DummyRegion):

        assert type(atoms_to_be_mutated) == list
        self.atoms_to_be_mutated = atoms_to_be_mutated
        self.dummy_region = dummy_region
        self.tlc = dummy_region.tlc

    def _mutate_charge(
        self, psf: pm.charmm.CharmmPsfFile, lambda_value: float, offset: int
    ):

        total_charge = int(
            round(sum([atom.initial_charge for atom in psf.view[f":{self.tlc}"].atoms]))
        )
        # scale the charge of all atoms
        logger.info(f"Scaling charge on: {self.atoms_to_be_mutated}")
        for idx in self.atoms_to_be_mutated:
            odx = idx + offset
            atom = psf[odx]
            logger.debug(f"Scale charge on {atom}")
            logger.debug(f"Scaling charge with: {lambda_value}")
            logger.debug(f"Old charge: {atom.charge}")
            atom.charge = atom.initial_charge * lambda_value
            logger.debug(f"New charge: {atom.charge}")

        if lambda_value != 1:
            # compensate for the total change in charge the terminal atom
            self._compensate_charge(psf, total_charge, offset)

    def _mutate_vdw(
        self,
        psf: pm.charmm.CharmmPsfFile,
        lambda_value: float,
        vdw_atom_idx: List[int],
        offset: int,
        to_default: bool,
    ):

        if not set(vdw_atom_idx).issubset(set(self.atoms_to_be_mutated)):
            raise RuntimeError(
                f"Specified atom {vdw_atom_idx} is not in atom_idx list {self.atoms_to_be_mutated}. Aborting."
            )

        logger.info(f"Acting on atoms: {vdw_atom_idx}")
        offset = min([a.idx for a in psf.view[f":{self.tlc.upper()}"].atoms])

        for i in vdw_atom_idx:
            atom = psf[i + offset]
            if to_default:
                logger.info("Mutate to default")
                atom_type_suffix = "DDX"
                atom.rmin = 1.5
                atom.epsilon = -0.15
            else:
                logger.info("Mutate to dummy")
                atom_type_suffix = f"DDD"
                self._scale_epsilon(atom, lambda_value)
                self._scale_rmin(atom, lambda_value)
            # NOTEthere is always a type change
            self._modify_type(atom, psf, atom_type_suffix)

    def mutate(
        self,
        psf: pm.charmm.CharmmPsfFile,
        lambda_value_electrostatic: float = 1.0,
        lambda_value_vdw: float = 1.0,
        vdw_atom_idx: List[int] = [],
        steric_mutation_to_default: bool = False,
    ):
        """Performs the mutation"""

        if lambda_value_electrostatic < 0.0 or lambda_value_electrostatic > 1.0:
            raise RuntimeError("Lambda value for LJ needs to be between 0.0 and 1.0.")

        if lambda_value_vdw < 0.0 or lambda_value_vdw > 1.0:
            raise RuntimeError("Lambda value for vdw needs to be between 0.0 and 1.0.")

        logger.debug(f"LJ scaling factor: {lambda_value_electrostatic}")
        logger.debug(f"VDW scaling factor: {lambda_value_vdw}")

        offset = min([a.idx for a in psf.view[f":{self.tlc.upper()}"].atoms])

        if lambda_value_electrostatic < 1.0:
            self._mutate_charge(psf, lambda_value_electrostatic, offset)

        if lambda_value_vdw < 1.0:
            self._mutate_vdw(
                psf, lambda_value_vdw, vdw_atom_idx, offset, steric_mutation_to_default
            )

    def _compensate_charge(
        self, psf: pm.charmm.CharmmPsfFile, total_charge: int, offset: int
    ):
        """
        _compensate_charge This function compensates the charge changes of a dummy region on the terminal real atom
        that connects the specific dummy group to the real region.

        Parameters
        ----------
        psf : pm.charmm.CharmmPsfFile
            [description]
        total_charge : int
            [description]
        offset : int
            [description]

        Raises
        ------
        RuntimeError
            [description]
        """

        # get dummy retions
        connected_dummy_regions = self.dummy_region.connected_dummy_regions
        logger.debug(f"Compensating charge ...")
        # save the atoms that are used for charge compenstation. This is done because if two regions
        # use the same atom, a special handling needs to be invoced
        compensating_on_this_real_atom = []
        # check for each dummy region how much charge has changed and compensate on atom that connects
        # the real region with specific dummy regions
        for dummy_idx in connected_dummy_regions:
            logger.debug(f"Dummy idx region: {dummy_idx}")
            connecting_real_atom_for_this_dummy_region = (
                self.dummy_region.return_connecting_real_atom(dummy_idx)
            )
            logger.debug(
                f"Connecting atom: {connecting_real_atom_for_this_dummy_region}"
            )
            if connecting_real_atom_for_this_dummy_region == None:
                raise RuntimeError(
                    "Something went wrong with the charge compensation. Aborting."
                )

            charge_acceptor = psf[connecting_real_atom_for_this_dummy_region + offset]

            charge_to_compenstate_for_region = 0.0
            for atom_idx in dummy_idx:
                charge_to_compenstate_for_region += (
                    psf[atom_idx + offset].initial_charge
                    - psf[atom_idx + offset].charge
                )

            logger.debug(f"Charge to compensate: {charge_to_compenstate_for_region}")
            # adding charge difference to initial charge on real terminal atom
            if (
                connecting_real_atom_for_this_dummy_region
                in compensating_on_this_real_atom
            ):
                charge_acceptor.charge = (
                    charge_acceptor.charge + charge_to_compenstate_for_region
                )
            else:
                charge_acceptor.charge = (
                    charge_acceptor.initial_charge + charge_to_compenstate_for_region
                )

            compensating_on_this_real_atom.append(
                connecting_real_atom_for_this_dummy_region
            )

        # check if rest charge is missing
        new_charge = sum(
            [atom.charge for atom in psf.view[f":{self.tlc.upper()}"].atoms]
        )

        if not (np.isclose(new_charge, total_charge, rtol=1e-4)):
            print(psf)
            raise RuntimeError(
                f"Charge compensation failed. Introducing non integer total charge: {new_charge}. Target total charge: {total_charge}."
            )

    @staticmethod
    def _scale_epsilon(atom, lambda_value: float):
        logger.debug(atom)
        logger.debug(atom.initial_epsilon)
        atom.epsilon = atom.initial_epsilon * lambda_value

    @staticmethod
    def _scale_rmin(atom, lambda_value: float):
        logger.debug(atom)
        logger.debug(atom.initial_rmin)
        atom.rmin = atom.initial_rmin * lambda_value

    @staticmethod
    def _modify_type(atom, psf, atom_type_suffix: str):

        if hasattr(atom, "initial_type"):
            # only change parameters
            pass
        else:
            atom.initial_type = atom.type
            if atom_type_suffix == "DDD":
                psf.number_of_dummys += 1
                new_type = f"{atom_type_suffix}{psf.number_of_dummys}"
            elif atom_type_suffix == "DDX":
                psf.mutations_to_default += 1
                new_type = f"{atom_type_suffix}{psf.mutations_to_default}"

            atom.type = new_type


def mutate_pure_tautomers(
    s1_to_s2: ProposeMutationRoute,
    system1: SystemStructure,
    system2: SystemStructure,
    configuration,
    single_state=False,
    nr_of_bonded_windows: int = 4,
):

    from transformato import (
        IntermediateStateFactory,
    )

    # setup mutation and StateFactory
    mutation_list = s1_to_s2.generate_mutations_to_common_core_for_mol1()
    i = IntermediateStateFactory(
        system=system1,
        configuration=configuration,
    )

    # write out states
    # start with charge
    intst = 1
    charges = mutation_list["charge"]
    output_files_t1 = []
    for lambda_value in np.linspace(1, 0, 2):
        # turn off charges
        o, intst = i.write_state(
            mutation_conf=charges,
            lambda_value_electrostatic=lambda_value,
            intst_nr=intst,
        )
        output_files_t1.append(o)
        if single_state:
            return (output_files_t1, [])
    # turn off the lj of the hydrogen
    lj = mutation_list["lj"]
    o, intst = i.write_state(
        mutation_conf=lj,
        lambda_value_vdw=0.0,
        intst_nr=intst,
    )
    output_files_t1.append(o)

    # transform common core
    for lambda_value in np.linspace(1, 0, nr_of_bonded_windows + 1)[1:]:
        # turn off charges
        o, intst = i.write_state(
            mutation_conf=mutation_list["transform"],
            common_core_transformation=lambda_value,
            intst_nr=intst,
        )
        output_files_t1.append(o)

    # setup other tautomer
    mutation_list = s1_to_s2.generate_mutations_to_common_core_for_mol2()
    i = IntermediateStateFactory(
        system=system2,
        configuration=configuration,
    )
    # write out states
    # start with charge
    intst = 1
    output_files_t2 = []
    charges = mutation_list["charge"]
    for lambda_value in np.linspace(1, 0, 2):
        # turn off charges
        o, intst = i.write_state(
            mutation_conf=charges,
            lambda_value_electrostatic=lambda_value,
            intst_nr=intst,
        )
        output_files_t2.append(o)

    # turn off the lj of the hydrogen
    lj = mutation_list["lj"]
    o, intst = i.write_state(
        mutation_conf=lj,
        lambda_value_vdw=0.0,
        intst_nr=intst,
    )
    output_files_t2.append(o)

    return (output_files_t1, output_files_t2)
