import io
import logging
import os
from collections import namedtuple
from copy import deepcopy

import numpy as np
import parmed as pm
import rdkit
from IPython.core.display import display
from rdkit import Chem
from rdkit.Chem import AllChem, Draw, rdFMCS
from rdkit.Chem.Draw import IPythonConsole, rdMolDraw2D
from simtk import unit
from transformato import state
import networkx as nx 

logger = logging.getLogger(__name__)


class ProposeMutationRoute(object):

    def __init__(self, s1: state, s2: state):
        """
        A class that proposes the mutation route between two molecules with a 
        common core (same atom types) based on two mols and generates the mutation 
        objects to perform the mutation on the psf objects.
        Parameters
        ----------
        mol1: Chem.Mol
        mol2: Chem.Mol
        """

        mol1_name: str = 'm1'
        mol2_name: str = 'm2'

        self.mols: dict = {mol1_name: s1.mol, mol2_name: s2.mol}
        self.graphs: dict = {mol1_name: s1.graph, mol2_name: s2.graph}
        self.psfs: dict = {mol1_name: s1.waterbox_psf[f":{s1.tlc}"], mol2_name: s2.waterbox_psf[f":{s2.tlc}"]}
        self._substructure_match: dict = {mol1_name: [], mol2_name: []}
        self._calculate_common_core(mol1_name, mol2_name)
        self.removed_indeces: dict = {mol1_name: [], mol2_name: []}
        self.added_indeces: dict = {mol1_name: [], mol2_name: []}
        self.s1_tlc = s1.tlc
        self.s2_tlc = s2.tlc

    def add_idx_to_common_core_of_mol1(self, idx: int):
        self._add_common_core_atom('m1', idx)
        print(self.get_common_core_idx_mol1())

    def add_idx_to_common_core_of_mol2(self, idx: int):
        self._add_common_core_atom('m2', idx)
        print(self.get_common_core_idx_mol2())

    def _add_common_core_atom(self, name, idx):
        if idx in self.added_indeces[name] or idx in self._get_common_core(name):
            print(f"Idx: {idx} already in common core.")
            pass
        self.added_indeces[name].append(idx)

    def get_common_core_idx_mol1(self) -> list:
        """
        Returns the common core of mol1.
        """
        return self._get_common_core('m1')

    def get_common_core_idx_mol2(self) -> list:
        """
        Returns the common core of mol2.
        """
        return self._get_common_core('m2')

    def _get_common_core(self, name: str) -> list:
        """
        Helper Function - should not be called directly.
        Returns the common core.
        """
        keep_idx = []
        for idx in self._substructure_match[name] + self.added_indeces[name]:  # BEWARE: the ordering is important - don't cast set!
            if idx not in self.removed_indeces[name]:
                keep_idx.append(idx)
        return keep_idx

    def _calculate_common_core(self, mol1_name: str, mol2_name: str):
        """
        A class that proposes the mutation route between two molecules with a 
        common core (same atom types) based on two mols and generates the mutation 
        objects to perform the mutation on the psf objects.
        Parameters
        ----------
        mol1_name: str
        mol2_name: str
        """
        m1, m2 = [deepcopy(self.mols[mol1_name]), deepcopy(self.mols[mol2_name])]

        for m in [m1, m2]:
            logger.info('Mol in SMILES format: {}.'.format(Chem.MolToSmiles(m, True)))

        # make copy of mols
        changed_mols = [Chem.Mol(x) for x in [m1, m2]]

        # find substructure match (ignore bond order but enforce element matching)
        mcs = rdFMCS.FindMCS(changed_mols, bondCompare=rdFMCS.BondCompare.CompareAny,
                             timeout=120, atomCompare=rdFMCS.AtomCompare.CompareElements)
        logger.info('Substructure match: {}'.format(mcs.smartsString))

        # convert from SMARTS
        mcsp = Chem.MolFromSmarts(mcs.smartsString, False)

        s1 = (m1.GetSubstructMatch(mcsp))
        logger.info('Substructere match idx: {}'.format(s1))
        self._display_mol(m1)
        s2 = (m2.GetSubstructMatch(mcsp))
        logger.info('Substructere match idx: {}'.format(s2))
        self._display_mol(m2)

        self._substructure_match[mol1_name] = list(s1)
        self._substructure_match[mol2_name] = list(s2)

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
                mol.GetAtomWithIdx(idx).SetProp('molAtomMapNumber', str(mol.GetAtomWithIdx(idx).GetIdx()))
            return mol

        mol = mol_with_atom_index(mol)
        AllChem.Compute2DCoords(mol)
        display(mol)

    def show_common_core_on_mol1(self):
        """
        Shows common core on mol1        
        """
        return self._show_common_core(self.mols['m1'], self.get_common_core_idx_mol1())

    def show_common_core_on_mol2(self):
        """
        Shows common core on mol2        
        """
        return self._show_common_core(self.mols['m2'], self.get_common_core_idx_mol2())

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
            opts.atomLabels[i.GetIdx()] = str(i.GetProp('atom_index')) + ':' + i.GetProp('atom_type')

        drawer.DrawMolecule(mol, highlightAtoms=highlight)
        Draw.DrawingOptions.includeAtomNumbers = False
        drawer.FinishDrawing()
        svg = drawer.GetDrawingText().replace('svg:', '')
        return(svg)

    def generate_mutations_to_common_core_for_mol1(self, nr_of_steps_for_el: int, nr_of_steps_for_bonded_parameters: int) -> list:
        """
        Generates the mutation route to the common fore for mol1.
        Parameters
        ----------
        nr_of_steps_for_el : int
            nr of steps used for linearly scaling the charges to zero
        nr_of_steps_for_bonded_parameters : int
        Returns
        ----------
        mutations: list
            list of mutations

        """
        m = self._mutate_to_common_core('m1', self.get_common_core_idx_mol1(), nr_of_steps_for_el)
        t = self._transform_common_core(nr_of_steps_for_bonded_parameters)

        return m + t

    def generate_mutations_to_common_core_for_mol2(self, nr_of_steps_for_el: int) -> list:
        """
        Generates the mutation route to the common fore for mol2.
        Returns
        ----------
        mutations: list
            list of mutations        
        """

        m = self._mutate_to_common_core('m2', self.get_common_core_idx_mol2(), nr_of_steps_for_el)
        return m

    def _transform_common_core(self, nr_of_steps_for_bonded_parameters: int) -> list:
        """
        Common Core 1 is transformed to Common core 2. Bonded parameters and charges are adjusted. 
        """

        transformations = []

        # test if bonded mutations are necessary
        bonded_mutations_necessary = False
        for cc1, cc2 in zip(self.get_common_core_idx_mol1(), self.get_common_core_idx_mol2()):
            # did atom type change? if not don't add BondedMutations
            atom1 = self.psfs['m1'][cc1]
            atom2 = self.psfs['m2'][cc2]
            if atom1.type != atom2.type:
                logger.info('###########')
                logger.info('Atom that needs to be transformed: {}.'.format(atom1))
                logger.info('Atom type of atom in cc1: {}.'.format(atom1.type))
                logger.info('Template atom: {}.'.format(atom2))
                logger.info('Atom type of atom in cc2: {}.'.format(atom2.type))

                bonded_mutations_necessary = True

        # if necessary transform bonded parameters
        if bonded_mutations_necessary:
            logger.info('Bonded parameters need to be transformed for the cc1 topology.')
            # ugh
            t = BondedParameterMutation(self.get_common_core_idx_mol1(),
            self.get_common_core_idx_mol2(), self.psfs['m1'], self.psfs['m2'],
            nr_of_steps_for_bonded_parameters, self.s1_tlc, self.s2_tlc, only_charge=False)
            transformations.append(t)
        else:
            # Only Charge transformation
            logger.info('Charges at commen core need to be transformed')
            t = BondedParameterMutation(self.get_common_core_idx_mol1(),
            self.get_common_core_idx_mol2(), self.psfs['m1'], self.psfs['m2'],
            nr_of_steps_for_bonded_parameters, self.s1_tlc, self.s2_tlc, only_charge=True)
            transformations.append(t)

        return transformations

    @staticmethod
    def _find_terminal_atom(cc_idx: list, mol):
        """
        Find atoms that connect the rest of the molecule to the common core.

        Args:
            cc_idx (list): common core index atoms
            mol ([type]): rdkit mol object
        """
        terminal_atoms = []
        
        for atom in mol.GetAtoms():
            idx = atom.GetIdx()
            if idx not in cc_idx:
                neighbors = [x.GetIdx() for x in atom.GetNeighbors()]
                if any([n in cc_idx for n in neighbors]):
                    terminal_atoms.append(idx)

        logger.debug(f"Terminal atoms: {str(list(set(terminal_atoms)))}")
        return list(set(terminal_atoms))

    def _mutate_to_common_core(self, name: str, cc_idx: list, nr_of_steps_for_el: int) -> list:
        """
        Helper function - do not call directly.
        Generates the mutation route to the common fore for mol2.
        """
        mol = self.mols[name]
        hydrogens = []
        charge_mutations = []
        lj_mutations = []
        atoms_to_be_mutated = []
        for atom in mol.GetAtoms():
            idx = atom.GetIdx()
            if idx not in cc_idx:
                if atom.GetSymbol() == 'H':
                    hydrogens.append(idx)
                atoms_to_be_mutated.append(idx)
                logger.info('Will be decoupled: Idx:{} Element:{}'.format(idx, atom.GetSymbol()))

        if atoms_to_be_mutated:
            ######################
            # scale all EL of all atoms to zero
            charge_mutations.append(ChargeToZeroMutation(atom_idx=atoms_to_be_mutated,
                                                nr_of_steps=nr_of_steps_for_el, common_core=cc_idx))

            
            ######################
            # scale LJ
            # here we save the last mutation steps
            terminal_atoms = self._find_terminal_atom(cc_idx, mol)
            lj_terminal_mutations = []

            # start with mutation of LJ of hydrogens
            lj_mutations.append(StericToZeroMutation(hydrogens))
            already_mutated = [] 
            # continue with scaling of heavy atoms LJ
            l = []
            for n in nx.dfs_edges(self.graphs[name]):
                logger.debug(n)
                l.append(n)
            
            for idx1, idx2 in l:
                if idx1 in atoms_to_be_mutated and idx1 not in hydrogens and idx1 not in already_mutated:
                    
                    if idx1 in terminal_atoms:
                        lj_terminal_mutations.append(StericToZeroMutation([idx1]))
                    else:
                        lj_mutations.append(StericToZeroMutation([idx1]))
                    already_mutated.append(idx1)

                if idx2 in atoms_to_be_mutated and idx2 not in hydrogens and idx2 not in already_mutated:
                    if idx2 in terminal_atoms:
                        lj_terminal_mutations.append(StericToZeroMutation([idx2]))
                    else:
                        lj_mutations.append(StericToZeroMutation([idx2]))
                    already_mutated.append(idx2)
            # test that all mutations are included
            # TODO: test that all mutations are covered
            
            mutations = charge_mutations + lj_mutations + lj_terminal_mutations

            for m in mutations:
                if type(m) == ChargeMutation:
                    logger.debug(f"charge mutation on: {str(m.atom_idx)}")
                elif type(m) == StericMutation:
                    logger.debug(f"steric mutation on: {str(m.atom_idx)}")
                else:
                    logger.debug(f"mutation on: {str(m.atom_idx)}")
                    
            nr_of_lj_mutations = len(lj_mutations) + len(lj_terminal_mutations)
            if nr_of_lj_mutations != len(atoms_to_be_mutated) - len(hydrogens) + 1:
                # test if we have a single mutation for every heavy atom and a mutation for all hydrogens
                logger.critical(f"Nr of lj mutations: {nr_of_lj_mutations}")
                logger.critical(f"Nr of atoms to be mutated (nr of atoms - nr of hydrogens): {len(atoms_to_be_mutated) - len(hydrogens)}")
                logger.critical(f"Atoms to be mutated: {str(atoms_to_be_mutated)}")
                logger.critical(mutations)
                raise RuntimeError('There are atoms missing in the steric mutation step ')
        else:
            logger.info("No atoms will be decoupled.")
            mutations = []
        return mutations

    def _find_cliques(self, atoms_idx: list, mol: Chem.Mol) -> list:
        mutation_order = []
        for atom in atoms_idx:
            if atom in mutation_order:  # already sorted
                continue

        return mutation_order


class BondedParameterMutation(object):

    def __init__(self, cc1_idx: list, cc2_idx: list, cc1_psf: pm.charmm.CharmmPsfFile, cc2_psf: pm.charmm.CharmmPsfFile, nr_of_steps: int, tlc_cc1: str, tlc_cc2: str, only_charge: bool=False):
        """
        Scale the bonded parameters inside the common core.
        Parameters
        ----------
        cc1_idx : list
            indices of cc1
        cc2_idx : list
            indices of cc2 (in the same order as cc1)
        cc1_psf : pm.charmm.CharmmPsfFile (copy of only ligand)
        cc2_psf : pm.charmm.CharmmPsfFile (copy of only ligand)
            the target psf that is used to generate the new bonded parmaeters
        nr_of_steps : int
        tlc_cc1 : str
            three letter code of ligand in cc1
        tlc_cc2 : str
            three letter code of ligand in cc2
        """
        self.cc1_idx = cc1_idx
        self.cc2_idx = cc2_idx
        self.cc2_psf = cc2_psf
        self.cc1_psf = cc1_psf
        self.nr_of_steps = nr_of_steps
        assert(self.nr_of_steps >= 2)
        self.tlc_cc1 = tlc_cc1
        self.tlc_cc2 = tlc_cc2
        self.atom_names_mapping = self._get_atom_mapping()
        self._prepare_cc2_psf_for_charge_transfer()
        self.only_charge = only_charge

    def _prepare_cc2_psf_for_charge_transfer(self):
        # prepare cc2 psf
        offset = min([atom.idx for atom in self.cc2_psf.view[f":{self.tlc_cc2.upper()}"].atoms])
        atoms_to_be_mutated = []
        for atom in self.cc2_psf.view[f":{self.tlc_cc2.upper()}"].atoms:
            atom.initial_charge = atom.charge
            idx = atom.idx - offset
            if idx not in self.cc2_idx:
                atoms_to_be_mutated.append(idx)
        logger.debug(f"Atoms for which charge is set to zero: {atoms_to_be_mutated}")
        m = ChargeToZeroMutation(atoms_to_be_mutated, 1, self.cc2_idx)
        m.mutate(self.cc2_psf, self.tlc_cc2, 1)

    def _get_atom_mapping(self):
        match_atom_names_cc1_to_cc2 = {}
        for cc1, cc2 in zip(self.cc1_idx, self.cc2_idx):
            cc1_a = self.cc1_psf[cc1]
            cc2_a = self.cc2_psf[cc2]
            match_atom_names_cc1_to_cc2[cc1_a.name] = cc2_a.name

        return match_atom_names_cc1_to_cc2

    def _mutate_charges(self, psf: pm.charmm.CharmmPsfFile, tlc: str, scale: float):

        for cc1_atom in psf.view[f":{tlc}"]:
            if cc1_atom.name not in self.atom_names_mapping:
                continue
            found = False
            for cc2_atom in self.cc2_psf:
                if self.atom_names_mapping[cc1_atom.name] == cc2_atom.name:
                    found = True
                    # are the atoms different?
                    logger.debug(f"Modifying atom: {cc1_atom}")
                    logger.debug(f"Template atom: {cc2_atom}")

                    # scale epsilon
                    logger.debug(f"Real charge: {cc1_atom.charge}")
                    modified_charge = (1.0 - scale) * cc1_atom.initial_charge + scale * cc2_atom.charge
                    logger.debug(f"New epsilon: {modified_charge}")
                    cc1_atom.charge = modified_charge

            if not found:
                raise RuntimeError('No corresponding atom in cc2 found')


    def _mutate_atoms(self, psf: pm.charmm.CharmmPsfFile, tlc: str, scale: float):
        mod_type = namedtuple('Atom', 'epsilon, rmin')
        for cc1_atom in psf.view[f":{tlc}"]:
            if cc1_atom.name not in self.atom_names_mapping:
                continue

            found = False
            for cc2_atom in self.cc2_psf:
                if self.atom_names_mapping[cc1_atom.name] == cc2_atom.name:
                    found = True
                    # are the atoms different?
                    if cc1_atom.type != cc2_atom.type:
                        self._modify_type(cc1_atom, psf)
                        logger.debug(f"Modifying atom: {cc1_atom}")
                        logger.debug(f"Template atom: {cc2_atom}")

                        # scale epsilon
                        logger.debug(f"Real epsilon: {cc1_atom.epsilon}")
                        modified_epsilon = (1.0 - scale) * cc1_atom.epsilon + scale * cc2_atom.epsilon
                        logger.debug(f"New epsilon: {modified_epsilon}")

                        # scale rmin
                        logger.debug(f"Real rmin: {cc1_atom.rmin}")
                        modified_rmin = (1.0 - scale) * cc1_atom.rmin + scale * cc2_atom.rmin
                        logger.debug(f"New rmin: {modified_rmin}")

                        cc1_atom.mod_type = mod_type(modified_epsilon, modified_rmin)

            if not found:
                raise RuntimeError('No corresponding atom in cc2 found')

    def _mutate_bonds(self, psf: pm.charmm.CharmmPsfFile, tlc: str, scale: float):

        mod_type = namedtuple('Bond', 'k, req')
        for cc1_bond in psf.view[f":{tlc}"].bonds:

            cc1_a1 = cc1_bond.atom1.name
            cc1_a2 = cc1_bond.atom2.name
            # all atoms of the bond must be in cc
            # everything outside the cc are bonded terms between dummies or
            # between real atoms and dummies and we can ignore them for now
            if not all(elem in self.atom_names_mapping for elem in [cc1_a1, cc1_a2]):
                continue

            found = False
            for cc2_bond in self.cc2_psf.bonds:
                cc2_a1 = cc2_bond.atom1.name
                cc2_a2 = cc2_bond.atom2.name
                # all atoms of the bond must be in cc
                if not all(elem in self.atom_names_mapping.values() for elem in [cc2_a1, cc2_a2]):
                    continue

                # match the two bonds
                if sorted([self.atom_names_mapping[e] for e in [cc1_a1, cc1_a2]]) == sorted([cc2_a1, cc2_a2]):
                    found = True
                    # are the bonds different?
                    if sorted([cc1_bond.atom1.type, cc1_bond.atom2.type]) == sorted([cc2_bond.atom1.type, cc2_bond.atom2.type]):
                        continue
                    logger.debug('##############################')
                    logger.debug(scale)
                    logger.debug(f"Modifying bond: {cc1_bond}")

                    logger.debug(f"Template bond: {cc2_bond}")
                    logger.debug('Original value for k: {}'.format(cc1_bond.type.k))
                    logger.debug(f"Target k: {cc2_bond.type.k}")
                    new_k = ((1.0 - scale) * cc1_bond.type.k) + (scale * cc2_bond.type.k)
                    logger.debug(new_k)

                    modified_k = new_k

                    logger.debug(f"New k: {modified_k}")

                    logger.debug(f"Old req: {cc1_bond.type.req}")
                    modified_req = ((1.0 - scale) * cc1_bond.type.req) + (scale * cc2_bond.type.req)
                    logger.debug(f"Modified bond: {cc1_bond}")

                    cc1_bond.mod_type = mod_type(modified_k, modified_req)
                    logger.debug(cc1_bond.mod_type)

            if not found:
                logger.critical(cc1_bond)
                raise RuntimeError('No corresponding bond in cc2 found: {}'.format(cc1_bond))

    def _mutate_angles(self, psf: pm.charmm.CharmmPsfFile, tlc: str, scale: float):

        mod_type = namedtuple('Angle', 'k, theteq')
        for cc1_angle in psf.view[f":{tlc}"].angles:
            cc1_a1 = cc1_angle.atom1.name
            cc1_a2 = cc1_angle.atom2.name
            cc1_a3 = cc1_angle.atom3.name

            # only angles in cc
            if not all(elem in self.atom_names_mapping for elem in [cc1_a1, cc1_a2, cc1_a3]):
                continue

            found = False
            for cc2_angle in self.cc2_psf.angles:
                cc2_a1 = cc2_angle.atom1.name
                cc2_a2 = cc2_angle.atom2.name
                cc2_a3 = cc2_angle.atom3.name
                # only angles in cc
                if not all(elem in self.atom_names_mapping.values() for elem in [cc2_a1, cc2_a2, cc2_a3]):
                    continue

                if sorted([self.atom_names_mapping[e] for e in [cc1_a1, cc1_a2, cc1_a3]]) == sorted([cc2_a1, cc2_a2, cc2_a3]):
                    found = True
                    if sorted([cc1_angle.atom1.type, cc1_angle.atom2.type, cc1_angle.atom3.type]) == \
                            sorted([cc2_angle.atom1.type, cc2_angle.atom2.type, cc2_angle.atom3.type]):
                        continue

                    logger.debug(f"Modifying angle: {cc1_angle}")
                    logger.debug(f"Template bond: {cc2_angle}")
                    logger.debug('Scaling k and theteq')

                    logger.debug(f"Old k: {cc1_angle.type.k}")
                    modified_k = (1.0 - scale) * cc1_angle.type.k + scale * cc2_angle.type.k
                    logger.debug(f"New k: {modified_k}")

                    logger.debug(f"Old k: {cc1_angle.type.theteq}")
                    modified_theteq = (1.0 - scale) * cc1_angle.type.theteq + scale * cc2_angle.type.theteq
                    logging.debug(f"New k: {modified_theteq}")

                    cc1_angle.mod_type = mod_type(modified_k, modified_theteq)

            if not found:
                logger.critical(cc1_angle)
                raise RuntimeError('No corresponding angle in cc2 found')

    def _mutate_torsions(self, psf: pm.charmm.CharmmPsfFile, tlc: str, scale: float):

        mod_type = namedtuple('Torsion', 'phi_k, per, phase, scee, scnb')

        # get all torsions present in initial topology
        for cc1_torsion in psf.view[f":{tlc}"].dihedrals:
            cc1_a1 = cc1_torsion.atom1.name
            cc1_a2 = cc1_torsion.atom2.name
            cc1_a3 = cc1_torsion.atom3.name
            cc1_a4 = cc1_torsion.atom4.name
            # all atoms must be in the cc
            if not all(elem in self.atom_names_mapping for elem in [cc1_a1, cc1_a2, cc1_a3, cc1_a4]):
                continue

            # get corresponding torsion types in the new topology
            for cc2_torsion in self.cc2_psf.dihedrals:
                cc2_a1 = cc2_torsion.atom1.name
                cc2_a2 = cc2_torsion.atom2.name
                cc2_a3 = cc2_torsion.atom3.name
                cc2_a4 = cc2_torsion.atom4.name
                # only torsion in cc
                if not all(elem in self.atom_names_mapping.values() for elem in [cc2_a1, cc2_a2, cc2_a3, cc2_a4]):
                    continue

                if sorted([self.atom_names_mapping[e] for e in [cc1_a1, cc1_a2, cc1_a3, cc1_a4]]) == sorted([cc2_a1, cc2_a2, cc2_a3, cc2_a4]):
                    found = True
                    if sorted([cc1_torsion.atom1.type, cc1_torsion.atom2.type, cc1_torsion.atom3.type, cc1_torsion.atom3.type]) == \
                            sorted([cc2_torsion.atom1.type, cc2_torsion.atom2.type, cc2_torsion.atom3.type, cc2_torsion.atom4.type]):
                        continue

                    mod_types = []
                    if scale <= 0.5:
                        # torsion present at cc1 needs to be turned fully off starting from self.nr_of_steps/2
                        for torsion_t in cc1_torsion.type:
                            modified_phi_k = torsion_t.phi_k * max(((1.0 - scale * 2)), 0.0)
                            mod_types.append(mod_type(modified_phi_k, torsion_t.per, torsion_t.phase,
                                                      torsion_t.scee, torsion_t.scnb))
                    else:
                        # torsion present at cc1 needs to be turned fully off starting from self.nr_of_steps/2
                        for torsion_t in cc2_torsion.type:
                            modified_phi_k = torsion_t.phi_k * max((scale - 0.5) * 2, 0.0)
                            mod_types.append(mod_type(modified_phi_k, torsion_t.per, torsion_t.phase,
                                                      torsion_t.scee, torsion_t.scnb))

                    cc1_torsion.mod_type = mod_types
            if not found:
                logger.critical(cc1_torsion)
                raise RuntimeError('No corresponding torsion in cc2 found')

    def mutate(self, psf: pm.charmm.CharmmPsfFile, tlc: str, current_step: int):
        """
        Mutates the bonded parameters of cc1 to cc2.
        Parameters
        ----------
        psf : pm.charmm.CharmmPsfFile
            psf that gets mutated
        tlc : str
        current_step : int
            the current step in the mutation protocoll
        only_charge : bool
            only charge is scaled from cc1 to cc2
        """

        assert(type(psf) == pm.charmm.CharmmPsfFile)
        scale = current_step / (self.nr_of_steps)
        if self.only_charge:
            logger.info(f" -- Charge parameters from cc1 are transformed to cc2.")
            logger.info(f"Scaling factor:{scale}")
            # scale charge
            self._mutate_charges(psf, tlc, scale)
        else:
            logger.info(f" -- Charge/Atom/Bond/Angle/Torsion parameters from cc1 are transformed to cc2.")
            logger.info(f"Scaling factor:{scale}")
            # scale charge
            self._mutate_charges(psf, tlc, scale)
            # scale atoms
            self._mutate_atoms(psf, tlc, scale)
            # scale bonds
            self._mutate_bonds(psf, tlc, scale)
            # scale angles
            self._mutate_angles(psf, tlc, scale)
            # scale torsions
            self._mutate_torsions(psf, tlc, scale)

    def _modify_type(self, atom: pm.Atom, psf: pm.charmm.CharmmPsfFile):

        if (hasattr(atom, 'initial_type')):
            # only change parameters
            pass
        else:
            logger.info(f"Setting RRR atomtype for atom: {atom}.")
            atom.type = f"RRR{psf.number_of_dummys}"
            atom.initial_type = atom.type
            psf.number_of_dummys += 1


class BaseMutation(object):

    def __init__(self, atom_idx: list, nr_of_steps: int):
        assert(type(atom_idx) == list)
        self.atom_idx = atom_idx
        self.nr_of_steps = nr_of_steps


class ChargeMutation(BaseMutation):

    def __init__(self, atom_idx: list, nr_of_steps: int, common_core: list):
        if (nr_of_steps <= 2):
            logger.warning('Rapid charge change detected. Proceed with caution.')
        super().__init__(atom_idx, nr_of_steps)
        self.common_core = common_core

    def _scale_charge(self, atom, new_charge):
        """
        Scale atom with charge and compensate the charge change.
        """
        diff_charge = atom.charge - new_charge
        atom.charge = new_charge
        return diff_charge

    def _compensate_charge(self, psf, offset: int, diff_charge: float, tlc: str):

        nr_of_atoms_to_spread_charge = len(self.common_core)
        logger.info('##############')
        logger.info(f"Charge to compensate: {diff_charge}")
        charge_part = diff_charge / nr_of_atoms_to_spread_charge
        logger.info(charge_part)
        logger.info('##############')
        for idx in self.common_core:
            odx = idx + offset
            psf[odx].charge += charge_part


class StericMutation(BaseMutation):

    def __init__(self, atom_idx: list, nr_of_steps: int):
        super().__init__(atom_idx, nr_of_steps)

    def _scale_epsilon(self, atom, multiplicator):
        atom.epsilon = atom.initial_epsilon * multiplicator

    def _scale_rmin(self, atom, multiplicator):
        atom.rmin = atom.initial_rmin * multiplicator

    def _modify_type(self, atom, psf):

        if (hasattr(atom, 'initial_type')):
            # only change parameters
            pass
        else:
            atom.initial_type = atom.type
            atom.type = f"DDD{psf.number_of_dummys}"
            psf.number_of_dummys += 1


class ChargeToZeroMutation(ChargeMutation):

    def __init__(self, atom_idx: list, nr_of_steps: int, common_core: list):
        """
        Scale the electrostatics of atoms specified in the atom_idx list to zero.
        Parameters
        ----------
        atom_idx : list
            atoms for which charges are scaled to zero
        mr_of_steps : int
            determines the scaling factor : multiplicator = 1 - (current_step / (nr_of_steps))
        common_core : list
        """
        super().__init__(atom_idx, nr_of_steps, common_core)

    def __str__(self):
        return "charges to zero mutation"

    def mutate(self, psf: pm.charmm.CharmmPsfFile, tlc: str, current_step: int):

        """ Performs the mutation """

        old_total_charge = round(sum([a.charge for a in psf[f":{tlc.upper()}"].atoms]))
        offset = min([a.idx for a in psf.view[f":{tlc.upper()}"].atoms])
        diff_charge = 0
        multiplicator = 1 - (current_step / (self.nr_of_steps))

        logger.info(f" -- Charge to zero mutation.")
        logger.info(f"Scaling factor: {multiplicator}")

        for idx in self.atom_idx:
            odx = idx + offset
            atom = psf[odx]
            logger.info(f"Scale charge on {atom}")
            logger.info(f"Old charge: {atom.initial_charge}")
            new_charge = float(np.round(atom.initial_charge * multiplicator, 4))
            logger.info(f"New charge: {new_charge}")
            diff_charge += self._scale_charge(atom, new_charge)

        if multiplicator != 1:
            # compensate for the total change in charge
            self._compensate_charge(psf, offset, diff_charge, tlc)
            new_charges = [a.charge for a in psf[f":{tlc.upper()}"].atoms]
            try:
                assert(np.isclose(sum(new_charges), old_total_charge))
            except AssertionError:
                new_charges = (np.array(new_charges) - np.average(np.array(new_charges)))
                for a, new_charge in zip(psf.view[f":{tlc.upper()}"].atoms, new_charges):
                    a.charge = new_charge
                new_charges = [a.charge for a in psf[f":{tlc.upper()}"].atoms]
                assert(np.isclose(sum(new_charges), old_total_charge))


class StericToZeroMutation(StericMutation):

    def __init__(self, atom_idx: list):
        """
        Set the steric terms of atoms specified in the atom_idx list to zero.
        Parameters
        ----------
        atom_list : list
            atoms for which charges are scaled to zero
        """
        super().__init__(atom_idx, 1)

    def __str__(self):
        return "Steric to zero mutation"

    def __unicode__(self):
        return u"Steric to zero mutation"

    def mutate(self, psf, tlc: str, current_step: int):

        """ Performs the actual mutation """

        logger.info(f" -- Steric interactions to zero mutation.")
        logger.info(f"Acting on atoms: {self.atom_idx}")
        offset = min([a.idx for a in psf.view[f":{tlc.upper()}"].atoms])

        for i in self.atom_idx:
            atom = psf[i + offset]
            multiplicator = 0
            self._modify_type(atom, psf)
            self._scale_epsilon(atom, multiplicator)
            self._scale_rmin(atom, multiplicator)


# TODO: RemoveImproperMutation()

        # # scale improper
        # for torsion in psf.impropers:
        #     # test if one of the torsions that needs to change
        #     if not hasattr(torsion, 'improper_present_in'):
        #         continue

        #     # torsion present at cc1 needs to be turned fully off starting from self.nr_of_steps/2
        #     if torsion.improper_present_in == 'cc1':
        #         # scale the initial torsions down
        #         torsion.type.psi_k = torsion.initial_psi_k * max(((1.0 - scale * 2)), 0.0)
        #     # torsion present at cc1 needs to be turned fully off starting from self.nr_of_steps/2
        #     elif torsion.improper_present_in == 'cc2':
        #         # scale the new torsinos up
        #         torsion.type.psi_k = torsion.initial_psi_k * max((scale -0.5) * 2, 0.0)
