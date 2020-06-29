import io
import logging
import os
from collections import namedtuple

import networkx as nx
import parmed as pm
import rdkit
from rdkit import Chem
from simtk import unit

from .utils import get_toppar_dir

logger = logging.getLogger(__name__)


class SystemStructure(object):

    def __init__(self, configuration:dict, structure:str):
        """
        A class that contains all informations for a single small ligand in different environments.
        Everything is constructed automatically from the charmm-gui folder.
        Parameters
        ----------
        configuration: dict
            the configuration dictionary obtained with utils.load_config_yaml
        structure: str
            either 'structure1' or 'structure2'
        """


        self.structure:str = structure
        self.name:str = configuration['system'][structure]['name']
        self.tlc:str = configuration['system'][structure]['tlc']
        self.charmm_gui_base:str = configuration['system'][structure]['charmm_gui_dir']
        self.psf_mapping:dict = {}
        self.vacuum_psf = None
        self.waterbox_psf = None
        self.complex_psf = None
        
        # running a binding-free energy calculation? 
        if configuration['simulation']['free-energy-type'] == 'binding-free-energy':
            self.envs:set = set(['complex', 'waterbox'])
            self.parameter:pm.charmm.CharmmParameterSet = self._read_parameters(configuration, 'complex')

            # set up complex objects
            self.complex_psf:pm.charmm.CharmmPsfFile = self._initialize_system(configuration, 'complex')
            # load parameters
            self.complex_psf.load_parameters(self.parameter)
            # get offset
            self.complex_offset:int = self._determine_offset_and_set_possible_dummy_properties(self.complex_psf)
            
            # set up waterbox objects
            self.waterbox_psf:pm.charmm.CharmmPsfFile = self._initialize_system(configuration, 'waterbox')
            # load parameters
            self.waterbox_psf.load_parameters(self.parameter)
            # get offset
            self.waterbox_offset:int = self._determine_offset_and_set_possible_dummy_properties(self.waterbox_psf)

            # generate rdkit mol object of small molecule
            self.mol:Chem.Mol = self._generate_rdkit_mol('complex', self.complex_psf[f":{self.tlc}"])
            self.graph:nx.Graph = self._mol_to_nx(self.mol)

        elif configuration['simulation']['free-energy-type'] == 'solvation-free-energy':
            self.envs:set = set(['waterbox', 'vacuum'])
            self.parameter:pm.charmm.CharmmParameterSet = self._read_parameters(configuration, 'vacuum')
            # set up complex objects
            self.vacuum_psf:pm.charmm.CharmmPsfFile = self._initialize_system(configuration, 'vacuum')
            # load parameters
            self.vacuum_psf.load_parameters(self.parameter)
            # get offset
            self.vacuum_offset:int = self._determine_offset_and_set_possible_dummy_properties(self.vacuum_psf)

            # set up waterbox objects
            self.waterbox_psf:pm.charmm.CharmmPsfFile = self._initialize_system(configuration, 'waterbox')
            # load parameters
            self.waterbox_psf.load_parameters(self.parameter)
            # get offset
            self.waterbox_offset:int = self._determine_offset_and_set_possible_dummy_properties(self.waterbox_psf)

            # generate rdkit mol object of small molecule
            self.mol:Chem.Mol = self._generate_rdkit_mol('waterbox', self.waterbox_psf[f":{self.tlc}"])
            self.graph:nx.Graph = self._mol_to_nx(self.mol)
        else:
            raise NotImplementedError('only binding and solvation free energy implemented.')

        self.psf_mapping = {'complex' : self.complex_psf,
                            'waterbox': self.waterbox_psf,
                            'vacuum'  : self.vacuum_psf} 

    def _mol_to_nx(self, mol:Chem.Mol):
        G = nx.Graph()

        for atom in mol.GetAtoms():
            G.add_node(atom.GetIdx(),
                    atomic_num=atom.GetAtomicNum(),
                    formal_charge=atom.GetFormalCharge(),
                    chiral_tag=atom.GetChiralTag(),
                    hybridization=atom.GetHybridization(),
                    num_explicit_hs=atom.GetNumExplicitHs(),
                    is_aromatic=atom.GetIsAromatic())
        
        for bond in mol.GetBonds():
            G.add_edge(bond.GetBeginAtomIdx(),
                    bond.GetEndAtomIdx(),
                    bond_type=bond.GetBondType())
        return G



    def _read_parameters(self, configuration:dict, env:str)->pm.charmm.CharmmParameterSet:
        """
        Reads in topparameters from a toppar dir and ligand specific parameters.
        Parameters
        ----------
        configuration: dict
            the configuration dictionary obtained with utils.load_config_yaml
        env: str
            waterbox,complex or vacuum
        Returns
        ----------
        parameter : pm.charmm.CharmmParameterSet
            parameters obtained from the CHARMM-GUI output dir.      
        """

        # the parameters for the vacuum system is parsed from the waterbox charmm-gui directory
        if env == 'vacuum':
            env = 'waterbox'

        charmm_gui_env = self.charmm_gui_base + env
        tlc = self.tlc       
        tlc_lower = str(tlc).lower()
        parameter_files = f"{charmm_gui_env}/{tlc_lower}/{tlc_lower}.rtf", f"{charmm_gui_env}/{tlc_lower}/{tlc_lower}.prm"         
        toppar_dir = get_toppar_dir()
        parameter_files =  parameter_files + (f"{toppar_dir}/top_all36_prot.rtf",)
        parameter_files =  parameter_files + (f"{toppar_dir}/par_all36m_prot.prm",)
        parameter_files =  parameter_files + (f"{toppar_dir}/par_all36_na.prm",)
        parameter_files =  parameter_files + (f"{toppar_dir}/top_all36_na.rtf",)
        parameter_files =  parameter_files + (f"{toppar_dir}/top_all36_cgenff.rtf",) 
        parameter_files =  parameter_files + (f"{toppar_dir}/par_all36_cgenff.prm",)
        parameter_files =  parameter_files + (f"{toppar_dir}/par_all36_lipid.prm",)
        #parameter_files =  parameter_files + (f"{toppar_dir}/par_all36_na.prm",) //Double
        parameter_files =  parameter_files + (f"{toppar_dir}/top_all36_lipid.rtf",)
        parameter_files =  parameter_files + (f"{toppar_dir}/toppar_water_ions.str",)
            
        # set up parameter objec
        parameter = pm.charmm.CharmmParameterSet(*parameter_files)
        return parameter

    def _initialize_system(self, configuration:dict, env:str)->pm.charmm.CharmmPsfFile:
        """
        Generates the psf file and sets the coordinates from the CHARMM-GUI files.
        Parameters
        ----------
        configuration: dict
            the configuration dictionary obtained with utils.load_config_yaml
        env: str
            waterbox,complex or vacuum
        Returns
        ----------
        psf : pm.charmm.CharmmPsfFile
        """
        
        if env == 'vacuum':
            # take the structures from the waterbox system and extract only the ligand
            taken_from = 'waterbox'           
            psf_file_name = configuration['system'][self.structure][taken_from]['psf_file_name']
            crd_file_name = configuration['system'][self.structure][taken_from]['crd_file_name']

            psf_file_path = f"{self.charmm_gui_base}/{taken_from}/openmm/{psf_file_name}.psf"
            crd_file_path = f"{self.charmm_gui_base}/{taken_from}/openmm/{crd_file_name}.crd"
            psf = pm.charmm.CharmmPsfFile(psf_file_path)
            coord = pm.charmm.CharmmCrdFile(crd_file_path)
            psf.coordinates = coord.coordinates
            # extract only ligand to generate vacuum system
            psf = psf[f":{self.tlc}"]
        else:
            psf_file_name = configuration['system'][self.structure][env]['psf_file_name']
            crd_file_name = configuration['system'][self.structure][env]['crd_file_name']

            psf_file_path = f"{self.charmm_gui_base}/{env}/openmm/{psf_file_name}.psf"
            crd_file_path = f"{self.charmm_gui_base}/{env}/openmm/{crd_file_name}.crd"
            psf = pm.charmm.CharmmPsfFile(psf_file_path)
            coord = pm.charmm.CharmmCrdFile(crd_file_path)
            psf.coordinates = coord.coordinates

        return psf


    def _determine_offset_and_set_possible_dummy_properties(self, psf:pm.charmm.CharmmPsfFile)->int:
        """
        Determines the offset and sets possible properties on the psf.
        Parameters
        ----------
        psf : pm.charmm.CharmmPsfFile
        env: str
            waterbox,complex or vacuum
        Returns
        ----------
        """
        assert(type(psf) == pm.charmm.CharmmPsfFile)
        if len(psf.view[f":{self.tlc}"].atoms) < 1:
            raise RuntimeError(f"No ligand selected for tlc: {self.tlc}")

        psf.number_of_dummys = 0

        idx_list = []
        for atom in psf.view[f":{self.tlc}"].atoms:
            idx_list.append(int(atom.idx))
            
            # charge, epsilon and rmin are directly modiefied
            atom.initial_charge = atom.charge
            atom.initial_epsilon = atom.epsilon
            atom.initial_rmin = atom.rmin

        return min(idx_list)
    
    def _generate_rdkit_mol(self, env:str, psf:pm.charmm.CharmmPsfFile)->Chem.Mol:
        """
        Generates the rdkit mol object.
        Parameters
        ----------
        env: str
            waterbox,complex or vacuum
        psf: pm.charmm.CharmmPsfFile
        Returns
        ----------
        mol: rdkit.Chem.mol
        """
        from itertools import product
        
        assert(type(psf) == pm.charmm.CharmmPsfFile)
        charmm_gui_env = self.charmm_gui_base + env
        tlc = self.tlc

        filenames = [str(tlc), str(tlc).lower(), str(tlc).upper()]
        dir_names = [str(tlc), str(tlc).lower(), str(tlc).upper()]

        for name in filenames:
            for dir_name in dir_names:
                try:
                    file = f"{charmm_gui_env}/{dir_name}/{name}.sdf"  # NOTE: maybe also tlc and tlc_lower for dir?      
                    suppl = Chem.SDMolSupplier(file, removeHs=False)
                    mol = next(suppl)
                    break
                except IOError:
                    logger.info(f"SDF file not found: {file}")
                    pass

                # try:
                #     file = f"{charmm_gui_env}/{dir_name}/{name}.mol2"  # NOTE: maybe also tlc and tlc_lower for dir?      
                #     mol = Chem.MolFromMol2File(file, removeHs=False)
                #     break
                # except IOError:
                #     logger.info(f"MOL file not found: {file}")
                #     pass


        atom_idx_to_atom_name, _, atom_name_to_atom_type, atom_idx_to_atom_partial_charge = self.generate_atom_tables_from_psf(psf)

        for atom in mol.GetAtoms():
            atom.SetProp('atom_name', atom_idx_to_atom_name[atom.GetIdx()])
            atom.SetProp('atom_type', atom_name_to_atom_type[atom_idx_to_atom_name[atom.GetIdx()]])
            atom.SetProp('atom_index', str(atom.GetIdx()))
            atom.SetProp('atom_charge', str(atom_idx_to_atom_partial_charge[atom.GetIdx()]))
            
        # check if psf and sdf have same indeces
        for a in mol.GetAtoms():
            if str(psf[a.GetIdx()].element_name) == str(a.GetSymbol()):
                pass
            else:
                raise RuntimeError('PSF to mol conversion did not work! Aborting.')

        return mol


    def generate_atom_tables_from_psf(self, psf:pm.charmm.CharmmPsfFile)->(dict,dict,dict,dict):
        """
        Generate mapping dictionaries for a molecule in a psf. 
        Parameters
        ----------
        psf: pm.charmm.CharmmPsfFile
        Returns
        ----------
        dict's
        """

        atom_idx_to_atom_name = dict()
        atom_name_to_atom_idx = dict()
        atom_name_to_atom_type = dict()
        atom_idx_to_atom_partial_charge = dict()

        for atom in psf.view[f":{self.tlc}"].atoms:
            atom_name = atom.name
            atom_index = atom.idx
            atom_type = atom.type
            atom_charge = atom.charge

            atom_idx_to_atom_name[atom_index] = atom_name
            atom_name_to_atom_idx[atom_name] = atom_index
            atom_name_to_atom_type[atom_name] = atom_type
            atom_idx_to_atom_partial_charge[atom_index] = atom_charge

        return (atom_idx_to_atom_name, atom_name_to_atom_idx, atom_name_to_atom_type, atom_idx_to_atom_partial_charge)
