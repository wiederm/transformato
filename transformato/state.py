import logging
import os
import shutil

import parmed as pm
import transformato

from .utils import get_toppar_dir

logger = logging.getLogger(__name__)


class IntermediateStateFactory(object):

    def __init__(self, system:transformato.system, mutation_list:list, configuration:dict):
        """
        Generate the intermediate directories with for the provided systems with the provided mutations.
        Parameters
        ----------
        system : transformato.system
            definition of the two states for a given system (either waterbox and vacuum or waterbox and complex)
        mutation_list : list
            list of mutations defined by the transformato.ProposeMutationRoute object
        configuration : dict
            configuration dictionary
        """

        self.system = system
        self.mutation_list = mutation_list
        self.path = f"{configuration['system_dir']}/{self.system.name}"
        self._init_base_dir()
        self.configuration = configuration

    def generate_specific_intermediate_state(self, mutation, state:int):

        output_file_base = self._init_intermediate_state_dir(state)
        logger.info('Writing to {}'.format(output_file_base))
        logger.info('#########################################')
        for env in self.system.envs:
            if env == 'vacuum':
                psf = self.system.vacuum_psf
            elif env == 'waterbox':
                psf = self.system.waterbox_psf
            elif env == 'complex':
                psf = self.system.complex_psf
            else:
                raise RuntimeError(f"Unknown system env :{env}")
            mutation.mutate(psf, self.system.tlc, state)
            self._write_psf(psf, output_file_base, env)

        self._write_rtf_file(psf, output_file_base, self.system.tlc)
        self._write_prm_file(psf, output_file_base, self.system.tlc)
        self._write_toppar_str(output_file_base, self.system.tlc)
        self._copy_files(output_file_base)
        return output_file_base

    
    def generate_intermediate_states(self, strategy='seperate'):
        """
        Generate the intermediate states as defined the the mutation list.
        """

        intst_nr = 1
        if strategy == 'seperate':
            # no mixing of the different mutation states - first electrostatics is turend off,
            # then VdW and the the bonded terms are transformed
            nr_of_total_mutations = 1 # include the endstate at 0
            start_step = 1 
            for m in self.mutation_list:
                for current_step in range(start_step, m.nr_of_steps+1):
                    nr_of_total_mutations += 1

            logger.info('#########################################')
            logger.info('#########################################')
            logger.info(f"Preparing for a total of {nr_of_total_mutations} mutation steps")
            logger.info(f"Writing endstate")
            self._write_state(None, current_step=0, intst_nr=intst_nr, mutate=False)            
            intst_nr += 1
            for m in self.mutation_list:
                start_step = 1 # don't write out the first, unmodified state
                for current_step in range(start_step, m.nr_of_steps+1):
                    self._write_state(m, current_step, intst_nr, mutate=True)
                    intst_nr += 1



    def _write_state(self, mutation, current_step:int, intst_nr:int, mutate:bool=True):
        
        logger.info('#########################################')
        logger.info('#########################################')
        logger.info('Current step: {}'.format(current_step))
        output_file_base = self._init_intermediate_state_dir(intst_nr)
        for env in self.system.envs:
            if mutate:
                mutation.mutate(self.system.psf_mapping[env], self.system.tlc, current_step)
            self._write_psf(self.system.psf_mapping[env], output_file_base, env)
        self._write_rtf_file(self.system.psf_mapping[env], output_file_base, self.system.tlc)
        self._write_prm_file(self.system.psf_mapping[env], output_file_base, self.system.tlc)
        self._write_toppar_str(output_file_base, self.system.tlc)
        self._copy_files(output_file_base)



    def _add_serializer(self, file):
                # adding serializer functions
        f = open(file, 'a')
        f.write(
'''
# mw: adding xml serializer to the simulation script
file_name = str(args.psffile).replace('.psf', '')
print(file_name)
serialized_integrator = XmlSerializer.serialize(integrator)
outfile = open(file_name + '_integrator.xml','w')
outfile.write(serialized_integrator)
outfile.close()
serialized_system = XmlSerializer.serialize(system)
outfile = open(file_name + '_system.xml','w')
outfile.write(serialized_system)
outfile.close()
'''
        )
        f.close()

    
    def _copy_files_for_binding_free_energy_calculations(self, basedir, intermediate_state_file_path):

        # parse omm simulation paramter
        for env in self.system.envs:
            omm_simulation_parameter_source = f"{basedir}/{env}/openmm/{self.configuration['system'][self.system.structure][env]['simulation_parameter']}" 
            omm_simulation_parameter_target = f"{intermediate_state_file_path}/{self.configuration['system'][self.system.structure][env]['intermediate-filename']}"
            self._overwrite_simulation_script_parameters(omm_simulation_parameter_source, omm_simulation_parameter_target)

        omm_simulation_submit_script_source = f"{self.configuration['bin_dir']}/simulation-binding-free-energy.sh"
        omm_simulation_submit_script_target = f"{intermediate_state_file_path}/simulation.sh"
        shutil.copyfile(omm_simulation_submit_script_source, omm_simulation_submit_script_target)



    def _copy_files_for_solvation_free_energy_calculations(self, basedir, intermediate_state_file_path):

        # parse omm simulation paramter
        for env in self.system.envs:
            if env == 'waterbox':
                omm_simulation_parameter_source = f"{basedir}/{env}/openmm/{self.configuration['system'][self.system.structure][env]['simulation_parameter']}" 
                omm_simulation_parameter_target = f"{intermediate_state_file_path}/{self.configuration['system'][self.system.structure][env]['intermediate-filename']}"
                self._overwrite_simulation_script_parameters(omm_simulation_parameter_source, omm_simulation_parameter_target)
            else: # vacuum
                used_env = 'waterbox'
                omm_simulation_parameter_source = f"{basedir}/{used_env}/openmm/{self.configuration['system'][self.system.structure][used_env]['simulation_parameter']}" 
                omm_simulation_parameter_target = f"{intermediate_state_file_path}/{self.configuration['system'][self.system.structure][env]['intermediate-filename']}"
                self._overwrite_simulation_script_parameters(omm_simulation_parameter_source, omm_simulation_parameter_target)
        
        omm_simulation_submit_script_source = f"{self.configuration['bin_dir']}/simulation-solvation-free-energy.sh"
        omm_simulation_submit_script_target = f"{intermediate_state_file_path}/simulation.sh"
        shutil.copyfile(omm_simulation_submit_script_source, omm_simulation_submit_script_target)

        omm_simulation_script_source = f"{self.configuration['bin_dir']}/openmm_run_vacuum.py"
        omm_simulation_script_target = f"{intermediate_state_file_path}/openmm_run_vacuum.py"
        shutil.copyfile(omm_simulation_script_source, omm_simulation_script_target)
        self._add_serializer(omm_simulation_script_target)

    def _get_simulations_parameters(self):
        prms = {}
        for key in self.configuration['simulation']['parameters']:
            prms[key] = self.configuration['simulation']['parameters'][key]
        return prms

    def _copy_files(self, intermediate_state_file_path):
        """
        Copy the files from the original CHARMM-GUI output folder in the intermediate directories.
        """
        
        basedir = self.system.charmm_gui_base
        
        if self.configuration['simulation']['free-energy-type'] == 'solvation-free-energy':
            self._copy_files_for_solvation_free_energy_calculations(basedir, intermediate_state_file_path)
        elif self.configuration['simulation']['free-energy-type'] == 'binding-free-energy':
            self._copy_files_for_binding_free_energy_calculations(basedir, intermediate_state_file_path)
        else:
            raise RuntimeError(f"Only solvation/binding free energies implemented")

        # copy rst files
        for env in self.system.envs:
            rst_file_source = f"{basedir}/{env}/openmm/{self.configuration['system'][self.system.structure][env]['rst_file_name']}.rst"
            rst_file_target = f"{intermediate_state_file_path}/lig_in_{env}.rst"
            try:
                shutil.copyfile(rst_file_source , rst_file_target)
            except FileNotFoundError:
                logger.warning(f"No restart file found for {env} -- starting simulation from crd file.")

        # copy crd files
        for env in self.system.envs:
            crd_file_source = f"{basedir}/{env}/openmm/{self.configuration['system'][self.system.structure][env]['crd_file_name']}.crd"
            crd_file_target = f"{intermediate_state_file_path}/lig_in_{env}.crd"
            try:
                shutil.copyfile(crd_file_source , crd_file_target)
            except FileNotFoundError:
                logger.warning(f"No crd file found for {env} -- using parmed system structure to create crd file.")
                crd_file_target = f"{intermediate_state_file_path}/lig_in_{env}.crd"
                pm.charmm.CharmmCrdFile.write(self.system.psf_mapping[env], crd_file_target)

        # copy ligand rtf file
        ligand_rtf = f"{basedir}/waterbox/{self.system.tlc.lower()}/{self.system.tlc.lower()}_g.rtf"
        toppar_target = f"{intermediate_state_file_path}/{self.system.tlc.lower()}_g.rtf" 
        shutil.copyfile(ligand_rtf, toppar_target)

        # copy ligand prm file
        ligand_prm = f"{basedir}/waterbox/{self.system.tlc.lower()}/{self.system.tlc.lower()}.prm"
        toppar_target = f"{intermediate_state_file_path}/{self.system.tlc.lower()}.prm" 
        shutil.copyfile(ligand_prm, toppar_target)

        # copy diverse set of helper functions
        FILES = ['omm_barostat.py', 'omm_readinputs.py', 'omm_readparams.py', 'omm_restraints.py', 'omm_rewrap.py', 'omm_vfswitch.py', 'omm_hmr.py']
        for f in FILES:
            try:
                omm_source = f"{basedir}/waterbox/openmm/{f}"
                omm_target = f"{intermediate_state_file_path}/{f}"
                shutil.copyfile(omm_source, omm_target)
            except OSError:
                logger.debug(f'Could not find file: {f}')

        # copy toppar folder
        toppar_dir = get_toppar_dir()
        toppar_source = f"{toppar_dir}"
        toppar_target = f"{intermediate_state_file_path}/toppar" 
        shutil.copytree(toppar_source, toppar_target)

        # copy omm simulation script
        omm_simulation_script_source = f"{basedir}/waterbox/openmm/openmm_run.py"
        omm_simulation_script_target = f"{intermediate_state_file_path}/openmm_run.py"      
        shutil.copyfile(omm_simulation_script_source, omm_simulation_script_target)  

        # add serialization
        self._add_serializer(omm_simulation_script_target)


    def _overwrite_simulation_script_parameters(self, omm_simulation_parameter_source:str, omm_simulation_parameter_target:str):

        overwrite_parameters = self._get_simulations_parameters()

        input_simulation_parameter = open(omm_simulation_parameter_source, 'r')
        output_simulation_parameter = open(omm_simulation_parameter_target + '.inp', 'w+')
    
        for l in input_simulation_parameter.readlines():
            if l.strip():
                t1, t2 = l.split('=')
                t1 = t1.strip()
                t2, comment = t2.split('#')
                t2 = t2.strip()
                comment = comment.strip()
                if t1 in overwrite_parameters.keys():
                    t2 = overwrite_parameters[t1]
                output_simulation_parameter.write(f"{t1:<25} = {t2:<25} # {comment:<30}\n")
            else:
                output_simulation_parameter.write('\n')
        input_simulation_parameter.close()
        output_simulation_parameter.close()


    
    def _write_rtf_file(self, psf, output_file_base, tlc): # NOTE: this needs some refactoring!
        """
        Generates the dummy atom parameter rtf.
        """

        header_rtf = '''* Dummy atom parameters 
* generated by transformato
*
36  1
'''
        rtf_file_handler = open(output_file_base +'/dummy_atom_definitions.rtf', 'w')
        rtf_file_handler.write(header_rtf)
        for atom in psf.view[f":{tlc}"].atoms:            
            if hasattr(atom, 'initial_type'):
                logger.debug('- Setting dummy parameters ...')
                logger.debug(f"  + Atom-Name: {atom.name}")
                logger.debug(f"  + Atom-Type: {atom.initial_type}")
                logger.debug(f"  + Atom Dummy Type: {atom.type}")

                rtf_file_handler.write('{:7} {:6} {:6} {:6}\n'.format('MASS', '-1', atom.type, atom.mass))
            
        rtf_file_handler.close()    


    def _write_prm_file(self, psf, output_file_base, tlc):
        """Generates the prm file

        Args:
            psf ([type]): [description]
            output_file_base ([type]): [description]
            tlc ([type]): [description]
        """
    
        header_prm = '''* Parameters generated by analogy by
* CHARMM General Force Field (CGenFF) program version 1.0.0
*
! Automatically obtained dummy parameters 
! from transformato
'''

        prm_file_handler = open(f"{output_file_base}/dummy_parameters.prm", 'w')
        prm_file_handler.write(header_prm)
        prm_file_handler.write('\nATOMS\n')

        view = psf.view[f":{tlc}"]
        # writing atom parameters
        for atom in view.atoms:
            if hasattr(atom, 'initial_type'):
                logger.debug('- Setting dummy parameters ...')
                logger.debug(f"  + Atom-Name: {atom.name}")
                logger.debug(f"  + Atom-Type: {atom.initial_type}")
                logger.debug(f"  + Atom Dummy Type: {atom.type}")
                prm_file_handler.write('{:7} {:6} {:6} {:9.5f}\n'.format('MASS', '-1', atom.type, atom.mass))
            
        prm_file_handler.write('\n\n')

        ##############################################################################
        # write bond parameters - again there are two ways to use this:
        # - keeping bonded terms between real/dummy and dummy atoms intact
        # - changing bonded parameters between real atoms - this again needs dummy atoms

        prm_file_handler.write('BONDS\n')
        for bond in view.bonds:
            atom1, atom2 = bond.atom1, bond.atom2
            if any(hasattr(atom, 'initial_type') for atom in [atom1, atom2]):
                logger.debug(' >> Setting dummy bond parameters for: {} - {}'.format(str(atom1.type),str(atom2.type)))
                try:
                    logger.debug('{:7} {:7} {:9.5f} {:9.5f} \n'.format(str(atom1.type), str(atom2.type), bond.mod_type.k ,bond.mod_type.req))
                    prm_file_handler.write('{:7} {:7} {:9.5f} {:9.5f} \n'.format(str(atom1.type), str(atom2.type), bond.mod_type.k ,bond.mod_type.req))
                except AttributeError:
                    logger.debug('{:7} {:7} {:9.5f} {:9.5f} \n'.format(str(atom1.type), str(atom2.type), bond.type.k ,bond.type.req))
                    prm_file_handler.write('{:7} {:7} {:9.5f} {:9.5f} \n'.format(str(atom1.type), str(atom2.type), bond.type.k ,bond.type.req))

        #################################################################
        prm_file_handler.write('\n\n')
        prm_file_handler.write('ANGLES\n')
        for angle in view.angles:
            atom1, atom2, atom3 = angle.atom1, angle.atom2, angle.atom3
            if any(hasattr(atom, 'initial_type') for atom in [atom1, atom2, atom3]):            
                logger.debug(' >> Setting dummy angle parameters for: {}-{}-{}'.format(str(atom1.type),str(atom2.type),str(atom3.type)))
                try:
                    prm_file_handler.write('{:7} {:7} {:7} {:9.5f} {:9.5f} \n'.format(str(atom1.type), str(atom2.type), str(atom3.type), angle.mod_type.k , angle.mod_type.theteq))
                    logger.debug('{:7} {:7} {:7} {:9.5f} {:9.5f} \n'.format(str(atom1.type), str(atom2.type), str(atom3.type), angle.mod_type.k , angle.mod_type.theteq))
                except AttributeError:
                    prm_file_handler.write('{:7} {:7} {:7} {:9.5f} {:9.5f} \n'.format(str(atom1.type), str(atom2.type), str(atom3.type), angle.type.k , angle.type.theteq))
                    logger.debug('{:7} {:7} {:7} {:9.5f} {:9.5f} \n'.format(str(atom1.type), str(atom2.type), str(atom3.type), angle.type.k , angle.type.theteq))
                    


        #################################################################
        prm_file_handler.write('\n\n')
        prm_file_handler.write('DIHEDRALS\n')
        for dihedral in view.dihedrals:
            atom1, atom2, atom3, atom4 = dihedral.atom1, dihedral.atom2, dihedral.atom3, dihedral.atom4
            if any(hasattr(atom, 'initial_type') for atom in [atom1, atom2, atom3, atom4]):            
                logger.debug(' >> Setting dummy dihedral parameters for: {}-{}-{}-{}'.format(str(atom1.type),str(atom2.type),str(atom3.type),str(atom4.type)))
                try:
                    for dihedral_type in dihedral.mod_type:
                        prm_file_handler.write('{:7} {:7} {:7} {:7} {:6.5f} {:9.5f} {:9.5f} \n'.format(str(atom1.type), str(atom2.type), str(atom3.type), str(atom4.type), dihedral_type.phi_k ,dihedral_type.per, dihedral_type.phase))
                except AttributeError:
                    for dihedral_type in dihedral.type:
                        prm_file_handler.write('{:7} {:7} {:7} {:7} {:6.5f} {:9.5f} {:9.5f} \n'.format(str(atom1.type), str(atom2.type), str(atom3.type), str(atom4.type), dihedral_type.phi_k ,dihedral_type.per, dihedral_type.phase))
                    
        #################################################################
        # get all unique improper and parameters
        prm_file_handler.write('\n\n')
        prm_file_handler.write('IMPROPERS\n')
        for impr in view.impropers:
            atom1, atom2, atom3, atom4 = impr.atom1, impr.atom2, impr.atom3, impr.atom4
            if any(hasattr(atom, 'initial_type') for atom in [atom1, atom2, atom3, atom4]):            
                #print('>> Setting dummy improper parameters for: {}-{}-{}-{}'.format(str(atom1.type),str(atom2.type),str(atom3.type),str(atom4.type)))
                # carefull with this solution - > central atom has to be set in the beginning
                prm_file_handler.write('{:7} {:7} {:7} {:7} {:9.5f} {:9.5f} \n'.format(str(atom1.type), str(atom2.type), str(atom3.type), str(atom4.type), impr.type.psi_k , impr.type.psi_eq))

        #################################################################
        prm_file_handler.write('\n\n')
        prm_file_handler.write('''NONBONDED nbxmod  5 atom cdiel fshift vatom vdistance vfswitch -
cutnb 14.0 ctofnb 12.0 ctonnb 10.0 eps 1.0 e14fac 1.0 wmin 1.5''')
        prm_file_handler.write('\n\n')

        for atom in view.atoms:
            if hasattr(atom, 'initial_type'):
                try:
                    prm_file_handler.write('{:7} {:6} {:9.5f} {:9.5f}\n'.format(atom.type, 0.0, 
                                            atom.mod_type.epsilon, 
                                            atom.mod_type.rmin))
                except AttributeError:
                    prm_file_handler.write('{:7} {:6} {:9.5f} {:9.5f}\n'.format(atom.type, 0.0, 
                                            atom.epsilon, 
                                            atom.rmin))

        prm_file_handler.write('\n')
        prm_file_handler.write('END')
        prm_file_handler.close()



    def _init_base_dir(self):
        """
        Generates the base directory which all intermediate states are located.
        """
       
        if os.path.isdir(self.path):
            shutil.rmtree(self.path)
            os.makedirs(self.path)
        else:
            os.makedirs(self.path)
    
    def _write_toppar_str(self, output_file_base, tlc):

        toppar_format = """
toppar/top_all36_prot.rtf
toppar/par_all36m_prot.prm
toppar/top_all36_na.rtf
toppar/par_all36_na.prm
toppar/top_all36_carb.rtf
toppar/par_all36_carb.prm
toppar/top_all36_lipid.rtf
toppar/par_all36_lipid.prm
toppar/top_all36_cgenff.rtf
toppar/par_all36_cgenff.prm
toppar/toppar_water_ions.str
toppar/toppar_dum_noble_gases.str
toppar/toppar_all36_prot_d_aminoacids.str
toppar/toppar_all36_prot_fluoro_alkanes.str
toppar/toppar_all36_prot_heme.str
toppar/toppar_all36_prot_na_combined.str
toppar/toppar_all36_prot_retinol.str
toppar/toppar_all36_na_nad_ppi.str
toppar/toppar_all36_lipid_bacterial.str
toppar/toppar_all36_lipid_cardiolipin.str
toppar/toppar_all36_lipid_cholesterol.str
toppar/toppar_all36_lipid_inositol.str
toppar/toppar_all36_lipid_lps.str
toppar/toppar_all36_lipid_miscellaneous.str
toppar/toppar_all36_lipid_model.str
toppar/toppar_all36_lipid_prot.str
toppar/toppar_all36_lipid_pyrophosphate.str
toppar/toppar_all36_lipid_sphingo.str
{}_g.rtf
{}.prm
dummy_atom_definitions.rtf
dummy_parameters.prm
""".format(tlc.lower(), tlc.lower())
        
        f = open(f"{output_file_base}/toppar.str", 'w+')
        f.write(toppar_format)
        f.close()


    def _write_psf(self, psf, output_file_base:str, env:str):
        """
        Writes the new psf.
        """
           
        psf.write_psf(f"{output_file_base}/lig_in_{env}.psf")
        psf.write_pdb(f"{output_file_base}/lig_in_{env}.pdb")
                    


    def _init_intermediate_state_dir(self, nr:int):
        """
        Generates the intermediate state directory.
        """
        output_file_base = f"{self.path}/intst{nr}/" 

        logger.info(f" - Created directory: - {os.path.abspath(output_file_base)}")
        os.makedirs(output_file_base)
        logger.info(f" - Writing in - {os.path.abspath(output_file_base)}")
        return output_file_base
