import transformato
import os
import shutil
from .utils import get_toppar_dir
import logging

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
        self.path = f"{configuration['analysis_dir_base']}/{self.system.name}"
        self._init_base_dir()
        self.configuration = configuration

    def generate_specific_intermediate_state(self, mutation, state:int):

        output_file_base = self._init_intermediate_state_dir(state)
        logger.info('Writing to {}'.format(output_file_base))
        logger.info('#########################################')
        for psf, env in zip([self.system.complex_psf, self.system.waterbox_psf], ['complex', 'waterbox']):
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
            start_idx = 0 # for the first iteration it is zero to get one endstate
            for m in self.mutation_list:
                for current_step in range(start_idx, m.nr_of_steps):
                    logger.info('Current step: {}'.format(current_step))
                    output_file_base = self._init_intermediate_state_dir(intst_nr)
                    logger.info('#########################################')
                    logger.info('#########################################')
                    for psf, env in zip([self.system.complex_psf, self.system.waterbox_psf], ['complex', 'waterbox']):
                        m.mutate(psf, self.system.tlc, current_step)
                        self._write_psf(psf, output_file_base, env)
                    self._write_rtf_file(psf, output_file_base, self.system.tlc)
                    self._write_prm_file(psf, output_file_base, self.system.tlc)
                    self._write_toppar_str(output_file_base, self.system.tlc)
                    self._copy_files(output_file_base)
                    intst_nr += 1
                
    def _copy_files(self, intermediate_state_file_path):
        """
        Copy the files from the original CHARMM-GUI output folder in the intermediate directories.
        """
        # copy crd files
        basedir = self.system.charmm_gui_base
        for env in ['waterbox', 'complex']:
            crd_file_source = f"{basedir}/{env}/openmm/{self.configuration['system'][self.system.structure][env]['crd_file_name']}.crd"
            crd_file_target = f"{intermediate_state_file_path}/lig_in_{env}.crd"
            shutil.copyfile(crd_file_source , crd_file_target)


        # copy rst files
        for env in ['waterbox', 'complex']:
            rst_file_source = f"{basedir}/{env}/openmm/{self.configuration['system'][self.system.structure][env]['rst_file_name']}.rst"
            rst_file_target = f"{intermediate_state_file_path}/lig_in_{env}.rst"
            shutil.copyfile(rst_file_source , rst_file_target)


        # copy ligand rtf file
        ligand_rtf = f"{basedir}/complex/{self.system.tlc.lower()}/{self.system.tlc.lower()}_g.rtf"
        toppar_target = f"{intermediate_state_file_path}/{self.system.tlc.lower()}_g.rtf" 
        shutil.copyfile(ligand_rtf, toppar_target)

        # copy ligand prm file
        ligand_prm = f"{basedir}/complex/{self.system.tlc.lower()}/{self.system.tlc.lower()}.prm"
        toppar_target = f"{intermediate_state_file_path}/{self.system.tlc.lower()}.prm" 
        shutil.copyfile(ligand_prm, toppar_target)



        # copy diverse set of helper functions
        omm_barostat_source = f"{basedir}/complex/openmm/omm_barostat.py"
        omm_barostat_target = f"{intermediate_state_file_path}/omm_barostat.py"
        shutil.copyfile(omm_barostat_source, omm_barostat_target)

        omm_readinputs_source = f"{basedir}/complex/openmm//omm_readinputs.py"
        omm_readinputs_target = f"{intermediate_state_file_path}/omm_readinputs.py"
        shutil.copyfile(omm_readinputs_source, omm_readinputs_target)

        omm_readparams_source = f"{basedir}/complex/openmm/omm_readparams.py"
        omm_readparams_target = f"{intermediate_state_file_path}/omm_readparams.py"
        shutil.copyfile(omm_readparams_source, omm_readparams_target)

        omm_restraints_source = f"{basedir}/complex/openmm/omm_restraints.py"
        omm_restraints_target = f"{intermediate_state_file_path}/omm_restraints.py"
        shutil.copyfile(omm_restraints_source, omm_restraints_target)

        omm_rewrap_source = f"{basedir}/complex/openmm/omm_rewrap.py"
        omm_rewrap_target = f"{intermediate_state_file_path}/omm_rewrap.py"
        shutil.copyfile(omm_rewrap_source, omm_rewrap_target)

        omm_vfswitch_source = f"{basedir}/complex/openmm/omm_vfswitch.py"
        omm_vfswitch_target = f"{intermediate_state_file_path}/omm_vfswitch.py"
        shutil.copyfile(omm_vfswitch_source, omm_vfswitch_target)


        # parse omm simulation paramter
        for env in ['waterbox', 'complex']:
            omm_simulation_parameter_source = f"{basedir}/{env}/openmm/{self.configuration['system'][self.system.structure][env]['simulation_parameter']}" 
            omm_simulation_parameter_target = f"{intermediate_state_file_path}/{self.configuration['system'][self.system.structure][env]['intermediate-filename']}"
            input_simulation_parameter = open(omm_simulation_parameter_source, 'r')
            output_simulation_parameter = open(omm_simulation_parameter_target + '.inp', 'w+')
        
            for l in input_simulation_parameter.readlines():
                if l.strip():
                    t1, t2 = l.split('=')
                    t1 = t1.strip()
                    t2, comment = t2.split('#')
                    t2 = t2.strip()
                    comment = comment.strip()
                    if t1 == 'nstep':
                        t2 = self.configuration['simulation']['nsteps']
                    output_simulation_parameter.write(f"{t1:<25} = {t2:<25} # {comment:<30}\n")
                else:
                    output_simulation_parameter.write('\n')
            input_simulation_parameter.close()
            output_simulation_parameter.close()


        # copy omm simulation script
        omm_simulation_script_source = f"{basedir}/complex/openmm/openmm_run.py"
        omm_simulation_script_target = f"{intermediate_state_file_path}/openmm_run.py"
        shutil.copyfile(omm_simulation_script_source, omm_simulation_script_target)

        # adding serializer functions
        f = open(omm_simulation_script_target, 'a')
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

        # copy toppar folder
        toppar_dir = get_toppar_dir()
        toppar_source = f"{toppar_dir}"
        toppar_target = f"{intermediate_state_file_path}/toppar" 
        shutil.copytree(toppar_source, toppar_target)

        omm_simulation_submit_script_source = f"{self.configuration['bin_dir']}/simulation.sh"
        omm_simulation_submit_script_target = f"{intermediate_state_file_path}/simulation.sh"
        shutil.copyfile(omm_simulation_submit_script_source, omm_simulation_submit_script_target)  
    
    
    
    def _write_rtf_file(self, psf, output_file_base, tlc): # NOTE: thisneeds some refactoring!
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
                logging.info('- Setting dummy parameters ...')
                logging.info(f"  + Atom-Name: {atom.name}")
                logging.info(f"  + Atom-Type: {atom.initial_type}")
                logging.info(f"  + Atom Dummy Type: {atom.type}")

                rtf_file_handler.write('{:7} {:6} {:6} {:6}\n'.format('MASS', '-1', atom.type, atom.mass))
            
        rtf_file_handler.close()    




    def _write_prm_file(self, psf, output_file_base, tlc):
    
        header_prm = '''* Parameters generated by analogy by
* CHARMM General Force Field (CGenFF) program version 1.0.0
*
! Automatically obtained dummy parameters 
! from transformato
'''

        prm_file_handler = open(output_file_base + '/dummy_parameters.prm', 'w')
        prm_file_handler.write(header_prm)
        prm_file_handler.write('\nATOMS\n')

        view = psf.view[f":{tlc}"]
        # writing atom parameters
        for atom in view.atoms:
            if hasattr(atom, 'initial_type'):
                logging.info('- Setting dummy parameters ...')
                logging.info(f"  + Atom-Name: {atom.name}")
                logging.info(f"  + Atom-Type: {atom.initial_type}")
                logging.info(f"  + Atom Dummy Type: {atom.type}")
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
                logger.info(' >> Setting dummy bond parameters for: {} - {}'.format(str(atom1.type),str(atom2.type)))
                try:
                    logger.info('{:7} {:7} {:9.5f} {:9.5f} \n'.format(str(atom1.type), str(atom2.type), bond.mod_type.k ,bond.mod_type.req))
                    prm_file_handler.write('{:7} {:7} {:9.5f} {:9.5f} \n'.format(str(atom1.type), str(atom2.type), bond.mod_type.k ,bond.mod_type.req))
                except AttributeError:
                    logger.info('{:7} {:7} {:9.5f} {:9.5f} \n'.format(str(atom1.type), str(atom2.type), bond.type.k ,bond.type.req))
                    prm_file_handler.write('{:7} {:7} {:9.5f} {:9.5f} \n'.format(str(atom1.type), str(atom2.type), bond.type.k ,bond.type.req))

        #################################################################
        prm_file_handler.write('\n\n')
        prm_file_handler.write('ANGLES\n')
        for angle in view.angles:
            atom1, atom2, atom3 = angle.atom1, angle.atom2, angle.atom3
            if any(hasattr(atom, 'initial_type') for atom in [atom1, atom2, atom3]):            
                logger.info(' >> Setting dummy angle parameters for: {}-{}-{}'.format(str(atom1.type),str(atom2.type),str(atom3.type)))
                try:
                    prm_file_handler.write('{:7} {:7} {:7} {:9.5f} {:9.5f} \n'.format(str(atom1.type), str(atom2.type), str(atom3.type), angle.mod_type.k , angle.mod_type.theteq))
                    logger.info('{:7} {:7} {:7} {:9.5f} {:9.5f} \n'.format(str(atom1.type), str(atom2.type), str(atom3.type), angle.mod_type.k , angle.mod_type.theteq))
                except AttributeError:
                    prm_file_handler.write('{:7} {:7} {:7} {:9.5f} {:9.5f} \n'.format(str(atom1.type), str(atom2.type), str(atom3.type), angle.type.k , angle.type.theteq))
                    logger.info('{:7} {:7} {:7} {:9.5f} {:9.5f} \n'.format(str(atom1.type), str(atom2.type), str(atom3.type), angle.type.k , angle.type.theteq))
                    


        #################################################################
        prm_file_handler.write('\n\n')
        prm_file_handler.write('DIHEDRALS\n')
        for dihedral in view.dihedrals:
            atom1, atom2, atom3, atom4 = dihedral.atom1, dihedral.atom2, dihedral.atom3, dihedral.atom4
            if any(hasattr(atom, 'initial_type') for atom in [atom1, atom2, atom3, atom4]):            
                logger.info(' >> Setting dummy dihedral parameters for: {}-{}-{}-{}'.format(str(atom1.type),str(atom2.type),str(atom3.type),str(atom4.type)))
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
        
        f = open(output_file_base + '/toppar.str', 'w+')
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

        logger.info(' - Created directory: - {}'.format(os.path.abspath(output_file_base)))
        os.makedirs(output_file_base)
        logger.info(' - Writing in - {}'.format(os.path.abspath(output_file_base)))
        return output_file_base