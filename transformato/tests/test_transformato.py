"""
Unit and regression test for the transformato package.
"""

import copy
import logging
import os
import pathlib
import shutil
import subprocess
import sys

import numpy as np
import parmed as pm
import pytest
# Import package, test suite, and other packages as needed
import transformato
# read in specific topology with parameters
from parmed.charmm.parameters import CharmmParameterSet
from transformato import (IntermediateStateFactory, ProposeMutationRoute,
                          SystemStructure, load_config_yaml)


def read_params(output_file_base):
    extlist = ['rtf', 'prm', 'str']
    print(output_file_base)
    parFiles = ()
    toppar_file = f"{output_file_base}/toppar.str"
    for line in open(toppar_file, 'r'):
        if '!' in line:
            line = line.split('!')[0]
        parfile = line.strip()
        if len(parfile) != 0:
            ext = parfile.lower().split('.')[-1]
            if not ext in extlist:
                continue
            parFiles += (f"{output_file_base}/{parfile}", )

    params = CharmmParameterSet(*parFiles)
    return params


def generate_psf(output_file_base, env):
    parms = read_params(output_file_base)
    target_psf = pm.charmm.CharmmPsfFile(f"{output_file_base}/lig_in_{env}.psf")
    target_psf.load_parameters(parms)
    return target_psf


def test_transformato_imported():
    """Sample test, will always pass so long as import statement worked"""
    assert "transformato" in sys.modules


def test_read_yaml():
    """Sample test, will check ability to read yaml files"""
    settingsMap = load_config_yaml(config='config/test-2oj9-solvation-free-energy.yaml',
                                   input_dir='.', output_dir='data/')

    assert(settingsMap['system']['name'] == '2OJ9-test1-2OJ9-test2-solvation-free-energy')


def test_initialize_systems():
    configuration = load_config_yaml(config='config/test-2oj9-solvation-free-energy.yaml',
                                     input_dir='data/', output_dir='.')

    s1 = SystemStructure(configuration, 'structure1')
    assert(int(s1.waterbox_offset) == 0)
    assert(int(s1.vacuum_offset) == 0)

    s2 = SystemStructure(configuration, 'structure2')
    assert(int(s2.waterbox_offset) == 0)
    assert(int(s2.vacuum_offset) == 0)

    assert('vacuum' in s1.envs and 'vacuum' in s2.envs)
    assert('waterbox' in s1.envs and 'waterbox' in s2.envs)

    configuration = load_config_yaml(config='config/test-2oj9-binding-free-energy.yaml',
                                     input_dir='data/', output_dir='.')

    s1 = SystemStructure(configuration, 'structure1')
    assert(int(s1.waterbox_offset) == 0)
    assert(int(s1.complex_offset) == 4811)

    s2 = SystemStructure(configuration, 'structure2')
    assert(int(s2.waterbox_offset) == 0)
    assert(int(s2.complex_offset) == 4692)

    assert('complex' in s1.envs and 'complex' in s2.envs)
    assert ('waterbox' in s1.envs and 'waterbox' in s2.envs)


def test_proposed_mutation():
    for conf in ['config/test-2oj9-solvation-free-energy.yaml', 'config/test-2oj9-binding-free-energy.yaml']:
        configuration = load_config_yaml(config=conf,
                                         input_dir='data/', output_dir='.')
        s1 = SystemStructure(configuration, 'structure1')
        s2 = SystemStructure(configuration, 'structure2')

        a = ProposeMutationRoute(s1, s2)
        cc1 = set([0, 3, 6, 5, 4, 14, 24, 23, 26, 22, 25, 17, 16, 28, 27, 29, 46, 47, 48, 45,
                   41, 44, 2, 7, 11, 9, 8, 10, 13, 12, 39, 38, 36, 37, 34, 35, 30, 32, 33, 31])
        assert(set(a.get_common_core_idx_mol1()) == cc1)
        cc2 = set([0, 3, 6, 5, 4, 14, 19, 18, 21, 17, 20, 16, 15, 23, 22, 24, 43, 44, 45, 42,
                   40, 41, 2, 7, 11, 9, 8, 10, 13, 12, 38, 37, 35, 36, 33, 34, 29, 31, 32, 30])
        assert(set(a.get_common_core_idx_mol2()) == cc2)
        assert(str(a.s1_tlc) == 'BMI')
        assert(str(a.s2_tlc) == 'UNK')


def test_endpoint():

    for conf in ['config/test-2oj9-solvation-free-energy.yaml', 'config/test-2oj9-binding-free-energy.yaml']:
        configuration = load_config_yaml(config=conf,
                                         input_dir='data/', output_dir='.')
        s1 = SystemStructure(configuration, 'structure1')
        s2 = SystemStructure(configuration, 'structure2')

        s1_to_s2 = ProposeMutationRoute(s1, s2)
        s2_to_s1 = ProposeMutationRoute(s2, s1)
        for a, system in zip([s1_to_s2, s2_to_s1], [s1, s2]):
            mutation_list = a.generate_mutations_to_common_core_for_mol1(
                nr_of_steps_for_el=3, nr_of_steps_for_bonded_parameters=3)
            i = IntermediateStateFactory(system=system, mutation_list=mutation_list, configuration=configuration)
            # extract first mutation
            m = mutation_list[0]
            current_step = 0
            try:
                # write out structure
                output_file_base = i.generate_specific_intermediate_state(m, current_step)
                # get a copy of the original structure
                original_psf = {}
                for env in system.envs:
                    original_psf[env] = copy.deepcopy(system.psf_mapping[env])

                # test for waterbox and complex
                for env in system.envs:
                    # read in psf and parameters
                    new_psf = generate_psf(output_file_base, env)
                    offset = min([a.idx for a in original_psf[env].view[f":{system.tlc.upper()}"].atoms])

                    # test if atom parameters are the same
                    for idx in m.atom_idx:
                        assert(original_psf[env].atoms[idx+offset].epsilon == new_psf.atoms[idx+offset].epsilon)
                        assert(original_psf[env].atoms[idx+offset].charge == new_psf.atoms[idx+offset].charge)
                        assert(original_psf[env].atoms[idx+offset].rmin == new_psf.atoms[idx+offset].rmin)

                        # test all bond parameters
                        for o_bond, n_bond in zip(original_psf[env].atoms[idx+offset].bonds, new_psf.atoms[idx+offset].bonds):
                            assert(o_bond.type.k == n_bond.type.k)

                        # test all angle parameters
                        for o_angle, n_angle in zip(original_psf[env].atoms[idx+offset].angles, new_psf.atoms[idx+offset].angles):
                            assert(o_angle.type.k == n_angle.type.k)
            finally:
                shutil.rmtree(output_file_base)


def test_charge_mutation():

    for conf in ['config/test-2oj9-solvation-free-energy.yaml', 'config/test-2oj9-binding-free-energy.yaml']:
        configuration = load_config_yaml(config=conf,
                                        input_dir='data/', output_dir='.')
        s1 = SystemStructure(configuration, 'structure1')
        s2 = SystemStructure(configuration, 'structure2')

        s1_to_s2 = ProposeMutationRoute(s1, s2)
        s2_to_s1 = ProposeMutationRoute(s2, s1)
        for a, system in zip([s1_to_s2, s2_to_s1], [s1, s2]):
            mutation_list = a.generate_mutations_to_common_core_for_mol1(
                nr_of_steps_for_el=4, nr_of_steps_for_bonded_parameters=4)
            i = IntermediateStateFactory(system=system, mutation_list=mutation_list, configuration=configuration)

            # test endpoint - charge not scaled
            m = mutation_list[0]
            current_step = 0
            try:
                output_file_base = i.generate_specific_intermediate_state(m, current_step)
                # get a copy of the original structure
                original_psf = {}
                for env in system.envs:
                    original_psf[env] = copy.deepcopy(system.psf_mapping[env])

                for env in system.envs:
                    offset = min([a.idx for a in original_psf[env].view[f":{system.tlc.upper()}"].atoms])

                    new_psf = generate_psf(output_file_base, env)
                    scale = current_step/(m.nr_of_steps)
                    for idx in m.atom_idx:
                        assert(np.isclose(original_psf[env].atoms[idx+offset].charge *
                                        (1 - scale), new_psf.atoms[idx+offset].charge))
                        assert(np.isclose(original_psf[env].atoms[idx+offset].charge, new_psf.atoms[idx+offset].charge))
            finally:
                shutil.rmtree(output_file_base)

            # scale charges with 0.25
            current_step = 1
            try:
                output_file_base = i.generate_specific_intermediate_state(m, current_step)
                for env in system.envs:
                    offset = min([a.idx for a in original_psf[env].view[f":{system.tlc.upper()}"].atoms])
                    new_psf = generate_psf(output_file_base, env)
                    scale = current_step/(m.nr_of_steps)
                    print('Scaling charges with: {}'.format((1 - scale)))
                    for idx in m.atom_idx:
                        assert(np.isclose(original_psf[env].atoms[idx+offset].charge *
                                        (1 - scale), new_psf.atoms[idx+offset].charge, rtol=1e-03))
            finally:
                shutil.rmtree(output_file_base)

            # scale charges with 1.0
            current_step = 4
            try:
                output_file_base = i.generate_specific_intermediate_state(m, current_step)
                for env in system.envs:
                    offset = min([a.idx for a in original_psf[env].view[f":{system.tlc.upper()}"].atoms])
                    new_psf = generate_psf(output_file_base, env)
                    scale = current_step/(m.nr_of_steps)
                    print('Scaling charges with: {}'.format((1 - scale)))
                    for idx in m.atom_idx:
                        idxo = idx+offset
                        assert(np.isclose(original_psf[env].atoms[idxo].charge *
                                        (1 - scale), new_psf.atoms[idxo].charge, rtol=1e-03))
                        assert(np.isclose(0.0, new_psf.atoms[idxo].charge))
            finally:
                shutil.rmtree(output_file_base)


def test_vdw_mutation():

    for conf in ['config/test-2oj9-solvation-free-energy.yaml', 'config/test-2oj9-binding-free-energy.yaml']:
        configuration = load_config_yaml(config=conf,
                                        input_dir='data/', output_dir='.')
        s1 = SystemStructure(configuration, 'structure1')
        s2 = SystemStructure(configuration, 'structure2')

        s1_to_s2 = ProposeMutationRoute(s1, s2)
        s2_to_s1 = ProposeMutationRoute(s2, s1)
        for a, system in zip([s1_to_s2, s2_to_s1], [s1, s2]):
            mutation_list = a.generate_mutations_to_common_core_for_mol1(
                nr_of_steps_for_el=4, nr_of_steps_for_bonded_parameters=4)
            i = IntermediateStateFactory(system=system, mutation_list=mutation_list, configuration=configuration)

            original_psf = {}
            for env in system.envs:
                original_psf[env] = copy.deepcopy(system.psf_mapping[env])

            m1 = mutation_list[1]
            current_step = 0
            try:
                output_file_base = i.generate_specific_intermediate_state(m1, current_step)
                # get a copy of the original structure

                for env in system.envs:

                    offset = min([a.idx for a in original_psf[env].view[f":{system.tlc.upper()}"].atoms])
                    new_psf = generate_psf(output_file_base, env)
                    print('Set epsilon/rmin to zero for selected atoms')
                    for idx in m1.atom_idx:
                        idxo = idx + offset
                        assert(np.isclose(original_psf[env].atoms[idxo].epsilon * 0, new_psf.atoms[idxo].epsilon))
                        assert(np.isclose(original_psf[env].atoms[idxo].rmin * 0, new_psf.atoms[idxo].rmin))

                    # make sure that all other idx are not touched
                    for idx in range(len(original_psf[env].atoms)):
                        idxo = idx - offset  # NOTE: the '-'
                        if idxo not in m1.atom_idx:
                            assert(np.isclose(original_psf[env].atoms[idx].epsilon, new_psf.atoms[idx].epsilon))
                            assert(np.isclose(original_psf[env].atoms[idx].rmin, new_psf.atoms[idx].rmin))
            finally:
                shutil.rmtree(output_file_base)

            m2 = mutation_list[2]
            current_step = 1
            try:
                output_file_base = i.generate_specific_intermediate_state(m2, current_step)
                for env in system.envs:

                    offset = min([a.idx for a in original_psf[env].view[f":{system.tlc.upper()}"].atoms])
                    new_psf = generate_psf(output_file_base, env)
                    print('Set epsilon/rmin to zero for selected atoms')
                    for idx in m2.atom_idx:
                        idxo = idx + offset
                        assert(np.isclose(original_psf[env].atoms[idxo].epsilon * 0, new_psf.atoms[idxo].epsilon))
                        assert(np.isclose(original_psf[env].atoms[idxo].rmin * 0, new_psf.atoms[idxo].rmin))

                    # make sure that all other idx are not touched
                    print(m2.atom_idx + m1.atom_idx)
                    for idx in range(len(original_psf[env].atoms)):
                        idxo = idx - offset  # NOTE: the '-'
                        if idxo not in m2.atom_idx + m1.atom_idx:
                            assert(np.isclose(original_psf[env].atoms[idx].epsilon, new_psf.atoms[idx].epsilon))
                            assert(np.isclose(original_psf[env].atoms[idx].rmin, new_psf.atoms[idx].rmin))

            finally:
                shutil.rmtree(output_file_base)


@pytest.mark.slowtest
def test_bonded_mutation():

    for conf in ['config/test-2oj9-solvation-free-energy.yaml', 'config/test-2oj9-binding-free-energy.yaml']:
        configuration = load_config_yaml(config=conf,
                                        input_dir='data/', output_dir='.')
        s1 = SystemStructure(configuration, 'structure1')
        s2 = SystemStructure(configuration, 'structure2')

        s1_to_s2 = ProposeMutationRoute(s1, s2)
        s2_to_s1 = ProposeMutationRoute(s2, s1)

        for a, system, template in zip([s1_to_s2, s2_to_s1], [SystemStructure(configuration, 'structure1'), SystemStructure(configuration, 'structure2')],
                                    [SystemStructure(configuration, 'structure2'), SystemStructure(configuration, 'structure1')]):

            original_psf = {}
            template_psf = {}
            for env in system.envs:
                original_psf[env] = copy.deepcopy(system.psf_mapping[env])
                template_psf[env] = copy.deepcopy(template.psf_mapping[env])

            mutation_list = a.generate_mutations_to_common_core_for_mol1(
                nr_of_steps_for_el=4, nr_of_steps_for_bonded_parameters=4)
            i = IntermediateStateFactory(system=system, mutation_list=mutation_list, configuration=configuration)

            print(mutation_list)
            # we select the bonded mutation which should be at postition [-1]
            m1 = mutation_list[-1]
            assert(type(m1) == transformato.mutate.BondedParameterMutation)
            current_step = 1
            try:
                output_file_base = i.generate_specific_intermediate_state(m1, current_step)

                for env in system.envs:
                    print('Env : {}'.format(env))
                    new_psf = generate_psf(output_file_base, env)

                    # atom names are the same in new and orginal psf
                    for natom, oatom in zip(new_psf.view[f":{system.tlc.upper()}"].atoms,
                                            original_psf[env].view[f":{system.tlc.upper()}"].atoms):

                        assert(natom.name == oatom.name)

                    # changes are only in the ligand
                    for natom, oatom in zip(new_psf.view[f"!(:{system.tlc.upper()})"].atoms,
                                            original_psf[env].view[f"!(:{system.tlc.upper()})"].atoms):

                        assert(np.isclose(natom.charge, oatom.charge))
                        assert(np.isclose(natom.epsilon, oatom.epsilon))

                    # get mapping between original/new and template psf
                    match_atom_names_cc1_to_cc2 = dict()
                    cc1_offset = min([a.idx for a in original_psf[env].view[f":{m1.tlc_cc1.upper()}"].atoms])
                    cc2_offset = min([a.idx for a in template_psf[env].view[f":{m1.tlc_cc2.upper()}"].atoms])
                    scale = current_step/(m1.nr_of_steps)

                    # test atom parameters
                    for cc1, cc2 in zip(m1.cc1_idx, m1.cc2_idx):
                        # did atom type change? if not continue

                        cc1_oidx = cc1 + cc1_offset
                        cc2_oidx = cc2 + cc2_offset
                        cc1_a = original_psf[env][cc1_oidx]
                        cc2_a = template_psf[env][cc2_oidx]
                        match_atom_names_cc1_to_cc2[cc1_a.name] = cc2_a.name

                        assert(np.isclose((1.0 - scale) * cc1_a.epsilon + scale * cc2_a.epsilon,
                                        new_psf[cc1_oidx].epsilon))
                        assert(np.isclose((1.0 - scale) * cc1_a.sigma + scale * cc2_a.sigma,
                                        new_psf[cc1_oidx].sigma, rtol=1e-03))
                    

                    # get mapping between original/new and template psf
                    for cc1_bond, new_bond in zip(original_psf[env].view[f":{m1.tlc_cc1.upper()}"].bonds,
                                                new_psf.view[f":{m1.tlc_cc1.upper()}"].bonds):
                        cc1_a1 = cc1_bond.atom1.name
                        cc1_a2 = cc1_bond.atom2.name
                        # all atoms of the bond must be in cc
                        # everything outside the cc are bonded terms between dummies or
                        # between real atoms and dummies and we can ignore them for now
                        if not all(elem in match_atom_names_cc1_to_cc2 for elem in [cc1_a1, cc1_a2]):
                            assert(np.isclose(cc1_bond.type.k, new_bond.type.k))
                            continue

                        for cc2_bond in template_psf[env].view[f":{m1.tlc_cc2.upper()}"].bonds:
                            cc2_a1 = cc2_bond.atom1.name
                            cc2_a2 = cc2_bond.atom2.name
                            # all atoms of the bond must be in cc
                            if not all(elem in match_atom_names_cc1_to_cc2.values() for elem in [cc2_a1, cc2_a2]):
                                continue

                            # match the two bonds
                            if sorted([match_atom_names_cc1_to_cc2[e] for e in [cc1_a1, cc1_a2]]) == sorted([cc2_a1, cc2_a2]):
                                scaled = (1.0 - scale) * cc1_bond.type.k + scale * cc2_bond.type.k
                                assert(np.isclose(scaled, new_bond.type.k))

                    # make sure everything else in not changed
                    for cc1_bond, new_bond in zip(original_psf[env].view[f"!:{m1.tlc_cc1.upper()}"].bonds,
                                                new_psf.view[f"!:{m1.tlc_cc1.upper()}"].bonds):
                        assert(np.isclose(cc1_bond.type.k, new_bond.type.k))

                    # get mapping between original/new and template psf
                    for cc1_angle, new_angle in zip(original_psf[env].view[f":{m1.tlc_cc1.upper()}"].angles,
                                                    new_psf.view[f":{m1.tlc_cc1.upper()}"].angles):
                        cc1_a1 = cc1_angle.atom1.name
                        cc1_a2 = cc1_angle.atom2.name
                        cc1_a3 = cc1_angle.atom3.name
                        # all atoms of the bond must be in cc
                        # everything outside the cc are bonded terms between dummies or
                        # between real atoms and dummies and we can ignore them for now
                        if not all(elem in match_atom_names_cc1_to_cc2 for elem in [cc1_a1, cc1_a2, cc1_a3]):
                            assert(np.isclose(cc1_angle.type.k, new_angle.type.k))
                            continue

                        for cc2_angle in template_psf[env].view[f":{m1.tlc_cc2.upper()}"].angles:
                            cc2_a1 = cc2_angle.atom1.name
                            cc2_a2 = cc2_angle.atom2.name
                            cc2_a3 = cc2_angle.atom3.name
                            # all atoms of the bond must be in cc
                            if not all(elem in match_atom_names_cc1_to_cc2.values() for elem in [cc2_a1, cc2_a2, cc2_a3]):
                                continue

                            # match the two bonds
                            if sorted([match_atom_names_cc1_to_cc2[e] for e in [cc1_a1, cc1_a2, cc1_a3]]) == sorted([cc2_a1, cc2_a2, cc2_a3]):
                                scaled = (1.0 - scale) * cc1_angle.type.k + scale * cc2_angle.type.k
                                assert (np.isclose(scaled, new_angle.type.k))
                                
            finally:
                shutil.rmtree(output_file_base)


@pytest.mark.slowtest
def test_charge_cc1_to_cc2_transformation1():
    from parmed.charmm.psf import CharmmPsfFile

    for conf in ['config/test-ethane-ethanol-solvation-free-energy.yaml']:
        configuration = load_config_yaml(config=conf,
                                        input_dir='data/', output_dir='data')

        # load systems
        s1 = SystemStructure(configuration, 'structure1')
        s2 = SystemStructure(configuration, 'structure2')
        a = ProposeMutationRoute(s1, s2)

        # manually matching Oxygen (0) with Hydrogen (4) 
        a.add_idx_to_common_core_of_mol1(4)
        a.add_idx_to_common_core_of_mol2(0)

        # generate mutation route
        mutation_list = a.generate_mutations_to_common_core_for_mol1(
            nr_of_steps_for_el=5, nr_of_steps_for_bonded_parameters=5)

        assert (len(mutation_list) == 1)
        assert (type(mutation_list[0]) == transformato.mutate.BondedParameterMutation)
        assert (mutation_list[0].nr_of_steps == 5)
        
        try:
            # write intermediate states for systems
            cc1_i = IntermediateStateFactory(system=s1, mutation_list=mutation_list, configuration=configuration)
            cc1_i.generate_intermediate_states()

            # generate mutation route
            mutation_list = a.generate_mutations_to_common_core_for_mol2(nr_of_steps_for_el=5)
            # write intermediate states
            cc2_i = IntermediateStateFactory(system=s2, mutation_list=mutation_list, configuration=configuration)
            cc2_i.generate_intermediate_states()

            for env in s1.envs:
                print('Env : {}'.format(env))

                cc_ethane = CharmmPsfFile(f"{cc1_i.path}/intst6/lig_in_{env}.psf")
                cc_ethanol = CharmmPsfFile(f"{cc2_i.path}/intst7/lig_in_{env}.psf")
                cc1_offset = min([atom.idx for atom in cc_ethane.view[f":{a.s1_tlc.upper()}"].atoms])
                cc2_offset = min([atom.idx for atom in cc_ethanol.view[f":{a.s2_tlc.upper()}"].atoms])

                # test atom parameters
                for cc1, cc2 in zip(list(a.get_common_core_idx_mol1()), list(a.get_common_core_idx_mol2())):
                    # did atom type change? if not continue
                    cc1_oidx = cc1 + cc1_offset
                    cc2_oidx = cc2 + cc2_offset
                    assert(np.isclose(cc_ethane[cc1_oidx].charge, cc_ethanol[cc2_oidx].charge, rtol=1e-03))
        finally:
            shutil.rmtree(cc1_i.path)
            shutil.rmtree(cc2_i.path)


@pytest.mark.slowtest
def test_charge_cc1_to_cc2_transformation2():
    from parmed.charmm.psf import CharmmPsfFile
    for conf in ['config/test-ethane-ethanol-solvation-free-energy.yaml']:
        configuration = load_config_yaml(config=conf,
                                        input_dir='data/', output_dir='data')

        # load systems
        s1 = SystemStructure(configuration, 'structure1')
        s2 = SystemStructure(configuration, 'structure2')
        a = ProposeMutationRoute(s1, s2)

        # generate mutation route
        mutation_list = a.generate_mutations_to_common_core_for_mol1(
            nr_of_steps_for_el=5, nr_of_steps_for_bonded_parameters=5)

        try:
            # write intermediate states for systems
            cc1_i = IntermediateStateFactory(system=s1, mutation_list=mutation_list, configuration=configuration)
            cc1_i.generate_intermediate_states()

            # generate mutation route
            mutation_list = a.generate_mutations_to_common_core_for_mol2(nr_of_steps_for_el=5)
            # write intermediate states
            cc2_i = IntermediateStateFactory(system=s2, mutation_list=mutation_list, configuration=configuration)
            cc2_i.generate_intermediate_states()

            for env in s1.envs:
                print('Env : {}'.format(env))

                cc_ethane = CharmmPsfFile(f"{cc1_i.path}/intst12/lig_in_{env}.psf")
                cc_ethanol = CharmmPsfFile(f"{cc2_i.path}/intst8/lig_in_{env}.psf")
                cc1_offset = min([atom.idx for atom in cc_ethane.view[f":{a.s1_tlc.upper()}"].atoms])
                cc2_offset = min([atom.idx for atom in cc_ethanol.view[f":{a.s2_tlc.upper()}"].atoms])

                # test atom parameters
                for cc1, cc2 in zip(list(a.get_common_core_idx_mol1()), list(a.get_common_core_idx_mol2())):
                    # did atom type change? if not continue
                    cc1_oidx = cc1 + cc1_offset
                    cc2_oidx = cc2 + cc2_offset
                    assert(np.isclose(cc_ethane[cc1_oidx].charge, cc_ethanol[cc2_oidx].charge, rtol=1e-03))
        finally:
            shutil.rmtree(cc1_i.path)
            shutil.rmtree(cc2_i.path)




@pytest.mark.slowtest
def test_run_test_systems():
    for conf in ['config/test-2oj9-solvation-free-energy.yaml', 'config/test-2oj9-binding-free-energy.yaml']:
        configuration = load_config_yaml(config=conf,
                                         input_dir='data/', output_dir='data/')

        # load systems
        s1 = SystemStructure(configuration, 'structure1')
        s2 = SystemStructure(configuration, 'structure2')
        a = ProposeMutationRoute(s1, s2)

        # generate mutation route
        mutation_list = a.generate_mutations_to_common_core_for_mol1(
            nr_of_steps_for_el=5, nr_of_steps_for_bonded_parameters=5)
        # write intermediate states for systems
        i = IntermediateStateFactory(system=s1, mutation_list=mutation_list, configuration=configuration)
        i.generate_intermediate_states()

        # generate mutation route
        mutation_list = a.generate_mutations_to_common_core_for_mol2(nr_of_steps_for_el=5)
        # write intermediate states
        i = IntermediateStateFactory(system=s2, mutation_list=mutation_list, configuration=configuration)
        try:
            i.generate_intermediate_states()
            print(pathlib.Path(i.path).parent)
        finally:
            shutil.rmtree(pathlib.Path(i.path).parent)


@pytest.mark.slowtest
@pytest.mark.skipif(
    os.environ.get("TRAVIS", None) == "true", reason="Skip slow test on travis."
)
def test_run_example1_systems_solvation_free_energy():
    from transformato import FreeEnergyCalculator
    for conf in ['config/test-2oj9-solvation-free-energy.yaml']:
        configuration = load_config_yaml(config=conf,
                                         input_dir='data/', output_dir='data/')

        # load systems
        s1 = SystemStructure(configuration, 'structure1')
        s2 = SystemStructure(configuration, 'structure2')
        a = ProposeMutationRoute(s1, s2)

        # generate mutation route
        mutation_list = a.generate_mutations_to_common_core_for_mol1(
            nr_of_steps_for_el=5, nr_of_steps_for_bonded_parameters=5)

        try:
            # write intermediate states for systems
            i = IntermediateStateFactory(system=s1, mutation_list=mutation_list, configuration=configuration)
            i.generate_intermediate_states()
            paths = pathlib.Path(i.path).glob('**/*.sh')
            for path in sorted(paths):
                run_dir = path.parent
                # because path is object not string
                print(f"Start sampling for: {path}")
                print(f"In directory: {run_dir}")
                try:
                    exe = subprocess.run(['bash', str(path), str(run_dir)], check=True, capture_output=True, text=True)
                except TypeError:
                    exe = subprocess.run(['bash', str(path), str(run_dir)], check=True,
                                        stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
                print(exe.stdout)
                print('Capture stderr')
                print(exe.stderr)

            # generate mutation route
            mutation_list = a.generate_mutations_to_common_core_for_mol2(nr_of_steps_for_el=5)
            # write intermediate states
            i = IntermediateStateFactory(system=s2, mutation_list=mutation_list, configuration=configuration)
            i.generate_intermediate_states()

            paths = pathlib.Path(i.path).glob('**/*.sh')
            for path in sorted(paths):
                run_dir = path.parent
                # because path is object not string
                print(f"Start sampling for: {path}")
                print(f"In directory: {run_dir}")
                try:
                    exe = subprocess.run(['bash', str(path), str(run_dir)], check=True, capture_output=True, text=True)
                except TypeError:
                    exe = subprocess.run(['bash', str(path), str(run_dir)], check=True,
                                        stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
                print(exe.stdout)
                print('Capture stderr')
                print(exe.stderr)

            f = FreeEnergyCalculator(configuration, '2OJ9-e1')
            f.load_trajs(thinning=1)
            f.calculate_dG_to_common_core()
            ddG, dddG = f.end_state_free_energy_difference
            print(f"Free energy difference: {ddG}")
            print(f"Uncertanty: {dddG}")
            #assert(ddG == 10.0)

            f.show_summary()

            f = FreeEnergyCalculator(configuration, '2OJ9-e2')
            f.load_trajs(thinning=1)
            f.calculate_dG_to_common_core()

            f.show_summary()
        finally:
            shutil.rmtree(pathlib.Path(i.path).parent)


@pytest.mark.slowtest
@pytest.mark.skipif(
    os.environ.get("TRAVIS", None) == "true", reason="Skip slow test on travis."
)
def test_run_example2_systems_solvation_free_energy():
    from transformato import FreeEnergyCalculator
    for conf in ['config/test-ethane-ethanol-solvation-free-energy.yaml']:
        configuration = load_config_yaml(config=conf,
                                         input_dir='data/', output_dir='data/')

        # load systems
        s1 = SystemStructure(configuration, 'structure1')
        s2 = SystemStructure(configuration, 'structure2')
        a = ProposeMutationRoute(s1, s2)

        # manually matching Oxygen (0) with Hydrogen (4) 
        a.add_idx_to_common_core_of_mol1(4)
        a.add_idx_to_common_core_of_mol2(0)

        # generate mutation route
        mutation_list = a.generate_mutations_to_common_core_for_mol1(
            nr_of_steps_for_el=5, nr_of_steps_for_bonded_parameters=5)

        try:
            # write intermediate states for systems
            i = IntermediateStateFactory(system=s1, mutation_list=mutation_list, configuration=configuration)
            i.generate_intermediate_states()
            paths = pathlib.Path(i.path).glob('**/*.sh')
            for path in sorted(paths):
                run_dir = path.parent
                # because path is object not string
                print(f"Start sampling for: {path}")
                print(f"In directory: {run_dir}")
                try:
                    exe = subprocess.run(['bash', str(path), str(run_dir)], check=True, capture_output=True, text=True)
                except TypeError:
                    exe = subprocess.run(['bash', str(path), str(run_dir)], check=True,
                                        stdout=subprocess.PIPE, stderr=subprocess.PIPE)
                print(exe.stdout)
                print('Capture stderr')
                print(exe.stderr)

            # generate mutation route
            mutation_list = a.generate_mutations_to_common_core_for_mol2(nr_of_steps_for_el=5)
            # write intermediate states
            i = IntermediateStateFactory(system=s2, mutation_list=mutation_list, configuration=configuration)
            i.generate_intermediate_states()

            paths = pathlib.Path(i.path).glob('**/*.sh')
            for path in sorted(paths):
                run_dir = path.parent
                # because path is object not string
                print(f"Start sampling for: {path}")
                print(f"In directory: {run_dir}")
                try:
                    exe = subprocess.run(['bash', str(path), str(run_dir)], check=True, capture_output=True, text=True)
                except TypeError:
                    exe = subprocess.run(['bash', str(path), str(run_dir)], check=True,
                                        stdout=subprocess.PIPE, stderr=subprocess.PIPE)
                print(exe.stdout)
                print('Capture stderr')
                print(exe.stderr)

            f = FreeEnergyCalculator(configuration, 'ethane')
            f.load_trajs(thinning=2)
            f.calculate_dG_to_common_core()
            ddG, dddG = f.end_state_free_energy_difference
            print(f"Free energy difference: {ddG}")
            print(f"Uncertanty: {dddG}")
            #assert(ddG == 10.0)
            f.show_summary()

            f = FreeEnergyCalculator(configuration, 'ethanol')
            f.load_trajs(thinning=2)
            f.calculate_dG_to_common_core()
            ddG, dddG = f.end_state_free_energy_difference
            print(f"Free energy difference: {ddG}")
            print(f"Uncertanty: {dddG}")
            f.show_summary()
        finally:
            pass
            #shutil.rmtree(pathlib.Path(i.path).parent)
