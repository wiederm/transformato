"""
Unit and regression test for the transformato package.
"""

import copy
import os
import shutil
import logging

import numpy as np
import parmed as pm
import pytest

# Import package, test suite, and other packages as needed
import transformato
from simtk import unit

# read in specific topology with parameters
from parmed.charmm.parameters import CharmmParameterSet
from transformato import (
    IntermediateStateFactory,
    ProposeMutationRoute,
    SystemStructure,
    load_config_yaml,
)


def read_params(output_file_base):
    extlist = ["rtf", "prm", "str"]
    print(output_file_base)
    parFiles = ()
    toppar_file = f"{output_file_base}/toppar.str"
    for line in open(toppar_file, "r"):
        if "!" in line:
            line = line.split("!")[0]
        parfile = line.strip()
        if len(parfile) != 0:
            ext = parfile.lower().split(".")[-1]
            if not ext in extlist:
                continue
            parFiles += (f"{output_file_base}/{parfile}",)

    params = CharmmParameterSet(*parFiles)
    return params


def generate_psf(output_file_base, env):
    parms = read_params(output_file_base)
    psf = pm.charmm.CharmmPsfFile(f"{output_file_base}/lig_in_{env}.psf")
    psf.load_parameters(parms)
    return psf, parms


def generate_crd(output_file_base, env):
    return pm.charmm.CharmmCrdFile(f"{output_file_base}/lig_in_{env}.crd")


def generate_sim(output_file_base, env):
    import simtk.openmm as mm
    import simtk.openmm.app as app

    # ParmEd Imports
    from parmed import unit as u

    system, psf = generate_system(output_file_base, env)
    crd = generate_crd(output_file_base, env)

    # Create the integrator to do Langevin dynamics
    integrator = mm.LangevinIntegrator(
        300 * u.kelvin,  # Temperature of heat bath
        1.0 / u.picoseconds,  # Friction coefficient
        2.0 * u.femtoseconds,  # Time step
    )

    # Create the Simulation object
    sim = app.Simulation(psf.topology, system, integrator)

    # Set the particle positions
    sim.context.setPositions(crd.positions)
    return sim


def generate_system(output_file_base, env):
    import simtk.openmm as mm
    import simtk.openmm.app as app

    # ParmEd Imports
    from parmed import unit as u

    psf, parms = generate_psf(output_file_base, env)

    system = psf.createSystem(parms, nonbondedMethod=app.NoCutoff)
    return system, psf


def _set_output_files_2oj9_tautomer_pair():
    output_files_t1 = [
        "data/2OJ9-original-2OJ9-tautomer-rsfe/2OJ9-original/intst1/",
        "data/2OJ9-original-2OJ9-tautomer-rsfe/2OJ9-original/intst2/",
        "data/2OJ9-original-2OJ9-tautomer-rsfe/2OJ9-original/intst3/",
        "data/2OJ9-original-2OJ9-tautomer-rsfe/2OJ9-original/intst4/",
        "data/2OJ9-original-2OJ9-tautomer-rsfe/2OJ9-original/intst5/",
        "data/2OJ9-original-2OJ9-tautomer-rsfe/2OJ9-original/intst6/",
        "data/2OJ9-original-2OJ9-tautomer-rsfe/2OJ9-original/intst7/",
    ]
    output_files_t2 = [
        "data/2OJ9-original-2OJ9-tautomer-rsfe/2OJ9-tautomer/intst1/",
        "data/2OJ9-original-2OJ9-tautomer-rsfe/2OJ9-tautomer/intst2/",
        "data/2OJ9-original-2OJ9-tautomer-rsfe/2OJ9-tautomer/intst3/",
    ]

    return output_files_t1, output_files_t2


def _set_output_files_acetylaceton_tautomer_pair():
    output_files_enol = [
        "data/acetylacetone-keto-acetylacetone-enol-rsfe/acetylacetone-enol/intst1/",
        "data/acetylacetone-keto-acetylacetone-enol-rsfe/acetylacetone-enol/intst2/",
        "data/acetylacetone-keto-acetylacetone-enol-rsfe/acetylacetone-enol/intst3/",
    ]
    output_files_keto = [
        "data/acetylacetone-keto-acetylacetone-enol-rsfe/acetylacetone-keto/intst1/",
        "data/acetylacetone-keto-acetylacetone-enol-rsfe/acetylacetone-keto/intst2/",
        "data/acetylacetone-keto-acetylacetone-enol-rsfe/acetylacetone-keto/intst3/",
        "data/acetylacetone-keto-acetylacetone-enol-rsfe/acetylacetone-keto/intst4/",
        "data/acetylacetone-keto-acetylacetone-enol-rsfe/acetylacetone-keto/intst5/",
        "data/acetylacetone-keto-acetylacetone-enol-rsfe/acetylacetone-keto/intst6/",
        "data/acetylacetone-keto-acetylacetone-enol-rsfe/acetylacetone-keto/intst7/",
    ]

    return output_files_enol, output_files_keto


def _set_output_files_toluene_methane_pair():
    output_files_methane = [
        "data/toluene-methane-rsfe/methane/intst1/",
        "data/toluene-methane-rsfe/methane/intst2/",
        "data/toluene-methane-rsfe/methane/intst3/",
    ]
    output_files_toluene = [
        "data/toluene-methane-rsfe/toluene/intst1/",
        "data/toluene-methane-rsfe/toluene/intst2/",
        "data/toluene-methane-rsfe/toluene/intst3/",
        "data/toluene-methane-rsfe/toluene/intst4/",
        "data/toluene-methane-rsfe/toluene/intst5/",
        "data/toluene-methane-rsfe/toluene/intst6/",
        "data/toluene-methane-rsfe/toluene/intst7/",
        "data/toluene-methane-rsfe/toluene/intst8/",
        "data/toluene-methane-rsfe/toluene/intst9/",
        "data/toluene-methane-rsfe/toluene/intst10/",
        "data/toluene-methane-rsfe/toluene/intst11/",
        "data/toluene-methane-rsfe/toluene/intst12/",
        "data/toluene-methane-rsfe/toluene/intst13/",
    ]

    return output_files_methane, output_files_toluene


def test_proposed_mutation_mcs():

    from rdkit.Chem import rdFMCS

    for conf in [
        "transformato/tests/config/test-2oj9-rsfe.yaml",
    ]:
        configuration = load_config_yaml(config=conf, input_dir="data/", output_dir=".")
        s1 = SystemStructure(configuration, "structure1")
        s2 = SystemStructure(configuration, "structure2")

        a = ProposeMutationRoute(s1, s2)
        # find mcs
        a._find_mcs("m1", "m2")

        assert str(a.s1_tlc) == "BMI"
        assert str(a.s2_tlc) == "UNK"

        cc1 = set(
            [
                0,
                3,
                6,
                5,
                4,
                14,
                24,
                23,
                26,
                22,
                25,
                17,
                16,
                28,
                27,
                29,
                46,
                47,
                48,
                45,
                41,
                44,
                2,
                7,
                11,
                9,
                8,
                10,
                13,
                12,
                39,
                38,
                36,
                37,
                34,
                35,
                30,
                32,
                33,
                31,
            ]
        )
        cc2 = set(
            [
                0,
                3,
                6,
                5,
                4,
                14,
                19,
                18,
                21,
                17,
                20,
                16,
                15,
                23,
                22,
                24,
                43,
                44,
                45,
                42,
                40,
                41,
                2,
                7,
                11,
                9,
                8,
                10,
                13,
                12,
                38,
                37,
                35,
                36,
                33,
                34,
                29,
                31,
                32,
                30,
            ]
        )

        print(a.get_common_core_idx_mol1())
        print(a.get_common_core_idx_mol2())

        assert set(a.get_common_core_idx_mol1()) == cc1
        assert set(a.get_common_core_idx_mol2()) == cc2
        a.bondCompare = rdFMCS.BondCompare.CompareOrder
        # find mcs
        a._find_mcs("m1", "m2")

        cc1 = set(
            [
                0,
                3,
                6,
                33,
                31,
                14,
                24,
                23,
                26,
                22,
                25,
                17,
                16,
                28,
                27,
                29,
                46,
                47,
                48,
                45,
                41,
                44,
            ]
        )
        cc2 = set(
            [
                0,
                3,
                6,
                32,
                30,
                14,
                19,
                18,
                21,
                17,
                20,
                16,
                15,
                23,
                22,
                24,
                43,
                44,
                45,
                42,
                40,
                41,
            ]
        )

        print(a.get_common_core_idx_mol1())
        print(a.get_common_core_idx_mol2())

        assert set(a.get_common_core_idx_mol2()) == cc2
        assert set(a.get_common_core_idx_mol1()) == cc1

    for conf in ["transformato/tests/config/test-toluene-methane-rsfe.yaml"]:
        configuration = load_config_yaml(config=conf, input_dir="data/", output_dir=".")
        s1 = SystemStructure(configuration, "structure1")
        s2 = SystemStructure(configuration, "structure2")

        a = ProposeMutationRoute(s1, s2)
        # find mcs
        a._find_mcs("m1", "m2")

        assert str(a.s1_tlc) == "UNL"
        assert str(a.s2_tlc) == "LIG"

        print(set(a.get_common_core_idx_mol1()))
        print(set(a.get_common_core_idx_mol2()))
        cc1 = set([8, 5, 6, 7])
        cc2 = set([0, 1, 2, 3])
        assert set(a.get_common_core_idx_mol1()) == cc1
        assert set(a.get_common_core_idx_mol2()) == cc2

        a.bondCompare = rdFMCS.BondCompare.CompareOrder
        # find mcs
        a._find_mcs("m1", "m2")

        cc1 = set([8, 5, 6, 7])
        cc2 = set([0, 1, 2, 3])
        assert set(a.get_common_core_idx_mol1()) == cc1
        assert set(a.get_common_core_idx_mol2()) == cc2


def test_proposed_mutation_terminal_dummy_real_atom_match():
    from rdkit.Chem import rdFMCS

    for conf in [
        "transformato/tests/config/test-7-CPI-2-CPI-rsfe.yaml",
    ]:
        configuration = load_config_yaml(config=conf, input_dir="data/", output_dir=".")
        s1 = SystemStructure(configuration, "structure1")
        s2 = SystemStructure(configuration, "structure2")

        a = ProposeMutationRoute(s1, s2)
        # find mcs
        a.bondCompare = rdFMCS.BondCompare.CompareOrderExact
        a.completeRingsOnly = True
        a._find_mcs("m1", "m2")
        a.remove_idx_from_common_core_of_mol1([14])
        a.remove_idx_from_common_core_of_mol2([6])

        # find terminal dummy/real atoms
        a._set_common_core_parameters()
        # match terminal real/dummy atoms
        print(a.terminal_real_atom_cc1)
        print(a.terminal_real_atom_cc2)
        assert set(a.terminal_real_atom_cc1) == set([15])
        assert set(a.terminal_dummy_atom_cc1) == set([14])
        assert set(a.terminal_real_atom_cc2) == set([15])
        assert set(a.terminal_dummy_atom_cc2) == set([6])

        match_terminal_atoms_cc1 = a._match_terminal_real_and_dummy_atoms_for_mol1()
        match_terminal_atoms_cc2 = a._match_terminal_real_and_dummy_atoms_for_mol2()

        # are the correct terminal common core atoms identified?
        assert match_terminal_atoms_cc1[15] == set([14])
        assert match_terminal_atoms_cc2[15] == set([6])

        # terminal atoms match between the two common cores
        assert a.matching_terminal_atoms_between_cc[0] == (15, 15)
        # INFO     transformato.mutate:mutate.py:139 Matching terminal atoms from cc1 to cc2. cc1: 0 : cc2: 0
        # INFO     transformato.mutate:mutate.py:139 Matching terminal atoms from cc1 to cc2. cc1: 16 : cc2: 15


@pytest.mark.slowtest
@pytest.mark.skipif(
    os.environ.get("TRAVIS", None) == "true", reason="Skip slow test on travis."
)
def test_find_connected_dummy_regions1():

    from rdkit.Chem import rdFMCS

    conf = "transformato/tests/config/test-7-CPI-2-CPI-rsfe.yaml"
    configuration = load_config_yaml(config=conf, input_dir="data/", output_dir=".")
    s1 = SystemStructure(configuration, "structure1")
    s2 = SystemStructure(configuration, "structure2")

    a = ProposeMutationRoute(s1, s2)
    # find mcs
    a.bondCompare = rdFMCS.BondCompare.CompareOrderExact
    a.completeRingsOnly = True
    a._find_mcs("m1", "m2")
    a.remove_idx_from_common_core_of_mol1([14])
    a.remove_idx_from_common_core_of_mol2([6])

    # find terminal dummy/real atoms
    a._set_common_core_parameters()
    # match terminal real/dummy atoms
    match_terminal_atoms_cc1 = a._match_terminal_real_and_dummy_atoms_for_mol1()
    match_terminal_atoms_cc2 = a._match_terminal_real_and_dummy_atoms_for_mol2()
    e = next(iter(match_terminal_atoms_cc1[15]))
    assert e == 14
    assert len(match_terminal_atoms_cc1[15]) == 1
    e = next(iter(match_terminal_atoms_cc2[15]))
    assert e == 6
    assert len(match_terminal_atoms_cc2[15]) == 1

    # find connected dummy regions
    connected_dummy_regions_cc1 = a._find_connected_dummy_regions(
        "m1", match_terminal_atoms_cc1
    )
    connected_dummy_regions_cc2 = a._find_connected_dummy_regions(
        "m2", match_terminal_atoms_cc2
    )

    lj_default_cc1, lj_default_cc2 = a._match_terminal_dummy_atoms_between_common_cores(
        match_terminal_atoms_cc1,
        match_terminal_atoms_cc2,
    )

    assert set(connected_dummy_regions_cc1[0]) == set(
        [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14]
    )
    assert set(connected_dummy_regions_cc2[0]) == set(
        [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14]
    )


def test_find_connected_dummy_regions2():

    from rdkit.Chem import rdFMCS

    ##################################################
    conf = "transformato/tests/config/test-2oj9-rsfe.yaml"
    configuration = load_config_yaml(config=conf, input_dir="data/", output_dir=".")
    s1 = SystemStructure(configuration, "structure1")
    s2 = SystemStructure(configuration, "structure2")

    a = ProposeMutationRoute(s1, s2)
    # find mcs
    a._find_mcs("m1", "m2")
    a._set_common_core_parameters()
    # match the real/dummy atoms
    match_terminal_atoms_cc1 = a._match_terminal_real_and_dummy_atoms_for_mol1()
    match_terminal_atoms_cc2 = a._match_terminal_real_and_dummy_atoms_for_mol2()

    # find connected dummy regions
    connected_dummy_regions_cc1 = a._find_connected_dummy_regions(
        "m1", match_terminal_atoms_cc1
    )
    connected_dummy_regions_cc2 = a._find_connected_dummy_regions(
        "m2", match_terminal_atoms_cc2
    )

    lj_default_cc1, lj_default_cc2 = a._match_terminal_dummy_atoms_between_common_cores(
        match_terminal_atoms_cc1,
        match_terminal_atoms_cc2,
    )

    dummy_region_m1 = transformato.mutate.DummyRegion(
        "m1",
        match_terminal_atoms_cc1,
        connected_dummy_regions_cc1,
        s1.tlc,
        lj_default=lj_default_cc1,
    )

    print(connected_dummy_regions_cc1)
    print(connected_dummy_regions_cc2)

    dummy_region_m2 = transformato.mutate.DummyRegion(
        "m2",
        match_terminal_atoms_cc2,
        connected_dummy_regions_cc2,
        s2.tlc,
        lj_default=lj_default_cc2,
    )

    print(dummy_region_m1.connected_dummy_regions)
    print(dummy_region_m2.connected_dummy_regions)

    # match
    assert dummy_region_m1.connected_dummy_regions[0] == {
        40,
        42,
        43,
        15,
        18,
        19,
        20,
        21,
    }

    assert dummy_region_m1.connected_dummy_regions[1] == {1}
    assert dummy_region_m2.connected_dummy_regions[0] == {1, 26, 27, 28}
    assert dummy_region_m2.connected_dummy_regions[1] == {25}
    assert dummy_region_m2.connected_dummy_regions[2] == {39}

    # return real atom that connects dummy region to mol
    assert (
        dummy_region_m1.return_connecting_real_atom(
            dummy_region_m1.connected_dummy_regions[0]
        )
        == 16
    )

    assert (
        dummy_region_m1.return_connecting_real_atom(
            dummy_region_m1.connected_dummy_regions[1]
        )
        == 0
    )

    print(f"Matched dummy region: {dummy_region_m1.lj_default}")
    print(f"Matched dummy region: {dummy_region_m2.lj_default}")
    assert dummy_region_m1.lj_default == [1, 18]
    assert dummy_region_m2.lj_default == [1, 39]


def test_common_core_for_multiple_systems():

    for conf in [
        "transformato/tests/config/test-toluene-methane-rsfe.yaml",
        "transformato/tests/config/test-neopentane-methane-rsfe.yaml",
        "transformato/tests/config/test-ethane-methanol-rsfe.yaml",
    ]:
        configuration = load_config_yaml(config=conf, input_dir="data/", output_dir=".")
        s1 = SystemStructure(configuration, "structure1")
        s2 = SystemStructure(configuration, "structure2")

        a = ProposeMutationRoute(s1, s2)
        if conf == "transformato/tests/config/test-neopentane-methane-rsfe.yaml":
            a.propose_common_core()
            a.finish_common_core(
                connected_dummy_regions_cc1=[
                    {0, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16}
                ]
            )
        else:
            # find mcs, find terminal dummy/real atoms, generate charge compensated psfs
            a.calculate_common_core()


def setup_systems(conf):
    configuration = load_config_yaml(config=conf, input_dir="data/", output_dir=".")
    s1 = SystemStructure(configuration, "structure1")
    s2 = SystemStructure(configuration, "structure2")

    s1_to_s2 = ProposeMutationRoute(s1, s2)
    s1_to_s2.calculate_common_core()
    i_s1 = IntermediateStateFactory(
        system=s1,
        configuration=configuration,
    )
    i_s2 = IntermediateStateFactory(
        system=s2,
        configuration=configuration,
    )

    mutation_list_mol1 = s1_to_s2.generate_mutations_to_common_core_for_mol1()
    mutation_list_mol2 = s1_to_s2.generate_mutations_to_common_core_for_mol2()

    return (configuration, mutation_list_mol1, mutation_list_mol2, i_s1, i_s2)


def test_generate_mutation_list_for_multiple_systems():

    for conf, system_name in zip(
        [
            "transformato/tests/config/test-toluene-methane-rsfe.yaml",
            "transformato/tests/config/test-neopentane-methane-rsfe.yaml",
            "transformato/tests/config/test-ethane-methanol-rsfe.yaml",
        ],
        ["toluene-methane", "neopentane-methane", "ethane-ethanol"],
    ):

        if system_name == "ethane-methanol":
            (
                configuration,
                mutation_list_mol1,
                mutation_list_mol2,
                i_s1,
                i_s2,
            ) = setup_systems(conf)
            assert set(mutation_list_mol1.keys()) == set(
                ["charge", "default-lj", "transform"]
            )
        if system_name == "toluene-methane":
            (
                configuration,
                mutation_list_mol1,
                mutation_list_mol2,
                i_s1,
                i_s2,
            ) = setup_systems(conf)
            assert set(mutation_list_mol1.keys()) == set(
                ["charge", "hydrogen-lj", "lj", "transform", "default-lj"]
            )

        if system_name == "neopentane-methane":
            configuration = load_config_yaml(
                config=conf, input_dir="data/", output_dir="."
            )
            s1 = SystemStructure(configuration, "structure1")
            s2 = SystemStructure(configuration, "structure2")

            s1_to_s2 = ProposeMutationRoute(s1, s2)
            s1_to_s2.propose_common_core()
            s1_to_s2.finish_common_core(
                connected_dummy_regions_cc1=[
                    {0, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16}
                ]
            )
            mutation_list = s1_to_s2.generate_mutations_to_common_core_for_mol1()
            assert set(mutation_list.keys()) == set(
                ["charge", "hydrogen-lj", "lj", "transform", "default-lj"]
            )


def test_write_endpoint_state():
    # test physical endpoint systems
    for conf in [
        "transformato/tests/config/test-toluene-methane-rsfe.yaml",
        "transformato/tests/config/test-ethane-methanol-rsfe.yaml",
    ]:
        (
            _,
            _,
            _,
            i_s1,
            i_s2,
        ) = setup_systems(conf)

        output_file_base, _ = i_s1.write_state(mutation_conf=[], intst_nr=1)
        shutil.rmtree(output_file_base)

        output_file_base, _ = i_s2.write_state(mutation_conf=[], intst_nr=1)
        shutil.rmtree(output_file_base)


def test_charges_at_endstate():

    for conf, system_name in zip(
        [
            "transformato/tests/config/test-toluene-methane-rsfe.yaml",
            "transformato/tests/config/test-ethane-methanol-rsfe.yaml",
        ],
        ["toluene-methane" "ethane-ethanol"],
    ):

        # try writing endstate in all directions
        (
            configuration,
            mutation_list_mol1,
            mutation_list_mol2,
            i_s1,
            i_s2,
        ) = setup_systems(conf)
        for i in [i_s1, i_s2]:
            output_file_base, _ = i.write_state(mutation_conf=[], intst_nr=1)

            # original psfs without charge change
            original_psf = {}
            for env in i.system.envs:
                original_psf[env] = copy.deepcopy(i.system.psfs[env])

            for env in i.system.envs:
                offset = i.system.offset[env]

                mutated_psf, param = generate_psf(output_file_base, env)
                for atom in i.system.mol.GetAtoms():
                    idx = atom.GetIdx()
                    print(original_psf[env].atoms[idx + offset].charge)
                    print(mutated_psf.atoms[idx + offset].charge)
                    assert np.isclose(
                        original_psf[env].atoms[idx + offset].charge,
                        mutated_psf.atoms[idx + offset].charge,
                        rtol=1e-3,
                    )
            shutil.rmtree(output_file_base)


def test_setup_dual_junction_system():

    conf = "transformato/tests/config/test-2oj9-rsfe.yaml"
    configuration, mutation_list_mol1, mutation_list_mol2, i_s1, i_s2 = setup_systems(
        conf
    )
    # write out endpoint
    output_files = []
    output_file_base, intst = i_s1.write_state(mutation_conf=[], intst_nr=1)
    output_files.append(output_file_base)

    charges = mutation_list_mol1["charge"]
    # start with charges
    # turn off charges
    output_file_base, intst = i_s1.write_state(
        mutation_conf=charges,
        lambda_value_electrostatic=0.0,
        intst_nr=intst,
    )
    output_files.append(output_file_base)

    # Turn off hydrogens
    hydrogen_lj_mutations = mutation_list_mol1["hydrogen-lj"]
    output_file_base, intst = i_s1.write_state(
        mutation_conf=hydrogen_lj_mutations,
        lambda_value_vdw=0.0,
        intst_nr=intst,
    )
    output_files.append(output_file_base)
    for f in output_files:
        shutil.rmtree(f)


def test_charge_mutation_for_multiple_systems():

    for conf, system_name in zip(
        [
            "transformato/tests/config/test-toluene-methane-rsfe.yaml",
            "transformato/tests/config/test-neopentane-methane-rsfe.yaml",
            "transformato/tests/config/test-ethane-methanol-rsfe.yaml",
        ],
        ["toluene-methane", "neopentane-methane", "ethane-methanol"],
    ):

        configuration = load_config_yaml(config=conf, input_dir="data/", output_dir=".")
        # scale charges with 0.5
        s1 = SystemStructure(configuration, "structure1")
        s2 = SystemStructure(configuration, "structure2")

        s1_to_s2 = ProposeMutationRoute(s1, s2)
        # original psfs without charge change
        original_psf = {}
        for env in s1.envs:
            original_psf[env] = copy.deepcopy(s1.psfs[env])

        for lambda_charge in [1.0, 0.5, 0.25, 0.0]:
            for a, system in zip([s1_to_s2], [s1]):
                if system_name == "neopentane-methane":
                    a.propose_common_core()
                    a.finish_common_core(
                        connected_dummy_regions_cc1=[
                            {0, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16}
                        ]
                    )
                else:
                    a.calculate_common_core()

                i = IntermediateStateFactory(
                    system=system,
                    configuration=configuration,
                )

                mutation_list = a.generate_mutations_to_common_core_for_mol1()
                charges = mutation_list["charge"]
                output_file_base, intst_nr = i.write_state(
                    mutation_conf=charges,
                    lambda_value_electrostatic=lambda_charge,
                    intst_nr=0,
                )
                for env in system.envs:
                    offset = system.offset[env]
                    # read in newly generated psf
                    mutated_psf, params = generate_psf(output_file_base, env)
                    for idx in charges[0].atoms_to_be_mutated:
                        assert np.isclose(
                            original_psf[env].atoms[idx + offset].charge
                            * lambda_charge,
                            mutated_psf.atoms[idx + offset].charge,
                            rtol=1e-03,
                        )

                shutil.rmtree(output_file_base)


def test_vdw_mutation_for_hydrogens_system1():
    # testing terminal lj
    from rdkit.Chem import rdFMCS

    for conf, system_name in zip(
        [
            "transformato/tests/config/test-toluene-methane-rsfe.yaml",
            "transformato/tests/config/test-neopentane-methane-rsfe.yaml",
            "transformato/tests/config/test-ethane-methanol-rsfe.yaml",
        ],
        ["toluene-methane", "neopentane-methane", "ethane-methanol"],
    ):
        configuration = load_config_yaml(config=conf, input_dir="data/", output_dir=".")
        s1 = SystemStructure(configuration, "structure1")
        s2 = SystemStructure(configuration, "structure2")

        s1_to_s2 = ProposeMutationRoute(s1, s2)
        if system_name == "neopentane-methane":
            s1_to_s2.propose_common_core()
            s1_to_s2.finish_common_core(
                connected_dummy_regions_cc1=[
                    {0, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16}
                ]
            )
        else:
            s1_to_s2.calculate_common_core()

        mutation_list = s1_to_s2.generate_mutations_to_common_core_for_mol1()
        i = IntermediateStateFactory(
            system=s1,
            configuration=configuration,
        )

        original_psf = {}
        for env in s1.envs:
            original_psf[env] = copy.deepcopy(s1.psfs[env])
        terminal_lj_mutations = mutation_list["default-lj"]
        output_file_base, intst_nr = i.write_state(
            mutation_conf=terminal_lj_mutations,
            lambda_value_vdw=0.0,
            intst_nr=0,
        )
        print("Set epsilon/rmin to zero for selected atoms")

        for env in s1.envs:
            new_psf, params = generate_psf(output_file_base, env)
            # are the terminal lj parameter correctly set?
            offset = s1.offset[env]
            for terminal_lj in terminal_lj_mutations:
                for idx in terminal_lj.vdw_atom_idx:
                    idxo = idx + offset
                    assert np.isclose(
                        -0.15,
                        new_psf.atoms[idxo].epsilon,
                        rtol=1e-3,
                    )
                    assert np.isclose(
                        1.5,
                        new_psf.atoms[idxo].rmin,
                        rtol=1e-3,
                    )

            # make sure that all other idx are not touched
            for idx in range(len(original_psf[env].atoms)):
                idxo = idx - offset  # NOTE: the '-'
                if idxo not in terminal_lj.vdw_atom_idx:
                    assert np.isclose(
                        original_psf[env].atoms[idx].epsilon,
                        new_psf.atoms[idx].epsilon,
                        rtol=1e-3,
                    )
                    assert np.isclose(
                        original_psf[env].atoms[idx].rmin,
                        new_psf.atoms[idx].rmin,
                        rtol=1e-3,
                    )

        shutil.rmtree(output_file_base)


def test_vdw_mutation_for_hydrogens_system2():
    from rdkit.Chem import rdFMCS

    for conf, system_name in zip(
        [
            "transformato/tests/config/test-7-CPI-2-CPI-rsfe.yaml",
        ],
        ["7-CPI-2-CPI"],
    ):
        configuration = load_config_yaml(config=conf, input_dir="data/", output_dir=".")
        s1 = SystemStructure(configuration, "structure1")
        s2 = SystemStructure(configuration, "structure2")

        s1_to_s2 = ProposeMutationRoute(s1, s2)
        s1_to_s2.bondCompare = rdFMCS.BondCompare.CompareOrderExact
        s1_to_s2.completeRingsOnly = True
        s1_to_s2.calculate_common_core()

        mutation_list = s1_to_s2.generate_mutations_to_common_core_for_mol1()
        i = IntermediateStateFactory(
            system=s1,
            configuration=configuration,
        )

        original_psf = {}
        for env in s1.envs:
            original_psf[env] = copy.deepcopy(s1.psfs[env])

        for lambda_value in [1.0, 0.5, 0.25]:
            # lj hydrogens scaling
            hydrogen_lj_mutations = mutation_list["hydrogen-lj"]
            output_file_base, _ = i.write_state(
                mutation_conf=hydrogen_lj_mutations,
                lambda_value_vdw=lambda_value,
                intst_nr=0,
            )
            print("Set epsilon/rmin for selected atoms")

            for env in s1.envs:
                new_psf, params = generate_psf(output_file_base, env)
                # are the terminal lj parameter correctly set?
                offset = s1.offset[env]
                for hydrogen_lj in hydrogen_lj_mutations:
                    for idx in hydrogen_lj.vdw_atom_idx:
                        idxo = idx + offset
                        assert np.isclose(
                            original_psf[env].atoms[idxo].epsilon * lambda_value,
                            new_psf.atoms[idxo].epsilon,
                            rtol=1e-3,
                        )
                        assert np.isclose(
                            original_psf[env].atoms[idxo].rmin * lambda_value,
                            new_psf.atoms[idxo].rmin,
                            rtol=1e-3,
                        )

                # make sure that all other idx are not touched
                for idx in range(len(original_psf[env].atoms)):
                    idxo = idx - offset  # NOTE: the '-'
                    if idxo not in hydrogen_lj.vdw_atom_idx:
                        assert np.isclose(
                            original_psf[env].atoms[idx].epsilon,
                            new_psf.atoms[idx].epsilon,
                            rtol=1e-3,
                        )
                        assert np.isclose(
                            original_psf[env].atoms[idx].rmin,
                            new_psf.atoms[idx].rmin,
                            rtol=1e-3,
                        )

            shutil.rmtree(output_file_base)


@pytest.mark.slowtest
def test_bonded_mutation():

    for conf in [
        "transformato/tests/config/test-toluene-methane-rsfe.yaml",
    ]:

        configuration = load_config_yaml(config=conf, input_dir="data/", output_dir=".")
        s1 = SystemStructure(configuration, "structure1")
        s2 = SystemStructure(configuration, "structure2")

        s1_to_s2 = ProposeMutationRoute(s1, s2)
        s1_to_s2.calculate_common_core()

        mutation_list = s1_to_s2.generate_mutations_to_common_core_for_mol1()
        i = IntermediateStateFactory(
            system=s1,
            configuration=configuration,
        )
        original_psf = {}
        output_files = []
        for env in s1.envs:
            original_psf[env] = copy.deepcopy(s1.psfs[env])

        # mutate everything else before touching bonded terms
        charges = mutation_list["charge"]
        # turn off charges
        output_file_base, intst_nr = i.write_state(
            mutation_conf=charges,
            lambda_value_electrostatic=0.0,
            intst_nr=1,
        )
        output_files.append(output_file_base)

        # Turn of hydrogens
        hydrogen_lj_mutations = mutation_list["hydrogen-lj"]
        output_file_base, intst_nr = i.write_state(
            mutation_conf=hydrogen_lj_mutations,
            lambda_value_vdw=0.0,
            intst_nr=intst_nr,
        )
        output_files.append(output_file_base)

        # turn off heavy atoms
        output_file_base, intst_nr = i.write_state(
            mutation_conf=mutation_list["lj"],
            lambda_value_vdw=0.0,
            intst_nr=intst_nr,
        )
        output_files.append(output_file_base)

        # generate terminal lj
        output_file_base, intst_nr = i.write_state(
            mutation_conf=mutation_list["default-lj"],
            lambda_value_vdw=0.0,
            intst_nr=intst_nr,
        )
        output_files.append(output_file_base)

        m = mutation_list["transform"]
        for lambda_value in np.linspace(0.75, 0, 3):
            print(lambda_value)
            # turn off charges
            output_file_base, intst_nr = i.write_state(
                mutation_conf=m,
                common_core_transformation=lambda_value,
                intst_nr=intst_nr,
            )
            output_files.append(output_file_base)
        shutil.rmtree(output_file_base)


@pytest.mark.slowtest
def test_equivalent_endstates_vacuum():

    import simtk.openmm as mm
    import simtk.openmm.app as app

    # ParmEd Imports
    from parmed import unit as u

    output_files_t1, output_files_t2 = _set_output_files_2oj9_tautomer_pair()

    env = "vacuum"
    psf, parms = generate_psf(output_files_t1[-1], env)
    coord = generate_crd(output_files_t1[-1], env).positions
    mask = []
    nr_of_atoms = len(psf.atoms)
    assert nr_of_atoms == len(coord)

    for a in psf.atoms:
        if str(a.type).startswith("DD"):
            mask.append(False)
        else:
            mask.append(True)
    psf = psf[mask]
    for idx, e in enumerate(mask):
        if not e:
            print(coord.pop(idx))

    system = psf.createSystem(parms, nonbondedMethod=app.NoCutoff)

    # Create the integrator to do Langevin dynamics
    integrator = mm.LangevinIntegrator(
        300 * u.kelvin,  # Temperature of heat bath
        1.0 / u.picoseconds,  # Friction coefficient
        2.0 * u.femtoseconds,  # Time step
    )
    assert nr_of_atoms == psf.topology.getNumAtoms() + 1
    print(psf.topology.getNumAtoms())
    # Create the Simulation object
    sim = app.Simulation(psf.topology, system, integrator)
    # Set the particle positions
    sim.context.setPositions(coord)
    e1 = (
        sim.context.getState(getEnergy=True)
        .getPotentialEnergy()
        .value_in_unit(unit.kilocalorie_per_mole)
    )

    #####################################
    #####################################
    psf, parms = generate_psf(output_files_t2[-1], env)
    # coord = generate_crd(output_files_t2[-1], env).positions
    mask = []
    nr_of_atoms = len(psf.atoms)
    assert nr_of_atoms == len(coord) + 1

    mask = []
    for a in psf.atoms:
        if str(a.type).startswith("DD"):
            mask.append(False)
        else:
            mask.append(True)
    psf = psf[mask]

    system = psf.createSystem(parms, nonbondedMethod=app.NoCutoff)
    # Create the integrator to do Langevin dynamics
    integrator = mm.LangevinIntegrator(
        300 * u.kelvin,  # Temperature of heat bath
        1.0 / u.picoseconds,  # Friction coefficient
        2.0 * u.femtoseconds,  # Time step
    )

    # Create the Simulation object
    sim = app.Simulation(psf.topology, system, integrator)
    # Set the particle positions
    sim.context.setPositions(coord)
    e2 = (
        sim.context.getState(getEnergy=True)
        .getPotentialEnergy()
        .value_in_unit(unit.kilocalorie_per_mole)
    )

    assert np.isclose(e1, e2)


@pytest.mark.slowtest
def test_equivalent_endstates_waterbox():

    import simtk.openmm as mm
    import simtk.openmm.app as app

    # ParmEd Imports
    from parmed import unit as u

    output_files_t1, output_files_t2 = _set_output_files_2oj9_tautomer_pair()

    env = "waterbox"
    psf, parms = generate_psf(output_files_t1[-1], env)
    coords = generate_crd(output_files_t1[-1], env).positions

    psf.box = (30, 30, 30, 90, 90, 90)

    mask = []
    nr_of_atoms = len(psf.atoms)
    assert nr_of_atoms == len(coords)

    for a in psf.atoms:
        if str(a.type).startswith("DD"):
            mask.append(False)
        else:
            mask.append(True)

    psf = psf[mask]
    nr_of_atoms_t1 = psf.topology.getNumAtoms()

    new_coords = []
    for idx, e in enumerate(mask):
        if e:
            new_coords.append(coords[idx])

    coords = new_coords
    system = psf.createSystem(
        parms,
        nonbondedMethod=app.PME,
        nonbondedCutoff=12.0 * u.angstroms,
        switchDistance=10.0 * u.angstroms,
    )

    # Create the integrator to do Langevin dynamics
    integrator = mm.LangevinIntegrator(
        300 * u.kelvin,  # Temperature of heat bath
        1.0 / u.picoseconds,  # Friction coefficient
        2.0 * u.femtoseconds,  # Time step
    )
    assert nr_of_atoms == psf.topology.getNumAtoms() + 1
    # Create the Simulation object
    sim = app.Simulation(psf.topology, system, integrator)
    # Set the particle positions
    sim.context.setPositions(coords)
    e1 = (
        sim.context.getState(getEnergy=True)
        .getPotentialEnergy()
        .value_in_unit(unit.kilocalorie_per_mole)
    )
    nr_of_atoms_t1 = psf.topology.getNumAtoms()
    #####################################
    #####################################
    psf, parms = generate_psf(output_files_t2[-1], env)
    psf.box = (30, 30, 30, 90, 90, 90)
    # coords = generate_crd(output_files_t2[-1], env).positions

    mask = []
    for a in psf.atoms:
        if str(a.type).startswith("DD"):
            mask.append(False)
        else:
            mask.append(True)

    # remove waters from top
    mask[-12:] = [False] * 12
    psf = psf[mask]
    nr_of_atoms_t2 = psf.topology.getNumAtoms()
    assert nr_of_atoms_t1 == nr_of_atoms_t2

    system = psf.createSystem(
        parms,
        nonbondedMethod=app.PME,
        nonbondedCutoff=12.0 * u.angstroms,
        switchDistance=10.0 * u.angstroms,
    )
    # Create the integrator to do Langevin dynamics
    integrator = mm.LangevinIntegrator(
        300 * u.kelvin,  # Temperature of heat bath
        1.0 / u.picoseconds,  # Friction coefficient
        2.0 * u.femtoseconds,  # Time step
    )

    # Create the Simulation object
    sim = app.Simulation(psf.topology, system, integrator)
    # Set the particle positions
    sim.context.setPositions(coords)
    e2 = (
        sim.context.getState(getEnergy=True)
        .getPotentialEnergy()
        .value_in_unit(unit.kilocalorie_per_mole)
    )
    assert np.isclose(e1, e2)


@pytest.mark.slowtest
def test_bonded_mutation_energies_t1_s1(caplog):
    caplog.set_level(logging.CRITICAL)
    output_files_t1, _ = _set_output_files_2oj9_tautomer_pair()
    e_t1_s1 = (
        generate_sim(output_files_t1[0], "vacuum")
        .context.getState(getEnergy=True)
        .getPotentialEnergy()
    )
    assert np.isclose(
        e_t1_s1.value_in_unit(unit.kilocalorie_per_mole), -17.638044396797515
    )


@pytest.mark.slowtest
def test_bonded_mutation_energies_t1_s2(caplog):
    caplog.set_level(logging.CRITICAL)
    output_files_t1, _ = _set_output_files_2oj9_tautomer_pair()

    e_t1_s2 = (
        generate_sim(output_files_t1[1], "vacuum")
        .context.getState(getEnergy=True)
        .getPotentialEnergy()
    )

    assert np.isclose(
        e_t1_s2.value_in_unit(unit.kilocalorie_per_mole), 5.50502085150725
    )


@pytest.mark.slowtest
def test_bonded_mutation_energies_t1_s3(caplog):
    caplog.set_level(logging.CRITICAL)
    output_files_t1, _ = _set_output_files_2oj9_tautomer_pair()

    e_t1_s3 = (
        generate_sim(output_files_t1[2], "vacuum")
        .context.getState(getEnergy=True)
        .getPotentialEnergy()
    )

    assert np.isclose(
        e_t1_s3.value_in_unit(unit.kilocalorie_per_mole), 5.680121579945277
    )


@pytest.mark.slowtest
def test_bonded_mutation_energies_t1_s4(caplog):
    caplog.set_level(logging.CRITICAL)
    output_files_t1, _ = _set_output_files_2oj9_tautomer_pair()

    e_t1_s4 = (
        generate_sim(output_files_t1[3], "vacuum")
        .context.getState(getEnergy=True)
        .getPotentialEnergy()
    )

    assert np.isclose(
        e_t1_s4.value_in_unit(unit.kilocalorie_per_mole), 19.32431518895557
    )


@pytest.mark.slowtest
def test_bonded_mutation_energies_t1_s5(caplog):
    caplog.set_level(logging.CRITICAL)
    output_files_t1, _ = _set_output_files_2oj9_tautomer_pair()

    e_t1_s5 = (
        generate_sim(output_files_t1[4], "vacuum")
        .context.getState(getEnergy=True)
        .getPotentialEnergy()
    )

    assert np.isclose(
        e_t1_s5.value_in_unit(unit.kilocalorie_per_mole), 39.24392077135464
    )


@pytest.mark.slowtest
def test_bonded_mutation_energies_t1_s6(caplog):
    caplog.set_level(logging.CRITICAL)
    output_files_t1, _ = _set_output_files_2oj9_tautomer_pair()

    e_t1_s6 = (
        generate_sim(output_files_t1[5], "vacuum")
        .context.getState(getEnergy=True)
        .getPotentialEnergy()
    )

    assert np.isclose(
        e_t1_s6.value_in_unit(unit.kilocalorie_per_mole), 54.94647553965466
    )


@pytest.mark.slowtest
def test_bonded_mutation_energies_t1_s7(caplog):
    caplog.set_level(logging.CRITICAL)
    output_files_t1, _ = _set_output_files_2oj9_tautomer_pair()

    e_t1_s7 = (
        generate_sim(output_files_t1[6], "vacuum")
        .context.getState(getEnergy=True)
        .getPotentialEnergy()
    )

    assert np.isclose(
        e_t1_s7.value_in_unit(unit.kilocalorie_per_mole), 62.026441265363836
    )


@pytest.mark.slowtest
def test_bonded_mutation_energies_t2_s1(caplog):
    caplog.set_level(logging.CRITICAL)
    _, output_files_t2 = _set_output_files_2oj9_tautomer_pair()

    e_t2_s1 = (
        generate_sim(output_files_t2[0], "vacuum")
        .context.getState(getEnergy=True)
        .getPotentialEnergy()
    )

    assert np.isclose(
        e_t2_s1.value_in_unit(unit.kilocalorie_per_mole), 12.152228076282555, rtol=1e-4
    )


@pytest.mark.slowtest
def test_bonded_mutation_energies_t2_s2(caplog):
    caplog.set_level(logging.CRITICAL)
    _, output_files_t2 = _set_output_files_2oj9_tautomer_pair()

    e_t2_s2 = (
        generate_sim(output_files_t2[1], "vacuum")
        .context.getState(getEnergy=True)
        .getPotentialEnergy()
    )

    assert np.isclose(
        e_t2_s2.value_in_unit(unit.kilocalorie_per_mole), 44.88436132326585
    )


@pytest.mark.slowtest
def test_bonded_mutation_energies_t2_s3(caplog):
    caplog.set_level(logging.CRITICAL)
    _, output_files_t2 = _set_output_files_2oj9_tautomer_pair()

    e_t2_s3 = (
        generate_sim(output_files_t2[2], "vacuum")
        .context.getState(getEnergy=True)
        .getPotentialEnergy()
    )

    assert np.isclose(
        e_t2_s3.value_in_unit(unit.kilocalorie_per_mole), 45.06542169452752
    )


@pytest.mark.slowtest
def test_bonded_mutation_atoms(caplog):
    caplog.set_level(logging.CRITICAL)

    from .test_mutation import setup_2OJ9_tautomer_pair_rsfe

    (output_files_t1, output_files_t2), _, p = setup_2OJ9_tautomer_pair_rsfe()
    psf_at_endstate_t1, _ = generate_psf(output_files_t1[0], "vacuum")
    prm_at_endstate_t1 = {
        a.idx: (a.charge, a.sigma, a.epsilon) for a in psf_at_endstate_t1.atoms
    }
    psf_at_t1_cc, _ = generate_psf(output_files_t1[-1], "vacuum")
    prm_at_t1_cc = {a.idx: (a.charge, a.sigma, a.epsilon) for a in psf_at_t1_cc.atoms}
    psf_at_t2_cc, _ = generate_psf(output_files_t2[-1], "vacuum")
    prm_at_t2_cc = {a.idx: (a.charge, a.sigma, a.epsilon) for a in psf_at_t2_cc.atoms}
    psf_at_endstate_t2, _ = generate_psf(output_files_t2[0], "vacuum")
    prm_at_endstate_t2 = {
        a.idx: (a.charge, a.sigma, a.epsilon) for a in psf_at_endstate_t2.atoms
    }
    print("@@@@@@@@@@@@@@@@@@")

    cc1_idx, cc2_idx = p.get_common_core_idx_mol1(), p.get_common_core_idx_mol2()
    for atom_t1_idx, atom_t2_idx in zip(cc1_idx, cc2_idx):
        atom_t1 = prm_at_t1_cc[atom_t1_idx]
        atom_t2 = prm_at_t2_cc[atom_t2_idx]
        assert atom_t1 == atom_t2

    for atom_t1 in psf_at_t1_cc.atoms:
        if atom_t1.idx not in p.get_common_core_idx_mol1():
            print(atom_t1)
    for atom_t2 in psf_at_t2_cc.atoms:
        if atom_t2.idx not in p.get_common_core_idx_mol2():
            print(atom_t2)

    for atom_t1_idx, atom_t2_idx in zip(cc1_idx, cc2_idx):
        atom_t1 = prm_at_endstate_t1[atom_t1_idx]
        atom_t2 = prm_at_endstate_t2[atom_t2_idx]
        try:
            assert atom_t1 == atom_t2
        except AssertionError:
            pass


@pytest.mark.slowtest
def test_bonded_mutation_bonds(caplog):
    caplog.set_level(logging.CRITICAL)

    from .test_mutation import setup_2OJ9_tautomer_pair_rsfe

    (output_files_t1, output_files_t2), _, p = setup_2OJ9_tautomer_pair_rsfe()
    ##################
    psf_at_endstate_t1, _ = generate_psf(output_files_t1[0], "vacuum")
    prm_at_endstate_t1 = {
        idx: (a.type.k, a.type.req) for idx, a in enumerate(psf_at_endstate_t1.bonds)
    }
    ##################
    psf_at_t1_cc, _ = generate_psf(output_files_t1[-1], "vacuum")
    prm_at_t1_cc = {
        idx: (a.type.k, a.type.req) for idx, a in enumerate(psf_at_t1_cc.bonds)
    }
    ##################
    psf_at_t2_cc, _ = generate_psf(output_files_t2[-1], "vacuum")
    prm_at_t2_cc = {
        idx: (a.type.k, a.type.req) for idx, a in enumerate(psf_at_t2_cc.bonds)
    }
    ##################
    psf_at_endstate_t2, _ = generate_psf(output_files_t2[0], "vacuum")
    prm_at_endstate_t2 = {
        idx: (a.type.k, a.type.req) for idx, a in enumerate(psf_at_endstate_t2.bonds)
    }  ##################
    print("@@@@@@@@@@@@@@@@@@")

    cc1_idx, cc2_idx = p.get_common_core_idx_mol1(), p.get_common_core_idx_mol2()
    # compare cc enstates
    for bond_t1_idx, bond_t1 in enumerate(psf_at_t1_cc.bonds):
        atom1_t1_idx = bond_t1.atom1.idx
        atom2_t1_idx = bond_t1.atom2.idx
        if atom1_t1_idx not in cc1_idx or atom2_t1_idx not in cc1_idx:
            continue

        # get index in common core
        idx1 = cc1_idx.index(atom1_t1_idx)
        idx2 = cc1_idx.index(atom2_t1_idx)

        atom1_t2 = psf_at_t2_cc[cc2_idx[idx1]]
        atom2_t2 = psf_at_t2_cc[cc2_idx[idx2]]
        bond_t2 = None
        for bond_t2_idx, bond_t2 in enumerate(psf_at_t2_cc.bonds):
            if atom1_t2 in bond_t2 and atom2_t2 in bond_t2:
                prm_at_t1_cc[bond_t1_idx] = prm_at_t2_cc[bond_t2_idx]


@pytest.mark.slowtest
def test_bonded_mutation_angles(caplog):
    caplog.set_level(logging.CRITICAL)
    from copy import copy
    from .test_mutation import setup_2OJ9_tautomer_pair_rsfe

    (output_files_t1, output_files_t2), _, p = setup_2OJ9_tautomer_pair_rsfe()
    ##################
    # psf_at_endstate_t1 = generate_psf(output_files_t1[0], "vacuum")
    # prm_at_endstate_t1 = {
    #     idx: (copy(a.type.k), copy(a.type.theteq))
    #     for idx, a in enumerate(psf_at_endstate_t1.angles)
    # }
    # ##################
    # psf_at_endstate_t2 = generate_psf(output_files_t2[0], "vacuum")
    # prm_at_endstate_t2 = {
    #     idx: (copy(a.type.k), copy(a.type.theteq))
    #     for idx, a in enumerate(psf_at_endstate_t2.angles)
    # }
    ##################
    psf_at_t1_cc, _ = generate_psf(output_files_t1[-1], "vacuum")
    prm_at_t1_cc = {
        idx: (copy(a.type.k), copy(a.type.theteq))
        for idx, a in enumerate(psf_at_t1_cc.angles)
    }
    ##################
    psf_at_t2_cc, _ = generate_psf(output_files_t2[-1], "vacuum")
    prm_at_t2_cc = {
        idx: (copy(a.type.k), copy(a.type.theteq))
        for idx, a in enumerate(psf_at_t2_cc.angles)
    }
    print("@@@@@@@@@@@@@@@@@@")
    ##################

    cc1_idx, cc2_idx = p.get_common_core_idx_mol1(), p.get_common_core_idx_mol2()
    for angle_t1_idx, angle_t1 in enumerate(psf_at_t1_cc.angles):
        atom1_t1_idx = angle_t1.atom1.idx
        atom2_t1_idx = angle_t1.atom2.idx
        atom3_t1_idx = angle_t1.atom3.idx
        if (
            atom1_t1_idx not in cc1_idx
            or atom2_t1_idx not in cc1_idx
            or atom3_t1_idx not in cc1_idx
        ):
            continue

        # get index in common core
        idx1 = cc1_idx.index(atom1_t1_idx)
        idx2 = cc1_idx.index(atom2_t1_idx)
        idx3 = cc1_idx.index(atom3_t1_idx)

        atom1_t2 = psf_at_t2_cc[cc2_idx[idx1]]
        atom2_t2 = psf_at_t2_cc[cc2_idx[idx2]]
        atom3_t2 = psf_at_t2_cc[cc2_idx[idx3]]
        faulty = False
        for angle_t2_idx, angle_t2 in enumerate(psf_at_t2_cc.angles):
            if atom1_t2 in angle_t2 and atom2_t2 in angle_t2 and atom3_t2 in angle_t2:

                if not (prm_at_t1_cc[angle_t1_idx] == prm_at_t2_cc[angle_t2_idx]):

                    print("###################")
                    print(prm_at_t1_cc[angle_t1_idx])
                    print(prm_at_t2_cc[angle_t2_idx])
                    print(angle_t1)
                    print(angle_t2)
                    faulty = True
        if faulty:
            raise AssertionError()


@pytest.mark.slowtest
def test_bonded_mutation_dihedrals(caplog):
    caplog.set_level(logging.CRITICAL)

    from .test_mutation import setup_2OJ9_tautomer_pair_rsfe

    (output_files_t1, output_files_t2), _, p = setup_2OJ9_tautomer_pair_rsfe()
    # ##################
    # psf_at_endstate_t1 = generate_psf(output_files_t1[0], "vacuum")
    # prm_at_endstate_t1 = {
    #     idx: [(a.type.phi_k, a.type.per, a.type.phase) for a in l.type]
    #     for idx, l in enumerate(psf_at_endstate_t1.dihedrals)
    # }
    # ##################
    # psf_at_endstate_t2 = generate_psf(output_files_t2[0], "vacuum")
    # prm_at_endstate_t2 = {
    #     idx: [(a.type.phi_k, a.type.per, a.type.phase) for a in l]
    #     for idx, a in enumerate(psf_at_endstate_t2.dihedrals)
    # }
    ####################################
    ####################################
    psf_at_t1_cc, _ = generate_psf(output_files_t1[-1], "vacuum")
    prm_at_t1_cc = {
        idx: [(a.phi_k, a.per, a.phase) for a in l.type]
        for idx, l in enumerate(psf_at_t1_cc.dihedrals)
    }
    t1_cc_nr_of_dihedrals = len(psf_at_t1_cc.dihedrals)
    e_at_t1_cc = (
        generate_sim(output_files_t1[-1], "vacuum")
        .context.getState(getEnergy=True)
        .getPotentialEnergy()
    )
    assert np.isclose(
        e_at_t1_cc.value_in_unit(unit.kilocalorie_per_mole), 62.026441265363836
    )
    ####################################
    ####################################
    psf_at_t2_cc, _ = generate_psf(output_files_t2[-1], "vacuum")
    prm_at_t2_cc = {
        idx: [(a.phi_k, a.per, a.phase) for a in l.type]
        for idx, l in enumerate(psf_at_t2_cc.dihedrals)
    }
    t2_cc_nr_of_dihedrals = len(psf_at_t2_cc.dihedrals)
    e_at_t2_cc = (
        generate_sim(output_files_t2[-1], "vacuum")
        .context.getState(getEnergy=True)
        .getPotentialEnergy()
    )
    assert np.isclose(
        e_at_t2_cc.value_in_unit(unit.kilocalorie_per_mole), 63.36457961831904
    )
    ####################################
    ####################################
    assert t1_cc_nr_of_dihedrals == t2_cc_nr_of_dihedrals
    # compare the common core parameters at the cc state
    cc1_idx, cc2_idx = p.get_common_core_idx_mol1(), p.get_common_core_idx_mol2()

    for dihedral_t1_idx, dihedral_t1 in enumerate(psf_at_t1_cc.dihedrals):
        atom1_t1_idx = dihedral_t1.atom1.idx
        atom2_t1_idx = dihedral_t1.atom2.idx
        atom3_t1_idx = dihedral_t1.atom3.idx
        atom4_t1_idx = dihedral_t1.atom4.idx
        if (
            atom1_t1_idx not in cc1_idx
            or atom2_t1_idx not in cc1_idx
            or atom3_t1_idx not in cc1_idx
            or atom4_t1_idx not in cc1_idx
        ):
            print("Not present in cc2:")
            print(dihedral_t1)
            print("#####################")
            continue

        # get index in common core
        idx1 = cc1_idx.index(atom1_t1_idx)
        idx2 = cc1_idx.index(atom2_t1_idx)
        idx3 = cc1_idx.index(atom3_t1_idx)
        idx4 = cc1_idx.index(atom4_t1_idx)

        atom1_t2 = psf_at_t2_cc[cc2_idx[idx1]]
        atom2_t2 = psf_at_t2_cc[cc2_idx[idx2]]
        atom3_t2 = psf_at_t2_cc[cc2_idx[idx3]]
        atom4_t2 = psf_at_t2_cc[cc2_idx[idx4]]

        dihedral_t2 = None
        faulty = False
        for dihedral_t2_idx, dihedral in enumerate(psf_at_t2_cc.dihedrals):
            if (
                atom1_t2 in dihedral
                and atom2_t2 in dihedral
                and atom3_t2 in dihedral
                and atom4_t2 in dihedral
            ):
                dihedral_t2 = dihedral
                break
        else:
            print("Not present in cc1:")
            print(dihedral)
            print("#####################")

        assert dihedral_t2 != None
        if not (prm_at_t1_cc[dihedral_t1_idx] == prm_at_t2_cc[dihedral_t2_idx]):

            print("###################")
            print(prm_at_t1_cc[dihedral_t1_idx])
            print(prm_at_t2_cc[dihedral_t2_idx])
            print(dihedral_t1)
            print(dihedral_t2)
            faulty = True
        if faulty:
            raise AssertionError()


def test_vdw_mutation_for_hydrogens_and_heavy_atoms():
    from rdkit.Chem import rdFMCS

    for conf, system_name in zip(
        [
            "transformato/tests/config/test-toluene-methane-rsfe.yaml",
            "transformato/tests/config/test-neopentane-methane-rsfe.yaml",
            "transformato/tests/config/test-ethane-methanol-rsfe.yaml",
        ],
        ["toluene-methane", "neopentane-methane", "ethane-methanol"],
    ):
        configuration = load_config_yaml(config=conf, input_dir="data/", output_dir=".")
        s1 = SystemStructure(configuration, "structure1")
        s2 = SystemStructure(configuration, "structure2")

        s1_to_s2 = ProposeMutationRoute(s1, s2)
        if system_name == "neopentane-methane":
            s1_to_s2.propose_common_core()
            s1_to_s2.finish_common_core(
                connected_dummy_regions_cc1=[
                    {0, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16}
                ]
            )
        elif system_name == "7-CPI-2-CPI":
            # find mcs
            s1_to_s2.bondCompare = rdFMCS.BondCompare.CompareOrderExact
            s1_to_s2.completeRingsOnly = True
            s1_to_s2.calculate_common_core()
        else:
            s1_to_s2.calculate_common_core()

        mutation_list = s1_to_s2.generate_mutations_to_common_core_for_mol1()
        i = IntermediateStateFactory(
            system=s1,
            configuration=configuration,
        )

        original_psf = {}
        for env in s1.envs:
            original_psf[env] = copy.deepcopy(s1.psfs[env])

        #
        intst_nr = 1
        for lambda_vdw in [1.0, 0.0]:
            print(f"Lambda: {lambda_vdw}")
            output_files = []
            all_atoms_for_which_lj_turned_off = []
            print(mutation_list.keys())
            # turn off hydrogen lj
            hydrogen_lj_mutations = mutation_list["hydrogen-lj"]
            print(
                f"Turn off lj for hydrogen atoms : {[e.atoms_to_be_mutated for e in hydrogen_lj_mutations]}"
            )
            output_file_base, intst_nr = i.write_state(
                mutation_conf=hydrogen_lj_mutations,
                lambda_value_vdw=lambda_vdw,
                intst_nr=intst_nr,
            )
            output_files.append(output_file_base)
            for mutation in hydrogen_lj_mutations:
                all_atoms_for_which_lj_turned_off.extend(mutation.vdw_atom_idx)

            # turn off heavy atom lj
            for mutation in mutation_list["lj"]:
                print(f"Turn off lj for heavy atom : {mutation.atoms_to_be_mutated}")

                output_file_base, intst_nr = i.write_state(
                    mutation_conf=[mutation],
                    lambda_value_vdw=lambda_vdw,
                    intst_nr=intst_nr,
                )
                all_atoms_for_which_lj_turned_off.extend(mutation.vdw_atom_idx)
                output_files.append(output_file_base)

            # change to default lj
            terminal_lj_mutations = mutation_list["default-lj"]
            terminal_idx = []
            for mutation in terminal_lj_mutations:
                all_atoms_for_which_lj_turned_off.extend(mutation.vdw_atom_idx)
                terminal_idx.extend(mutation.vdw_atom_idx)
            print(
                f"Turn off lj for terminal atom : {all_atoms_for_which_lj_turned_off}"
            )

            output_file_base, intst_nr = i.write_state(
                mutation_conf=terminal_lj_mutations,
                lambda_value_vdw=0.0,
                intst_nr=intst_nr,
            )
            output_files.append(output_file_base)

            print(f"Set epsilon/rmin to base * {lambda_vdw} for selected atoms")

            print(all_atoms_for_which_lj_turned_off)
            for env in s1.envs:
                print(env)
                # read in generated psf at last mutation step
                new_psf, params = generate_psf(output_file_base, env)
                offset = s1.offset[env]
                for idx in all_atoms_for_which_lj_turned_off:
                    idxo = idx + offset
                    # match terminal jl
                    if idx in terminal_idx:
                        assert np.isclose(
                            -0.15,
                            new_psf.atoms[idxo].epsilon,
                            rtol=1e-3,
                        )
                        assert np.isclose(
                            1.5,
                            new_psf.atoms[idxo].rmin,
                            rtol=1e-3,
                        )
                    else:
                        # match all other lj
                        assert np.isclose(
                            original_psf[env].atoms[idxo].epsilon * lambda_vdw,
                            new_psf.atoms[idxo].epsilon,
                            rtol=1e-3,
                        )
                        assert np.isclose(
                            original_psf[env].atoms[idxo].rmin * lambda_vdw,
                            new_psf.atoms[idxo].rmin,
                            rtol=1e-3,
                        )

                # make sure that all other idx are not touched
                for idx in range(s1.mol.GetNumAtoms()):
                    if idx not in all_atoms_for_which_lj_turned_off:
                        print(idx)
                        assert np.isclose(
                            original_psf[env].atoms[idx].epsilon,
                            new_psf.atoms[idx].epsilon,
                            rtol=1e-3,
                        )
                        assert np.isclose(
                            original_psf[env].atoms[idx].rmin,
                            new_psf.atoms[idx].rmin,
                            rtol=1e-3,
                        )

        shutil.rmtree(f"{system_name}-rsfe")


def setup_2OJ9_tautomer_pair_rsfe(
    single_state=False, conf_path="", nr_of_bonded_windows=4
):
    from ..mutate import mutate_pure_tautomers
    from ..constants import check_platform

    if not conf_path:
        conf_path = "transformato/tests/config/test-2oj9-tautomer-pair-rsfe.yaml"
    else:
        print(conf_path)

    configuration = load_config_yaml(
        config=conf_path, input_dir="data/", output_dir="."
    )
    check_platform(configuration)

    s1 = SystemStructure(configuration, "structure1")
    s2 = SystemStructure(configuration, "structure2")
    s1_to_s2 = ProposeMutationRoute(s1, s2)
    s1_to_s2.calculate_common_core()
    return (
        mutate_pure_tautomers(
            s1_to_s2,
            s1,
            s2,
            configuration,
            single_state=single_state,
            nr_of_bonded_windows=4,
        ),
        configuration,
        s1_to_s2,
    )


def setup_2OJ9_tautomer_pair_rbfe(
    single_state=False, conf_path="", nr_of_bonded_windows=4
):
    from ..mutate import mutate_pure_tautomers
    from ..constants import check_platform

    if not conf_path:
        conf_path = "transformato/tests/config/test-2oj9-tautomer-pair-rsfe.yaml"
    else:
        print(conf_path)

    configuration = load_config_yaml(
        config=conf_path, input_dir="data/", output_dir="."
    )
    check_platform(configuration)

    s1 = SystemStructure(configuration, "structure1")
    s2 = SystemStructure(configuration, "structure2")
    s1_to_s2 = ProposeMutationRoute(s1, s2)
    s1_to_s2.calculate_common_core()
    return (
        mutate_pure_tautomers(
            s1_to_s2,
            s1,
            s2,
            configuration,
            single_state=single_state,
            nr_of_bonded_windows=4,
        ),
        configuration,
        s1_to_s2,
    )


def setup_acetylacetone_tautomer_pair(
    single_state=False, conf_path="", nr_of_bonded_windows=4
):
    from ..mutate import mutate_pure_tautomers
    from ..constants import check_platform

    if not conf_path:
        conf_path = "transformato/tests/config/test-acetylacetone-tautomer-rsfe.yaml"
    else:
        print(conf_path)

    configuration = load_config_yaml(
        config=conf_path, input_dir="data/", output_dir="."
    )
    check_platform(configuration)

    s1 = SystemStructure(configuration, "structure1")
    s2 = SystemStructure(configuration, "structure2")
    s1_to_s2 = ProposeMutationRoute(s1, s2)
    s1_to_s2.calculate_common_core()
    return (
        mutate_pure_tautomers(
            s1_to_s2,
            s1,
            s2,
            configuration,
            single_state=single_state,
            nr_of_bonded_windows=nr_of_bonded_windows,
        ),
        configuration,
        s1_to_s2,
    )


def test_acetylacetone_tautomer_pair(caplog):
    caplog.set_level(logging.DEBUG)
    setup_acetylacetone_tautomer_pair()
    shutil.rmtree("acetylacetone-keto-acetylacetone-enol-rsfe")


def test_2OJ9_tautomer_pair(caplog):
    caplog.set_level(logging.DEBUG)
    setup_2OJ9_tautomer_pair_rsfe()
    shutil.rmtree("2OJ9-original-2OJ9-tautomer-rsfe")


def test_full_mutation_system1(caplog):
    caplog.set_level(logging.WARNING)

    for conf, system_name in zip(
        [
            "transformato/tests/config/test-toluene-methane-rsfe.yaml",
            "transformato/tests/config/test-neopentane-methane-rsfe.yaml",
            "transformato/tests/config/test-ethane-methanol-rsfe.yaml",
        ],
        ["toluene-methane", "neopentane-methane", "ethane-methanol"],
    ):
        print(system_name)
        configuration = load_config_yaml(config=conf, input_dir="data/", output_dir=".")
        s1 = SystemStructure(configuration, "structure1")
        s2 = SystemStructure(configuration, "structure2")

        s1_to_s2 = ProposeMutationRoute(s1, s2)
        if system_name == "neopentane-methane":
            s1_to_s2.propose_common_core()
            s1_to_s2.finish_common_core(
                connected_dummy_regions_cc1=[
                    {0, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16}
                ]
            )
        else:
            s1_to_s2.calculate_common_core()

        mutation_list = s1_to_s2.generate_mutations_to_common_core_for_mol1()
        i = IntermediateStateFactory(
            system=s1,
            configuration=configuration,
        )
        intst = 1
        charges = mutation_list["charge"]
        for lambda_value in np.linspace(0, 1, 5):
            # turn off charges
            output_file_base, intst = i.write_state(
                mutation_conf=charges,
                lambda_value_electrostatic=1 - lambda_value,
                intst_nr=intst,
            )

        original_psf = {}
        for env in s1.envs:
            original_psf[env] = copy.deepcopy(s1.psfs[env])

        #
        lambda_vdw = 0.0

        all_atoms_for_which_lj_turned_off = []
        print(mutation_list.keys())
        # Turn of hydrogens
        terminal_lj_mutations = mutation_list["default-lj"]
        _, intst = i.write_state(
            mutation_conf=terminal_lj_mutations,
            lambda_value_vdw=lambda_vdw,
            intst_nr=intst,
        )
        for mutation in terminal_lj_mutations:
            all_atoms_for_which_lj_turned_off.extend(mutation.vdw_atom_idx)

        shutil.rmtree(f"{system_name}-rsfe")


def test_full_mutation_system2():

    for conf, system_name in zip(
        [
            "transformato/tests/config/test-toluene-methane-rsfe.yaml",
            "transformato/tests/config/test-neopentane-methane-rsfe.yaml",
        ],
        ["toluene-methane", "neopentane-methane"],
    ):
        configuration = load_config_yaml(config=conf, input_dir="data/", output_dir=".")
        s1 = SystemStructure(configuration, "structure1")
        s2 = SystemStructure(configuration, "structure2")

        s1_to_s2 = ProposeMutationRoute(s1, s2)
        if system_name == "neopentane-methane":
            s1_to_s2.propose_common_core()
            s1_to_s2.finish_common_core(
                connected_dummy_regions_cc1=[
                    {0, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16}
                ]
            )
        else:
            s1_to_s2.calculate_common_core()

        mutation_list = s1_to_s2.generate_mutations_to_common_core_for_mol1()
        i = IntermediateStateFactory(
            system=s1,
            configuration=configuration,
        )

        intst = 1
        charges = mutation_list["charge"]
        for lambda_value in np.linspace(0, 1, 5):
            # turn off charges
            output_file_base, intst = i.write_state(
                mutation_conf=charges,
                lambda_value_electrostatic=1 - lambda_value,
                intst_nr=intst,
            )

        original_psf = {}
        for env in s1.envs:
            original_psf[env] = copy.deepcopy(s1.psfs[env])

        #
        lambda_vdw = 0.0
        print(f"Lambda: {lambda_vdw}")
        all_atoms_for_which_lj_turned_off = []
        print(mutation_list.keys())
        # Turn of hydrogens
        hydrogen_lj_mutations = mutation_list["hydrogen-lj"]
        output_file_base, intst = i.write_state(
            mutation_conf=hydrogen_lj_mutations,
            lambda_value_vdw=lambda_vdw,
            intst_nr=intst,
        )

        for mutation in hydrogen_lj_mutations:
            all_atoms_for_which_lj_turned_off.extend(mutation.vdw_atom_idx)

        # turn off heavy atoms
        for mutation in mutation_list["lj"]:

            output_file_base, intst = i.write_state(
                mutation_conf=[mutation],
                lambda_value_vdw=lambda_vdw,
                intst_nr=intst,
            )
            all_atoms_for_which_lj_turned_off.extend(mutation.vdw_atom_idx)

        new_psf, params = generate_psf(output_file_base, env)
        print(f"Set epsilon/rmin to base * {lambda_vdw} for selected atoms")

        print(all_atoms_for_which_lj_turned_off)
        offset = s1.offset[env]
        for idx in all_atoms_for_which_lj_turned_off:
            idxo = idx + offset
            assert np.isclose(
                original_psf[env].atoms[idxo].epsilon * lambda_vdw,
                new_psf.atoms[idxo].epsilon,
                rtol=1e-3,
            )
            assert np.isclose(
                original_psf[env].atoms[idxo].rmin * lambda_vdw,
                new_psf.atoms[idxo].rmin,
                rtol=1e-3,
            )

        # make sure that all other idx are not touched
        for idx in range(len(original_psf[env].atoms)):
            idxo = idx - offset  # NOTE: the '-'
            if idxo not in all_atoms_for_which_lj_turned_off:
                assert np.isclose(
                    original_psf[env].atoms[idx].epsilon,
                    new_psf.atoms[idx].epsilon,
                    rtol=1e-3,
                )
                assert np.isclose(
                    original_psf[env].atoms[idx].rmin,
                    new_psf.atoms[idx].rmin,
                    rtol=1e-3,
                )

        shutil.rmtree(f"{system_name}-rsfe")
