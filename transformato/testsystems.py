import numpy as np

import transformato
from transformato.mutate import ProposeMutationRoute
from transformato.state import IntermediateStateFactory
from transformato.system import SystemStructure
from transformato.utils import load_config_yaml
from transformato.constants import check_platform

transformato_systems_dir = "/home/mwieder/Work/Projects/transformato-systems/"


def mutate_methane_to_methane_cc(conf: str = "", output_dir: str = "."):

    configuration = load_config_yaml(
        config=conf, input_dir=transformato_systems_dir, output_dir=output_dir
    )
    check_platform(configuration)

    s1 = SystemStructure(configuration, "structure1")
    s2 = SystemStructure(configuration, "structure2")

    s1_to_s2 = ProposeMutationRoute(s1, s2)
    s1_to_s2.calculate_common_core()

    mutation_list = s1_to_s2.generate_mutations_to_common_core_for_mol2()
    i = IntermediateStateFactory(
        system=s2,
        configuration=configuration,
    )

    # write out endpoint
    output_files = []
    output_file_base, intst = i.write_state(mutation_conf=[], intst_nr=1)
    output_files.append(output_file_base)

    charges = mutation_list["charge"]
    # turn off charges
    output_file_base, intst = i.write_state(
        mutation_conf=charges,
        lambda_value_electrostatic=0.0,
        intst_nr=intst,
    )
    output_files.append(output_file_base)

    # generate terminal lj
    output_file_base, intst = i.write_state(
        mutation_conf=mutation_list["default-lj"],
        lambda_value_vdw=0.0,
        intst_nr=intst,
    )
    output_files.append(output_file_base)

    return output_files, configuration


def testing_mutate_toluene_to_methane_cc(conf: str = "", output_dir: str = "."):

    configuration = load_config_yaml(
        config=conf, input_dir=transformato_systems_dir, output_dir=output_dir
    )
    check_platform(configuration)

    s1 = SystemStructure(configuration, "structure1")
    s2 = SystemStructure(configuration, "structure2")

    s1_to_s2 = ProposeMutationRoute(s1, s2)
    s1_to_s2.calculate_common_core()

    mutation_list = s1_to_s2.generate_mutations_to_common_core_for_mol1()
    i = IntermediateStateFactory(
        system=s1,
        configuration=configuration,
    )

    # write out endpoint
    output_files = []
    output_file_base, intst = i.write_state(mutation_conf=[], intst_nr=1)
    output_files.append(output_file_base)

    charges = mutation_list["charge"]
    # start with charges
    # turn off charges
    output_file_base, intst = i.write_state(
        mutation_conf=charges,
        lambda_value_electrostatic=0.0,
        intst_nr=intst,
    )
    output_files.append(output_file_base)

    # Turn off hydrogens
    hydrogen_lj_mutations = mutation_list["hydrogen-lj"]
    output_file_base, intst = i.write_state(
        mutation_conf=hydrogen_lj_mutations,
        lambda_value_vdw=0.5,
        intst_nr=intst,
    )
    output_files.append(output_file_base)

    # Turn off hydrogens
    hydrogen_lj_mutations = mutation_list["hydrogen-lj"]
    output_file_base, intst = i.write_state(
        mutation_conf=hydrogen_lj_mutations,
        lambda_value_vdw=0.0,
        intst_nr=intst,
    )
    output_files.append(output_file_base)

    # turn off heavy atoms
    d = transformato.utils.map_lj_mutations_to_atom_idx(mutation_list["lj"])
    m = [d[(13,)]]
    output_file_base, intst = i.write_state(
        mutation_conf=m,
        lambda_value_vdw=0.5,
        intst_nr=intst,
    )
    output_files.append(output_file_base)

    # turn off heavy atoms
    d = transformato.utils.map_lj_mutations_to_atom_idx(mutation_list["lj"])
    m = [d[(13,)]]
    output_file_base, intst = i.write_state(
        mutation_conf=m,
        lambda_value_vdw=0.0,
        intst_nr=intst,
    )
    output_files.append(output_file_base)


def mutate_toluene_to_methane_cc(conf: str = "", output_dir: str = "."):

    configuration = load_config_yaml(
        config=conf, input_dir=transformato_systems_dir, output_dir=output_dir
    )
    check_platform(configuration)
    s1 = SystemStructure(configuration, "structure1")
    s2 = SystemStructure(configuration, "structure2")

    s1_to_s2 = ProposeMutationRoute(s1, s2)
    s1_to_s2.calculate_common_core()

    mutation_list = s1_to_s2.generate_mutations_to_common_core_for_mol1()
    i = IntermediateStateFactory(
        system=s1,
        configuration=configuration,
    )

    # write out endpoint
    output_files = []
    output_file_base, intst = i.write_state(mutation_conf=[], intst_nr=1)
    output_files.append(output_file_base)

    charges = mutation_list["charge"]
    # start with charges
    # turn off charges
    output_file_base, intst = i.write_state(
        mutation_conf=charges,
        lambda_value_electrostatic=0.0,
        intst_nr=intst,
    )
    output_files.append(output_file_base)

    # Turn off hydrogens
    hydrogen_lj_mutations = mutation_list["hydrogen-lj"]
    output_file_base, intst = i.write_state(
        mutation_conf=hydrogen_lj_mutations,
        lambda_value_vdw=0.0,
        intst_nr=intst,
    )
    output_files.append(output_file_base)

    # turn off heavy atoms
    d = transformato.utils.map_lj_mutations_to_atom_idx(mutation_list["lj"])
    m = [d[(13,)]]

    output_file_base, intst = i.write_state(
        mutation_conf=m,
        lambda_value_vdw=0.0,
        intst_nr=intst,
    )
    output_files.append(output_file_base)

    # turn off heavy atoms
    m = [d[(11,)]]

    output_file_base, intst = i.write_state(
        mutation_conf=m,
        lambda_value_vdw=0.0,
        intst_nr=intst,
    )
    output_files.append(output_file_base)

    # turn off heavy atoms
    m = [d[(9,)]]

    output_file_base, intst = i.write_state(
        mutation_conf=m,
        lambda_value_vdw=0.0,
        intst_nr=intst,
    )
    output_files.append(output_file_base)

    # turn off heavy atoms
    m = [d[(3,)]]

    output_file_base, intst = i.write_state(
        mutation_conf=m,
        lambda_value_vdw=0.0,
        intst_nr=intst,
    )
    output_files.append(output_file_base)

    # turn off heavy atoms
    m = [d[(1,)]]

    output_file_base, intst = i.write_state(
        mutation_conf=m,
        lambda_value_vdw=0.0,
        intst_nr=intst,
    )
    output_files.append(output_file_base)

    # generate terminal lj
    output_file_base, intst = i.write_state(
        mutation_conf=mutation_list["default-lj"],
        lambda_value_vdw=0.0,
        intst_nr=intst,
    )
    output_files.append(output_file_base)

    m = mutation_list["transform"]
    for lambda_value in np.linspace(0.75, 0, 4):
        print(lambda_value)
        # turn off charges
        output_file_base, intst = i.write_state(
            mutation_conf=m,
            common_core_transformation=lambda_value,
            intst_nr=intst,
        )
        output_files.append(output_file_base)
    return output_files, configuration


def mutate_ethane_to_methane_cc(conf: str = "", output_dir: str = "."):

    configuration = load_config_yaml(
        config=conf, input_dir=transformato_systems_dir, output_dir=output_dir
    )
    check_platform(configuration)

    s1 = SystemStructure(configuration, "structure1")
    s2 = SystemStructure(configuration, "structure2")

    s1_to_s2 = ProposeMutationRoute(s1, s2)
    s1_to_s2.calculate_common_core()

    mutation_list = s1_to_s2.generate_mutations_to_common_core_for_mol1()
    i = IntermediateStateFactory(
        system=s1,
        configuration=configuration,
    )

    # write out endpoint
    output_files = []
    output_file_base, intst = i.write_state(mutation_conf=[], intst_nr=1)
    output_files.append(output_file_base)

    charges = mutation_list["charge"]
    # start with charges
    # turn off charges
    output_file_base, intst = i.write_state(
        mutation_conf=charges,
        lambda_value_electrostatic=0.0,
        intst_nr=intst,
    )
    output_files.append(output_file_base)

    # Turn off hydrogens
    hydrogen_lj_mutations = mutation_list["hydrogen-lj"]
    output_file_base, intst = i.write_state(
        mutation_conf=hydrogen_lj_mutations,
        lambda_value_vdw=0.0,
        intst_nr=intst,
    )
    output_files.append(output_file_base)

    # generate terminal lj
    output_file_base, intst = i.write_state(
        mutation_conf=mutation_list["default-lj"],
        lambda_value_vdw=0.0,
        intst_nr=intst,
    )
    output_files.append(output_file_base)

    m = mutation_list["transform"]
    for lambda_value in np.linspace(0.75, 0, 4):
        print(lambda_value)
        # turn off charges
        output_file_base, intst = i.write_state(
            mutation_conf=m,
            common_core_transformation=lambda_value,
            intst_nr=intst,
        )
        output_files.append(output_file_base)
    return output_files, configuration


def mutate_methanol_to_methane_cc(conf: str = "", output_dir: str = "."):

    configuration = load_config_yaml(
        config=conf, input_dir=transformato_systems_dir, output_dir=output_dir
    )
    check_platform(configuration)

    s1 = SystemStructure(configuration, "structure1")
    s2 = SystemStructure(configuration, "structure2")

    s1_to_s2 = ProposeMutationRoute(s1, s2)
    s1_to_s2.calculate_common_core()

    mutation_list = s1_to_s2.generate_mutations_to_common_core_for_mol1()
    i = IntermediateStateFactory(
        system=s1,
        configuration=configuration,
    )

    # write out endpoint
    output_files = []
    output_file_base, intst = i.write_state(mutation_conf=[], intst_nr=1)
    output_files.append(output_file_base)

    charges = mutation_list["charge"]
    # start with charges
    # turn off charges
    output_file_base, intst = i.write_state(
        mutation_conf=charges,
        lambda_value_electrostatic=0.8,
        intst_nr=intst,
    )
    output_files.append(output_file_base)

    # start with charges
    # turn off charges
    output_file_base, intst = i.write_state(
        mutation_conf=charges,
        lambda_value_electrostatic=0.6,
        intst_nr=intst,
    )
    output_files.append(output_file_base)

    # start with charges
    # turn off charges
    output_file_base, intst = i.write_state(
        mutation_conf=charges,
        lambda_value_electrostatic=0.4,
        intst_nr=intst,
    )
    output_files.append(output_file_base)

    # start with charges
    # turn off charges
    output_file_base, intst = i.write_state(
        mutation_conf=charges,
        lambda_value_electrostatic=0.2,
        intst_nr=intst,
    )
    output_files.append(output_file_base)

    # start with charges
    # turn off charges
    output_file_base, intst = i.write_state(
        mutation_conf=charges,
        lambda_value_electrostatic=0.0,
        intst_nr=intst,
    )
    output_files.append(output_file_base)

    # Turn off hydrogens
    hydrogen_lj_mutations = mutation_list["hydrogen-lj"]
    output_file_base, intst = i.write_state(
        mutation_conf=hydrogen_lj_mutations,
        lambda_value_vdw=0.0,
        intst_nr=intst,
    )
    output_files.append(output_file_base)

    # generate terminal lj
    output_file_base, intst = i.write_state(
        mutation_conf=mutation_list["default-lj"],
        lambda_value_vdw=0.0,
        intst_nr=intst,
    )
    output_files.append(output_file_base)

    m = mutation_list["transform"]
    for lambda_value in np.linspace(0.75, 0, 4):
        # turn off charges
        output_file_base, intst = i.write_state(
            mutation_conf=m,
            common_core_transformation=lambda_value,
            intst_nr=intst,
        )
        output_files.append(output_file_base)
    return output_files, configuration


def mutate_ethane_to_methanol_cc(
    conf: str = "", output_dir: str = "."
):  # not a loeffler system

    configuration = load_config_yaml(
        config=conf, input_dir=transformato_systems_dir, output_dir=output_dir
    )
    check_platform(configuration)

    s1 = SystemStructure(configuration, "structure1")
    s2 = SystemStructure(configuration, "structure2")

    s1_to_s2 = ProposeMutationRoute(s1, s2)
    s1_to_s2.calculate_common_core()

    mutation_list = s1_to_s2.generate_mutations_to_common_core_for_mol1()
    i = IntermediateStateFactory(
        system=s1,
        configuration=configuration,
    )

    # write out endpoint
    output_files = []
    output_file_base, intst = i.write_state(mutation_conf=[], intst_nr=1)
    output_files.append(output_file_base)

    charges = mutation_list["charge"]
    # start with charges
    # turn off charges
    output_file_base, intst = i.write_state(
        mutation_conf=charges,
        lambda_value_electrostatic=0.0,
        intst_nr=intst,
    )
    output_files.append(output_file_base)

    # Turn off hydrogens
    hydrogen_lj_mutations = mutation_list["hydrogen-lj"]
    output_file_base, intst = i.write_state(
        mutation_conf=hydrogen_lj_mutations,
        lambda_value_vdw=0.0,
        intst_nr=intst,
    )
    output_files.append(output_file_base)

    # generate terminal lj
    output_file_base, intst = i.write_state(
        mutation_conf=mutation_list["default-lj"],
        lambda_value_vdw=0.0,
        intst_nr=intst,
    )
    output_files.append(output_file_base)

    m = mutation_list["transform"]
    for lambda_value in np.linspace(0.75, 0, 4):
        # turn off charges
        output_file_base, intst = i.write_state(
            mutation_conf=m,
            common_core_transformation=lambda_value,
            intst_nr=intst,
        )
        output_files.append(output_file_base)
    return output_files, configuration


def mutate_7_CPI_to_2_CPI_cc(
    conf: str = "", output_dir: str = "."
):  # will be tested later on

    configuration = load_config_yaml(
        config=conf, input_dir=transformato_systems_dir, output_dir=output_dir
    )
    check_platform(configuration)

    s1 = SystemStructure(configuration, "structure1")
    s2 = SystemStructure(configuration, "structure2")

    s1_to_s2 = ProposeMutationRoute(s1, s2)
    s1_to_s2.completeRingsOnly = True
    s1_to_s2.propose_common_core()
    s1_to_s2.remove_idx_from_common_core_of_mol1([14])
    s1_to_s2.remove_idx_from_common_core_of_mol2([6])
    s1_to_s2.finish_common_core()

    mutation_list = s1_to_s2.generate_mutations_to_common_core_for_mol2()

    i = IntermediateStateFactory(
        system=s2,
        configuration=configuration,
    )
    # write out endpoint
    output_files = []
    output_file_base, intst = i.write_state(mutation_conf=[], intst_nr=1)
    output_files.append(output_file_base)

    # start with charges
    # turn off charges
    charges = mutation_list["charge"]
    for lambda_value in np.linspace(1, 0, 4)[1:]:
        output_file_base, intst = i.write_state(
            mutation_conf=charges,
            lambda_value_electrostatic=lambda_value,
            intst_nr=intst,
        )
        output_files.append(output_file_base)

    # Turn off hydrogens
    hydrogen_lj_mutations = mutation_list["hydrogen-lj"]
    output_file_base, intst = i.write_state(
        mutation_conf=hydrogen_lj_mutations,
        lambda_value_vdw=0.0,
        intst_nr=intst,
    )
    output_files.append(output_file_base)

    # turn off heavy atoms
    d = transformato.utils.map_lj_mutations_to_atom_idx(mutation_list["lj"])
    # turn off heavy atoms
    for m in [
        [d[(13,)]],
        [d[(11,)], d[(8,)]],
        [d[(0,)]],
        [d[(2,)], d[(10,)]],
        [d[(4,)], d[(7,)]],
    ]:
        output_file_base, intst = i.write_state(
            mutation_conf=m,
            lambda_value_vdw=0.0,
            intst_nr=intst,
        )
        output_files.append(output_file_base)

    # generate terminal lj
    output_file_base, intst = i.write_state(
        mutation_conf=mutation_list["default-lj"],
        lambda_value_vdw=0.0,
        intst_nr=intst,
    )
    output_files.append(output_file_base)

    return output_files, configuration


def mutate_2_CPI_to_7_CPI_cc(
    conf: str = "", output_dir: str = "."
):  # will be tested later on

    configuration = load_config_yaml(
        config=conf, input_dir=transformato_systems_dir, output_dir=output_dir
    )
    check_platform(configuration)

    s1 = SystemStructure(configuration, "structure1")
    s2 = SystemStructure(configuration, "structure2")

    s1_to_s2 = ProposeMutationRoute(s1, s2)
    s1_to_s2.completeRingsOnly = True
    s1_to_s2.propose_common_core()
    s1_to_s2.remove_idx_from_common_core_of_mol1([14])
    s1_to_s2.remove_idx_from_common_core_of_mol2([6])
    s1_to_s2.finish_common_core()

    mutation_list = s1_to_s2.generate_mutations_to_common_core_for_mol1()

    i = IntermediateStateFactory(
        system=s1,
        configuration=configuration,
    )
    # write out endpoint
    output_files = []
    output_file_base, intst = i.write_state(mutation_conf=[], intst_nr=1)
    output_files.append(output_file_base)

    # start with charges
    # turn off charges
    charges = mutation_list["charge"]
    for lambda_value in np.linspace(1, 0, 4)[1:]:
        output_file_base, intst = i.write_state(
            mutation_conf=charges,
            lambda_value_electrostatic=lambda_value,
            intst_nr=intst,
        )
        output_files.append(output_file_base)

    # Turn off hydrogens
    hydrogen_lj_mutations = mutation_list["hydrogen-lj"]
    output_file_base, intst = i.write_state(
        mutation_conf=hydrogen_lj_mutations,
        lambda_value_vdw=0.0,
        intst_nr=intst,
    )
    output_files.append(output_file_base)

    # turn off heavy atoms
    d = transformato.utils.map_lj_mutations_to_atom_idx(mutation_list["lj"])
    # turn off heavy atoms
    for m in [
        [d[(2,)], d[(4,)]],
        [d[(0,)], d[(6,)]],
        [d[(11,)]],
        [d[(8,)]],
        [d[(12,)], d[(9,)]],
    ]:

        output_file_base, intst = i.write_state(
            mutation_conf=m,
            lambda_value_vdw=0.0,
            intst_nr=intst,
        )
        output_files.append(output_file_base)

    # generate terminal lj
    output_file_base, intst = i.write_state(
        mutation_conf=mutation_list["default-lj"],
        lambda_value_vdw=0.0,
        intst_nr=intst,
    )
    output_files.append(output_file_base)

    m = mutation_list["transform"]
    for lambda_value in np.linspace(1, 0, 5)[1:]:
        # turn off charges
        output_file_base, intst = i.write_state(
            mutation_conf=m,
            common_core_transformation=lambda_value,
            intst_nr=intst,
        )
        output_files.append(output_file_base)
    return output_files, configuration


def mutate_2_methylfuran_to_methane_cc(conf: str = "", output_dir: str = "."):

    configuration = load_config_yaml(
        config=conf, input_dir=transformato_systems_dir, output_dir=output_dir
    )
    check_platform(configuration)

    s1 = SystemStructure(configuration, "structure1")
    s2 = SystemStructure(configuration, "structure2")

    s1_to_s2 = ProposeMutationRoute(s1, s2)
    s1_to_s2.calculate_common_core()

    mutation_list = s1_to_s2.generate_mutations_to_common_core_for_mol1()
    i = IntermediateStateFactory(
        system=s1,
        configuration=configuration,
    )

    # write out endpoint
    output_files = []
    output_file_base, intst = i.write_state(mutation_conf=[], intst_nr=1)
    output_files.append(output_file_base)

    charges = mutation_list["charge"]
    # start with charges
    # turn off charges
    output_file_base, intst = i.write_state(
        mutation_conf=charges,
        lambda_value_electrostatic=0.0,
        intst_nr=intst,
    )
    output_files.append(output_file_base)

    # Turn off hydrogens
    hydrogen_lj_mutations = mutation_list["hydrogen-lj"]
    output_file_base, intst = i.write_state(
        mutation_conf=hydrogen_lj_mutations,
        lambda_value_vdw=0.0,
        intst_nr=intst,
    )
    output_files.append(output_file_base)

    # turn off heavy atoms
    d = transformato.utils.map_lj_mutations_to_atom_idx(mutation_list["lj"])

    # turn off heavy atoms
    m = [d[(10,)]]

    output_file_base, intst = i.write_state(
        mutation_conf=m,
        lambda_value_vdw=0.0,
        intst_nr=intst,
    )
    output_files.append(output_file_base)

    # turn off heavy atoms
    m = [d[(8,)]]
    output_file_base, intst = i.write_state(
        mutation_conf=m,
        lambda_value_vdw=0.0,
        intst_nr=intst,
    )
    output_files.append(output_file_base)

    # turn off heavy atoms
    m = [d[(2,)]]

    output_file_base, intst = i.write_state(
        mutation_conf=m,
        lambda_value_vdw=0.0,
        intst_nr=intst,
    )
    output_files.append(output_file_base)

    # turn off heavy atoms
    m = [d[(0,)]]

    output_file_base, intst = i.write_state(
        mutation_conf=m,
        lambda_value_vdw=0.0,
        intst_nr=intst,
    )
    output_files.append(output_file_base)

    # generate terminal lj

    output_file_base, intst = i.write_state(
        mutation_conf=mutation_list["default-lj"],
        lambda_value_vdw=0.0,
        intst_nr=intst,
    )
    output_files.append(output_file_base)

    m = mutation_list["transform"]
    for lambda_value in np.linspace(0.75, 0, 4):
        # turn off charges
        output_file_base, intst = i.write_state(
            mutation_conf=m,
            common_core_transformation=lambda_value,
            intst_nr=intst,
        )
        output_files.append(output_file_base)
    return output_files, configuration


def mutate_neopentane_to_methane_cc(conf: str = "", output_dir: str = "."):

    configuration = load_config_yaml(
        config=conf, input_dir=transformato_systems_dir, output_dir=output_dir
    )
    check_platform(configuration)

    s1 = SystemStructure(configuration, "structure1")
    s2 = SystemStructure(configuration, "structure2")

    s1_to_s2 = ProposeMutationRoute(s1, s2)
    s1_to_s2.propose_common_core()
    s1_to_s2.finish_common_core(
        connected_dummy_regions_cc1=[{0, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16}]
    )

    mutation_list = s1_to_s2.generate_mutations_to_common_core_for_mol1()
    i = IntermediateStateFactory(
        system=s1,
        configuration=configuration,
    )

    # write out endpoint
    output_files = []
    output_file_base, intst = i.write_state(mutation_conf=[], intst_nr=1)
    output_files.append(output_file_base)

    # start with charges
    # turn off charges
    charges = mutation_list["charge"]
    nr_of_mutation_steps_charge = (
        5  # defines the number of mutation steps for charge mutation
    )
    for lambda_value in np.linspace(1, 0, nr_of_mutation_steps_charge + 1)[1:]:
        output_file_base, intst = i.write_state(
            mutation_conf=charges,
            lambda_value_electrostatic=lambda_value,
            intst_nr=intst,
        )
        output_files.append(output_file_base)

    # Turn off hydrogens
    hydrogen_lj_mutations = mutation_list["hydrogen-lj"]
    output_file_base, intst = i.write_state(
        mutation_conf=hydrogen_lj_mutations,
        lambda_value_vdw=0.0,
        intst_nr=intst,
    )
    output_files.append(output_file_base)

    # turn off heavy atoms
    d = transformato.utils.map_lj_mutations_to_atom_idx(mutation_list["lj"])

    # turn off heavy atoms
    m = [d[(13,)]]
    output_file_base, intst = i.write_state(
        mutation_conf=m,
        lambda_value_vdw=0.0,
        intst_nr=intst,
    )
    output_files.append(output_file_base)

    # turn off heavy atoms
    m = [d[(9,)]]
    output_file_base, intst = i.write_state(
        mutation_conf=m,
        lambda_value_vdw=0.0,
        intst_nr=intst,
    )
    output_files.append(output_file_base)

    # turn off heavy atoms
    m = [d[(5,)]]
    output_file_base, intst = i.write_state(
        mutation_conf=m,
        lambda_value_vdw=0.0,
        intst_nr=intst,
    )
    output_files.append(output_file_base)

    # generate terminal lj
    output_file_base, intst = i.write_state(
        mutation_conf=mutation_list["default-lj"],
        lambda_value_vdw=0.0,
        intst_nr=intst,
    )
    output_files.append(output_file_base)

    m = mutation_list["transform"]
    nr_of_mutation_steps_cc_bonded_terms = 5

    for lambda_value in np.linspace(1, 0, nr_of_mutation_steps_cc_bonded_terms + 1)[1:]:
        # turn off charges
        output_file_base, intst = i.write_state(
            mutation_conf=m,
            common_core_transformation=lambda_value,
            intst_nr=intst,
        )
        output_files.append(output_file_base)
    return output_files, configuration


def mutate_2_methylindole_to_methane_cc(conf: str = "", output_dir: str = "."):

    configuration = load_config_yaml(
        config=conf, input_dir=transformato_systems_dir, output_dir=output_dir
    )
    check_platform(configuration)

    s1 = SystemStructure(configuration, "structure1")
    s2 = SystemStructure(configuration, "structure2")

    s1_to_s2 = ProposeMutationRoute(s1, s2)
    s1_to_s2.calculate_common_core()

    mutation_list = s1_to_s2.generate_mutations_to_common_core_for_mol1()
    i = IntermediateStateFactory(
        system=s1,
        configuration=configuration,
    )
    # write out endpoint
    output_files = []
    output_file_base, intst = i.write_state(mutation_conf=[], intst_nr=1)
    output_files.append(output_file_base)

    charges = mutation_list["charge"]
    # start with charges
    # turn off charges
    output_file_base, intst = i.write_state(
        mutation_conf=charges,
        lambda_value_electrostatic=0.5,
        intst_nr=intst,
    )
    output_files.append(output_file_base)

    # start with charges
    # turn off charges
    output_file_base, intst = i.write_state(
        mutation_conf=charges,
        lambda_value_electrostatic=0.0,
        intst_nr=intst,
    )
    output_files.append(output_file_base)

    # Turn off hydrogens
    hydrogen_lj_mutations = mutation_list["hydrogen-lj"]
    output_file_base, intst = i.write_state(
        mutation_conf=hydrogen_lj_mutations,
        lambda_value_vdw=0.0,
        intst_nr=intst,
    )
    output_files.append(output_file_base)

    # turn off heavy atoms
    d = transformato.utils.map_lj_mutations_to_atom_idx(mutation_list["lj"])

    # turn off heavy atoms
    m = [d[(17,)]]
    output_file_base, intst = i.write_state(
        mutation_conf=m,
        lambda_value_vdw=0.0,
        intst_nr=intst,
    )
    output_files.append(output_file_base)

    # turn off heavy atoms
    m = [d[(15,)]]
    output_file_base, intst = i.write_state(
        mutation_conf=m,
        lambda_value_vdw=0.0,
        intst_nr=intst,
    )
    output_files.append(output_file_base)

    # turn off heavy atoms
    m = [d[(9,)]]
    output_file_base, intst = i.write_state(
        mutation_conf=m,
        lambda_value_vdw=0.0,
        intst_nr=intst,
    )
    output_files.append(output_file_base)

    # turn off heavy atoms
    m = [d[(7,)]]
    output_file_base, intst = i.write_state(
        mutation_conf=m,
        lambda_value_vdw=0.0,
        intst_nr=intst,
    )
    output_files.append(output_file_base)

    # turn off heavy atoms
    m = [d[(3,)]]
    output_file_base, intst = i.write_state(
        mutation_conf=m,
        lambda_value_vdw=0.0,
        intst_nr=intst,
    )
    output_files.append(output_file_base)

    # turn off heavy atoms
    m = [d[(2,)]]
    output_file_base, intst = i.write_state(
        mutation_conf=m,
        lambda_value_vdw=0.0,
        intst_nr=intst,
    )
    output_files.append(output_file_base)

    # turn off heavy atoms
    m = [d[(5,)]]
    output_file_base, intst = i.write_state(
        mutation_conf=m,
        lambda_value_vdw=0.0,
        intst_nr=intst,
    )
    output_files.append(output_file_base)

    # turn off heavy atoms
    m = [d[(0,)]]
    output_file_base, intst = i.write_state(
        mutation_conf=m,
        lambda_value_vdw=0.0,
        intst_nr=intst,
    )
    output_files.append(output_file_base)

    # generate terminal lj
    output_file_base, intst = i.write_state(
        mutation_conf=mutation_list["default-lj"],
        lambda_value_vdw=0.0,
        intst_nr=intst,
    )
    output_files.append(output_file_base)

    m = mutation_list["transform"]
    for lambda_value in np.linspace(0.75, 0, 4):
        # turn off charges
        output_file_base, intst = i.write_state(
            mutation_conf=m,
            common_core_transformation=lambda_value,
            intst_nr=intst,
        )
        output_files.append(output_file_base)
    return output_files, configuration


def mutate_acetylaceton_methyl_common_core(
    conf_path: str, input_dir: str, output_dir: str
):
    configuration = load_config_yaml(
        config=conf_path, input_dir=input_dir, output_dir=output_dir
    )
    s1 = SystemStructure(configuration, "structure1")
    s2 = SystemStructure(configuration, "structure2")
    s1_to_s2 = ProposeMutationRoute(s1, s2)

    # modify MCS matching
    from rdkit.Chem import rdFMCS

    s1_to_s2.matchValences = True
    s1_to_s2.bondCompare = rdFMCS.BondCompare.CompareOrderExact
    s1_to_s2.propose_common_core()
    s1_to_s2.remove_idx_from_common_core_of_mol1([2, 10])
    s1_to_s2.remove_idx_from_common_core_of_mol2([2, 10])

    # manually set the dummy region
    s1_to_s2.finish_common_core(
        connected_dummy_regions_cc1=[{11, 2, 10, 3, 6, 4, 13, 14, 12}]
    )

    ###############################
    ########### KETO ##############
    ###############################
    # generate the mutation list for keto acetylaceton
    mutation_list = s1_to_s2.generate_mutations_to_common_core_for_mol1()
    print(mutation_list.keys())
    output_files = []

    i = IntermediateStateFactory(
        system=s1,
        configuration=configuration,
    )

    # write endpoint mutation
    output_file_base, intst = i.write_state(mutation_conf=[], intst_nr=1)
    output_files.append(output_file_base)

    # start with charges
    # turn off charges
    charges = mutation_list["charge"]
    nr_of_mutation_steps_charge = (
        5  # defines the number of mutation steps for charge mutation
    )
    for lambda_value in np.linspace(1, 0, nr_of_mutation_steps_charge + 1)[1:]:
        output_file_base, intst = i.write_state(
            mutation_conf=charges,
            lambda_value_electrostatic=lambda_value,
            intst_nr=intst,
        )
        output_files.append(output_file_base)

    # Turn off hydrogens
    hydrogen_lj_mutations = mutation_list["hydrogen-lj"]
    output_file_base, intst = i.write_state(
        mutation_conf=hydrogen_lj_mutations,
        lambda_value_vdw=0.0,
        intst_nr=intst,
    )
    output_files.append(output_file_base)

    # turn off lj of heavy atoms
    d = transformato.utils.map_lj_mutations_to_atom_idx(mutation_list["lj"])
    for m in [d[(4,)], d[(6,)], d[(3,)]]:
        output_file_base, intst = i.write_state(
            mutation_conf=[m],
            lambda_value_vdw=0.0,
            intst_nr=intst,
        )
        output_files.append(output_file_base)

    # generate terminal lj
    output_file_base, intst = i.write_state(
        mutation_conf=mutation_list["default-lj"],
        lambda_value_vdw=0.0,
        intst_nr=intst,
    )
    output_files.append(output_file_base)

    # change charge on common core
    m = mutation_list["transform"]
    for lambda_value in np.linspace(0.75, 0, 4):
        # interpolate between parameters
        output_file_base, intst = i.write_state(
            mutation_conf=m,
            common_core_transformation=lambda_value,
            intst_nr=intst,
        )
        output_files.append(output_file_base)

    ###############################
    ########### ENOL ##############
    ###############################

    mutation_list = s1_to_s2.generate_mutations_to_common_core_for_mol2()

    i = IntermediateStateFactory(
        system=s2,
        configuration=configuration,
    )

    output_files = []
    output_file_base, intst = i.write_state(mutation_conf=[], intst_nr=1)
    output_files.append(output_file_base)

    # start with charges
    # turn off charges
    charges = mutation_list["charge"]
    nr_of_mutation_steps_charge = (
        5  # defines the number of mutation steps for charge mutation
    )
    for lambda_value in np.linspace(1, 0, nr_of_mutation_steps_charge + 1)[1:]:
        output_file_base, intst = i.write_state(
            mutation_conf=charges,
            lambda_value_electrostatic=lambda_value,
            intst_nr=intst,
        )
        output_files.append(output_file_base)

    # Turn off hydrogens
    hydrogen_lj_mutations = mutation_list["hydrogen-lj"]
    output_file_base, intst = i.write_state(
        mutation_conf=hydrogen_lj_mutations,
        lambda_value_vdw=0.0,
        intst_nr=intst,
    )
    output_files.append(output_file_base)
    d = transformato.utils.map_lj_mutations_to_atom_idx(mutation_list["lj"])

    # turn off heavy atom lj
    for m in [d[(0,)], d[(5,)], d[(1,)]]:
        output_file_base, intst = i.write_state(
            mutation_conf=[m],
            lambda_value_vdw=0.0,
            intst_nr=intst,
        )
        output_files.append(output_file_base)

    # generate terminal lj
    output_file_base, intst = i.write_state(
        mutation_conf=mutation_list["default-lj"],
        lambda_value_vdw=0.0,
        intst_nr=intst,
    )
    output_files.append(output_file_base)


def mutate_bmi_small_common_core(conf_path: str, input_dir: str, output_dir: str):
    configuration = load_config_yaml(
        config=conf_path, input_dir=input_dir, output_dir=output_dir
    )
    print("Setup system ...")
    print(configuration)
    s1 = SystemStructure(configuration, "structure1")
    s2 = SystemStructure(configuration, "structure2")
    print("Propose mutation ...")
    s1_to_s2 = ProposeMutationRoute(s1, s2)
    s1_to_s2.propose_common_core()
    print("Remove atom idx ...")

    s1_to_s2.remove_idx_from_common_core_of_mol1(
        [2, 3, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23]
    )
    s1_to_s2.remove_idx_from_common_core_of_mol2(
        [2, 3, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23]
    )

    # manually set the dummy region
    print("Manually specify dummy regions ...")

    s1_to_s2.finish_common_core(
        connected_dummy_regions_cc1=[
            {2, 3, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23},
            {38},
        ],
        connected_dummy_regions_cc2=[
            {2, 3, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23},
            {40},
        ],
    )

    ###############################
    ####### 2OJ9 - original #######
    ###############################
    # generate the mutation list for the original
    mutation_list = s1_to_s2.generate_mutations_to_common_core_for_mol1()
    print(mutation_list.keys())
    output_files = []

    i = IntermediateStateFactory(
        system=s1,
        configuration=configuration,
    )

    # write endpoint mutation
    output_file_base, intst = i.write_state(mutation_conf=[], intst_nr=1)
    output_files.append(output_file_base)

    # start with charges
    # turn off charges
    charges = mutation_list["charge"]
    nr_of_mutation_steps_charge = (
        5  # defines the number of mutation steps for charge mutation
    )
    for lambda_value in np.linspace(1, 0, nr_of_mutation_steps_charge + 1)[1:]:
        output_file_base, intst = i.write_state(
            mutation_conf=charges,
            lambda_value_electrostatic=lambda_value,
            intst_nr=intst,
        )
        output_files.append(output_file_base)

    # Turn off hydrogens
    hydrogen_lj_mutations = mutation_list["hydrogen-lj"]
    output_file_base, intst = i.write_state(
        mutation_conf=hydrogen_lj_mutations,
        lambda_value_vdw=0.0,
        intst_nr=intst,
    )
    output_files.append(output_file_base)

    # turn off lj of heavy atoms
    d = transformato.utils.map_lj_mutations_to_atom_idx(mutation_list["lj"])
    for m in [
        [d[(18,)], d[(22,)]],
        [d[(14,)]],
        [d[(21,)], d[(16,)]],
        [d[(20,)]],
        [d[(11,)]],
    ]:
        output_file_base, intst = i.write_state(
            mutation_conf=m,
            lambda_value_vdw=0.0,
            intst_nr=intst,
        )
        output_files.append(output_file_base)

    # generate terminal lj
    output_file_base, intst = i.write_state(
        mutation_conf=mutation_list["default-lj"],
        lambda_value_vdw=0.0,
        intst_nr=intst,
    )
    output_files.append(output_file_base)

    # change bonded parameters on common core
    m = mutation_list["transform"]
    for lambda_value in np.linspace(0.75, 0, 4):
        # interpolate between parameters
        output_file_base, intst = i.write_state(
            mutation_conf=m,
            common_core_transformation=lambda_value,
            intst_nr=intst,
        )
        output_files.append(output_file_base)

    ###############################
    ###### 2OJ9 - tautomer ########
    ###############################

    mutation_list = s1_to_s2.generate_mutations_to_common_core_for_mol2()

    i = IntermediateStateFactory(
        system=s2,
        configuration=configuration,
    )

    output_files = []
    output_file_base, intst = i.write_state(mutation_conf=[], intst_nr=1)
    output_files.append(output_file_base)

    # start with charges
    # turn off charges
    charges = mutation_list["charge"]
    nr_of_mutation_steps_charge = (
        5  # defines the number of mutation steps for charge mutation
    )
    for lambda_value in np.linspace(1, 0, nr_of_mutation_steps_charge + 1)[1:]:
        output_file_base, intst = i.write_state(
            mutation_conf=charges,
            lambda_value_electrostatic=lambda_value,
            intst_nr=intst,
        )
        output_files.append(output_file_base)

    # Turn off hydrogens
    hydrogen_lj_mutations = mutation_list["hydrogen-lj"]
    output_file_base, intst = i.write_state(
        mutation_conf=hydrogen_lj_mutations,
        lambda_value_vdw=0.0,
        intst_nr=intst,
    )
    output_files.append(output_file_base)
    d = transformato.utils.map_lj_mutations_to_atom_idx(mutation_list["lj"])

    # turn off lj of heavy atoms
    d = transformato.utils.map_lj_mutations_to_atom_idx(mutation_list["lj"])
    for m in [
        [d[(18,)], d[(22,)], d[(14,)]],
        [d[(21,)], d[(16,)]],
        [d[(20,)]],
        [d[(11,)]],
    ]:
        output_file_base, intst = i.write_state(
            mutation_conf=m,
            lambda_value_vdw=0.0,
            intst_nr=intst,
        )
        output_files.append(output_file_base)

    # generate terminal lj
    output_file_base, intst = i.write_state(
        mutation_conf=mutation_list["default-lj"],
        lambda_value_vdw=0.0,
        intst_nr=intst,
    )
    output_files.append(output_file_base)
