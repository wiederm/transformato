"""
Unit and regression test for the transformato package.
"""

import os

import numpy as np
import pytest
import logging
from ..constants import initialize_NUM_PROC

# read in specific topology with parameters
from transformato import (
    load_config_yaml,
)
from .test_mutation import (
    _set_output_files_2oj9_tautomer_pair,
    _set_output_files_acetylaceton_tautomer_pair,
    _set_output_files_toluene_methane_pair,
)


def postprocessing(
    configuration: dict,
    name: str = "methane",
    engine: str = "openMM",
    max_snapshots: int = 300,
    show_summary: bool = False,
    different_path_for_dcd: str = "",
    only_vacuum: bool = False,
):
    from transformato import FreeEnergyCalculator

    f = FreeEnergyCalculator(configuration, name)
    if only_vacuum:
        f.envs = ["vacuum"]

    if different_path_for_dcd:
        # this is needed if the trajectories are stored at a different location than the
        # potential definitions
        path = f.base_path
        f.base_path = different_path_for_dcd
        f.load_trajs(nr_of_max_snapshots=max_snapshots)
        f.base_path = path
    else:
        f.load_trajs(nr_of_max_snapshots=max_snapshots)

    f.calculate_dG_to_common_core(engine=engine)
    if only_vacuum:
        return -1, -1, f
    else:
        ddG, dddG = f.end_state_free_energy_difference
        print(f"Free energy difference: {ddG}")
        print(f"Uncertanty: {dddG}")
        if show_summary:
            f.show_summary()
        return ddG, dddG, f


###########################################
# 2OJ9-tautomer system
###########################################
@pytest.mark.system_2oj9
def test_compare_energies_2OJ9_original_vacuum(caplog):
    caplog.set_level(logging.WARNING)
    from transformato import FreeEnergyCalculator
    import mdtraj as md

    env = "vacuum"

    base = "data/2OJ9-original-2OJ9-tautomer-rsfe/2OJ9-original/"
    output_files_t1, _ = _set_output_files_2oj9_tautomer_pair()

    conf = "transformato/tests/config/test-2oj9-tautomer-pair-rsfe.yaml"

    configuration = load_config_yaml(
        config=conf, input_dir="data/", output_dir="data"
    )  # NOTE: for preprocessing input_dir is the output dir

    f = FreeEnergyCalculator(configuration, "2OJ9-original")
    for idx, b in enumerate(output_files_t1):
        traj = md.load_dcd(
            f"{b}/lig_in_{env}.dcd",
            f"{b}/lig_in_{env}.psf",
        )
        traj.save_dcd(f"{base}/traj.dcd")
        l_charmm = f._evaluated_e_on_all_snapshots_CHARMM(traj, idx + 1, env)
        l_openMM = f._evaluated_e_on_all_snapshots_openMM(traj, idx + 1, env)

        assert len(l_charmm) == len(l_openMM)
        s = abs(np.array(l_charmm) - np.array(l_openMM))
        print(s)
        mae = np.sum(s) / len(s)
        assert mae < 0.005
        for e_charmm, e_openMM in zip(l_charmm, l_openMM):
            assert np.isclose(e_charmm, e_openMM, rtol=0.1)


@pytest.mark.system_2oj9
def test_compare_energies_2OJ9_original_waterbox(caplog):
    caplog.set_level(logging.WARNING)
    from transformato import FreeEnergyCalculator
    import mdtraj as md

    env = "waterbox"

    base = "data/2OJ9-original-2OJ9-tautomer-rsfe/2OJ9-original/"
    output_files_t1, _ = _set_output_files_2oj9_tautomer_pair()

    conf = "transformato/tests/config/test-2oj9-tautomer-pair-rsfe.yaml"

    configuration = load_config_yaml(
        config=conf, input_dir="data/", output_dir="data"
    )  # NOTE: for preprocessing input_dir is the output dir

    f = FreeEnergyCalculator(configuration, "2OJ9-original")
    for idx, b in enumerate(output_files_t1):
        traj = md.load_dcd(
            f"{b}/lig_in_{env}.dcd",
            f"{b}/lig_in_{env}.psf",
        )
        traj.save_dcd(f"{base}/traj.dcd")
        l_charmm = f._evaluated_e_on_all_snapshots_CHARMM(traj, idx + 1, env)
        l_openMM = f._evaluated_e_on_all_snapshots_openMM(traj, idx + 1, env)

        assert len(l_charmm) == len(l_openMM)
        s = abs(np.array(l_charmm) - np.array(l_openMM))
        print(s)
        mae = np.sum(s) / len(s)
        assert mae < 1.0
        for e_charmm, e_openMM in zip(l_charmm, l_openMM):
            assert np.isclose(e_charmm, e_openMM, rtol=0.1)


@pytest.mark.system_2oj9
def test_compare_energies_2OJ9_tautomer_vacuum(caplog):
    caplog.set_level(logging.WARNING)
    from transformato import FreeEnergyCalculator
    import mdtraj as md

    env = "vacuum"
    base = "data/2OJ9-original-2OJ9-tautomer-rsfe/2OJ9-tautomer/"
    _, output_files_t2 = _set_output_files_2oj9_tautomer_pair()

    conf = "transformato/tests/config/test-2oj9-tautomer-pair-rsfe.yaml"

    configuration = load_config_yaml(
        config=conf, input_dir="data/", output_dir="data"
    )  # NOTE: for preprocessing input_dir is the output dir

    f = FreeEnergyCalculator(configuration, "2OJ9-tautomer")
    for idx, b in enumerate(output_files_t2):
        traj = md.load_dcd(
            f"{b}/lig_in_{env}.dcd",
            f"{b}/lig_in_{env}.psf",
        )
        traj.save_dcd(f"{base}/traj.dcd")
        l_charmm = f._evaluated_e_on_all_snapshots_CHARMM(traj, idx + 1, env)
        l_openMM = f._evaluated_e_on_all_snapshots_openMM(traj, idx + 1, env)

        assert len(l_charmm) == len(l_openMM)
        s = abs(np.array(l_charmm) - np.array(l_openMM))
        print(s)
        mae = np.sum(s) / len(s)
        assert mae < 0.005
        for e_charmm, e_openMM in zip(l_charmm, l_openMM):
            assert np.isclose(e_charmm, e_openMM, rtol=1e-2)


@pytest.mark.system_2oj9
def test_compare_energies_2OJ9_tautomer_waterbox(caplog):
    caplog.set_level(logging.WARNING)
    from transformato import FreeEnergyCalculator
    import mdtraj as md

    env = "waterbox"
    base = "data/2OJ9-original-2OJ9-tautomer-rsfe/2OJ9-tautomer/"
    _, output_files_t2 = _set_output_files_2oj9_tautomer_pair()

    conf = "transformato/tests/config/test-2oj9-tautomer-pair-rsfe.yaml"

    configuration = load_config_yaml(
        config=conf, input_dir="data/", output_dir="data"
    )  # NOTE: for preprocessing input_dir is the output dir
    f = FreeEnergyCalculator(configuration, "2OJ9-tautomer")
    for idx, b in enumerate(output_files_t2):
        print(b)
        traj = md.load_dcd(
            f"{b}/lig_in_{env}.dcd",
            f"{b}/lig_in_{env}.psf",
        )
        # traj = traj.center_coordinates()
        traj.save_dcd(f"{base}/traj.dcd", force_overwrite=True)
        l_charmm = f._evaluated_e_on_all_snapshots_CHARMM(traj, idx + 1, env)
        l_openMM = f._evaluated_e_on_all_snapshots_openMM(traj, idx + 1, env)
        assert len(l_charmm) == len(l_openMM)
        s = abs(np.array(l_charmm) - np.array(l_openMM))
        mae = np.sum(s) / len(s)
        print(mae)
        assert mae < 1.0
        for e_charmm, e_openMM in zip(l_charmm, l_openMM):
            print(f"{e_charmm}, {e_openMM}: {e_charmm - e_openMM}")
            assert np.isclose(e_charmm, e_openMM, rtol=0.1)


@pytest.mark.system_2oj9
@pytest.mark.slowtest
@pytest.mark.skipif(
    os.environ.get("TRAVIS", None) == "true", reason="Skip slow test on travis."
)
def test_2oj9_calculate_rsfe_with_openMM_mp(caplog):
    caplog.set_level(logging.DEBUG)

    initialize_NUM_PROC(4)
    conf = "transformato/tests/config/test-2oj9-tautomer-pair-rsfe.yaml"
    configuration = load_config_yaml(
        config=conf, input_dir="data/", output_dir="data"
    )  # NOTE: for preprocessing input_dir is the output dir

    # 2OJ9-original to tautomer common core
    ddG_openMM, dddG, f_openMM = postprocessing(
        configuration, name="2OJ9-original", engine="openMM", max_snapshots=600
    )

    assert np.isclose(ddG_openMM, 3.3739872639241213)


@pytest.mark.system_2oj9
@pytest.mark.slowtest
@pytest.mark.skipif(
    os.environ.get("TRAVIS", None) == "true", reason="Skip slow test on travis."
)
def test_2oj9_calculate_rsfe_with_different_engines():

    conf = "transformato/tests/config/test-2oj9-tautomer-pair-rsfe.yaml"
    configuration = load_config_yaml(
        config=conf, input_dir="data/", output_dir="data"
    )  # NOTE: for preprocessing input_dir is the output dir

    # 2OJ9-original to tautomer common core
    ddG_charmm, dddG, f_charmm = postprocessing(
        configuration, name="2OJ9-original", engine="CHARMM", max_snapshots=600
    )
    ddG_openMM, dddG, f_openMM = postprocessing(
        configuration, name="2OJ9-original", engine="openMM", max_snapshots=600
    )

    assert np.isclose(
        f_charmm.vacuum_free_energy_differences[0, -1],
        f_openMM.vacuum_free_energy_differences[0, -1],
        rtol=1e-4,
    )

    assert np.isclose(
        f_charmm.waterbox_free_energy_differences[0, -1],
        f_openMM.waterbox_free_energy_differences[0, -1],
        rtol=1e-2,
    )

    assert np.isclose(ddG_openMM, ddG_charmm, rtol=1e-2)
    assert np.isclose(ddG_openMM, 3.3739872639241213)
    assert np.isclose(ddG_charmm, 3.3656925062767016)

    # 2OJ9-tautomer to tautomer common core
    ddG_charmm, dddG, f_charmm = postprocessing(
        configuration, name="2OJ9-tautomer", engine="CHARMM", max_snapshots=600
    )
    ddG_openMM, dddG, f_openMM = postprocessing(
        configuration, name="2OJ9-tautomer", engine="openMM", max_snapshots=600
    )

    assert np.isclose(
        f_charmm.vacuum_free_energy_differences[0, -1],
        f_openMM.vacuum_free_energy_differences[0, -1],
        atol=1e-2,
    )

    assert np.isclose(
        f_charmm.waterbox_free_energy_differences[0, -1],
        f_openMM.waterbox_free_energy_differences[0, -1],
        rtol=1e-2,
    )

    assert np.isclose(ddG_openMM, ddG_charmm, rtol=1e-2)
    assert np.isclose(ddG_openMM, 4.721730274995082)
    assert np.isclose(ddG_charmm, 4.722406415490632)


@pytest.mark.system_2oj9
@pytest.mark.slowtest
@pytest.mark.skipif(
    os.environ.get("TRAVIS", None) == "true", reason="Skip slow test on travis."
)
def test_2oj9_calculate_rsfe_with_different_switches(caplog):
    caplog.set_level(logging.WARNING)
    from .test_mutation import setup_2OJ9_tautomer_pair_rsfe
    from .test_run_production import run_simulation
    from ..analysis import FreeEnergyCalculator

    # vfswitch
    conf_path = "transformato/tests/config/test-2oj9-tautomer-pair-rsfe_vfswitch.yaml"
    (output_files_t1, output_files_t2), _, _ = setup_2OJ9_tautomer_pair_rsfe(
        conf_path=conf_path
    )
    run_simulation(output_files_t1)
    configuration = load_config_yaml(
        config=conf_path, input_dir="data/", output_dir="."
    )
    f = FreeEnergyCalculator(configuration, "2OJ9-original")

    # 2OJ9-original to tautomer common core
    ddG_charmm, dddG, f_charmm_vfswitch = postprocessing(
        configuration,
        name="2OJ9-original",
        engine="CHARMM",
        max_snapshots=600,
        different_path_for_dcd="data/2OJ9-original-2OJ9-tautomer-rsfe/2OJ9-original",
    )
    ddG_openMM, dddG, f_openMM_vfswitch = postprocessing(
        configuration,
        name="2OJ9-original",
        engine="openMM",
        max_snapshots=600,
        different_path_for_dcd="data/2OJ9-original-2OJ9-tautomer-rsfe/2OJ9-original",
    )

    assert np.isclose(
        f_charmm_vfswitch.vacuum_free_energy_differences[0, -1],
        f_openMM_vfswitch.vacuum_free_energy_differences[0, -1],
        atol=1e-2,
    )

    assert np.isclose(
        f_charmm_vfswitch.waterbox_free_energy_differences[0, -1],
        f_openMM_vfswitch.waterbox_free_energy_differences[0, -1],
        atol=1e-2,
    )

    # switch
    conf_path = "transformato/tests/config/test-2oj9-tautomer-pair-rsfe_vswitch.yaml"
    (output_files_t1, output_files_t2), _, _ = setup_2OJ9_tautomer_pair_rsfe(
        conf_path=conf_path
    )
    run_simulation(output_files_t1)
    configuration = load_config_yaml(
        config=conf_path, input_dir="data/", output_dir="."
    )  # NOTE: for preprocessing input_dir is the output dir
    f = FreeEnergyCalculator(configuration, "2OJ9-original")

    # 2OJ9-original to tautomer common core
    ddG_charmm, dddG, f_charmm_switch = postprocessing(
        configuration,
        name="2OJ9-original",
        engine="CHARMM",
        max_snapshots=600,
        different_path_for_dcd="data/2OJ9-original-2OJ9-tautomer-rsfe/2OJ9-original",
    )
    ddG_charmm, dddG, f_openMM_switch = postprocessing(
        configuration,
        name="2OJ9-original",
        engine="openMM",
        max_snapshots=600,
        different_path_for_dcd="data/2OJ9-original-2OJ9-tautomer-rsfe/2OJ9-original",
    )

    assert np.isclose(
        f_charmm_switch.vacuum_free_energy_differences[0, -1],
        f_openMM_switch.vacuum_free_energy_differences[0, -1],
        atol=1e-2,
    )

    assert np.isclose(
        f_charmm_switch.waterbox_free_energy_differences[0, -1],
        f_openMM_switch.waterbox_free_energy_differences[0, -1],
        atol=1e-2,
    )

    assert np.isclose(
        f_charmm_vfswitch.vacuum_free_energy_differences[0, -1],
        f_charmm_switch.vacuum_free_energy_differences[0, -1],
        atol=1e-2,
    )
    assert np.isclose(
        f_charmm_vfswitch.waterbox_free_energy_differences[0, -1],
        f_charmm_switch.waterbox_free_energy_differences[0, -1],
        atol=1e-2,
    )
    assert np.isclose(
        f_openMM_vfswitch.waterbox_free_energy_differences[0, -1],
        f_openMM_switch.waterbox_free_energy_differences[0, -1],
        atol=1e-2,
    )


@pytest.mark.system_2oj9
@pytest.mark.slowtest
@pytest.mark.skipif(
    os.environ.get("TRAVIS", None) == "true", reason="Skip slow test on travis."
)
def test_2oj9_calculate_rbfe_with_openMM():

    conf = "transformato/tests/config/test-2oj9-tautomer-pair-rbfe.yaml"
    configuration = load_config_yaml(
        config=conf, input_dir="data/", output_dir="data"
    )  # NOTE: for preprocessing input_dir is the output dir

    # 2OJ9-tautomer to tautomer common core
    ddG_openMM, dddG, f_openMM = postprocessing(
        configuration, name="2OJ9-tautomer", engine="openMM", max_snapshots=10
    )

    assert np.isclose(ddG_openMM, 1.0224154995479893)

    # 2OJ9-original to tautomer common core
    ddG_openMM, dddG, f_openMM = postprocessing(
        configuration, name="2OJ9-original", engine="openMM", max_snapshots=10
    )

    assert np.isclose(ddG_openMM, -23.233893466807686)


###########################################
# acetylacetone-tautomer system
###########################################
@pytest.mark.system_acetylacetone
def test_compare_energies_acetylacetone_enol_vacuum(caplog):
    caplog.set_level(logging.WARNING)
    from transformato import FreeEnergyCalculator
    import mdtraj as md

    env = "vacuum"

    base = "data/acetylacetone-keto-acetylacetone-enol-rsfe/acetylacetone-enol/"
    (
        output_files_enol,
        output_files_keto,
    ) = _set_output_files_acetylaceton_tautomer_pair()

    conf = "transformato/tests/config/test-acetylacetone-tautomer-rsfe.yaml"

    configuration = load_config_yaml(
        config=conf, input_dir="data/", output_dir="data"
    )  # NOTE: for preprocessing input_dir is the output dir

    f = FreeEnergyCalculator(configuration, "acetylacetone-enol")
    for idx, b in enumerate(output_files_enol):
        traj = md.load_dcd(
            f"{b}/lig_in_{env}.dcd",
            f"{b}/lig_in_{env}.psf",
        )
        traj.save_dcd(f"{base}/traj.dcd")
        l_charmm = f._evaluated_e_on_all_snapshots_CHARMM(traj, idx + 1, env)
        l_openMM = f._evaluated_e_on_all_snapshots_openMM(traj, idx + 1, env)

        assert len(l_charmm) == len(l_openMM)
        s = abs(np.array(l_charmm) - np.array(l_openMM))
        print(s)
        mae = np.sum(s) / len(s)
        assert mae < 0.005

        for e_charmm, e_openMM in zip(l_charmm, l_openMM):
            assert np.isclose(e_charmm, e_openMM, rtol=1e-2)


@pytest.mark.system_acetylacetone
def test_compare_energies_acetylacetone_enol_waterbox(caplog):
    caplog.set_level(logging.WARNING)
    from transformato import FreeEnergyCalculator
    import mdtraj as md

    env = "waterbox"

    base = "data/acetylacetone-keto-acetylacetone-enol-rsfe/acetylacetone-enol/"
    (
        output_files_enol,
        output_files_keto,
    ) = _set_output_files_acetylaceton_tautomer_pair()

    conf = "transformato/tests/config/test-acetylacetone-tautomer-rsfe.yaml"

    configuration = load_config_yaml(
        config=conf, input_dir="data/", output_dir="data"
    )  # NOTE: for preprocessing input_dir is the output dir

    f = FreeEnergyCalculator(configuration, "acetylacetone-enol")
    for idx, b in enumerate(output_files_enol):
        traj = md.load_dcd(
            f"{b}/lig_in_{env}.dcd",
            f"{b}/lig_in_{env}.psf",
        )
        traj.save_dcd(f"{base}/traj.dcd")
        l_charmm = f._evaluated_e_on_all_snapshots_CHARMM(traj, idx + 1, env)
        l_openMM = f._evaluated_e_on_all_snapshots_openMM(traj, idx + 1, env)

        assert len(l_charmm) == len(l_openMM)
        s = abs(np.array(l_charmm) - np.array(l_openMM))
        print(s)
        mae = np.sum(s) / len(s)
        assert mae < 1.0

        for e_charmm, e_openMM in zip(l_charmm, l_openMM):
            assert np.isclose(e_charmm, e_openMM, rtol=1e-2)


@pytest.mark.system_acetylacetone
def test_compare_energies_acetylacetone_keto_vacuum(caplog):
    caplog.set_level(logging.WARNING)
    from transformato import FreeEnergyCalculator
    import mdtraj as md

    env = "vacuum"
    base = "data/acetylacetone-keto-acetylacetone-enol-rsfe/acetylacetone-keto/"
    (
        output_files_enol,
        output_files_keto,
    ) = _set_output_files_acetylaceton_tautomer_pair()

    conf = "transformato/tests/config/test-acetylacetone-tautomer-rsfe.yaml"

    configuration = load_config_yaml(
        config=conf, input_dir="data/", output_dir="data"
    )  # NOTE: for preprocessing input_dir is the output dir

    f = FreeEnergyCalculator(configuration, "acetylacetone-keto")
    for idx, b in enumerate(output_files_keto):
        traj = md.load_dcd(
            f"{b}/lig_in_{env}.dcd",
            f"{b}/lig_in_{env}.psf",
        )
        traj.save_dcd(f"{base}/traj.dcd")
        l_charmm = f._evaluated_e_on_all_snapshots_CHARMM(traj, idx + 1, env)
        l_openMM = f._evaluated_e_on_all_snapshots_openMM(traj, idx + 1, env)

        assert len(l_charmm) == len(l_openMM)
        s = abs(np.array(l_charmm) - np.array(l_openMM))
        print(s)
        mae = np.sum(s) / len(s)
        assert mae < 0.005

        for e_charmm, e_openMM in zip(l_charmm, l_openMM):
            assert np.isclose(e_charmm, e_openMM, rtol=1e-2)


@pytest.mark.system_acetylacetone
def test_compare_energies_acetylacetone_keto_waterbox(caplog):
    caplog.set_level(logging.WARNING)
    from transformato import FreeEnergyCalculator
    import mdtraj as md

    env = "waterbox"
    base = "data/acetylacetone-keto-acetylacetone-enol-rsfe/acetylacetone-keto/"
    (
        output_files_enol,
        output_files_keto,
    ) = _set_output_files_acetylaceton_tautomer_pair()

    conf = "transformato/tests/config/test-acetylacetone-tautomer-rsfe.yaml"

    configuration = load_config_yaml(
        config=conf, input_dir="data/", output_dir="data"
    )  # NOTE: for preprocessing input_dir is the output dir

    f = FreeEnergyCalculator(configuration, "acetylacetone-keto")
    for idx, b in enumerate(output_files_keto):
        traj = md.load_dcd(
            f"{b}/lig_in_{env}.dcd",
            f"{b}/lig_in_{env}.psf",
        )
        traj.save_dcd(f"{base}/traj.dcd")
        l_charmm = f._evaluated_e_on_all_snapshots_CHARMM(traj, idx + 1, env)
        l_openMM = f._evaluated_e_on_all_snapshots_openMM(traj, idx + 1, env)

        assert len(l_charmm) == len(l_openMM)
        s = abs(np.array(l_charmm) - np.array(l_openMM))
        print(s)
        mae = np.sum(s) / len(s)
        assert mae < 1.0

        for e_charmm, e_openMM in zip(l_charmm, l_openMM):
            assert np.isclose(e_charmm, e_openMM, rtol=1e-2)


@pytest.mark.system_acetylacetone
@pytest.mark.slowtest
@pytest.mark.skipif(
    os.environ.get("TRAVIS", None) == "true", reason="Skip slow test on travis."
)
def test_acetylacetone_calculate_rsfe_with_different_engines():

    conf = "transformato/tests/config/test-acetylacetone-tautomer-rsfe.yaml"
    configuration = load_config_yaml(
        config=conf, input_dir="data/", output_dir="data"
    )  # NOTE: for preprocessing input_dir is the output dir

    # enol
    ddG_charmm, dddG, f_charmm = postprocessing(
        configuration, name="acetylacetone-enol", engine="CHARMM", max_snapshots=500
    )
    ddG_openMM, dddG, f_openMM = postprocessing(
        configuration, name="acetylacetone-enol", engine="openMM", max_snapshots=500
    )
    assert np.isclose(
        f_charmm.vacuum_free_energy_differences[0, -1],
        f_openMM.vacuum_free_energy_differences[0, -1],
        rtol=1e-4,
    )
    assert np.isclose(
        f_charmm.waterbox_free_energy_differences[0, -1],
        f_openMM.waterbox_free_energy_differences[0, -1],
        rtol=1e-4,
    )

    assert np.isclose(ddG_openMM, ddG_charmm, rtol=1e-1)
    print(ddG_openMM)
    print(ddG_charmm)
    assert np.isclose(ddG_openMM, -0.6532604462065663, rtol=1e-3)
    assert np.isclose(ddG_charmm, -0.6591611245563769, rtol=1e-3)

    # keto
    ddG_charmm, dddG, f_charmm = postprocessing(
        configuration, name="acetylacetone-keto", engine="CHARMM", max_snapshots=600
    )
    ddG_openMM, dddG, f_openMM = postprocessing(
        configuration, name="acetylacetone-keto", engine="openMM", max_snapshots=600
    )
    assert np.isclose(
        f_charmm.vacuum_free_energy_differences[0, -1],
        f_openMM.vacuum_free_energy_differences[0, -1],
        rtol=1e-4,
    )
    assert np.isclose(
        f_charmm.waterbox_free_energy_differences[0, -1],
        f_openMM.waterbox_free_energy_differences[0, -1],
        rtol=0.01,
    )
    assert np.isclose(ddG_openMM, ddG_charmm, rtol=1e-1)
    print(ddG_openMM)
    print(ddG_charmm)
    assert np.isclose(ddG_openMM, 2.691482858438775, rtol=0.01)
    assert np.isclose(ddG_charmm, 2.699116266252545, rtol=0.01)


@pytest.mark.system_acetylacetone
@pytest.mark.slowtest
@pytest.mark.skipif(
    os.environ.get("TRAVIS", None) == "true", reason="Skip slow test on travis."
)
def test_acetylacetone_calculate_rsfe_with_different_engines_only_vacuum():
    from ..constants import kT
    from simtk import unit

    conf = "transformato/tests/config/test-acetylacetone-tautomer-rsfe.yaml"
    configuration = load_config_yaml(
        config=conf, input_dir="data/", output_dir="data"
    )  # NOTE: for preprocessing input_dir is the output dir

    # enol
    ddG_charmm, dddG, f_charmm = postprocessing(
        configuration,
        name="acetylacetone-enol",
        engine="CHARMM",
        max_snapshots=10_000,
        only_vacuum=True,
    )
    print(ddG_charmm, dddG)

    ddG_openMM, dddG, f_openMM = postprocessing(
        configuration,
        name="acetylacetone-enol",
        engine="openMM",
        max_snapshots=10_000,
        only_vacuum=True,
    )
    print(ddG_charmm, dddG)

    assert np.isclose(
        f_charmm.vacuum_free_energy_differences[0, -1],
        f_openMM.vacuum_free_energy_differences[0, -1],
        rtol=1e-4,
    )

    # keto
    ddG_charmm, dddG, f_charmm = postprocessing(
        configuration,
        name="acetylacetone-keto",
        engine="CHARMM",
        max_snapshots=10_000,
        only_vacuum=True,
    )
    print(ddG_charmm, dddG)
    ddG_openMM, dddG, f_openMM = postprocessing(
        configuration,
        name="acetylacetone-keto",
        engine="openMM",
        max_snapshots=10_000,
        only_vacuum=True,
    )
    print(ddG_charmm, dddG)
    assert np.isclose(
        f_charmm.vacuum_free_energy_differences[0, -1],
        f_openMM.vacuum_free_energy_differences[0, -1],
        rtol=1e-4,
    )

    print(
        f"results-openMM: {(f_openMM.vacuum_free_energy_differences[0, -1] * kT).value_in_unit(unit.kilocalorie_per_mole)}"
    )
    print(
        f"results-CHARMM: {(f_charmm.vacuum_free_energy_differences[0, -1] * kT).value_in_unit(unit.kilocalorie_per_mole)}"
    )


###########################################
# toluene-methane system
###########################################
@pytest.mark.system_toluene_methane
def test_compare_energies_methane_vacuum(caplog):
    caplog.set_level(logging.WARNING)
    from transformato import FreeEnergyCalculator
    import mdtraj as md

    env = "vacuum"
    base = "data/toluene-methane-rsfe/methane/"
    (
        output_files_methane,
        output_files_toluene,
    ) = _set_output_files_toluene_methane_pair()

    conf = "transformato/tests/config/test-toluene-methane-rsfe.yaml"

    configuration = load_config_yaml(
        config=conf, input_dir="data/", output_dir="data"
    )  # NOTE: for preprocessing input_dir is the output dir

    f = FreeEnergyCalculator(configuration, "methane")
    for idx, b in enumerate(output_files_methane):
        traj = md.load_dcd(
            f"{b}/lig_in_{env}.dcd",
            f"{b}/lig_in_{env}.psf",
        )
        traj.save_dcd(f"{base}/traj.dcd")
        l_charmm = f._evaluated_e_on_all_snapshots_CHARMM(traj, idx + 1, env)
        l_openMM = f._evaluated_e_on_all_snapshots_openMM(traj, idx + 1, env)

        assert len(l_charmm) == len(l_openMM)
        s = abs(np.array(l_charmm) - np.array(l_openMM))
        mae = np.sum(s) / len(s)
        assert mae < 0.005
        for e_charmm, e_openMM in zip(l_charmm, l_openMM):
            assert np.isclose(e_charmm, e_openMM, rtol=1e-2)


@pytest.mark.system_toluene_methane
def test_compare_energies_methane_waterbox(caplog):
    caplog.set_level(logging.WARNING)
    from transformato import FreeEnergyCalculator
    import mdtraj as md

    env = "waterbox"
    base = "data/toluene-methane-rsfe/methane/"
    (
        output_files_methane,
        output_files_toluene,
    ) = _set_output_files_toluene_methane_pair()

    conf = "transformato/tests/config/test-toluene-methane-rsfe.yaml"

    configuration = load_config_yaml(
        config=conf, input_dir="data/", output_dir="data"
    )  # NOTE: for preprocessing input_dir is the output dir

    f = FreeEnergyCalculator(configuration, "methane")
    for idx, b in enumerate(output_files_methane):
        traj = md.load_dcd(
            f"{b}/lig_in_{env}.dcd",
            f"{b}/lig_in_{env}.psf",
        )
        traj.save_dcd(f"{base}/traj.dcd")
        l_charmm = f._evaluated_e_on_all_snapshots_CHARMM(traj, idx + 1, env)
        l_openMM = f._evaluated_e_on_all_snapshots_openMM(traj, idx + 1, env)

        assert len(l_charmm) == len(l_openMM)
        s = abs(np.array(l_charmm) - np.array(l_openMM))
        mae = np.sum(s) / len(s)
        assert mae < 0.7
        for e_charmm, e_openMM in zip(l_charmm, l_openMM):
            assert np.isclose(e_charmm, e_openMM, rtol=1e-2)


@pytest.mark.system_toluene_methane
def test_compare_energies_toluene_vacuum(caplog):
    caplog.set_level(logging.WARNING)
    from transformato import FreeEnergyCalculator
    import mdtraj as md

    env = "vacuum"
    base = "data/toluene-methane-rsfe/toluene/"
    (
        output_files_methane,
        output_files_toluene,
    ) = _set_output_files_toluene_methane_pair()

    conf = "transformato/tests/config/test-toluene-methane-rsfe.yaml"

    configuration = load_config_yaml(
        config=conf, input_dir="data/", output_dir="data"
    )  # NOTE: for preprocessing input_dir is the output dir

    f = FreeEnergyCalculator(configuration, "toluene")
    for idx, b in enumerate(output_files_toluene):
        traj = md.load_dcd(
            f"{b}/lig_in_{env}.dcd",
            f"{b}/lig_in_{env}.psf",
        )
        traj.save_dcd(f"{base}/traj.dcd")
        l_charmm = f._evaluated_e_on_all_snapshots_CHARMM(traj, idx + 1, env)
        l_openMM = f._evaluated_e_on_all_snapshots_openMM(traj, idx + 1, env)

        assert len(l_charmm) == len(l_openMM)
        s = abs(np.array(l_charmm) - np.array(l_openMM))
        mae = np.sum(s) / len(s)
        assert mae < 0.005
        for e_charmm, e_openMM in zip(l_charmm, l_openMM):
            assert np.isclose(e_charmm, e_openMM, rtol=1e-2)


@pytest.mark.system_toluene_methane
def test_compare_energies_toluene_waterbox(caplog):
    caplog.set_level(logging.WARNING)
    from transformato import FreeEnergyCalculator
    import mdtraj as md

    env = "waterbox"
    base = "data/toluene-methane-rsfe/toluene/"
    (
        output_files_methane,
        output_files_toluene,
    ) = _set_output_files_toluene_methane_pair()

    conf = "transformato/tests/config/test-toluene-methane-rsfe.yaml"

    configuration = load_config_yaml(
        config=conf, input_dir="data/", output_dir="data"
    )  # NOTE: for preprocessing input_dir is the output dir

    f = FreeEnergyCalculator(configuration, "toluene")
    for idx, b in enumerate(output_files_toluene):
        traj = md.load_dcd(
            f"{b}/lig_in_{env}.dcd",
            f"{b}/lig_in_{env}.psf",
        )
        traj.save_dcd(f"{base}/traj.dcd")
        l_charmm = f._evaluated_e_on_all_snapshots_CHARMM(traj, idx + 1, env)
        l_openMM = f._evaluated_e_on_all_snapshots_openMM(traj, idx + 1, env)

        assert len(l_charmm) == len(l_openMM)
        s = abs(np.array(l_charmm) - np.array(l_openMM))
        mae = np.sum(s) / len(s)
        print(mae)
        assert mae < 0.8
        for e_charmm, e_openMM in zip(l_charmm, l_openMM):
            assert np.isclose(e_charmm, e_openMM, rtol=1e-2)


@pytest.mark.system_toluene_methane
@pytest.mark.slowtest
@pytest.mark.skipif(
    os.environ.get("TRAVIS", None) == "true", reason="Skip slow test on travis."
)
def test_toluene_to_methane_calculate_rsfe_with_different_engines():

    conf = "transformato/tests/config/test-toluene-methane-rsfe.yaml"
    configuration = load_config_yaml(
        config=conf, input_dir="data/", output_dir="data"
    )  # NOTE: for preprocessing input_dir is the output dir
    # methane
    ddG_charmm, dddG, f_charmm = postprocessing(
        configuration, name="methane", engine="CHARMM", max_snapshots=600
    )
    ddG_openMM, dddG, f_openMM = postprocessing(
        configuration, name="methane", engine="openMM", max_snapshots=600
    )
    assert np.isclose(
        f_charmm.vacuum_free_energy_differences[0, -1],
        f_openMM.vacuum_free_energy_differences[0, -1],
        rtol=1e-4,
    )
    assert np.isclose(
        f_charmm.waterbox_free_energy_differences[0, -1],
        f_openMM.waterbox_free_energy_differences[0, -1],
        rtol=1e-3,
    )

    assert np.isclose(ddG_openMM, ddG_charmm, rtol=1e-1)
    print(f_charmm.vacuum_free_energy_differences[0, -1])
    print(f_openMM.vacuum_free_energy_differences[0, -1])
    assert np.isclose(ddG_openMM, -1.3681988336807516)
    assert np.isclose(ddG_charmm, -1.3674494407692004)

    # toluene
    ddG_charmm, dddG, f_charmm = postprocessing(
        configuration, name="toluene", engine="CHARMM", max_snapshots=600
    )
    ddG_openMM, dddG, f_openMM = postprocessing(
        configuration, name="toluene", engine="openMM", max_snapshots=600
    )
    assert np.isclose(
        f_charmm.vacuum_free_energy_differences[0, -1],
        f_openMM.vacuum_free_energy_differences[0, -1],
        rtol=1e-4,
    )
    assert np.isclose(
        f_charmm.waterbox_free_energy_differences[0, -1],
        f_openMM.waterbox_free_energy_differences[0, -1],
        rtol=1e-4,
    )

    assert np.isclose(ddG_openMM, ddG_charmm, rtol=1e-1)
    print(f_charmm.vacuum_free_energy_differences[0, -1])
    print(f_openMM.vacuum_free_energy_differences[0, -1])
    assert np.isclose(ddG_openMM, 5.651522313536532)
    assert np.isclose(ddG_charmm, 5.651678173410401)


def test_postprocessing_thinning():
    from transformato import FreeEnergyCalculator

    conf = "transformato/tests/config/test-toluene-methane-rsfe.yaml"
    configuration = load_config_yaml(
        config=conf, input_dir="data/", output_dir="data"
    )  # NOTE: for preprocessing input_dir is the output dir

    f = FreeEnergyCalculator(configuration, "methane")
    f.load_trajs(nr_of_max_snapshots=300)
    assert len(f.snapshots.keys()) == 2  # entry for vacuum and waterbox
    assert f.nr_of_states == 3  # nr of lambda states

    print(f.snapshots["vacuum"])
    assert len(f.snapshots["vacuum"]) == 900
    assert len(f.snapshots["waterbox"]) == 900

    f = FreeEnergyCalculator(configuration, "methane")
    f.load_trajs(nr_of_max_snapshots=200)

    print(f.snapshots["vacuum"])
    assert len(f.snapshots["vacuum"]) == 600
    assert len(f.snapshots["waterbox"]) == 600

    f = FreeEnergyCalculator(configuration, "toluene")
    f.load_trajs(nr_of_max_snapshots=200)
    assert len(f.snapshots.keys()) == 2  # entry for vacuum and waterbox
    assert f.nr_of_states == 13  # nr of lambda states

    print(f.snapshots["vacuum"])
    assert len(f.snapshots["vacuum"]) == 1950
    assert len(f.snapshots["waterbox"]) == 1950
