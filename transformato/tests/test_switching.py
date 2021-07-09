"""
Unit and regression test for the transformato package.
"""

import os
import logging
import numpy as np
import pytest
from transformato import load_config_yaml, FreeEnergyCalculator
import mdtraj as md
from .test_run_production import run_simulation


@pytest.mark.slowtest
@pytest.mark.skipif(
    os.environ.get("TRAVIS", None) == "true", reason="Skip slow test on travis."
)
def test_run_2OJ9_tautomer_pair_vfswitch(caplog):
    caplog.set_level(logging.WARNING)
    from .test_mutation import setup_2OJ9_tautomer_pair_rsfe

    conf_path = "transformato/tests/config/test-2oj9-tautomer-pair-rsfe_vfswitch.yaml"
    env = "waterbox"
    system_path = "2OJ9-original-2OJ9-tautomer-rsfe/2OJ9-original/"

    (output_files_t1, output_files_t2), _, _ = setup_2OJ9_tautomer_pair_rsfe(
        single_state=True, conf_path=conf_path
    )
    run_simulation(output_files_t1[:1])
    configuration = load_config_yaml(
        config=conf_path, input_dir="data/", output_dir="."
    )  # NOTE: for preprocessing input_dir is the output dir
    f = FreeEnergyCalculator(configuration, "2OJ9-original")

    # load pregenerated dcd
    traj = md.load_dcd(
        f"data/{system_path}/intst1/lig_in_{env}.dcd",
        f"data/{system_path}/intst1/lig_in_{env}.psf",
    )
    # save dcd in freshly generated system
    traj.save_dcd(f"{system_path}/traj.dcd", force_overwrite=True)
    l_charmm = f._evaluated_e_on_all_snapshots_CHARMM(traj, 1, env)
    l_openMM = f._evaluated_e_on_all_snapshots_openMM(traj, 1, env)
    assert len(l_charmm) == len(l_openMM)
    s = abs(np.array(l_charmm) - np.array(l_openMM))
    mae = np.sum(s) / len(s)

    assert np.isclose(l_charmm[0], -16377.666636494012, atol=1e-4)
    assert np.isclose(l_openMM[0], -16377.00295248849, atol=1e-4)
    assert mae < 0.6

    for e_charmm, e_openMM in zip(l_charmm, l_openMM):
        print(f"{e_charmm}, {e_openMM}: {e_charmm - e_openMM}")
        assert np.isclose(e_charmm, e_openMM, atol=1.5)


@pytest.mark.slowtest
@pytest.mark.skipif(
    os.environ.get("TRAVIS", None) == "true", reason="Skip slow test on travis."
)
def test_run_2OJ9_tautomer_pair_vswitch(caplog):
    caplog.set_level(logging.WARNING)
    from .test_mutation import setup_2OJ9_tautomer_pair_rsfe

    conf_path = "transformato/tests/config/test-2oj9-tautomer-pair-rsfe_vswitch.yaml"
    env = "waterbox"
    system_path = "2OJ9-original-2OJ9-tautomer-rsfe/2OJ9-original/"

    (output_files_t1, output_files_t2), _, _ = setup_2OJ9_tautomer_pair_rsfe(
        single_state=True, conf_path=conf_path
    )
    run_simulation(output_files_t1[:1])
    configuration = load_config_yaml(
        config=conf_path, input_dir="data/", output_dir="."
    )  # NOTE: for preprocessing input_dir is the output dir
    f = FreeEnergyCalculator(configuration, "2OJ9-original")

    # load pregenerated dcd
    traj = md.load_dcd(
        f"data/{system_path}/intst1/lig_in_{env}.dcd",
        f"data/{system_path}/intst1/lig_in_{env}.psf",
    )
    # save dcd in freshly generated system
    traj.save_dcd(f"{system_path}/traj.dcd", force_overwrite=True)
    l_charmm = f._evaluated_e_on_all_snapshots_CHARMM(traj, 1, env)
    l_openMM = f._evaluated_e_on_all_snapshots_openMM(traj, 1, env)
    assert len(l_charmm) == len(l_openMM)
    assert np.isclose(l_charmm[0], -16419.20921476934, atol=1e-4)
    assert np.isclose(l_openMM[0], -16418.247444995068, atol=1e-4)

    s = abs(np.array(l_charmm) - np.array(l_openMM))
    mae = np.sum(s) / len(s)
    print(mae)
    assert mae < 0.75
    for e_charmm, e_openMM in zip(l_charmm, l_openMM):
        print(f"{e_charmm}, {e_openMM}: {e_charmm - e_openMM}")
        assert np.isclose(e_charmm, e_openMM, atol=1.9)


@pytest.mark.slowtest
@pytest.mark.skipif(
    os.environ.get("TRAVIS", None) == "true", reason="Skip slow test on travis."
)
def test_run_2OJ9_tautomer_pair_no_switch(caplog):
    # NOTE: THIS DOES NOT WORK! IF NO SWITCH IS SPECIFIED, TRANFORMATO WILL USE VFSWITCH

    caplog.set_level(logging.WARNING)
    from .test_mutation import setup_2OJ9_tautomer_pair_rsfe

    conf_path = "transformato/tests/config/test-2oj9-tautomer-pair-rsfe_no-switch.yaml"
    env = "waterbox"
    system_path = "2OJ9-original-2OJ9-tautomer-rsfe/2OJ9-original/"

    (output_files_t1, output_files_t2), _, _ = setup_2OJ9_tautomer_pair_rsfe(
        single_state=True, conf_path=conf_path
    )
    run_simulation(output_files_t1[:1])
    configuration = load_config_yaml(
        config=conf_path, input_dir="data/", output_dir="."
    )  # NOTE: for preprocessing input_dir is the output dir
    f = FreeEnergyCalculator(configuration, "2OJ9-original")

    # load pregenerated dcd
    traj = md.load_dcd(
        f"data/{system_path}/intst1/lig_in_{env}.dcd",
        f"data/{system_path}/intst1/lig_in_{env}.psf",
    )
    # save dcd in freshly generated system
    traj.save_dcd(f"{system_path}/traj.dcd", force_overwrite=True)
    l_charmm = f._evaluated_e_on_all_snapshots_CHARMM(traj, 1, env)
    l_openMM = f._evaluated_e_on_all_snapshots_openMM(traj, 1, env)
    assert len(l_charmm) == len(l_openMM)
    assert np.isclose(l_charmm[0], -16377.666636494012, atol=1e-4)
    assert np.isclose(l_openMM[0], -16377.002950975046, atol=1e-4)

    s = abs(np.array(l_charmm) - np.array(l_openMM))
    mae = np.sum(s) / len(s)
    print(mae)
    assert mae < 0.75
    for e_charmm, e_openMM in zip(l_charmm, l_openMM):
        print(f"{e_charmm}, {e_openMM}: {e_charmm - e_openMM}")
        assert np.isclose(e_charmm, e_openMM, atol=1.9)
