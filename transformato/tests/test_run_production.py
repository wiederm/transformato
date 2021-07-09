"""
Unit and regression test for the transformato package.
"""

import os
import subprocess
import logging

import numpy as np
import pytest
import shutil


def run_simulation(output_files, engine="openMM", only_vacuum=False):
    for path in sorted(output_files):
        # because path is object not string
        print(f"Start sampling for: {path}")
        runfile = "simulation.sh"
        calculate_solv_and_vac = 2  # 2 means yes, 1 only vacuum
        if engine.upper() == "CHARMM":
            runfile = "simulation_charmm.sh"
        if only_vacuum:
            calculate_solv_and_vac = 1

        exe = subprocess.run(
            ["bash", f"{str(path)}/{runfile}", str(path), str(calculate_solv_and_vac)],
            check=True,
            capture_output=True,
            text=True,
        )
        print(exe.stdout)
        print("Capture stderr")
        print(exe.stderr)


@pytest.mark.slowtest
@pytest.mark.rsfe
@pytest.mark.skipif(
    os.environ.get("TRAVIS", None) == "true", reason="Skip slow test on travis."
)
def test_run_toluene_to_methane_cc_rsfe_with_openMM():
    from transformato.testsystems import mutate_toluene_to_methane_cc
    from .test_run_production import run_simulation

    output_files, configuration = mutate_toluene_to_methane_cc(
        conf="transformato/tests/config/test-toluene-methane-rsfe.yaml"
    )

    run_simulation(output_files)
    f = "/".join(output_files[0].split("/")[:-3])
    print(f)
    shutil.rmtree(f)


@pytest.mark.slowtest
@pytest.mark.rsfe
@pytest.mark.skipif(
    os.environ.get("TRAVIS", None) == "true", reason="Skip slow test on travis."
)
def test_run_methane_to_methane_cc_rsfe_with_openMM():
    from transformato.testsystems import mutate_methane_to_methane_cc
    from .test_run_production import run_simulation

    output_files, configuration = mutate_methane_to_methane_cc(
        conf="transformato/tests/config/test-toluene-methane-rsfe.yaml"
    )
    run_simulation(output_files)
    f = "/".join(output_files[0].split("/")[:-3])
    print(f)
    shutil.rmtree(f)


@pytest.mark.slowtest
@pytest.mark.rsfe
@pytest.mark.skipif(
    os.environ.get("TRAVIS", None) == "true", reason="Skip slow test on travis."
)
def test_run_methane_to_methane_cc_rsfe_with_CHARMM():
    from transformato.testsystems import mutate_methane_to_methane_cc
    from .test_run_production import run_simulation
    from .test_postprocessing import postprocessing

    output_files, configuration = mutate_methane_to_methane_cc(
        conf="transformato/tests/config/test-toluene-methane-rsfe.yaml",
        output_dir=".",
    )
    run_simulation(output_files)
    f = "/".join(output_files[0].split("/")[:-3])
    print(f)
    shutil.rmtree(f)


@pytest.mark.slowtest
@pytest.mark.rsfe
@pytest.mark.skipif(
    os.environ.get("TRAVIS", None) == "true", reason="Skip slow test on travis."
)
def test_run_acetylacetone_tautomer_pair_rsfe(caplog):
    caplog.set_level(logging.WARNING)
    from .test_mutation import setup_acetylacetone_tautomer_pair
    from .test_run_production import run_simulation

    (output_files_t1, output_files_t2), _, _ = setup_acetylacetone_tautomer_pair()
    run_simulation(output_files_t1)
    run_simulation(output_files_t2)
    f = "/".join(output_files_t1[0].split("/")[:-3])
    print(f)
    shutil.rmtree(f)


@pytest.mark.slowtest
@pytest.mark.skipif(
    os.environ.get("TRAVIS", None) == "true", reason="Skip slow test on travis."
)
def test_run_acetylacetone_tautomer_pair_only_in_vacuum(caplog):
    caplog.set_level(logging.WARNING)
    from .test_mutation import setup_acetylacetone_tautomer_pair
    from .test_run_production import run_simulation

    (output_files_t1, output_files_t2), _, _ = setup_acetylacetone_tautomer_pair(
        nr_of_bonded_windows=16
    )
    run_simulation(output_files_t1, only_vacuum=True)
    run_simulation(output_files_t2, only_vacuum=True)
    f = "/".join(output_files_t1[0].split("/")[:-3])
    print(f)
    shutil.rmtree(f)


@pytest.mark.slowtest
@pytest.mark.rsfe
@pytest.mark.skipif(
    os.environ.get("TRAVIS", None) == "true", reason="Skip slow test on travis."
)
def test_run_2OJ9_tautomer_pair_rsfe_openMM(caplog):
    caplog.set_level(logging.WARNING)
    from .test_mutation import setup_2OJ9_tautomer_pair_rsfe
    from .test_run_production import run_simulation

    conf_path = "transformato/tests/config/test-2oj9-tautomer-pair-rsfe.yaml"

    (output_files_t1, output_files_t2), _, _ = setup_2OJ9_tautomer_pair_rsfe(
        conf_path=conf_path
    )
    run_simulation(output_files_t1)
    run_simulation(output_files_t2)
    f = "/".join(output_files_t1[0].split("/")[:-3])
    print(f)
    shutil.rmtree(f)


@pytest.mark.slowtest
@pytest.mark.rbfe
@pytest.mark.skipif(
    os.environ.get("TRAVIS", None) == "true", reason="Skip slow test on travis."
)
def test_run_2OJ9_tautomer_pair_rbfe_openMM(caplog):
    caplog.set_level(logging.WARNING)
    from .test_mutation import setup_2OJ9_tautomer_pair_rbfe
    from .test_run_production import run_simulation

    conf_path = "transformato/tests/config/test-2oj9-tautomer-pair-rbfe.yaml"

    (output_files_t1, output_files_t2), _, _ = setup_2OJ9_tautomer_pair_rbfe(
        conf_path=conf_path
    )
    run_simulation(output_files_t1)
    run_simulation(output_files_t2)
    f = "/".join(output_files_t1[0].split("/")[:-3])
    print(f)
    shutil.rmtree(f)


@pytest.mark.slowtest
@pytest.mark.rsfe
@pytest.mark.skipif(
    os.environ.get("TRAVIS", None) == "true", reason="Skip slow test on travis."
)
def test_run_2OJ9_tautomer_pair_charmm(caplog):
    caplog.set_level(logging.WARNING)
    from .test_mutation import setup_2OJ9_tautomer_pair_rsfe
    from .test_run_production import run_simulation

    conf_path = "transformato/tests/config/test-2oj9-tautomer-pair-rsfe.yaml"

    (output_files_t1, output_files_t2), _, _ = setup_2OJ9_tautomer_pair_rsfe(
        conf_path=conf_path
    )
    run_simulation(output_files_t1, engine="CHARMM")
    run_simulation(output_files_t2, engine="CHARMM")
    f = "/".join(output_files_t1[0].split("/")[:-3])
    print(f)
    shutil.rmtree(f)


@pytest.mark.slowtest
@pytest.mark.skipif(
    os.environ.get("TRAVIS", None) == "true", reason="Skip slow test on travis."
)
def test_get_free_energy_acetylaceton_tautomer_pair(caplog):
    caplog.set_level(logging.WARNING)
    from .test_mutation import setup_acetylacetone_tautomer_pair
    from .test_run_production import run_simulation
    from .test_postprocessing import postprocessing

    (output_files_t1, output_files_t2), conf, _ = setup_acetylacetone_tautomer_pair()
    run_simulation(output_files_t1, only_vacuum=True)
    run_simulation(output_files_t2, only_vacuum=True)
    f = "/".join(output_files_t1[0].split("/")[:-3])
    print(f)
    shutil.rmtree(f)
