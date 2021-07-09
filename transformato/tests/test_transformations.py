"""
Unit and regression test for the transformato package.
"""

import sys

import numpy as np
import parmed as pm
import pytest
import shutil

# Import package, test suite, and other packages as needed
import transformato
import logging

# read in specific topology with parameters
from transformato import (
    load_config_yaml,
)


def test_transformato_imported():
    """Sample test, will always pass so long as import statement worked"""
    assert "transformato" in sys.modules


def test_generate_output_for_acetylacetone_tautomer_pair(caplog):
    caplog.set_level(logging.WARNING)
    from .test_mutation import setup_acetylacetone_tautomer_pair

    (output_files_t1, output_files_t2), _, _ = setup_acetylacetone_tautomer_pair(
        nr_of_bonded_windows=8
    )


def test_generate_output_for_methane_cc_solvation_free_energy():
    from transformato.testsystems import mutate_methane_to_methane_cc

    output_files, configuration = mutate_methane_to_methane_cc(
        conf="transformato/tests/config/test-toluene-methane-rsfe.yaml"
    )

    assert len(output_files) == 3
    f = "/".join(output_files[0].split("/")[:-3])
    print(f)
    shutil.rmtree(f)


def test_generate_output_for_toluene_cc_solvation_free_energy():
    from transformato.testsystems import mutate_toluene_to_methane_cc

    output_files, configuration = mutate_toluene_to_methane_cc(
        conf="transformato/tests/config/test-toluene-methane-rsfe.yaml"
    )

    f = "/".join(output_files[0].split("/")[:-3])
    print(f)
    shutil.rmtree(f)


def test_generate_output_for_neopentane_cc_solvation_free_energy():
    from transformato.testsystems import mutate_neopentane_to_methane_cc

    output_files, configuration = mutate_neopentane_to_methane_cc(
        conf="transformato/tests/config/test-neopentane-methane-rsfe.yaml"
    )
    print(output_files)
    f = "/".join(output_files[0].split("/")[:-3])
    print(f)
    shutil.rmtree(f)


def test_generate_output_for_methanol_cc_solvation_free_energy():
    from transformato.testsystems import mutate_methanol_to_methane_cc

    output_files, _ = mutate_methanol_to_methane_cc(
        conf="transformato/tests/config/test-methanol-methane-rsfe.yaml"
    )
    print(output_files)
    f = "/".join(output_files[0].split("/")[:-3])
    print(f)
    shutil.rmtree(f)


def test_generate_output_for_2_CPI_solvation_free_energy():
    from transformato.testsystems import mutate_2_CPI_to_7_CPI_cc

    output_files, _ = mutate_2_CPI_to_7_CPI_cc(
        conf="transformato/tests/config/test-7-CPI-2-CPI-rsfe.yaml"
    )
    print(output_files)
    f = "/".join(output_files[0].split("/")[:-3])
    print(f)
    shutil.rmtree(f)


def test_generate_output_for_7_CPI_solvation_free_energy():
    from transformato.testsystems import mutate_7_CPI_to_2_CPI_cc

    output_files, _ = mutate_7_CPI_to_2_CPI_cc(
        conf="transformato/tests/config/test-7-CPI-2-CPI-rsfe.yaml"
    )
    print(output_files)
    f = "/".join(output_files[0].split("/")[:-3])
    print(f)
    shutil.rmtree(f)
