import pytest


@pytest.mark.rsfe
def test_rsfe_mutate_acetylaceton_methyl_common_core():
    from ..testsystems import mutate_acetylaceton_methyl_common_core

    mutate_acetylaceton_methyl_common_core(
        conf_path="transformato/tests/config/test-acetylacetone-tautomer-rsfe.yaml",
        input_dir="data/",
        output_dir=".",
    )


@pytest.mark.rsfe
def test_rsfe_mutate_bmi_small_common_core():
    from ..testsystems import mutate_bmi_small_common_core

    mutate_bmi_small_common_core(
        conf_path="transformato/tests/config/test-2oj9-tautomer-pair-rsfe.yaml",
        input_dir="data/",
        output_dir=".",
    )


@pytest.mark.rbfe
def test_rbfe_mutate_2oj9():
    from ..mutate import mutate_pure_tautomers
    from transformato import (
        ProposeMutationRoute,
        SystemStructure,
        load_config_yaml,
    )

    conf_path = "transformato/tests/config/test-2oj9-tautomer-pair-rbfe.yaml"

    configuration = load_config_yaml(
        config=conf_path, input_dir="data/", output_dir="."
    )
    #check_platform(configuration)

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
            nr_of_bonded_windows=4,
        ),
        configuration,
        s1_to_s2,
    )
