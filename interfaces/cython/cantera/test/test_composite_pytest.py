import cantera as ct
import pytest
from ruamel import yaml


def load_yaml(yml_file):
    # Load YAML data from file using the "safe" loading option.
    try:
        yaml_ = yaml.YAML(typ="safe")
        with open(yml_file, "rt", encoding="utf-8") as stream:
            return yaml_.load(stream)
    except yaml.constructor.ConstructorError:
        with open(yml_file, "rt", encoding="utf-8") as stream:
            # Ensure that the loader remains backward-compatible with legacy
            # ruamel.yaml versions (prior to 0.17.0).
            return yaml.safe_load(stream)


def test_load_thermo_models(TEST_DATA_PATH):
    yml = load_yaml(TEST_DATA_PATH / "thermo-models.yaml")
    for ph in yml["phases"]:
        assert 0, ph


def test_empty_report():
    gas = ct.ThermoPhase()
    with pytest.raises(ct.CanteraError, match="NotImplementedError"):
        gas()


def test_empty_TP():
    with pytest.raises(ct.CanteraError, match="NotImplementedError"):
        ct.ThermoPhase().TP = 300, ct.one_atm


def test_empty_equilibrate():
    with pytest.raises(ct.CanteraError, match="NotImplementedError"):
        ct.ThermoPhase().equilibrate("TP")
