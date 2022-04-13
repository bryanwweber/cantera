from pathlib import Path

import cantera
import pytest


@pytest.fixture(scope="session", autouse=True)
def TEST_DATA_PATH():
    return Path(__file__).parent / "data"


@pytest.fixture(scope="session", autouse=True)
def CANTERA_DATA_PATH():
    return Path(__file__).parents[1] / "data"


@pytest.fixture
def allow_deprecated():
    cantera.suppress_deprecation_warnings()
    yield
    cantera.make_deprecation_warnings_fatal()


@pytest.fixture
def has_temperature_derivative_warnings():
    with pytest.warns(UserWarning, match="ddTScaledFromStruct"):
        # test warning raised for BlowersMasel and TwoTempPlasma derivatives
        yield
