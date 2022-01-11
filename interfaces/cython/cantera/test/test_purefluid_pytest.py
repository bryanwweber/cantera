import pytest
import math
import cantera as ct
from itertools import permutations
import numpy as np


@pytest.fixture
def water():
    return ct.Water()


def test_critical_properties(water):
    assert np.isclose(water.critical_pressure, 22.089e6)
    assert np.isclose(water.critical_temperature, 647.286)
    assert np.isclose(water.critical_density, 317.0)


def test_set_quality(water):
    water.PQ = 101325, 0.5
    assert np.isclose(water.P, 101325.0)
    assert np.isclose(water.Q, 0.5)

    water.TQ = 500, 0.8
    assert np.isclose(water.T, 500.0)
    assert np.isclose(water.Q, 0.8)


def states():
    s = {"T": 400.0, "V": 1.45, "P": 101325.0, "H": -1.45e7, "U": -1.45e7, "S": 5000.0}
    return "properties,values", [
        ("".join(p), (s[p[0]], s[p[1]])) for p in permutations(s.keys(), 2)
    ]


@pytest.mark.parametrize(*states())
def test_set(water, properties, values):
    try:
        setattr(water, properties, values)
    except AttributeError:
        pass
    else:
        prop_0 = properties[0].lower() if properties[0] in "VHUS" else properties[0]
        prop_1 = properties[1].lower() if properties[1] in "VHUS" else properties[1]
        assert np.isclose(getattr(water, prop_0), values[0])
        assert np.isclose(getattr(water, prop_1), values[1])


def test_negative_specific_volume(water):
    with pytest.raises(ct.CanteraError, match=".*Negative specific volume.*"):
        water.TV = 400, -1.0


def test_native_states(water):
    assert water._native_state == ("T", "D")
    assert "TPY" not in water._full_states.values()
    assert "TQ" in water._partial_states.values()


def test_set_Q(water):
    water.TQ = 500.0, 0.0
    p = water.P
    water.Q = 0.8
    assert np.isclose(water.P, p)
    assert np.isclose(water.T, 500.0)
    assert np.isclose(water.Q, 0.8)


def test_set_Q_above_critical_T(water):
    water.TP = 650.0, 101325.0
    with pytest.raises(ct.CanteraError, match="Illegal temperature value.*"):
        water.Q = 0.1


def test_set_Q_outside_vapor_dome(water):
    water.TP = 450.0, 101325.0
    with pytest.raises(ValueError, match="Cannot set vapor quality outside the.*"):
        water.Q = 0.1


def test_set_minmax(water):
    water.TP = water.min_temp, 101325.0
    assert np.isclose(water.T, water.min_temp)

    water.TP = water.max_temp, 101325.0
    assert np.isclose(water.T, water.max_temp)


def check_fd_properties(phase, T1, P1, T2, P2, tol):
    # Properties which are computed as finite differences
    phase.TP = T1, P1
    h1a = phase.enthalpy_mass
    cp1 = phase.cp_mass
    cv1 = phase.cv_mass
    k1 = phase.isothermal_compressibility
    alpha1 = phase.thermal_expansion_coeff
    h1b = phase.enthalpy_mass

    phase.TP = T2, P2
    h2a = phase.enthalpy_mass
    cp2 = phase.cp_mass
    cv2 = phase.cv_mass
    k2 = phase.isothermal_compressibility
    alpha2 = phase.thermal_expansion_coeff
    h2b = phase.enthalpy_mass

    assert np.isclose(cp1, cp2, rtol=tol)
    assert np.isclose(cv1, cv2, rtol=tol)
    assert np.isclose(k1, k2, rtol=tol)
    assert np.isclose(alpha1, alpha2, rtol=tol)

    # calculating these finite difference properties should not perturb the
    # state of the object (except for checks on edge cases)
    assert np.isclose(h1a, h1b, rtol=1e-9)
    assert np.isclose(h2a, h2b, rtol=1e-9)


def test_properties_near_min(water):
    T = water.min_temp
    check_fd_properties(
        water,
        T * (1 + 1e-5),
        101325.0,
        T * (1 + 1e-4),
        101325.0,
        1e-2,
    )


def test_properties_near_max(water):
    T = water.max_temp
    check_fd_properties(
        water,
        T * (1 - 1e-5),
        101325,
        T * (1 - 1e-4),
        101325,
        1e-2,
    )


def test_properties_near_sat1(water):
    for T in [340.0, 390.0, 420.0]:
        water.TQ = T, 0.0
        P = water.P
        check_fd_properties(water, T, P + 0.01, T, P + 0.5, 1e-4)


def test_properties_near_sat2(water):
    for T in [340.0, 390.0, 420.0]:
        water.TQ = T, 0.0
        P = water.P
        check_fd_properties(water, T, P - 0.01, T, P - 0.5, 1e-4)


def test_isothermal_compressibility_lowP(water):
    # Low-pressure limit corresponds to ideal gas
    ref = ct.Solution("h2o2.yaml", transport_model=None)
    ref.TPX = 450, 12, "H2O:1.0"
    water.TP = 450, 12
    assert np.isclose(
        ref.isothermal_compressibility, water.isothermal_compressibility, rtol=1e-5
    )


def test_thermal_expansion_coeff_lowP(water):
    # Low-pressure limit corresponds to ideal gas
    ref = ct.Solution("h2o2.yaml", transport_model=None)
    ref.TPX = 450, 12, "H2O:1.0"
    water.TP = 450, 12
    assert np.isclose(
        ref.thermal_expansion_coeff, water.thermal_expansion_coeff, rtol=1e-5
    )


def test_thermal_expansion_coeff_TD(water):
    for T in [440.0, 550.0, 660.0]:
        water.TD = T, 0.1
        assert np.isclose(T * water.thermal_expansion_coeff, 1.0, rtol=1e-2)


def test_pq_setter_triple_check(water):
    water.PQ = 101325, 0.2
    T = water.T
    # change T such that it would result in a Psat larger than P
    water.TP = 400, 101325
    # ensure that correct triple point pressure is recalculated
    # (necessary as this value is not stored by the C++ base class)
    water.PQ = 101325, 0.2
    assert np.isclose(T, water.T, rtol=1e-9)

    def test_pq_setter_error_below_triple_point(water):
        # min_temp is triple point temperature
        water.TP = water.min_temp, 101325
        P = water.P_sat  # triple-point pressure
        with pytest.raises(ct.CanteraError, match=".*below triple point.*"):
            water.PQ = 0.999 * P, 0.2


def test_quality_exceptions(water):
    # Critical point
    water.TP = 300.0, ct.one_atm
    water.TQ = water.critical_temperature, 0.5
    assert np.isclose(water.P, water.critical_pressure)
    water.TP = 300.0, ct.one_atm
    water.PQ = water.critical_pressure, 0.5
    assert np.isclose(water.T, water.critical_temperature)

    # Supercritical
    with pytest.raises(ct.CanteraError, match="supercritical"):
        water.TQ = 1.001 * water.critical_temperature, 0.0
        with pytest.raises(ct.CanteraError, match="supercritical"):
            water.PQ = 1.001 * water.critical_pressure, 0.0

    # Q negative
    with pytest.raises(ct.CanteraError, match="Invalid vapor fraction"):
        water.TQ = 373.15, -0.001
        with pytest.raises(ct.CanteraError, match="Invalid vapor fraction"):
            water.PQ = ct.one_atm, -0.001

    # Q larger than one
    with pytest.raises(ct.CanteraError, match="Invalid vapor fraction"):
        water.TQ = 373.15, 1.001
        with pytest.raises(ct.CanteraError, match="Invalid vapor fraction"):
            water.PQ = ct.one_atm, 1.001


def test_set_saturated_mixture_raises(water):
    water.TP = 300, ct.one_atm
    with pytest.raises(ct.CanteraError, match="Saturated mixture detected"):
        water.TP = 300, water.P_sat


def test_saturated_vapor_properties(water):
    w = ct.Water()

    # Saturated vapor
    water.TQ = 373.15, 1.0
    assert water.phase_of_matter == "liquid-gas-mix"
    w.TP = water.T, 0.999 * water.P_sat
    assert np.isclose(water.cp, w.cp, rtol=1.0e-3)
    assert np.isclose(water.cv, w.cv, rtol=1.0e-3)
    assert np.isclose(
        water.thermal_expansion_coeff, w.thermal_expansion_coeff, rtol=1.0e-3
    )
    assert np.isclose(
        water.isothermal_compressibility, w.isothermal_compressibility, rtol=1.0e-3
    )


def test_saturated_mix_properties(water):
    # Saturated mixture
    water.TQ = 373.15, 0.5
    assert water.phase_of_matter == "liquid-gas-mix"
    assert np.isinf(water.cp)
    assert np.isnan(water.cv)
    assert np.isinf(water.isothermal_compressibility)
    assert np.isinf(water.thermal_expansion_coeff)


def test_saturated_liquid_properties(water):
    w = ct.Water()
    # Saturated liquid
    water.TQ = 373.15, 0.0
    assert water.phase_of_matter == "liquid-gas-mix"
    w.TP = water.T, 1.001 * water.P_sat
    assert np.isclose(water.cp, w.cp, rtol=1.0e-3)
    assert np.isclose(water.cv, w.cv, rtol=1.0e-3)
    assert np.isclose(
        water.thermal_expansion_coeff, w.thermal_expansion_coeff, rtol=1.0e-3
    )
    assert np.isclose(
        water.isothermal_compressibility, w.isothermal_compressibility, rtol=1.0e-3
    )


def test_saturation_near_low_T_limit(water):
    # Low temperature limit (triple point)
    water.TP = 300, ct.one_atm
    water.P_sat  # ensure that solver buffers sufficiently different values
    water.TP = water.min_temp, ct.one_atm
    psat = water.P_sat
    water.TP = 300, ct.one_atm
    water.P_sat  # ensure that solver buffers sufficiently different values
    water.TP = 300, psat
    assert np.isclose(water.T_sat, water.min_temp)


def test_saturation_near_high_T_limit(water):
    # High temperature limit (critical point) - saturation temperature
    water.TP = 300, ct.one_atm
    water.P_sat  # ensure that solver buffers sufficiently different values
    water.TP = water.critical_temperature, water.critical_pressure
    assert np.isclose(water.T_sat, water.critical_temperature)

    # High temperature limit (critical point) - saturation pressure
    water.TP = 300, ct.one_atm
    water.P_sat  # ensure that solver buffers sufficiently different values
    water.TP = water.critical_temperature, water.critical_pressure
    assert np.isclose(water.P_sat, water.critical_pressure)


def test_supercritical_setter_raises(water):
    # Supercricital
    with pytest.raises(ct.CanteraError, match="Illegal temperature value"):
        water.TP = (
            1.001 * water.critical_temperature,
            water.critical_pressure,
        )
        water.P_sat
    with pytest.raises(ct.CanteraError, match="Illegal pressure value"):
        water.TP = (
            water.critical_temperature,
            1.001 * water.critical_pressure,
        )
        water.T_sat

    # Below triple point
    with pytest.raises(ct.CanteraError, match="Illegal temperature"):
        water.TP = 0.999 * water.min_temp, ct.one_atm
        water.P_sat
    # @todo: test disabled pending fix of GitHub issue #605
    # with self.assertRaisesRegex(ct.CanteraError, "Illegal pressure value"):
    #     self.water.TP = 300, .999 * psat
    #     self.water.T_sat


def test_TPQ(water):
    water.TQ = 400.0, 0.8
    T, P, Q = water.TPQ
    assert np.isclose(T, 400.0)
    assert np.isclose(Q, 0.8)

    # a supercritical state
    water.TPQ = 800.0, 3e7, 1.0
    assert np.isclose(water.T, 800.0)
    assert np.isclose(water.P, 3e7)

    water.TPQ = T, P, Q
    assert np.isclose(water.Q, 0.8)
    with pytest.raises(ct.CanteraError, match="inconsistent"):
        water.TPQ = T, 0.999 * P, Q
    with pytest.raises(ct.CanteraError, match="inconsistent"):
        water.TPQ = T, 1.001 * P, Q
    with pytest.raises(TypeError):
        water.TPQ = T, P, "spam"

    water.TPQ = 500, 1e5, 1  # superheated steam
    assert np.isclose(water.P, 1e5)
    with pytest.raises(ct.CanteraError, match="inconsistent"):
        water.TPQ = 500, 1e5, 0  # vapor fraction should be 1 (T < Tc)
    with pytest.raises(ct.CanteraError, match="inconsistent"):
        water.TPQ = 700, 1e5, 0  # vapor fraction should be 1 (T > Tc)


def test_phase_of_matter(water):
    water.TP = 300, 101325
    assert water.phase_of_matter == "liquid"
    water.TP = 500, 101325
    assert water.phase_of_matter == "gas"
    water.TP = water.critical_temperature * 2, 101325
    assert water.phase_of_matter == "supercritical"
    water.TP = 300, water.critical_pressure * 2
    assert water.phase_of_matter == "supercritical"
    water.TQ = 300, 0.4
    assert water.phase_of_matter == "liquid-gas-mix"

    # These cases work after fixing GH-786
    n2 = ct.Nitrogen()
    n2.TP = 100, 1000
    assert n2.phase_of_matter == "gas"

    co2 = ct.CarbonDioxide()
    assert co2.phase_of_matter == "gas"


@pytest.mark.parametrize(
    "backend,thermo_model",
    [("Reynolds", "PureFluid"), ("IAPWS95", "liquid-water-IAPWS95")],
)
def test_water_backends(backend, thermo_model):
    w = ct.Water(backend=backend)
    assert w.thermo_model == thermo_model


def test_unknown_water_backend_error():
    with pytest.raises(KeyError, match="Unknown backend"):
        ct.Water("foobar")


def test_water_iapws():
    w = ct.Water(backend="IAPWS95")
    assert np.isclose(w.critical_density, 322.0)
    assert np.isclose(w.critical_temperature, 647.096)
    assert np.isclose(w.critical_pressure, 22064000.0)

    # test internal TP setters (setters update temperature at constant
    # density before updating pressure)
    w.TP = 300, ct.one_atm
    dens = w.density
    w.TP = 2000, ct.one_atm  # supercritical
    assert w.phase_of_matter == "supercritical"
    w.TP = 300, ct.one_atm  # state goes from supercritical -> gas -> liquid
    assert np.isclose(w.density, dens)
    assert w.phase_of_matter == "liquid"

    # test setters for critical conditions
    w.TP = w.critical_temperature, w.critical_pressure
    assert np.isclose(w.density, 322.0)
    w.TP = 2000, ct.one_atm  # uses current density as initial guess
    w.TP = 273.16, ct.one_atm  # uses fixed density as initial guess
    assert np.isclose(w.density, 999.84376)
    assert w.phase_of_matter == "liquid"
    w.TP = w.T, w.P_sat
    assert w.phase_of_matter == "liquid"
    with pytest.raises(ct.CanteraError, match="assumes liquid phase"):
        w.TP = 273.1599999, ct.one_atm
    with pytest.raises(ct.CanteraError, match="assumes liquid phase"):
        w.TP = 500, ct.one_atm


@pytest.fixture(scope="class", params=("HFC-134a", "heptane", "carbon-dioxide", "hydrogen", "nitrogen", "oxygen", "water"))
def create_fluid(request):
    request.cls.fluid = ct.PureFluid("liquidvapor.yaml", request.param)
    from ruamel import yaml
    from .utilities import TEST_DATA_PATH
    reader = yaml.YAML(typ="safe")
    state_data = reader.load(
        TEST_DATA_PATH / "state-data.yaml"
    )[request.param]
    request.cls.states = state_data["states"]
    request.cls.reference_state = state_data["reference"]
    request.cls.tol = reader.load(TEST_DATA_PATH / "tolerance-data.yaml")[request.param]

@pytest.mark.usefixtures("create_fluid")
class TestPureFluid:

    def a(self, T, rho):
        """ Helmholtz free energy """
        self.fluid.TD = T, rho
        return self.fluid.u - T * self.fluid.s

    def test_has_phase_transition(self):
        assert self.fluid.has_phase_transition

    def test_consistency_temperature(self):
        for state in self.states:
            dT = 2e-5 * state["T"]
            self.fluid.TD = state["T"] - dT, state["rho"]
            s1 = self.fluid.s
            u1 = self.fluid.u
            self.fluid.TD = state["T"] + dT, state["rho"]
            s2 = self.fluid.s
            u2 = self.fluid.u

            # At constant volume, dU = T dS
            assert np.isclose((u2-u1)/(s2-s1), state["T"])

    def test_consistency_volume(self):
        for state in self.states:
            self.fluid.TD = state["T"], state["rho"]
            p = self.fluid.P
            V = 1.0 / state["rho"]
            dV = 5e-6 * V 

            a1 = self.a(state["T"], 1/(V-0.5*dV))
            a2 = self.a(state["T"], 1/(V+0.5*dV))

            # dP/drho is high for liquids, so relax tolerances
            tol = 300*self.tol["dAdV"] if state["phase"] == "liquid" else self.tol["dAdV"]

            # At constant temperature, dA = - p dV
            assert np.isclose(-(a2-a1)/dV, p, rtol=tol)

    def test_saturation(self):
        for state in self.states:
            if state["phase"] == "super":
                continue

            dT = 1e-6 * state["T"]
            self.fluid.TQ = state["T"], 0
            p1 = self.fluid.P
            vf = 1.0 / self.fluid.density
            hf = self.fluid.h
            sf = self.fluid.s

            self.fluid.TQ = state["T"] + dT, 0
            p2 = self.fluid.P

            self.fluid.TQ = state["T"], 1
            vg = 1.0 / self.fluid.density
            hg = self.fluid.h
            sg = self.fluid.s

            # Clausius-Clapeyron Relation
            assert np.isclose((p2-p1)/dT, (hg-hf)/(state["T"] * (vg-vf)), rtol=self.tol["dPdT"])

            # True for a change in state at constant pressure and temperature
            assert np.isclose(hg-hf, state["T"] * (sg-sf), rtol=self.tol["hTs"])

    def test_pressure(self):
        for state in self.states:
            self.fluid.TD = state["T"], state["rho"]
            # dP/drho is high for liquids, so relax tolerances
            tol = 50*self.tol["p"] if state["phase"] == "liquid" else self.tol["p"]
            tol *= state["tolMod"]
            assert np.isclose(self.fluid.P, state["p"], rtol=tol)

    def test_internal_energy(self):
        self.fluid.TD = self.reference_state["T"], self.reference_state["rho"]
        u_0 = self.fluid.u
        for state in self.states:
            self.fluid.TD = state["T"], state["rho"]
            assert np.isclose(self.fluid.u - u_0,
                            state["u"] - self.reference_state["u"],
                            rtol=self.tol["u"] * state["tolMod"])

    def test_entropy(self):
        self.fluid.TD = self.reference_state["T"], self.reference_state["rho"]
        s_0 = self.fluid.s
        for state in self.states:
            self.fluid.TD = state["T"], state["rho"]
            assert np.isclose(self.fluid.s - s_0,
                            state["s"] - self.reference_state["s"],
                            self.tol["s"] * state["tolMod"])

