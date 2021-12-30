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
    self.water.TP = 300.0, ct.one_atm
    water.TQ = water.critical_temperature, 0.5
    assertNear(water.P, water.critical_pressure)
    water.TP = 300.0, ct.one_atm
    water.PQ = water.critical_pressure, 0.5
    assertNear(water.T, water.critical_temperature)

    # Supercritical
    with assertRaisesRegex(ct.CanteraError, "supercritical"):
        water.TQ = 1.001 * water.critical_temperature, 0.0
        with assertRaisesRegex(ct.CanteraError, "supercritical"):
            water.PQ = 1.001 * water.critical_pressure, 0.0

    # Q negative
    with assertRaisesRegex(ct.CanteraError, "Invalid vapor fraction"):
        water.TQ = 373.15, -0.001
        with assertRaisesRegex(ct.CanteraError, "Invalid vapor fraction"):
            water.PQ = ct.one_atm, -0.001

    # Q larger than one
    with assertRaisesRegex(ct.CanteraError, "Invalid vapor fraction"):
        water.TQ = 373.15, 1.001
        with assertRaisesRegex(ct.CanteraError, "Invalid vapor fraction"):
            water.PQ = ct.one_atm, 1.001


def test_saturated_mixture(self):
    self.water.TP = 300, ct.one_atm
    with self.assertRaisesRegex(ct.CanteraError, "Saturated mixture detected"):
        self.water.TP = 300, self.water.P_sat

    w = ct.Water()

    # Saturated vapor
    self.water.TQ = 373.15, 1.0
    self.assertEqual(self.water.phase_of_matter, "liquid-gas-mix")
    w.TP = self.water.T, 0.999 * self.water.P_sat
    self.assertNear(self.water.cp, w.cp, 1.0e-3)
    self.assertNear(self.water.cv, w.cv, 1.0e-3)
    self.assertNear(
        self.water.thermal_expansion_coeff, w.thermal_expansion_coeff, 1.0e-3
    )
    self.assertNear(
        self.water.isothermal_compressibility, w.isothermal_compressibility, 1.0e-3
    )

    # Saturated mixture
    self.water.TQ = 373.15, 0.5
    self.assertEqual(self.water.phase_of_matter, "liquid-gas-mix")
    self.assertEqual(self.water.cp, np.inf)
    self.assertTrue(np.isnan(self.water.cv))
    self.assertEqual(self.water.isothermal_compressibility, np.inf)
    self.assertEqual(self.water.thermal_expansion_coeff, np.inf)

    # Saturated liquid
    self.water.TQ = 373.15, 0.0
    self.assertEqual(self.water.phase_of_matter, "liquid-gas-mix")
    w.TP = self.water.T, 1.001 * self.water.P_sat
    self.assertNear(self.water.cp, w.cp, 1.0e-3)
    self.assertNear(self.water.cv, w.cv, 1.0e-3)
    self.assertNear(
        self.water.thermal_expansion_coeff, w.thermal_expansion_coeff, 1.0e-3
    )
    self.assertNear(
        self.water.isothermal_compressibility, w.isothermal_compressibility, 1.0e-3
    )


def test_saturation_near_limits(self):
    # Low temperature limit (triple point)
    self.water.TP = 300, ct.one_atm
    self.water.P_sat  # ensure that solver buffers sufficiently different values
    self.water.TP = self.water.min_temp, ct.one_atm
    psat = self.water.P_sat
    self.water.TP = 300, ct.one_atm
    self.water.P_sat  # ensure that solver buffers sufficiently different values
    self.water.TP = 300, psat
    self.assertNear(self.water.T_sat, self.water.min_temp)

    # High temperature limit (critical point) - saturation temperature
    self.water.TP = 300, ct.one_atm
    self.water.P_sat  # ensure that solver buffers sufficiently different values
    self.water.TP = self.water.critical_temperature, self.water.critical_pressure
    self.assertNear(self.water.T_sat, self.water.critical_temperature)

    # High temperature limit (critical point) - saturation pressure
    self.water.TP = 300, ct.one_atm
    self.water.P_sat  # ensure that solver buffers sufficiently different values
    self.water.TP = self.water.critical_temperature, self.water.critical_pressure
    self.assertNear(self.water.P_sat, self.water.critical_pressure)

    # Supercricital
    with self.assertRaisesRegex(ct.CanteraError, "Illegal temperature value"):
        self.water.TP = (
            1.001 * self.water.critical_temperature,
            self.water.critical_pressure,
        )
        self.water.P_sat
        with self.assertRaisesRegex(ct.CanteraError, "Illegal pressure value"):
            self.water.TP = (
                self.water.critical_temperature,
                1.001 * self.water.critical_pressure,
            )
            self.water.T_sat

    # Below triple point
    with self.assertRaisesRegex(ct.CanteraError, "Illegal temperature"):
        self.water.TP = 0.999 * self.water.min_temp, ct.one_atm
        self.water.P_sat
        # @todo: test disabled pending fix of GitHub issue #605
        # with self.assertRaisesRegex(ct.CanteraError, "Illegal pressure value"):
        #     self.water.TP = 300, .999 * psat
        #     self.water.T_sat


def test_TPQ(self):
    self.water.TQ = 400, 0.8
    T, P, Q = self.water.TPQ
    self.assertNear(T, 400)
    self.assertNear(Q, 0.8)

    # a supercritical state
    self.water.TPQ = 800, 3e7, 1
    self.assertNear(self.water.T, 800)
    self.assertNear(self.water.P, 3e7)

    self.water.TPQ = T, P, Q
    self.assertNear(self.water.Q, 0.8)
    with self.assertRaisesRegex(ct.CanteraError, "inconsistent"):
        self.water.TPQ = T, 0.999 * P, Q
        with self.assertRaisesRegex(ct.CanteraError, "inconsistent"):
            self.water.TPQ = T, 1.001 * P, Q
            with self.assertRaises(TypeError):
                self.water.TPQ = T, P, "spam"

    self.water.TPQ = 500, 1e5, 1  # superheated steam
    self.assertNear(self.water.P, 1e5)
    with self.assertRaisesRegex(ct.CanteraError, "inconsistent"):
        self.water.TPQ = 500, 1e5, 0  # vapor fraction should be 1 (T < Tc)
        with self.assertRaisesRegex(ct.CanteraError, "inconsistent"):
            self.water.TPQ = 700, 1e5, 0  # vapor fraction should be 1 (T > Tc)


def test_phase_of_matter(self):
    self.water.TP = 300, 101325
    self.assertEqual(self.water.phase_of_matter, "liquid")
    self.water.TP = 500, 101325
    self.assertEqual(self.water.phase_of_matter, "gas")
    self.water.TP = self.water.critical_temperature * 2, 101325
    self.assertEqual(self.water.phase_of_matter, "supercritical")
    self.water.TP = 300, self.water.critical_pressure * 2
    self.assertEqual(self.water.phase_of_matter, "supercritical")
    self.water.TQ = 300, 0.4
    self.assertEqual(self.water.phase_of_matter, "liquid-gas-mix")

    # These cases work after fixing GH-786
    n2 = ct.Nitrogen()
    n2.TP = 100, 1000
    self.assertEqual(n2.phase_of_matter, "gas")

    co2 = ct.CarbonDioxide()
    self.assertEqual(co2.phase_of_matter, "gas")


def test_water_backends(self):
    w = ct.Water(backend="Reynolds")
    self.assertEqual(w.thermo_model, "PureFluid")
    w = ct.Water(backend="IAPWS95")
    self.assertEqual(w.thermo_model, "liquid-water-IAPWS95")
    with self.assertRaisesRegex(KeyError, "Unknown backend"):
        ct.Water("foobar")


def test_water_iapws(self):
    w = ct.Water(backend="IAPWS95")
    self.assertNear(w.critical_density, 322.0)
    self.assertNear(w.critical_temperature, 647.096)
    self.assertNear(w.critical_pressure, 22064000.0)

    # test internal TP setters (setters update temperature at constant
    # density before updating pressure)
    w.TP = 300, ct.one_atm
    dens = w.density
    w.TP = 2000, ct.one_atm  # supercritical
    self.assertEqual(w.phase_of_matter, "supercritical")
    w.TP = 300, ct.one_atm  # state goes from supercritical -> gas -> liquid
    self.assertNear(w.density, dens)
    self.assertEqual(w.phase_of_matter, "liquid")

    # test setters for critical conditions
    w.TP = w.critical_temperature, w.critical_pressure
    self.assertNear(w.density, 322.0)
    w.TP = 2000, ct.one_atm  # uses current density as initial guess
    w.TP = 273.16, ct.one_atm  # uses fixed density as initial guess
    self.assertNear(w.density, 999.84376)
    self.assertEqual(w.phase_of_matter, "liquid")
    w.TP = w.T, w.P_sat
    self.assertEqual(w.phase_of_matter, "liquid")
    with self.assertRaisesRegex(ct.CanteraError, "assumes liquid phase"):
        w.TP = 273.1599999, ct.one_atm
        with self.assertRaisesRegex(ct.CanteraError, "assumes liquid phase"):
            w.TP = 500, ct.one_atm
