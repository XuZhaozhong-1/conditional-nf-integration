import numpy as np
import pytest

from madgraph_process_integrand import (
    DibosonFourLeptonPhaseSpace,
    ThreeBodyPhaseSpace,
    TwoBodyPhaseSpace,
)


def test_two_body_phase_space_conserves_momentum_and_shells():
    phase_space = TwoBodyPhaseSpace(172.5, 172.5)
    rng = np.random.default_rng(20260920)
    unit = rng.random((256, 2))
    energy = rng.uniform(360.0, 2000.0, len(unit))
    checks = phase_space.validate(unit, energy)
    assert checks["max_momentum_residual"] < 1e-10
    assert checks["max_mass_shell_residual"] < 1e-8
    assert checks["minimum_jacobian"] > 0.0
    assert checks["finite_fraction"] == 1.0


def test_two_body_phase_space_rejects_threshold():
    phase_space = TwoBodyPhaseSpace(172.5, 172.5)
    unit = np.full((1, 2), 0.5)
    try:
        phase_space.map(unit, np.array([345.0]))
    except ValueError as error:
        assert "must exceed" in str(error)
    else:
        raise AssertionError("Threshold point should have been rejected")


def test_three_body_phase_space_conserves_momentum_and_shells():
    phase_space = ThreeBodyPhaseSpace(172.0, 172.0, 125.0)
    rng = np.random.default_rng(7)
    unit = rng.random((1000, 5))
    energy = rng.uniform(480.0, 2000.0, len(unit))
    checks = phase_space.validate(unit, energy)
    assert checks["max_momentum_residual"] < 1e-9
    assert checks["max_mass_shell_residual"] < 1e-7
    assert checks["minimum_jacobian"] > 0.0
    assert checks["finite_fraction"] == 1.0


def test_massless_three_body_phase_space_volume():
    phase_space = ThreeBodyPhaseSpace(0.0, 0.0, 0.0)
    rng = np.random.default_rng(11)
    unit = rng.random((200000, 5))
    energy = np.full(len(unit), 100.0)
    _, jacobian = phase_space.map(unit, energy)
    exact = energy[0] ** 2 / (256.0 * np.pi**3)
    assert np.mean(jacobian) == pytest.approx(exact, rel=4e-3)


def test_diboson_four_lepton_phase_space_conserves_momentum_and_shells():
    phase_space = DibosonFourLeptonPhaseSpace(80.419, 2.05)
    rng = np.random.default_rng(19)
    unit = rng.random((2048, 8))
    energy = rng.uniform(250.0, 2000.0, len(unit))
    checks = phase_space.validate(unit, energy)
    assert checks["max_momentum_residual"] < 1e-9
    assert checks["max_mass_shell_residual"] < 1e-7
    assert checks["minimum_jacobian"] > 0.0
    assert checks["finite_fraction"] == 1.0
