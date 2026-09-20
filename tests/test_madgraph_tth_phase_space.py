from pathlib import Path

import numpy as np

from madgraph_tth_integrand import TTHPhaseSpace


def test_tth_phase_space_conserves_momentum_and_shells(tmp_path: Path):
    card = tmp_path / "param_card.dat"
    card.write_text(
        """BLOCK MASS
  5 4.700000e+00
  6 1.725000e+02
 24 8.041900e+01
 25 1.250000e+02
DECAY 6 1.491500e+00
DECAY 24 2.047600e+00
DECAY 25 6.382000e-03
""",
        encoding="utf-8",
    )
    phase_space = TTHPhaseSpace(card, width_window=15.0)
    rng = np.random.default_rng(10)
    u = rng.random((128, 20))
    energy = rng.uniform(600.0, 2000.0, 128)
    checks = phase_space.validate(u, energy)
    assert checks["max_momentum_residual"] < 1e-9
    assert checks["max_mass_shell_residual"] < 1e-8
    assert checks["minimum_jacobian"] > 0.0
    assert checks["finite_fraction"] == 1.0
