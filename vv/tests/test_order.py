import math

import numpy as np
import pytest

from cfd_vv.errors import VVError
from cfd_vv.order import fit_order, gci_two_grids, mesh_size, richardson


def test_mesh_size_is_the_dth_root_of_mean_cell_volume() -> None:
    assert mesh_size(1000, 1.0, 3) == pytest.approx(0.1, rel=1e-15)
    assert mesh_size(400, 4.0, 2) == pytest.approx(0.1, rel=1e-15)
    with pytest.raises(VVError) as err:
        mesh_size(0, 1.0, 3)
    assert err.value.code == "V0101"


@pytest.mark.parametrize("order", [1.0, 2.0, 3.0])
def test_fit_recovers_the_exponent_of_a_power_law(order: float) -> None:
    h = np.array([0.4, 0.2, 0.1, 0.05, 0.025])
    fit = fit_order(h, 3.7 * h**order)
    assert fit.order == pytest.approx(order, abs=1e-12)
    assert fit.constant == pytest.approx(3.7, rel=1e-12)
    assert fit.r_squared == pytest.approx(1.0, abs=1e-14)
    assert all(p == pytest.approx(order, abs=1e-12) for p in fit.pairwise_orders)


def test_fit_brackets_the_order_of_noisy_data_and_sorts_its_input() -> None:
    rng = np.random.default_rng(20261031)
    h = np.array([0.025, 0.4, 0.1, 0.2, 0.05])
    errors = 2.0 * h**2 * np.exp(rng.normal(0.0, 0.02, h.size))
    fit = fit_order(h, errors)
    assert fit.order_ci95[0] < 2.0 < fit.order_ci95[1]
    assert fit.order == pytest.approx(2.0, abs=0.05)
    assert len(fit.pairwise_orders) == 4


def test_fit_rejects_non_positive_errors() -> None:
    with pytest.raises(VVError) as err:
        fit_order([0.2, 0.1], [1e-3, 0.0])
    assert err.value.code == "V0103"


def test_richardson_matches_celik_2008_table_1() -> None:
    # Celik et al. (2008), J. Fluids Eng. 130(7):078001, Table 1, first column (2D, N = 18000, 8000, 4500).
    h = [mesh_size(n, 1.0, 2) for n in (18000, 8000, 4500)]
    res = richardson(h, [6.063, 5.972, 5.863])
    assert res.order == pytest.approx(1.53, abs=0.005)
    assert res.extrapolated == pytest.approx(6.1685, abs=5e-5)
    assert res.relative_gci_fine == pytest.approx(0.022, abs=5e-4)
    assert res.refinement_ratios == pytest.approx((1.5, 4.0 / 3.0))


def test_richardson_is_exact_for_a_power_law_with_non_constant_refinement() -> None:
    f0, a, p = 1.25, -0.8, 1.7
    h = [0.013, 0.031, 0.05]
    res = richardson(h, [f0 + a * hi**p for hi in h])
    assert res.order == pytest.approx(p, abs=1e-10)
    assert res.extrapolated == pytest.approx(f0, abs=1e-12)
    # For an exact power law the fine-grid error is |a| h1^p, so GCI_fine = 1.25 |a| h1^p.
    assert res.gci_fine == pytest.approx(1.25 * abs(a) * h[0] ** p, rel=1e-9)


@pytest.mark.parametrize(
    ("values", "code"), [([1.0, 1.1, 1.05], "V0107"), ([1.0, 1.1, 1.15], "V0108")]
)
def test_richardson_rejects_oscillatory_and_divergent_triplets(
    values: list[float], code: str
) -> None:
    with pytest.raises(VVError) as err:
        richardson([0.1, 0.2, 0.4], values)
    assert err.value.code == code


def test_two_grid_gci_uses_safety_factor_three() -> None:
    gci = gci_two_grids([0.1, 0.2], [1.0, 1.03], order=2.0)
    assert gci == pytest.approx(3.0 * 0.03 / 3.0)
    assert math.isfinite(gci)
