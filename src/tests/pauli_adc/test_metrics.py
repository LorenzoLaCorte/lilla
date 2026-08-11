"""Quality metrics exposed for the six-coordinate backend."""

import numpy as np
import pytest
from numpy.testing import assert_allclose

from src.core.pauli_adc import RealXState, bell_weights_to_coordinates
from src.core.pauli_fourier.state import PauliFourierState, lambda_to_mu
from src.utils.utility_functions import (
    distillable_entanglement,
    get_mean_sf,
    pauli_fourier_to_fid,
    real_x_to_fid,
)


def test_real_x_fidelity_and_bb84_metric_reduce_to_pauli_fourier_metrics():
    weights = np.array([0.79, 0.08, 0.06, 0.07])
    mu = lambda_to_mu(weights)
    coordinates = bell_weights_to_coordinates(weights)
    assert_allclose(real_x_to_fid(coordinates), pauli_fourier_to_fid(mu))

    pmf = np.array([0.0, 0.3, 0.4, 0.2])
    x_func = np.tile(coordinates, (len(pmf), 1))
    mu_func = np.tile(mu, (len(pmf), 1))
    assert_allclose(
        get_mean_sf(pmf, x_func, state_type=RealXState),
        get_mean_sf(pmf, mu_func, state_type=PauliFourierState),
    )


def test_fidelity_only_entanglement_surrogate_rejects_locally_biased_states():
    coordinates = np.array([1.0, 0.18, 0.18, 0.5508, 0.648, -0.648])
    with pytest.raises(NotImplementedError, match="not justified"):
        distillable_entanglement(
            np.tile(coordinates, (3, 1)), state_type=RealXState
        )
