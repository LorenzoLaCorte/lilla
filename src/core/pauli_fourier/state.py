import numpy as np

from src.core.states import QuantumState

MuFunc = np.ndarray
Mus = np.ndarray


def lambda_to_mu(lambdas):
    """Convert Bell weights to Pauli-Fourier coordinates.

    Bell order is (phi+, psi+, psi-, phi-) = (I, X, Y, Z).
    """
    lambdas = np.asarray(lambdas)
    if lambdas.shape[-1] != 4:
        raise ValueError("Bell weight arrays must have last dimension 4")
    l0, l1, l2, l3 = np.moveaxis(lambdas, -1, 0)
    return np.stack(
        (
            l0 + l1 + l2 + l3,
            l0 + l1 - l2 - l3,
            l0 - l1 + l2 - l3,
            l0 - l1 - l2 + l3,
        ),
        axis=-1,
    )


def mu_to_lambda(mu):
    """Convert Pauli-Fourier coordinates to Bell weights."""
    mu = np.asarray(mu)
    if mu.shape[-1] != 4:
        raise ValueError("Pauli-Fourier arrays must have last dimension 4")
    mu0, mu1, mu2, mu3 = np.moveaxis(mu, -1, 0)
    return np.stack(
        (
            (mu0 + mu1 + mu2 + mu3) / 4.0,
            (mu0 + mu1 - mu2 - mu3) / 4.0,
            (mu0 - mu1 + mu2 - mu3) / 4.0,
            (mu0 - mu1 - mu2 + mu3) / 4.0,
        ),
        axis=-1,
    )


def mu_to_fid(mu):
    """Return lambda_phi_plus from Pauli-Fourier coordinates."""
    mu = np.asarray(mu)
    return np.sum(mu, axis=-1) / 4.0


def mu_func_to_lambda_func(mu_func: MuFunc):
    return mu_to_lambda(mu_func)


class PauliFourierState(QuantumState):
    """Bell-diagonal state represented by Pauli-Fourier coordinates."""

    def __init__(self, lambdas=None, mu=None):
        if (lambdas is None) == (mu is None):
            raise ValueError("Provide exactly one of lambdas or mu")

        if lambdas is not None:
            mu = lambda_to_mu(np.asarray(lambdas, dtype=float))
        else:
            mu = np.asarray(mu, dtype=float)

        if mu.shape != (4,):
            raise ValueError("Pauli-Fourier states require four coordinates")
        if not np.isclose(mu[0], 1.0):
            raise ValueError("normalized Pauli-Fourier states require mu0=1")

        lambdas_from_mu = mu_to_lambda(mu)
        if np.any(lambdas_from_mu < -1.0e-12):
            raise ValueError("mu does not describe a physical Bell-diagonal state")
        if not np.isclose(np.sum(lambdas_from_mu), 1.0):
            raise ValueError("mu does not describe a normalized Bell-diagonal state")

        self.mu: Mus = mu.astype(float, copy=False)

    def __repr__(self):
        return f"PauliFourierState(mu={self.mu})"

    def get_generation_sf(self, t_trunc) -> MuFunc:
        return np.array([self.mu] * t_trunc, dtype=float)


def polish_mu_func(mu_func: MuFunc, atol=1.0e-12) -> MuFunc:
    """Project small numerical drift back to physical Bell-diagonal states."""
    mu_arr = np.asarray(mu_func)
    one_dimensional = mu_arr.ndim == 1
    if one_dimensional:
        mu_arr = mu_arr[np.newaxis, :]

    mu_arr = np.real_if_close(mu_arr, tol=1000)
    if np.iscomplexobj(mu_arr):
        mu_arr = np.real(mu_arr)
    mu_arr = np.asarray(mu_arr, dtype=float)
    np.nan_to_num(mu_arr, copy=False, nan=0.0, posinf=0.0, neginf=0.0)
    mu_arr[:, 0] = 1.0

    lambdas = mu_to_lambda(mu_arr)
    np.nan_to_num(lambdas, copy=False, nan=0.25, posinf=0.25, neginf=0.25)

    small_negative = (lambdas < 0.0) & (lambdas >= -atol)
    lambdas[small_negative] = 0.0
    if np.any(lambdas < -atol):
        raise ValueError("Pauli-Fourier polishing found non-physical Bell weights")

    row_sums = np.sum(lambdas, axis=1)
    valid = row_sums > atol
    lambdas[valid] /= row_sums[valid, np.newaxis]
    lambdas[~valid] = 0.25

    polished = lambda_to_mu(lambdas)
    polished[:, 0] = 1.0
    if one_dimensional:
        return polished[0]
    return polished
