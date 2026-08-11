import numpy as np

from src.core.bell.protocol_units import memory_cut_off, run_time_cut_off
from src.core.pauli_fourier.state import MuFunc, mu_to_fid

__all__ = [
    "memory_cut_off",
    "fidelity_cut_off",
    "run_time_cut_off",
    "pauli_fourier_noise",
    "apply_noise",
    "get_one",
    "get_swap_mu_out",
    "get_dist_mu_out",
    "get_dist_prob_fail",
    "get_dist_prob_suc",
    "twirl_weighted_mu_from_inputs",
    "pauli_fourier_join",
]


def _is_rate_sequence(rate):
    return isinstance(rate, (list, tuple, np.ndarray))


def get_link_rate(rate_a, rate_b):
    return rate_a + rate_b


def get_links_rate(rates):
    values = list(np.asarray(rates, dtype=float).tolist())
    if len(values) == 2:
        return [get_link_rate(values[0], values[1])] * 2
    if len(values) == 3:
        return [
            get_link_rate(values[0], values[1]),
            get_link_rate(values[1], values[2]),
        ]
    raise AssertionError(
        f"SWAP/DIST rate list must have 2 or 3 elements, got {rates}"
    )


def _get_link_rates(rate):
    if _is_rate_sequence(rate):
        link_rates = get_links_rate(rate)
        return link_rates[0], link_rates[1]
    return rate, rate


def pauli_fourier_noise(mu, dt, depolar_rate, dephase_rate=0.0):
    mu = np.asarray(mu, dtype=float).copy()
    dep = np.exp(-dt * depolar_rate)
    deph = np.exp(-dt * dephase_rate)
    mu[0] = 1.0
    mu[1] *= dep * deph
    mu[2] *= dep * deph
    mu[3] *= dep
    return mu


def apply_noise(t1, t2, mu1, mu2, depolar_rate, dephase_rate=0.0):
    mu1 = np.asarray(mu1, dtype=float).copy()
    mu2 = np.asarray(mu2, dtype=float).copy()
    dt = abs(t1 - t2)
    if dt == 0:
        return mu1, mu2

    depolar_rate1, depolar_rate2 = _get_link_rates(depolar_rate)
    dephase_rate1, dephase_rate2 = _get_link_rates(dephase_rate)
    if t1 < t2:
        mu1 = pauli_fourier_noise(mu1, dt, depolar_rate1, dephase_rate1)
    else:
        mu2 = pauli_fourier_noise(mu2, dt, depolar_rate2, dephase_rate2)
    return mu1, mu2


def get_one(t1, t2, mu1, mu2, depolar_rate, dephase_rate=0.0, w_twirling=True):
    del t1, t2, mu1, mu2, depolar_rate, dephase_rate, w_twirling
    return 1.0


def get_swap_mu_out(t1, t2, mu_a, mu_b, depolar_rate, dephase_rate=0.0, w_twirling=True):
    del w_twirling
    mu_a, mu_b = apply_noise(t1, t2, mu_a, mu_b, depolar_rate, dephase_rate)
    out = mu_a * mu_b
    out[0] = 1.0
    return out


def get_dist_prob_suc(t1, t2, mu_a, mu_b, depolar_rate, dephase_rate=0.0, w_twirling=True):
    del w_twirling
    mu_a, mu_b = apply_noise(t1, t2, mu_a, mu_b, depolar_rate, dephase_rate)
    return (1.0 + mu_a[3] * mu_b[3]) / 2.0


def get_dist_prob_fail(t1, t2, mu_a, mu_b, depolar_rate, dephase_rate=0.0, w_twirling=True):
    return 1.0 - get_dist_prob_suc(
        t1, t2, mu_a, mu_b, depolar_rate, dephase_rate, w_twirling=w_twirling
    )


def twirl_weighted_mu_from_inputs(mu_a, mu_b):
    p_success = (1.0 + mu_a[3] * mu_b[3]) / 2.0
    n_phi_plus = (
        (1.0 + mu_a[3]) * (1.0 + mu_b[3])
        + (mu_a[1] + mu_a[2]) * (mu_b[1] + mu_b[2])
    ) / 8.0
    q = (4.0 * n_phi_plus - p_success) / 3.0
    return np.array([p_success, q, q, q], dtype=float)


def get_dist_mu_out(t1, t2, mu_a, mu_b, depolar_rate, dephase_rate=0.0, w_twirling=True):
    mu_a, mu_b = apply_noise(t1, t2, mu_a, mu_b, depolar_rate, dephase_rate)
    weighted = np.array(
        [
            (1.0 + mu_a[3] * mu_b[3]) / 2.0,
            (mu_a[1] * mu_b[1] + mu_a[2] * mu_b[2]) / 2.0,
            (mu_a[1] * mu_b[2] + mu_a[2] * mu_b[1]) / 2.0,
            (mu_a[3] + mu_b[3]) / 2.0,
        ],
        dtype=float,
    )
    if not w_twirling:
        return weighted
    q = (weighted[1] + weighted[2] + weighted[3]) / 3.0
    return np.array([weighted[0], q, q, q], dtype=float)


def fidelity_cut_off(
    t1,
    t2,
    mu1,
    mu2,
    mt_cut=np.iinfo(int).max,
    w_cut=1.0e-8,
    rt_cut=np.iinfo(int).max,
):
    del mt_cut, rt_cut
    f1, f2 = mu_to_fid(mu1), mu_to_fid(mu2)
    if t1 == t2:
        if f1 < w_cut or f2 < w_cut:
            return t1, False
        return t1, True
    if t1 > t2:
        t1, t2 = t2, t1
        f1, f2 = f2, f1
    if f1 < w_cut:
        return t1, False
    waiting = int(np.floor(np.log(f1 / w_cut)))
    if t1 + waiting < t2:
        return t1 + waiting, False
    if f2 < w_cut:
        return t2, False
    return t2, True


def _resolve_evaluate_func(evaluate_func):
    if evaluate_func == "one_rule":
        return get_one
    if evaluate_func == "swap_sf_rule":
        return get_swap_mu_out
    if evaluate_func == "dist_ps_rule":
        return get_dist_prob_suc
    if evaluate_func == "dist_pf_rule":
        return get_dist_prob_fail
    if evaluate_func == "dist_sf_rule":
        return get_dist_mu_out
    if isinstance(evaluate_func, str):
        raise ValueError(evaluate_func)
    return evaluate_func


def _resolve_cutoff_func(cut_type, cutoff):
    mt_cut = np.iinfo(int).max
    w_cut = 0.0
    rt_cut = np.iinfo(int).max
    if cut_type == "memory_time":
        cutoff_func = memory_cut_off
        mt_cut = cutoff
    elif cut_type == "fidelity":
        cutoff_func = fidelity_cut_off
        w_cut = cutoff
    elif cut_type == "run_time":
        cutoff_func = run_time_cut_off
        rt_cut = cutoff
    else:
        raise NotImplementedError("Unknown cut-off type")
    return cutoff_func, mt_cut, w_cut, rt_cut


def pauli_fourier_join(
    pmf1,
    pmf2,
    mu_func1: MuFunc,
    mu_func2: MuFunc,
    ycut=True,
    cutoff=np.iinfo(int).max,
    cut_type="memory_time",
    evaluate_func=get_one,
    depolar_rate=0.0,
    dephase_rate=0.0,
    w_twirling=True,
):
    cutoff_func, mt_cut, w_cut, rt_cut = _resolve_cutoff_func(cut_type, cutoff)
    evaluate_func = _resolve_evaluate_func(evaluate_func)
    size = len(pmf1)
    if np.asarray(mu_func1).ndim == 2:
        result = np.zeros((size, 4), dtype=np.float64)
    else:
        result = np.zeros(size, dtype=np.float64)

    for t1 in range(1, size):
        for t2 in range(1, size):
            waiting_time, selection_pass = cutoff_func(
                t1,
                t2,
                mu_func1[t1],
                mu_func2[t2],
                mt_cut,
                w_cut,
                rt_cut,
            )
            if not ycut:
                selection_pass = not selection_pass
            if selection_pass:
                output = evaluate_func(
                    t1,
                    t2,
                    mu_func1[t1],
                    mu_func2[t2],
                    depolar_rate,
                    dephase_rate,
                    w_twirling=w_twirling,
                )
                result[waiting_time] += pmf1[t1] * pmf2[t2] * output
    return result
