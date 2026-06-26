import numba as nb
import numpy as np

from src.core.bell.state import LFunc

__all__ = [
    "memory_cut_off", "fidelity_cut_off", "run_time_cut_off", "time_cut_off",
    "depolarizing_noise", "dephasing_noise",
    "get_one", "get_swap_lambda_out",
    "get_dist_lambda_out", "get_dist_prob_fail", "get_dist_prob_suc",
]

"""
This module contain the defined success probability and
resulting output parameters of each protocol unit.
TODO: refactor this as ``noise model'' and a module with swapping and distillation as ``protocol units''
"""
########################################################################


@nb.jit(nopython=True, error_model="numpy")
def get_link_rate(rate_A: int, rate_B: int):
    """
    Return the joint depolarizing or dephasing rate of two nodes' memories.
    """
    return rate_A + rate_B


@nb.jit(nopython=True, error_model="numpy")
def get_links_rate(rates):
    """
    Return the rate (depolar or dephase) of the links involved
        by considering the rates of the involved nodes.
    """
    assert len(rates) == 3 or len(rates) == 2, f"SWAP/DIST's coherence time list must have 2 or 3 elements, got {rates}"
    # Distillation
    if len(rates) == 2:
        links_depolar_rate = [get_link_rate(rates[0], rates[1])]*2
    # Swap
    else:
        links_depolar_rate = [get_link_rate(rates[0], rates[1]), 
                              get_link_rate(rates[1], rates[2])]
    return links_depolar_rate


########################################################################
"""
Success probability p and
the resulting output parameters of swap and distillation.

Parameters
----------
t1, t2: int
    The waiting time of the two input links.
lambdas1, lambdas2 : float
    The parameters of the two input links.
depolar_rate, dephase_rate :
    Error parameters

Returns
-------
waiting_time: int
    The time used for preparing this pair of input links with cut-off.
    This time is different for a failing or successful attempt
result: bool
    The result of the cut-off
"""
@nb.jit(nopython=True, error_model="numpy")
def apply_noise(t1, t2, lambdas1, lambdas2, depolar_rate, dephase_rate):
    """
    Applies depolarizing and dephasing noise to the older link.
    Here t1 and t2 are preparation/completion times, so the older stored link
    is the one with the smaller time.
    Returns updated lambdas1 and lambdas2.
    """
    dt = np.abs(t1-t2)
    if dt == 0:
        return lambdas1, lambdas2

    if isinstance(depolar_rate, list) or isinstance(dephase_rate, list):
        assert isinstance(depolar_rate, list) and isinstance(dephase_rate, list), \
            "Both depolar_rate and dephase_rate must be lists"
        depolar_rate1, depolar_rate2 = get_links_rate(depolar_rate)
        dephase_rate1, dephase_rate2 = get_links_rate(dephase_rate)
        if t1 < t2:
            lambdas1 = depolarizing_noise(lambdas1, dt, depolar_rate1)
            lambdas1 = dephasing_noise(lambdas1, dt, dephase_rate1)
        else:
            lambdas2 = depolarizing_noise(lambdas2, dt, depolar_rate2)
            lambdas2 = dephasing_noise(lambdas2, dt, dephase_rate2)
    else:
        if t1 < t2:
            lambdas1 = depolarizing_noise(lambdas1, dt, depolar_rate)
            lambdas1 = dephasing_noise(lambdas1, dt, dephase_rate)
        else:
            lambdas2 = depolarizing_noise(lambdas2, dt, depolar_rate)
            lambdas2 = dephasing_noise(lambdas2, dt, dephase_rate)
        
    if not np.isclose(sum(lambdas1), 1.0, atol=1e-1):
        pass # print("[WARNING] after noise sum(lambdasOut)=", t1, t2, sum(lambdas1), lambdas1, "normalizing...")
        # lambdas1 /= sum(lambdas1)
    if not np.isclose(sum(lambdas2), 1.0, atol=1e-1):
        pass # print("[WARNING] after noise sum(lambdasOut)=", t1, t2, sum(lambdas2), lambdas2, "normalizing...")
        # lambdas2 /= sum(lambdas2)

    return lambdas1, lambdas2


"""
Error model functions
"""
@nb.jit(nopython=True, error_model="numpy")
def depolarizing_noise(lambdas, t, depolar_rate):
    """
    Applies depolarizing noise to the Bell Diagonal state, ensuring normalization.
    Note that this is global depolarization on the pair.
    For a 4-dimensional system, the depolarizing channel:
        - with prob. p, replaces the system with completely mixed state I/4
        - with prob. (1-p), leaves the state untouched
    i.e.
        epsilon(rho)  = (p/4) * I + (1 - p) * rho
    with
        p(t) = 1 - np.exp(-t * depolar_rate)
    
    TODO: if at some point we think time is a better param,
        we can substitute this with 
            p(t) = 1 - np.exp(-t / T_coh)

    Example:
        with 
            lamdas = [0.85, 0.05, 0.05, 0.05]
            t = 1e05
            depolar_rate = 2e-05
        we get
            depolarized_lambdas = array([0.33120117, 0.22293294, 0.22293294, 0.22293294])
    """
    p = 1 - np.exp(- t * depolar_rate)
    depolarized_lambdas = (p / 4) + (1 - p) * np.asarray(lambdas)
    return depolarized_lambdas


@nb.jit(nopython=True, error_model="numpy")
def dephasing_noise(lambdas, t, dephase_rate):
    """
    Applies phase damping noise with normalization.
    In the BD state:
        - with prob. p, (phi^+ and phi^-), (psi^+ and psi^-) weights are swapped,
        - with prob. (1-p), the state is left untouched
    with
        p(t) = (1 - e^(-t * dephase_rate) / 2
    
    Example
        with 
            lamdas = [0.85, 0.05, 0.075, 0.025]
            t = 100
            gamma = 0.01
        we get
            array([0.58925027, 0.05790151, 0.06709849, 0.28574973])
        with 
            t = 1000
        we get
            array([0.43751873, 0.06249943, 0.06250057, 0.43748127])
    """
    p = (1 - np.exp(- t * dephase_rate)) / 2
    dephased_lambdas = np.asarray([
        lambdas[0] * (1 - p) + lambdas[3] * p,
        lambdas[1] * (1 - p) + lambdas[2] * p,
        lambdas[2] * (1 - p) + lambdas[1] * p,
        lambdas[3] * (1 - p) + lambdas[0] * p
    ])
    return dephased_lambdas


@nb.jit(nopython=True, error_model="numpy")
def get_one(t1, t2, lambdas1, lambdas2, depolar_rate, dephase_rate=0., w_twirling=True):
    """
    Get a trivial one
    """
    return 1.


@nb.jit(nopython=True, error_model="numpy")
def get_swap_lambda_out(t1, t2, lambdasA, lambdasB, depolar_rate, dephase_rate=0., w_twirling=True):
    """
    Get w_swap
    """
    lambdasA, lambdasB = apply_noise(t1, t2, lambdasA, lambdasB, depolar_rate, dephase_rate)

    lambdasOut = np.asarray([
        (lambdasA[0] * lambdasB[0] + lambdasA[1] * lambdasB[1] + lambdasA[2] * lambdasB[2] + lambdasA[3] * lambdasB[3]),
        (lambdasA[0] * lambdasB[1] + lambdasA[1] * lambdasB[0] + lambdasA[2] * lambdasB[3] + lambdasA[3] * lambdasB[2]),
        (lambdasA[0] * lambdasB[2] + lambdasA[1] * lambdasB[3] + lambdasA[2] * lambdasB[0] + lambdasA[3] * lambdasB[1]),
        (lambdasA[0] * lambdasB[3] + lambdasA[1] * lambdasB[2] + lambdasA[2] * lambdasB[1] + lambdasA[3] * lambdasB[0])
    ])

    if not np.isclose(sum(lambdasOut), 1.0, atol=1e-1):
        pass # print("[WARNING] sum(lambdasOut)=", t1, t2, sum(lambdasOut), lambdasOut, "normalizing...")
        # lambdasOut /= sum(lambdasOut)
    return lambdasOut


@nb.jit(nopython=True, error_model="numpy")
def get_dist_lambda_out(t1, t2, a, b, depolar_rate, dephase_rate=0., w_twirling=True):
    """
    Get p_dist * lambda_dist.
    """
    a, b = apply_noise(t1, t2, a, b, depolar_rate, dephase_rate)

    numerator = np.asarray([
        (a[0] * b[0] + a[3] * b[3]),
        (a[1] * b[1] + a[2] * b[2]),
        (a[1] * b[2] + a[2] * b[1]),
        (a[0] * b[3] + a[3] * b[0]),
    ])

    if w_twirling:
        p_dist = sum(numerator)
        lambdasOut = np.asarray([
            numerator[0],
            (p_dist - numerator[0]) / 3,
            (p_dist - numerator[0]) / 3,
            (p_dist - numerator[0]) / 3,
        ])    
    else:
        lambdasOut = numerator
    
    return lambdasOut


@nb.jit(nopython=True, error_model="numpy")
def get_dist_prob_fail(t1, t2, lambdas1, lambdas2, depolar_rate, dephase_rate=0., w_twirling=True):
    """
    Get 1 - p_dist
    """
    return 1. - get_dist_prob_suc(t1, t2, lambdas1, lambdas2, depolar_rate, dephase_rate, w_twirling=w_twirling)


@nb.jit(nopython=True, error_model="numpy")
def get_dist_prob_suc(t1, t2, lambdas1, lambdas2, depolar_rate, dephase_rate=0., w_twirling=True):
    """
    Get p_dist
    """
    lambdas1, lambdas2 = apply_noise(t1, t2, lambdas1, lambdas2, depolar_rate, dephase_rate)

    return ((lambdas1[0] + lambdas1[3])*(lambdas2[0] + lambdas2[3]) + (lambdas1[1] + lambdas1[2])*(lambdas2[1] + lambdas2[2])) 

########################################################################
"""
Cut-off functions

Parameters
----------
t1, t2: int
    The waiting time of the two input links.
lambdas1, lambdas2: float
    The Bell diagonal coefficients of the two input links.
mt_cut: int
    The memory time cut-off.
f_cut: float
    The fidelity parameter cut-off. (0 < lambda[0] < 1)
    Set a cut-off on the input links's bell diagonal coefficient
t_coh: int or float
    The memory coherence time.

Returns
-------
waiting_time: int
    The time used for preparing this pair of input links with cut-off.
    This time is different for a failing or successful attempt
result: bool
    The result of the cut-off
"""
@nb.jit(nopython=True, error_model="numpy")
def memory_cut_off(
        t1, t2, lambdas1, lambdas2,
        mt_cut=np.iinfo(int).max, w_cut=1.e-8, rt_cut=np.iinfo(int).max):
    """
    Memory storage cut-off. The two input links suvives only if
    |t1-t2|<=mt_cut
    """
    if abs(t1 - t2) > mt_cut:
        # constant shift mt_cut is added in the iterative convolution
        return min(t1, t2), False
    else:
        return max(t1, t2), True


@nb.jit(nopython=True, error_model="numpy")
def fidelity_cut_off(
    t1, t2, lambdas1, lambdas2,
    mt_cut=np.iinfo(int).max, f_cut=1.e-8, rt_cut=np.iinfo(int).max):
    """
    Fidelity-dependent cut-off, The two input links suvives only if
    lambdas1 <= f_cut and lambdas2 <= f_cut including decoherence.
    """
    f1, f2 = lambdas1[0], lambdas2[0]
    if t1 == t2:
        if f1 < f_cut or f2 < f_cut:
            return t1, False
        return t1, True
    if t1 > t2:  # make sure t1 < t2
        t1, t2 = t2, t1
        lambdas1, lambdas2 = lambdas2, lambdas1
        f1, f2 = lambdas1[0], lambdas2[0]
    # first link has low quality
    if f1 < f_cut:
        return t1, False  # waiting_time = min(t1, t2)
    waiting = int(np.floor(np.log(f1/f_cut))) # TODO: t_coh not here anymore, maybe apply noise
    # first link waits too long
    if t1 + waiting < t2:
        return t1 + waiting, False  # min(t1, t2) < waiting_time < max(t1, t2)
    # second link has low quality
    elif f2 < f_cut:
        return t2, False  # waiting_time = max(t1, t2)
    # both links are good
    else:
        return t2, True  # waiting_time = max(t1, t2)


@nb.jit(nopython=True, error_model="numpy")
def run_time_cut_off(
    t1, t2, lambdas1, lambdas2,
    mt_cut=np.iinfo(int).max, w_cut=1.e-8, rt_cut=np.iinfo(int).max):
    if t1 > rt_cut or t2 > rt_cut:
        return rt_cut, False
    else:
        return max(t1, t2), True


@nb.jit(nopython=True, error_model="numpy")
def time_cut_off(
    t1, t2, lambdas1, lambdas2,
    mt_cut=np.iinfo(int).max, w_cut=1.e-8, rt_cut=np.iinfo(int).max, ):
    waiting_time1, result1 = memory_cut_off(
        t1, t2, lambdas1, lambdas2, mt_cut=mt_cut, w_cut=w_cut, rt_cut=rt_cut)
    waiting_time2, result2 = run_time_cut_off(
        t1, t2, lambdas1, lambdas2, mt_cut=mt_cut, w_cut=w_cut, rt_cut=rt_cut)
    result1 += mt_cut
    result2 += rt_cut
    if result1 and result2:
        return max(waiting_time1, waiting_time2), True
    else:
        # the waiting time of failing cutoff is always
        # smaller than max(t1, t2), so we just need a min here.
        return min(waiting_time1, waiting_time2), False


########################################################################
def bell_join(
        pmf1, pmf2, lambda_func1, lambda_func2, ycut=True,
        cutoff=np.iinfo(int).max, 
        cut_type="memory_time", evaluate_func=get_one, 
        depolar_rate=0., dephase_rate=0., w_twirling=True):
    """
    Calculate P_s and P_f.
    Calculate sum_(t=tA+tB) Pr(TA=tA)*Pr(TB=tB)*f(tA, tB)
    where f is the value function to
    be evaluated for the joint distribution.

    Note
    ----
    For swap the success probability p is
    considered in the iterative convolution.

    For the memory time cut-off,
    the constant shift is added in the iterative convolution.


    Parameters
    ----------
    pmf1, pmf2: array-like
        The waiting time distribution of the two input links, Pr(T=t).
    lambda_func1, lambda_func2: array-like
        The Bell diagonal coefficient function, W(t).
    cutoff: int or float
        The cut-off threshold.
    ycut: bool
        Successful cut-off or failed cut-off.
    cutoff_type: str
        Type of cut-off.
        `memory_time`, `run_time` or `fidelity`.
    evaluate_func: str
        The function to be evaluated the returns a float number.
        It can be
        ``get_one`` for trival cases\n
        ``get_swap_lambda_out`` for lambdaswap\n
        ``get_dist_prob_suc`` for pdist\n
        ``get_dist_prob_fail`` for 1-pdist\n
        ``get_dist_lambda_out`` for pdist * lambdadist
    t_coh: int or float
        The coherence time of the memory.

    Returns
    -------
    result: array-like 1-D
        The resulting array of joining the two links.
    """
    mt_cut=np.iinfo(int).max
    w_cut=0.0
    rt_cut=np.iinfo(int).max
    twod_par = False
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
        raise NotImplementedError("Unknow cut-off type")

    # TODO: refactor these names to be "get_swap_state_out" instead of "f1f2"
    if evaluate_func == "1":
        evaluate_func = get_one
    elif evaluate_func == "f1f2":
        evaluate_func = get_swap_lambda_out
    elif evaluate_func == "0.5+0.5f1f2":
        evaluate_func = get_dist_prob_suc
    elif evaluate_func == "0.5-0.5f1f2":
        evaluate_func = get_dist_prob_fail
    elif evaluate_func == "f1+f2+4f1f2":
        evaluate_func = get_dist_lambda_out
    elif isinstance(evaluate_func, str):
        raise ValueError(evaluate_func)
    
    result = join_links_helper(
        pmf1, pmf2, lambda_func1, lambda_func2, cutoff_func=cutoff_func, evaluate_func=evaluate_func, ycut=ycut, 
        mt_cut=mt_cut, w_cut=w_cut, rt_cut=rt_cut, depolar_rate=depolar_rate, dephase_rate=dephase_rate, w_twirling=w_twirling)
    return result


@nb.jit(nopython=True, error_model="numpy")
def join_links_iterate(
        result, pmf1, pmf2, w_func1, w_func2,
        cutoff_func=memory_cut_off, evaluate_func=get_one, ycut=True, mt_cut=np.iinfo(int).max, w_cut=0.0, rt_cut=np.iinfo(int).max, 
        depolar_rate=0., dephase_rate=0., w_twirling=True):
    """
    Iterate over all possible t1 and t2
    """
    size = len(pmf1)
    for t1 in range(1, size):
        for t2 in range(1, size):
            waiting_time, selection_pass = cutoff_func(
                t1, t2, w_func1[t1], w_func2[t2],
                mt_cut, w_cut, rt_cut)
            if not ycut:
                selection_pass = not selection_pass
            if selection_pass:
                output = evaluate_func(t1, t2, w_func1[t1], w_func2[t2], depolar_rate, dephase_rate, w_twirling=w_twirling)
                result[waiting_time] += pmf1[t1] * pmf2[t2] * output
    return result


def join_links_helper(
        pmf1, pmf2, sf1, sf2,
        cutoff_func=memory_cut_off, evaluate_func=get_one, ycut=True, mt_cut=np.iinfo(int).max, w_cut=0.0, rt_cut=np.iinfo(int).max, 
        depolar_rate=0., dephase_rate=0., w_twirling=True):
    """
    Call the appropriate function based on if tiling is required (computing lambdas)
    """
    size = len(pmf1)
    if isinstance(sf1, LFunc) and sf1.ndim == 2:
        result = np.zeros((size, 4), dtype=np.float64)
    else:
        result = np.zeros(size, dtype=np.float64)
    return join_links_iterate(result, pmf1, pmf2, sf1, sf2, cutoff_func, evaluate_func, 
                              ycut, mt_cut, w_cut, rt_cut, depolar_rate, dephase_rate, w_twirling=w_twirling)
