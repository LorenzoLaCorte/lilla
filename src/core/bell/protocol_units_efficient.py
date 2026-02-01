"""
Efficient join for Bell-diagonal states subject to depolarizing noise.

if any unsupported configuration (such as dephasing or heterogeneous noise)
it falls back to the baseline implementation
"""

from __future__ import annotations
from typing import Callable, Union
import numpy as np

from .protocol_units import bell_join  # fallback implementation
from src.core.werner.protocol_units_efficient import werner_join_efficient


# Mapping from input index to output index for each component of the Bell swap update rule
_BELL_S_MAP = (
    (0, 1, 2, 3),  # c0: pairs (0,0),(1,1),(2,2),(3,3)
    (1, 0, 3, 2),  # c1: pairs (0,1),(1,0),(2,3),(3,2)
    (2, 3, 0, 1),  # c2: pairs (0,2),(1,3),(2,0),(3,1)
    (3, 2, 1, 0),  # c3: pairs (0,3),(1,2),(2,1),(3,0)
)


def bell_join_efficient(
    pmf1: np.ndarray,
    pmf2: np.ndarray,
    lambda_func1: np.ndarray,
    lambda_func2: np.ndarray,
    ycut: bool = True,
    cutoff: Union[int, float] = np.iinfo(np.int32).max,
    cut_type: str = "memory_time",
    evaluate_func: Union[str, Callable] = "1",
    depolar_rate: float = 0.0,
    dephase_rate: float = 0.0,
    twirling: bool = True,
) -> np.ndarray:
    """
    Efficient convolution for Bell-diagonal states under depolarizing noise.

    When ``evaluate_func`` is ``"1"`` or ``"f1f2"`` and the cut-off is the
    memory time cut-off, this routine reduces the convolution complexity
    from quadratic to linear (up to log factors) by exploiting a
    separable form of the Bell-diagonal swap.  If other evaluation
    functions are requested or heterogeneous noise/dephasing is present,
    the function delegates to :func:`~src.core.bell.protocol_units.bell_join`.

    Parameters
    ----------
    pmf1, pmf2 : np.ndarray
        Waiting time distributions.  These may be one- or two-dimensional;
        when two-dimensional (shape ``(t,4)``) each column is assumed
        identical and the first column is used internally.
    lambda_func1, lambda_func2 : np.ndarray
        Bell-diagonal coefficient functions as a function of time with
        shape ``(t,4)``.
    ycut : bool
        Whether to keep (``True``) or discard (``False``) waiting times
        that exceed the cut-off.
    cutoff : int or float
        The memory time cut-off threshold.
    cut_type : str
        The type of cut-off.  Only ``"memory_time"`` is supported here.
    evaluate_func : str or callable
        The function to evaluate.  Only ``"1"`` (probability) and
        ``"f1f2"`` (swap) are handled by the efficient implementation.
        All other values will result in a fallback to the baseline
        implementation.
    depolar_rate : float
        Depolarizing rate :math:`\gamma`; the coherence time is
        :math:`1/\gamma`.  If zero or not positive, no depolarization
        is applied and the dynamic part vanishes.
    dephase_rate : float
        Dephasing rate.  If non-zero the baseline implementation is used.
    twirling : bool
        Ignored here but included for signature compatibility.

    Returns
    -------
    np.ndarray
        The resulting distribution or state function.  For ``evaluate_func``
        equal to ``"1"`` the result has the same shape as the input
        distributions (i.e., ``(t,)`` or ``(t,4)``); for ``"f1f2"`` the
        result has shape ``(t,4)`` where each column corresponds to one
        Bell basis component of the swapped state.
    """
    # Fallback to the baseline implementation for unsupported scenarios
    if cut_type != "memory_time" or dephase_rate != 0.0:
        return bell_join(
            pmf1, pmf2,
            lambda_func1=lambda_func1,
            lambda_func2=lambda_func2,
            ycut=ycut, cutoff=cutoff, cut_type=cut_type,
            evaluate_func=evaluate_func,
            depolar_rate=depolar_rate, dephase_rate=dephase_rate,
            twirling=twirling,
        )

    # Reject non-string evaluation functions (user provided callables)
    if not isinstance(evaluate_func, str):
        return bell_join(
            pmf1, pmf2,
            lambda_func1=lambda_func1,
            lambda_func2=lambda_func2,
            ycut=ycut, cutoff=cutoff, cut_type=cut_type,
            evaluate_func=evaluate_func,
            depolar_rate=depolar_rate, dephase_rate=dephase_rate,
            twirling=twirling,
        )

    # Only handle evaluate_func values that are supported
    if evaluate_func not in ("1", "f1f2"):
        return bell_join(
            pmf1, pmf2,
            lambda_func1=lambda_func1,
            lambda_func2=lambda_func2,
            ycut=ycut, cutoff=cutoff, cut_type=cut_type,
            evaluate_func=evaluate_func,
            depolar_rate=depolar_rate, dephase_rate=dephase_rate,
            twirling=twirling,
        )

    # Collapse possibly tiled PMF arrays to one dimension.  The Bell code
    # duplicates the PMF across four columns, so we take the first one.
    base_pmf1 = pmf1[:, 0] if pmf1.ndim == 2 else pmf1
    base_pmf2 = pmf2[:, 0] if pmf2.ndim == 2 else pmf2

    # Determine the coherence time from the depolarizing rate.
    # When depolar_rate <= 0 (including zero), use an infinite coherence
    # time to avoid numerical issues; exponential factors become unity.
    if depolar_rate and depolar_rate > 0.0:
        t_coh = 1.0 / depolar_rate
    else:
        t_coh = np.inf

    # Compute the base waiting time distribution using the Werner routine.
    pmf_base = werner_join_efficient(
        base_pmf1, base_pmf2,
        np.ones_like(base_pmf1), np.ones_like(base_pmf2),
        cutoff=cutoff, ycut=ycut,
        cut_type="memory_time",
        evaluate_func="1",
        t_coh=t_coh,
    )

    # If the user only requests the probability distribution, replicate the
    # base distribution across four columns to match the baseline output.
    if evaluate_func == "1":
        if pmf1.ndim == 2:
            return np.tile(pmf_base[:, np.newaxis], (1, 4))
        else:
            return pmf_base

    # At this point evaluate_func == "f1f2".  We compute the dynamic
    # contributions for each Bell basis component separately and add the
    # constant term 1/4 times the base distribution.  The dynamic part
    # vanishes when the depolarizing rate is zero (t_coh = infinity).
    size = pmf_base.shape[0]
    result = np.zeros((size, 4), dtype=np.float64)

    for k in range(4):
        # Accumulate contributions for the k-th output component
        dyn_total = np.zeros_like(pmf_base)
        for i in range(4):
            # (a_i - 1/4) for the first link
            w_func1_i = lambda_func1[:, i] - 0.25
            # b_{s_k(i)} for the second link (permute according to the
            # swapping rule)
            w_func2_i = lambda_func2[:, _BELL_S_MAP[k][i]]
            # Convolve with decay using the Werner efficient routine.  The
            # evaluate_func="w1w2" asks for the product of the two state
            # functions with exponential decay of the older link.
            contrib = werner_join_efficient(
                base_pmf1, base_pmf2,
                w_func1_i, w_func2_i,
                cutoff=cutoff, ycut=ycut,
                cut_type="memory_time",
                evaluate_func="w1w2",
                t_coh=t_coh,
            )
            dyn_total = contrib
        # Add constant term: 1/4 times the base PMF
        result[:, k] = 0.25 * pmf_base + dyn_total

    return result