"""Full-reset Werner renewal with one resolvent for probability and state."""

import numpy as np

from src.core.renewal import _prepare_renewal, _convolve_with_resolvent


def renew_werner(cutoff_failure, valid_failure, valid_success, weighted_success, *, shift):
    """Return completion PMF and conditional Werner parameter.

    K is the cutoff rejection kernel delayed by cutoff duration plus abort
    heralding; F is the destructive-operation failure kernel. Both output
    columns share the causal inverse series R = 1/(1-K-F). Successful and
    destructive-operation kernels already include operation heralding.
    """
    retry = np.array(valid_failure, dtype=float, copy=True)
    if shift < len(retry):
        retry[shift:] += cutoff_failure[: len(retry) - shift]
    first = np.column_stack((valid_success, weighted_success))
    if not np.any(first[:, 0] > 0.0):
        return np.zeros(len(first)), np.ones(len(first))
    first, inverse = _prepare_renewal(retry, first, 0)
    weighted = _convolve_with_resolvent(first, inverse)
    pmf = np.array(weighted[:, 0], copy=True)
    numerator = weighted[:, 1]
    floor = (64.0 * np.finfo(float).eps * (1.0 + np.ceil(np.log2(max(2, len(pmf)))))
             * np.max(np.abs(pmf), initial=0.0))
    if np.any(pmf < -floor):
        raise ValueError("FFT renewal produced a material negative Werner probability")
    if np.any(numerator < -floor) or np.any(numerator > pmf + floor):
        raise ValueError("FFT renewal produced a materially invalid weighted Werner parameter")
    # Absolute FFT error is amplified by conditioning on vanishing mass.
    # Use the same reliability floor as the real-X weighted renewal.
    pmf[pmf <= floor] = 0.0
    numerator = np.clip(numerator, 0.0, pmf)
    state = np.ones_like(pmf)
    np.divide(numerator, pmf, out=state, where=pmf > 0.0)
    return pmf, state
