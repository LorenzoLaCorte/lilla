"""Shared causal FFT arithmetic for full-reset renewal processes.

The scalar inverse of ``1 - failure`` is built once and applied to one or
several weighted output columns. State-specific checks belong to the backend.
"""

import numpy as np


def _next_power_of_two(value):
    return 1 << max(0, int(value - 1).bit_length())


def _fft_convolve_truncated(left, right, length):
    left = np.asarray(left, dtype=float)
    right = np.asarray(right, dtype=float)
    if length <= 0:
        return np.empty(0, dtype=float)
    if left.ndim != 1 or right.ndim != 1:
        raise ValueError("FFT convolution inputs must be one-dimensional")
    if len(left) == 0 or len(right) == 0:
        return np.zeros(length, dtype=float)
    transform_length = _next_power_of_two(len(left) + len(right) - 1)
    product = np.fft.rfft(left, transform_length) * np.fft.rfft(right, transform_length)
    convolution = np.fft.irfft(product, transform_length)
    result = np.zeros(length, dtype=float)
    result[: min(length, len(convolution))] = convolution[:length]
    return result


def _power_series_inverse(coefficients, length):
    coefficients = np.asarray(coefficients, dtype=float)
    if coefficients.ndim != 1 or len(coefficients) < length:
        raise ValueError("power-series coefficients have an invalid shape")
    if coefficients[0] <= 1.0e-15:
        raise ValueError("renewal has a unit-probability zero-duration failure loop")
    inverse = np.array([1.0 / coefficients[0]], dtype=float)
    order = 1
    while order < length:
        next_order = min(2 * order, length)
        product = _fft_convolve_truncated(coefficients[:next_order], inverse, next_order)
        correction = -product
        correction[0] += 2.0
        inverse = _fft_convolve_truncated(inverse, correction, next_order)
        order = next_order
    return inverse[:length]


def _prepare_renewal(failure_kernel, first_success, shift):
    failure = np.asarray(failure_kernel, dtype=float)
    first = np.asarray(first_success, dtype=float)
    if failure.ndim != 1 or first.ndim not in (1, 2) or len(first) != len(failure):
        raise ValueError("renewal kernels must have the same leading dimension")
    if (not np.all(np.isfinite(failure)) or np.any(failure < -1.0e-12)
            or np.sum(failure) > 1.0 + 1.0e-10 or not np.all(np.isfinite(first))):
        raise ValueError("renewal kernels contain invalid probability data")
    if not isinstance(shift, (int, np.integer)) or shift < 0:
        raise ValueError("renewal shift must be a nonnegative integer")
    failure = np.maximum(failure, 0.0)
    size = len(failure)
    shifted = np.zeros(size, dtype=float)
    if shift < size:
        shifted[shift:] = failure[: size - shift]
    denominator = np.zeros(size, dtype=float)
    denominator[0] = 1.0
    denominator -= shifted
    return first, _power_series_inverse(denominator, size)


def _convolve_with_resolvent(first, inverse):
    size = len(inverse)
    transform_length = _next_power_of_two(len(first) + len(inverse) - 1)
    inverse_spectrum = np.fft.rfft(inverse, transform_length)
    if first.ndim == 1:
        result = np.fft.irfft(np.fft.rfft(first, transform_length) * inverse_spectrum,
                              transform_length)[:size]
    else:
        result = np.fft.irfft(
            np.fft.rfft(first, transform_length, axis=0) * inverse_spectrum[:, np.newaxis],
            transform_length, axis=0)[:size]
    leading_nonzero = np.flatnonzero(np.any(first != 0.0, axis=1) if first.ndim == 2 else first != 0.0)
    if len(leading_nonzero) == 0:
        result[...] = 0.0
    else:
        result[: leading_nonzero[0]] = 0.0
    return np.real_if_close(result, tol=1000).astype(float, copy=False)


def renewal_sum_fft(failure_kernel, first_success, *, shift=0):
    """Sum causal retries without wrapping a long tail into earlier bins.

    Supports scalar or multi-column numerators, including signed correlators.
    Only roundoff-sized negative values of nonnegative outputs are clipped.
    """
    first, inverse = _prepare_renewal(failure_kernel, first_success, shift)
    result = _convolve_with_resolvent(first, inverse)
    columns = [result] if result.ndim == 1 else result.T
    first_columns = [first] if first.ndim == 1 else first.T
    factor = 64.0 * np.finfo(float).eps * (1.0 + np.ceil(np.log2(max(2, len(inverse)))))
    for values, initial in zip(columns, first_columns):
        if np.all(initial >= 0.0):
            tolerance = factor * np.max(np.abs(values), initial=0.0)
            tiny_negative = (values < 0.0) & (values >= -tolerance)
            values[tiny_negative] = 0.0
    return result
