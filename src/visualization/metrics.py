"""Pooled BB84 objective and completion-time diagnostics (no tail extrapolation)."""
import numpy as np

from src.core.pauli_adc.state import RealXState, coordinates_to_matrix, polish_x_func
from src.utils.utility_functions import get_mean_sf, get_mean_waiting_time, secret_key_rate

ROUND_OFF = 1e-10
DISPLAY_THRESHOLD = 1e-12


def bell_populations(x):
    x = np.asarray(x)
    n, zz, xx, yy = x[..., 0], x[..., 3], x[..., 4], x[..., 5]
    return np.stack([n+zz+xx-yy, n-zz+xx+yy, n-zz-xx-yy, n+zz-xx+yy], axis=-1)/4


def binary_entropy(p):
    p = np.asarray(p, dtype=float)
    if np.any(p < -ROUND_OFF) or np.any(p > 1+ROUND_OFF):
        raise ValueError("BB84 error outside [0,1]")
    p = np.clip(p, 0, 1)  # only already-validated endpoint roundoff
    out = np.zeros_like(p)
    mask = (p > 0) & (p < 1)
    out[mask] = -p[mask]*np.log2(p[mask]) - (1-p[mask])*np.log2(1-p[mask])
    return out


def time_quantities(x):
    x = np.asarray(x)
    ex, ez = (1-x[..., 4])/2, (1-x[..., 3])/2
    return {"fidelity": bell_populations(x)[..., 0], "eX": ex, "eZ": ez,
            "secret_fraction": np.maximum(0, 1-binary_entropy(ex)-binary_entropy(ez))}


def validate_outputs(pmf, x, horizon):
    p, x = np.array(pmf, dtype=float, copy=True), np.array(x, dtype=float, copy=True)
    if p.shape != (horizon,) or x.shape != (horizon, 6):
        raise ValueError("unexpected backend output shape")
    if not np.all(np.isfinite(p)) or not np.all(np.isfinite(x)):
        raise ValueError("nonfinite backend output")
    if np.min(p) < -ROUND_OFF or abs(p[0]) > ROUND_OFF:
        raise ValueError("invalid probability mass or unsupported bin 0")
    cleanup = []
    if p[0] != 0:
        cleanup.append({"kind": "bin_zero_residue", "mass": float(p[0])})
        p[0] = 0
    negative = p < 0
    if np.any(negative):
        cleanup.append({"kind": "negative_mass_roundoff", "mass": float(-p[negative].sum())})
        p[negative] = 0
    coverage = p.sum()
    if coverage > 1+ROUND_OFF:
        raise ValueError("completion mass exceeds one")
    if coverage > 1:
        cleanup.append({"kind": "unit_mass_roundoff", "excess": float(coverage-1)})
        p /= coverage
    populated = p > 0
    if np.any(populated):
        rows = x[populated]
        if np.any(abs(rows[:, 0]-1) > ROUND_OFF):
            raise ValueError("completion states are not normalized")
        clean = polish_x_func(rows, atol=ROUND_OFF)
        # PSD coordinates are bounded by one; limit only endpoint roundoff.
        if np.any(abs(clean) > 1+ROUND_OFF):
            raise ValueError("invalid normalized coordinates")
        clean = np.clip(clean, -1, 1)
        change = float(np.max(abs(clean-rows)))
        if change:
            cleanup.append({"kind": "state_roundoff", "max_coordinate_change": change})
        x[populated] = clean
    # Empty rows are placeholders, never observations.
    x[~populated] = [1, 0, 0, 0, 0, 0]
    return p, x, cleanup


def summarize(p, x, coverage_target, slot_duration=None):
    coverage = float(p.sum())
    provisional = coverage < coverage_target
    diagnostics = {"provisional": provisional, "coverage_target": coverage_target,
                   "display_threshold": DISPLAY_THRESHOLD, "bin_zero_mass": float(p[0])}
    if coverage == 0:
        return {"status": "no_completion", "coverage": 0.0, "rate_bits_per_slot": 0.0,
                "rate_bits_per_second": 0.0 if slot_duration else None,
                "pooled_secret_fraction": None, "time_resolved_secret_fraction": None,
                "mean_wait_repository_slots": None, "mean_wait_observed_slots": None,
                "mean_wait_repository_seconds": None, "average_fidelity": None,
                "eX": None, "eZ": None, "pooled_coordinates": None}, diagnostics
    pooled = np.sum(p[:, None]*x, axis=0)/coverage
    q = time_quantities(pooled)
    sf = float(q["secret_fraction"])
    mean = float(get_mean_waiting_time(p, extrapolation=False))
    rate = float(secret_key_rate(p, x, state_type=RealXState, extrapolation=False))
    reference_sf = max(0, float(get_mean_sf(p, x, False, RealXState)))
    if not np.isfinite(rate) or not np.isclose(sf, reference_sf, atol=1e-12, rtol=1e-9):
        raise ValueError("independent pooled BB84 metric disagrees with repository helper")
    if not np.isclose(rate, sf/mean, atol=1e-12, rtol=1e-9):
        raise ValueError("headline rate disagrees with independent pooled quantities")
    matrix = coordinates_to_matrix(pooled)
    diagnostics.update(pooled_trace=float(np.trace(matrix).real),
                       pooled_min_eigenvalue=float(np.linalg.eigvalsh(matrix).min()))
    return {"status": "provisional" if provisional else "covered", "coverage": coverage,
            "rate_bits_per_slot": rate,
            "rate_bits_per_second": rate/slot_duration if slot_duration else None,
            "pooled_secret_fraction": sf,
            "time_resolved_secret_fraction": float(np.sum(p*time_quantities(x)["secret_fraction"])/coverage),
            "mean_wait_repository_slots": mean,
            "mean_wait_repository_seconds": mean*slot_duration if slot_duration else None,
            "mean_wait_observed_slots": float(np.sum(p*np.arange(len(p)))/coverage),
            "average_fidelity": float(q["fidelity"]), "eX": float(q["eX"]),
            "eZ": float(q["eZ"]), "pooled_coordinates": pooled.tolist()}, diagnostics
