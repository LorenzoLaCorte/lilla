"""Reference real-X-state kernel for Pauli noise and amplitude damping."""

from src.core.pauli_adc.protocol_units import (
    LocalAffineChannel,
    apply_local_channels,
    compose_channels,
    joint_pauli_adc_channel,
    pauli_eigenvalues,
    accepted_bell_swap,
    phi_plus_swap,
    psi_corrected_swap,
    recurrence_distillation,
    sequential_pauli_adc_channel,
)
from src.core.pauli_adc.state import (
    COORDINATE_LABELS,
    RealXState,
    bell_weights_to_coordinates,
    coordinates_to_matrix,
    is_physical,
    matrix_to_coordinates,
    normalize_weighted_x_func,
    phi_plus_fidelity,
    phi_plus_fidelity_func,
    polish_x_func,
    state_trace,
)
from src.core.pauli_adc.swap_join import AttemptTerms, real_x_attempt_join, renewal_sum
from src.core.pauli_adc.swap_join_efficient import (
    real_x_attempt_join_efficient,
    renewal_sum_efficient,
    renewal_sum_efficient_weighted,
)

__all__ = [
    "COORDINATE_LABELS",
    "LocalAffineChannel",
    "RealXState",
    "AttemptTerms",
    "accepted_bell_swap",
    "apply_local_channels",
    "bell_weights_to_coordinates",
    "compose_channels",
    "coordinates_to_matrix",
    "is_physical",
    "joint_pauli_adc_channel",
    "matrix_to_coordinates",
    "normalize_weighted_x_func",
    "pauli_eigenvalues",
    "phi_plus_fidelity",
    "phi_plus_fidelity_func",
    "phi_plus_swap",
    "polish_x_func",
    "psi_corrected_swap",
    "real_x_attempt_join",
    "real_x_attempt_join_efficient",
    "renewal_sum",
    "renewal_sum_efficient",
    "renewal_sum_efficient_weighted",
    "recurrence_distillation",
    "sequential_pauli_adc_channel",
    "state_trace",
]
