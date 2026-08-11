"""

TODO: types and exception defined in gp_utils should be moved here
"""
from argparse import ArgumentTypeError
from collections.abc import Iterable
import re
from typing import Tuple, TypedDict, Union, Literal, List
import numpy as np

from src.core.states import QuantumState
from src.core.werner.state import WernerState
from src.core.pauli_adc.state import RealXState

class ThresholdExceededError(Exception):
    """
    This exception is raised when the CDF coverage is below the threshold.
    """
    def __init__(self, message="CDF under threshold count incremented", extra_info=None):
        super().__init__(message)
        self.extra_info = extra_info

SymProtocol = Tuple[int] 
AsymProtocol = Tuple[str]
QuantumProtocol = Union[SymProtocol, AsymProtocol]

class SimParameters(TypedDict):
    """
    Type representing a set of parameters for a generic simulation of the algorithm
    """
    state: QuantumState
    protocol: QuantumProtocol
    t_coh: Union[int, List[int]]
    p_gen: Union[float, List[float]]
    p_swap: float
    t_trunc: int
    w0: Union[float, List[float]]
    lambdas: list[float]
    real_x_coordinates: list[float]
    pauli_mode_decay_rates: Union[float, list[float], list[list[float]]]
    amplitude_damping_rate: Union[float, list[float]]
    bell_outcomes: str
    pauli_adc_noise_model: str
    p_distillation: float
    swap_hardware_efficiency: float

PMF = np.ndarray

# Define the type for the optimizer and space_type
OptimizerType = Literal["bf", "gp"]
SpaceType = Literal["one_level", "strategy", "enumerate", "centerspace", "asymmetric"] # TODO: remove all except the one we use

def optimizerType(value: str) -> OptimizerType:
    """
    Validates the optimizer type passed in input
    """
    valid_options = ("gp", "bf")
    if value not in valid_options:
        raise ArgumentTypeError(f"Invalid optimizer type: {value}. Available options are: {', '.join(valid_options)}")
    return value

def spaceType(value: str) -> SpaceType:
    """
    Validates the space type passed in input
    """
    # TODO: refactor: valid_options = SpaceType.__args__
    valid_options = ("one_level", "strategy", "enumerate", "centerspace", "asymmetric")
    if value not in valid_options:
        raise ArgumentTypeError(f"Invalid space type: {value}. Available options are: {', '.join(valid_options)}")
    return value


def checkProtocolUnit(punit: str) -> bool:
    """
    Checks if a string is a valid protocol unit
        i.e. a string of one char 's' or 'd' and one (arbitrary high) number
        
    """
    return bool(re.match(r'^[sd]\d+$', punit))


def checkAsymProtocol(protocol: Tuple[str], S: int = None) -> Tuple[str]:
    """
    Validates a string passed in input for running an asymmetric protocol
    If the protocol is valid, the string is translated in an instance of the type
    Otherwise, an exception is thrown
    """
    swapped_segments = []
    for punit in protocol:
        operation = punit[0]
        segment = int(punit[1:])
        
        if operation == 's':
            swapped_segments.append(segment)
        
        # Check the protocol doesn't distill a index associated previously with a swapping
        elif operation == 'd':
            assert segment not in swapped_segments, "The protocol is bad formatted."

        assert checkProtocolUnit(punit), "The protocol is bad formatted."
    
    if not swapped_segments:
        S = 1
    else:
        S = max(swapped_segments) + 2 if S is None else S
    
    # Check if the number is between the allowed indexes for segments
    assert all([0 <= s <= S-2 for s in swapped_segments]) and len(swapped_segments) == S-1, "The protocol is bad formatted."
    return S


def validate_heterogeneous_parameters(parameters, number_of_segments, state_type: QuantumState):
    """
    Validate the parameters of a heterogeneous protocol.
    """
    if state_type == RealXState:
        for key in (
            "p_gen",
            "pauli_mode_decay_rates",
            "amplitude_damping_rate",
        ):
            if key not in parameters:
                raise ValueError(f"Missing required parameter: {key}")

        if "real_x_coordinates" in parameters:
            initial_states = np.asarray(parameters["real_x_coordinates"], dtype=float)
            if initial_states.shape != (number_of_segments, 6):
                raise ValueError(
                    "real_x_coordinates must have shape "
                    f"({number_of_segments}, 6)"
                )
            for coordinates in initial_states:
                RealXState(coordinates=coordinates)
        elif "lambdas" in parameters:
            initial_states = np.asarray(parameters["lambdas"], dtype=float)
            if initial_states.shape != (number_of_segments, 4):
                raise ValueError(
                    f"lambdas must have shape ({number_of_segments}, 4)"
                )
            for lambdas in initial_states:
                RealXState(lambdas=lambdas)
        elif "state" in parameters:
            states = parameters["state"]
            if not isinstance(states, Iterable) or len(states) != number_of_segments:
                raise ValueError(
                    "heterogeneous RealXState input must contain one state per segment"
                )
            if not all(isinstance(state, RealXState) for state in states):
                raise ValueError("every elementary state must be a RealXState")
        else:
            raise ValueError(
                "Missing elementary state: provide real_x_coordinates, lambdas, or state"
            )

        p_gen = np.asarray(parameters["p_gen"], dtype=float)
        if p_gen.shape != (number_of_segments,):
            raise ValueError("p_gen must contain one value per segment")
        if np.any(p_gen < 0.0) or np.any(p_gen > 1.0):
            raise ValueError("p_gen values must lie in [0,1]")

        node_count = number_of_segments + 1
        pauli_rates = np.asarray(parameters["pauli_mode_decay_rates"], dtype=float)
        adc_rates = np.asarray(parameters["amplitude_damping_rate"], dtype=float)
        if pauli_rates.shape != (node_count, 3):
            raise ValueError(
                "heterogeneous pauli_mode_decay_rates must have shape "
                f"({node_count}, 3)"
            )
        if adc_rates.shape != (node_count,):
            raise ValueError(
                "heterogeneous amplitude_damping_rate must have shape "
                f"({node_count},)"
            )
        if (
            not np.all(np.isfinite(pauli_rates))
            or not np.all(np.isfinite(adc_rates))
            or np.any(pauli_rates < 0.0)
            or np.any(adc_rates < 0.0)
        ):
            raise ValueError("memory-noise rates must be finite and nonnegative")
        jump_rates = np.stack(
            (
                pauli_rates[:, 1] + pauli_rates[:, 2] - pauli_rates[:, 0],
                pauli_rates[:, 0] + pauli_rates[:, 2] - pauli_rates[:, 1],
                pauli_rates[:, 0] + pauli_rates[:, 1] - pauli_rates[:, 2],
            ),
            axis=1,
        ) / 4.0
        if np.any(jump_rates < -1.0e-14):
            raise ValueError(
                "Pauli mode-decay rates violate the CP triangle constraints"
            )
        return

    if state_type == WernerState:
        required_keys = ["w0", "p_gen", "t_coh"]
        sf = parameters["w0"]
        noise = parameters["t_coh"]
    else:
        required_keys = ["lambdas", "p_gen", "depolarizing_rate"]
        sf = parameters["lambdas"]
        noise = parameters["depolarizing_rate"]
    for key in required_keys:
        if key not in parameters:
            raise ValueError(f"Missing required parameter: {key}")
        if not isinstance(parameters[key], Iterable):
            raise ValueError(f"{key} must be iterable.")

    if len(sf) != number_of_segments or len(parameters["p_gen"]) != number_of_segments:
        raise ValueError("The number of segments must match the number of p_gen and w0 values.")
    if len(noise) != number_of_segments + 1:
        raise ValueError("The number of nodes must match the number of t_coh values.")
