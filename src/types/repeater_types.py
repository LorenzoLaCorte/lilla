"""

TODO: types and exception defined in gp_utils should be moved here
"""
from argparse import ArgumentTypeError
from collections.abc import Iterable
import re
from typing import Tuple, TypedDict, Union, Literal, List
import numpy as np

from src.core.states import QuantumState

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


def validate_heterogeneous_parameters(parameters, number_of_segments):
    """
    Validate the parameters of a heterogeneous protocol.
    """
    if not isinstance(parameters["w0"], Iterable) or not isinstance(parameters["t_coh"], Iterable):
        raise ValueError("w0 and t_coh must be iterable.")
    if len(parameters["w0"]) != number_of_segments or len(parameters["p_gen"]) != number_of_segments:
        raise ValueError("The number of segments must match the number of p_gen and w0 values.")
    if len(parameters["t_coh"]) != number_of_segments + 1:
        raise ValueError("The number of nodes must match the number of t_coh values.")
