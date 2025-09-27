import pytest

import numpy as np
from numpy.testing import assert_allclose

from src.core.repeater_algorithm import repeater_sim, RepeaterChainEvaluation

from src.core.werner.state import WernerState
from src.utils.utility_functions import secret_key_rate


def test_secret_key_rate():
    """
    First, simple test with standard parameters.
    It tests a simple swap-only protocol, and check for the expected output secret key rate (with exponential extrapolation).
    """
    parameters = {
        "protocol": (0, ),
        "p_gen": 0.1,
        "p_swap": 0.5,
        "w0": 0.99,
        "tau": 5,
        "t_coh": 30,
        "t_trunc": 236
        }
    pmf, w_func = repeater_sim(parameters)
    skr = secret_key_rate(pmf, w_func, extrapolation=True)
    print(f"Secret key rate: {skr}")
    assert_allclose(skr, .01074, rtol=1.e-3)


"""
SETUPS for testing the algorithm.
"""

# Default solution for the test cases (uses non-efficient protocol_units.py)
def default_solution(parameters):
    simulator = RepeaterChainEvaluation(state_type=WernerState)
    simulator.efficient = False
    simulator.use_fft = False
    pmf, w_func = simulator.nested_protocol(parameters)
    return pmf, w_func

# Efficient protocol simulation setups (uses efficient protocol_units_efficient.py)
_convolution_simulator = RepeaterChainEvaluation(state_type=WernerState)
_convolution_simulator.use_fft = False

# FFT-based protocol simulation setups
_fft_simulator = RepeaterChainEvaluation(state_type=WernerState)
_fft_simulator.use_fft = True

# GPU-accelerated protocol simulation setups
_gpu_simulator = RepeaterChainEvaluation(state_type=WernerState)
_gpu_simulator.use_fft = True
_gpu_simulator.use_gpu = True
_gpu_simulator.gpu_threshold = 1


"""
PROTOCOLS for testing the algorithm.
These protocols are used to test the algorithm with different simulators and efficiency settings.
"""

swap_only_protocol = { 
    "protocol": (0, 0, 0), # GEN -> SWAP -> SWAP -> SWAP
    "p_gen": 0.5,
    "p_swap": 0.8,
    "sample_size": 100000,
    "w0": 1.,
    "t_coh": 50,
    "t_trunc": 100
}

dist_only_protocol = {
    "protocol": (1, 1), # GEN -> DIST -> DIST
    "p_gen": 0.5,
    "p_swap": 0.8,
    "sample_size": 100000,
    "w0": 1.,
    "t_coh": 20,
    "t_trunc": 100
}

memory_cutoff_swap_protocol = {
    "protocol": (0, 0, 0), # GEN -> SWAP -> SWAP -> SWAP
    "p_gen": 0.5,
    "p_swap": 0.8,
    "mt_cut": 5,
    "sample_size": 200000,
    "w0": 1.,
    "t_coh": 30,
    "t_trunc": 100
}

memory_cutoff_briegel_protocol = {
    "protocol": (1, 0, 1, 0, 1, 0), # GEN -> DIST -> SWAP -> DIST -> SWAP -> DIST -> SWAP [Briegel et al.]
    "p_gen": 0.5,
    "p_swap": 0.8,
    "mt_cut": (3, 6, 10, 14, 25, 100),
    "sample_size": 200000,
    "w0": 1.,
    "t_coh": 30,
    "t_trunc": 150
}

fidelity_cutoff_parameters = {
    "protocol": (1, 0), # GEN -> DIST -> SWAP
    "p_gen": 0.5,
    "p_swap": 0.8,
    "w0": 0.99,
    "t_coh": 50,
    "t_trunc": 100,
    "cut_type": "fidelity",
    "w_cut": 0.9,
    "sample_size": 1000000,
}

runtime_cutoff_parameters = {
    "protocol": (1, 0), # GEN -> DIST -> SWAP
    "p_gen": 0.5,
    "p_swap": 0.8,
    "w0": 0.99,
    "t_coh": 50,
    "t_trunc": 100,
    "cut_type": "run_time",
    "rt_cut": (5, 10),
    "sample_size": 1000000,
}


@pytest.mark.parametrize("parameters, expect",
    [
        pytest.param(swap_only_protocol, default_solution(swap_only_protocol), id="swap_only"),
        pytest.param(dist_only_protocol, default_solution(dist_only_protocol), id="dist_only"),
        pytest.param(memory_cutoff_swap_protocol, default_solution(memory_cutoff_swap_protocol), id="swap_memory_cutoff"),
        pytest.param(memory_cutoff_briegel_protocol, default_solution(memory_cutoff_briegel_protocol), id="swap_dist_memory_cutoff"),
        pytest.param(fidelity_cutoff_parameters, default_solution(fidelity_cutoff_parameters), id="fidelity_cutoff"),
        pytest.param(runtime_cutoff_parameters, default_solution(runtime_cutoff_parameters), id="runtime_cutoff"),
    ])
@pytest.mark.parametrize(("simulator", "efficient"),
    [
        pytest.param(_convolution_simulator, False, id="compartible-conv"),
        pytest.param(_convolution_simulator, True, id="efficient-conv"),
        pytest.param(_fft_simulator, True, id="compartible-fft"),
        pytest.param(_fft_simulator, False, id="efficient-fft"),
        pytest.param(_gpu_simulator, True, id="compartible-gpu"),
    ])
def test_algorithm(parameters, expect, simulator, efficient):
    """
    Test the algorithm with different simulators and efficiency settings.
    """
    default_pmf, default_w_func = expect
    simulator.efficient = True
    pmf, w_func = simulator.nested_protocol(parameters)
    cdf = np.cumsum(pmf)
    start_pos = next(x[0] for x in enumerate(cdf) if x[1] > 1.0e-2)
    end_pos = np.searchsorted(cdf, 0.99)
    assert_allclose(pmf[start_pos: end_pos], default_pmf[start_pos: end_pos], rtol=1.0e-7)
    assert_allclose(w_func[start_pos: end_pos], default_w_func[start_pos: end_pos], rtol=1.0e-7)


if __name__ == "__main__":
    pytest.main(["-sv", __file__])