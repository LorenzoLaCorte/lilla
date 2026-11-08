import numpy as np
from numpy.testing import assert_allclose
import pytest

from src.core.werner.protocol_units import *
from src.utils.logging_utilities import *
from src.utils.utility_functions import *
from src.optimization.optimize_cutoff import CutoffOptimizer
from src.core.repeater_algorithm import RepeaterChainEvaluation
from src.core.werner.state import WernerState

def test_fidelity_cut_off_function():
    w_cut = 0.95
    t_coh = 50
    assert(fidelity_cut_off(1, 3, 0.98, 0.98, w_cut=w_cut, t_coh=t_coh) == (2, False))
    assert(fidelity_cut_off(2, 3, 0.98, 0.98, w_cut=w_cut, t_coh=t_coh) == (3, True))
    assert(fidelity_cut_off(4, 3, 0.98, 0.98, w_cut=w_cut, t_coh=t_coh) == (4, True))
    assert(fidelity_cut_off(3, 3, 0.98, 0.98, w_cut=w_cut, t_coh=t_coh) == (3, True))
    assert(fidelity_cut_off(6, 3, 0.98, 0.98, w_cut=w_cut, t_coh=t_coh) == (4, False))
    assert(fidelity_cut_off(1, 4, 0.98, 0.94, w_cut=w_cut, t_coh=t_coh) == (2, False))
    assert(fidelity_cut_off(1, 2, 0.98, 0.94, w_cut=w_cut, t_coh=t_coh) == (2, False))
    assert(fidelity_cut_off(4, 4, 0.95, 0.95, w_cut=w_cut, t_coh=t_coh) == (4, True))
    assert(fidelity_cut_off(4, 4, 0.95, 0.9499, w_cut=w_cut, t_coh=t_coh) == (4, False))


def test_cutoff_dict_generation():    
    parameters = {
        "protocol": (0, 0),
        "p_gen": 0.5,
        "p_swap": 0.8,
        "w0": 1.0,
        "t_coh": 20,
        "t_trunc": 100,
        "cut_type": "fidelity",
        }

    cutoff_dict = create_cutoff_dict((100, 120, 0.5, 0.4), ["memory_time", "fidelity"], parameters)
    assert_allclose(cutoff_dict["memory_time"], np.array([100, 120]))
    assert_allclose(cutoff_dict["fidelity"], np.array([0.5, 0.4]))

    cutoff_dict = create_cutoff_dict((0.8, ), ["fidelity"], parameters)
    assert_allclose(cutoff_dict["fidelity"], np.array([0.8, 0.8]))

    cutoff_dict = create_cutoff_dict((100, ), ["memory_time"], parameters)
    assert_allclose(cutoff_dict["memory_time"], np.array([100, 100]))

    cutoff_dict = create_cutoff_dict((100, 0.5), ["memory_time", "fidelity"], parameters)
    assert_allclose(cutoff_dict["memory_time"], np.array([100, 100]))
    assert_allclose(cutoff_dict["fidelity"], np.array([0.5, 0.5]))


@pytest.mark.parametrize("communication,expected", [(False, (4,)), (True, (3,))])
def test_opt_adaptive_trunc(communication, expected):
    """
    Search adaptation preserves the optimum for the selected physical model.
    The reference cutoff 4 assumes zero classical communication; with the
    current communication model, an independent time-domain grid gives 3.
    """
    np.random.seed(1)
    parameters = {
        "protocol": (0, ),
        "p_gen": 0.1,
        "p_swap": 0.5,
        "w0": 0.99,
        "t_coh": 30,
        "t_trunc": 300
        }

    logging.info("Full tau optimization\n")
    opt = CutoffOptimizer(adaptive=True, workers=1, simulator=RepeaterChainEvaluation(
        state_type=WernerState, classical_communication=communication))
    best_cutoff_dict = opt.run(parameters)
    assert_allclose(best_cutoff_dict["memory_time"], expected)


@pytest.mark.parametrize("communication,expected", [(False, (3, 7)), (True, (1, 3))])
def test_opt_adaptive_search_range(communication, expected):
    """
    The search range should be restricted
    """
    np.random.seed(3)
    parameters = {
        "protocol": (0, 0),
        "p_gen": 0.2,
        "p_swap": 0.6,
        "w0": 0.95,
        "t_coh": 300,
        "t_trunc": 400
        }

    opt = CutoffOptimizer(adaptive=True, popsize=5, workers=1,
                          simulator=RepeaterChainEvaluation(
                              state_type=WernerState, classical_communication=communication))
    best_cutoff_dict = opt.run(parameters)
    assert_allclose(best_cutoff_dict["memory_time"], expected)


def test_opt_uniform():
    np.random.seed(0)
    parameters = {
        "protocol": (0, 0),
        "p_gen": 0.1,
        "p_swap": 0.8,
        "t_trunc": 500,
        "w0": 0.99,
        "t_coh": 400,
        }

    opt = CutoffOptimizer(
        opt_kind="uniform_de", adaptive=True)
    best_cutoff_dict = opt.run(parameters)
    assert_allclose(best_cutoff_dict["memory_time"], (25, 25))


if __name__ == "__main__":
    pytest.main(["-sv", __file__])
