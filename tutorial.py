from matplotlib import pyplot as plt
import numpy as np
from src.core.repeater_algorithm import RepeaterChainEvaluation
from src.core.bell.state import BellState
from src.core.werner.state import WernerState
from src.core.states import QuantumState
from src.plotting.basic_plots import plot_algorithm
from src.utils.utility_functions import werner_to_fid, bell_to_fid, get_mean_waiting_time, get_mean

# ------------------
# This is a notebook for testing purposes
# it will be converted in a python notebook file
# ------------------

def working_example():
    # Entanglement Swapping using Werner states
    # Generation through direct photon transmission over a fiber channel
    state_type : QuantumState = WernerState
    machinery = RepeaterChainEvaluation(state_type=state_type)

    # Here, we test a **nested** protocol.
    # This is a protocol that acts on $2^l+1$ nodes, e.g., 3 here ($l=1$)
    # and swaps the $l-1$ elementary links for $l$ nesting levels 
    parameters = {
        "protocol": (0,),
        "p_gen": 0.05,
        "p_swap": 0.5,
        "w0": 0.8,  # fidelity 0.85
        "t_coh": 1000, # depolarizing noise
        "t_trunc": 1000,
    }

    w_pmf, w_func = machinery.nested_protocol(parameters)
    w_fid_func = [werner_to_fid(w) for w in w_func]
    fig, axs = plot_algorithm(
        pmf=w_pmf,
        fid_func=w_fid_func,
    )
    plt.show()


    # Now, let's use a Bell state instead
    state_type : QuantumState = BellState
    machinery = RepeaterChainEvaluation(state_type=state_type)

    # Here, we can test a more fine-grained noise using a Bell diagonal state, 
    # with four parameters for the weights of the Bell states in the mixture
    # Thus, we allow for other types of noise, e.g. dephasing, and evaluate their effect 
    parameters = {
        "protocol": (0,),
        "p_gen": 0.05,
        "p_swap": 0.5,
        "lambdas": [0.85, 0.08, 0.05, 0.02],  # fidelity 0.85
        "depolarizing_rate": 0.001, # depolarizing noise
        "dephasing_rate": 0.05, # depolarizing noise
        "t_trunc": 1000,
    }

    b_pmf, l_func = machinery.nested_protocol(parameters)
    fig, axs = plot_algorithm(
        pmf=b_pmf,
        fid_func=l_func,
        legend_fid=["$\\lambda_{\\phi^+}$", "$\\lambda_{\\phi^-}$", "$\\lambda_{\\psi^+}$", "$\\lambda_{\\psi^-}$"],
    )
    plt.show()


    # We can compute the probability mass that is covered by this distribution
    coverage = np.sum(w_pmf)
    print("Total probability mass:", coverage)

    # We can also calculate the average waiting time and compare it with the analytical result.
    numerical_result = get_mean_waiting_time(w_pmf) # numerical
    p_gen = parameters["p_gen"]
    p_swap = parameters["p_swap"]
    analytical_result = (3-2*p_gen)/((2-p_gen)*p_gen*p_swap) # analytical
    print("Numerical result (Werner):", numerical_result)
    print("Analytical result:", analytical_result)

    # Let's try with Bell states
    numerical_result_bell = get_mean_waiting_time(b_pmf) # numerical
    print("Numerical result (Bell):", numerical_result_bell)

    # Notice that the numerical results should be very close to each other and to the analytical one
    # This is a good test of the validity of our evaluations

    # Now, let's consider fidelity
    # Here, we use `werner_to_fid` and `fid_to_werner` to transfer between the Werner parameter and the fidelity.
    # We use `bell_to_fid` to compute the fidelity of the Bell diagonal states.
    # We can compute the average fidelity of the final states produced by the repeater chain
    average_fidelity = get_mean(w_pmf, w_fid_func)
    print("Average fidelity (Werner):", average_fidelity)
    l_fid_func = [bell_to_fid(lambdas) for lambdas in l_func] # consider only lambda_phi+
    average_fidelity_bell = get_mean(b_pmf, l_fid_func)
    print("Average fidelity (Bell):", average_fidelity_bell)

    # Notice that the fidelity of the Bell state is lower, 
    # as expected due to the additional noise (dephasing) considered

    # ! Under the `test` module, an extensive testsuite evaluates the correctness of our tool
    # e.g., checking the two formalisms produce the same results if tested under the same condition

    # Distillation
    # Entanglement distillation, a.k.a. purification, is used to consume two links of lower fidelity
    # to produce one link of higher fidelity.
    # This is done probabilistically, and thus affects the waiting time.

    # In the nested protocol representation we encode distillation by placing a 1 in the protocol tuple
    # Thus, a nested protocol with one level of distillation can be defined as
    parameters["protocol"] = (1,)

    # We can evaluate the performance of this protocol with Bell states as before
    # By twirling after distillation, we get a Werner state (useful for comparison)
    state_type : QuantumState = BellState
    machinery = RepeaterChainEvaluation(state_type=state_type, twirling=True)

    dist_pmf, dist_sf = machinery.nested_protocol(parameters)
    fig, axs = plot_algorithm(
        pmf=dist_pmf,
        fid_func=dist_sf,
        legend_fid=["$\\lambda_{\\phi^+}$", "$\\lambda_{\\phi^-}$", "$\\lambda_{\\psi^+}$", "$\\lambda_{\\psi^-}$"]
    )
    plt.show()

    # By not twirling after distillation, we keep the full Bell diagonal state
    state_type : QuantumState = BellState
    machinery = RepeaterChainEvaluation(state_type=state_type, twirling=False)

    dist_pmf, dist_sf = machinery.nested_protocol(parameters)
    fig, axs = plot_algorithm(
        pmf=dist_pmf,
        fid_func=dist_sf,
        legend_fid=["$\\lambda_{\\phi^+}$", "$\\lambda_{\\phi^-}$", "$\\lambda_{\\psi^+}$", "$\\lambda_{\\psi^-}$"]
    )
    plt.show()

    # We can increase complexity by considering asymmetric protocols
    # These are protocols that differ from the *doubling* nested structure of the previous examples
    # They can be represented as binary trees of swapping and distillation operations where
    # - nodes represent entangled links
    # - nodes' labels represent the number of ro. of distillation on a link
    # - edges of the tree represent how link are combined

    # Since any swap takes either 0 or 2 links in input, the trees are full binary trees
    # For example, consider the protocol

    #         0
    #        / \
    #       0   0
    #      / \  
    #     1   0 
    asymmetric_protocol = ("d0", "s0", "s1")

    # The three leaves represent the three elementary links
    # The leftmost link undergoes one round of distillation before being swapped with the middle link
    # Finally, the resulting link is swapped with the rightmost link

    state_type : QuantumState = WernerState
    machinery = RepeaterChainEvaluation(state_type=state_type)
    parameters = {
        "protocol": asymmetric_protocol,
        "p_gen": 0.05,
        "p_swap": 0.5,
        "w0": 0.8,  # fidelity 0.85
        "t_coh": 1000, # depolarizing noise
        "t_trunc": 1000,
    }
    asym_pmf, asym_fid_func = machinery.asymmetric_homogeneous_protocol(parameters, number_of_segments=3)
    asym_fid_func = [werner_to_fid(w) for w in asym_fid_func]
    fig, axs = plot_algorithm(
        pmf=asym_pmf,
        fid_func=asym_fid_func,
    )
    plt.show()

    # The same protocol can be defined for Bell state
    state_type : QuantumState = BellState
    machinery = RepeaterChainEvaluation(state_type=state_type, twirling=False)
    parameters = {
        "protocol": asymmetric_protocol,
        "p_gen": 0.05,
        "p_swap": 0.5,
        "lambdas": [0.85, 0.05, 0.05, 0.05],  # fidelity 0.85
        "depolarizing_rate": 0.001, # depolarizing noise
        "t_trunc": 1000,
    }
    asym_bell_pmf, asym_l_func = machinery.asymmetric_homogeneous_protocol(parameters, number_of_segments=3)
    fig, axs = plot_algorithm(
        pmf=asym_bell_pmf,
        fid_func=asym_l_func,
        legend_fid=["$\\lambda_{\\phi^+}$", "$\\lambda_{\\phi^-}$", "$\\lambda_{\\psi^+}$", "$\\lambda_{\\psi^-}$"],
    )
    plt.show()

    # We can also consider asymmetric protocols with heterogeneous hardware
    # That is,
    # - each link can have different a different probability of generation and initial quality
    # - each node can have a different memory coherence time

    state_type : QuantumState = WernerState
    machinery = RepeaterChainEvaluation(state_type=state_type)
    parameters = {
        "p_gen": [0.1, 0.1, 0.05],
        "p_swap": 0.5,
        "w0": [0.9, 0.9, 0.9],  # fidelity for the 3 links
        "t_coh": [1000, 1000, 1000, 1000],  # coherence time for the 4 nodes
        "t_trunc": 1000,
    }
    # For example, consider the protocols
    #         0
    #        / \
    #       0   0
    #      / \  
    #     0   0 
    asymmetric_protocol_left = ("s0", "s1")
    # and
    #         0
    #        / \
    #       0   0
    #          / \
    #         0   0
    asymmetric_protocol_right = ("s1", "s0")

    parameters["protocol"] = asymmetric_protocol_left
    asym_het_pmf_left, asym_het_fid_func_left = machinery.asymmetric_heterogeneous_protocol(
        parameters,
        number_of_segments=3,
    )
    print("Mean waiting time (left-swap-first):", get_mean_waiting_time(asym_het_pmf_left))
    print("Average fidelity (left-swap-first):", get_mean(asym_het_pmf_left, [werner_to_fid(w) for w in asym_het_fid_func_left]))

    parameters["protocol"] = asymmetric_protocol_right
    asym_het_pmf_right, asym_het_fid_func_right = machinery.asymmetric_heterogeneous_protocol(
        parameters,
        number_of_segments=3,
    )
    print("Mean waiting time (right-swap-first):", get_mean_waiting_time(asym_het_pmf_right))
    print("Average fidelity (right-swap-first):", get_mean(asym_het_pmf_right, [werner_to_fid(w) for w in asym_het_fid_func_right]))

    # Since, on average, the rightmost link takes longer to generate, 
    # the protocol that swaps it last performs better (less waiting time, higher fidelity)
    # as, while waiting for it, the other two links are generated and then swapped

# """
# TODO: Bayesian optimization
# TODO: cutoff optimization?
# """

# We can also evaluate asymmetric heterogenous protocols with Bell diagonal states
# setting different generation success probabilities and lambdas for the three different links
# and different coherence times for the four nodes
state_type : QuantumState = BellState
machinery = RepeaterChainEvaluation(state_type=state_type, twirling=False)
parameters = {
    "p_gen": [0.1, 0.1, 0.05],
    "p_swap": 0.5,
    "lambdas": [[0.75, 0.14, 0.1, 0.01], 
                [0.75, 0.14, 0.1, 0.01], 
                [0.95, 0.01, 0.03, 0.01]],
    "depolarizing_rate": [0.001, 0.001, 0.001, 0.001],
    "dephasing_rate": [0.01, 0.01, 0.01, 0.01],
    "t_trunc": 1000,
}
# For example, consider the protocols
#         0
#        / \
#       0   0
#      / \  
#     0   0 
asymmetric_protocol_left = ("s0", "s1")
parameters["protocol"] = asymmetric_protocol_left
asym_het_bell_pmf_left, asym_het_bell_l_func_left = machinery.asymmetric_heterogeneous_protocol(
    parameters,
    number_of_segments=3,
)
print("Mean waiting time (left-swap-first, Bell):", get_mean_waiting_time(asym_het_bell_pmf_left))
l_fid_func_left = [bell_to_fid(lambdas) for lambdas in asym_het_bell_l_func_left]
print("Average fidelity (left-swap-first, Bell):", get_mean(asym_het_bell_pmf_left, l_fid_func_left))

fig, axs = plot_algorithm(
    pmf=asym_het_bell_pmf_left,
    fid_func=asym_het_bell_l_func_left,
    legend_fid=["$\\lambda_{\\phi^+}$", "$\\lambda_{\\phi^-}$", "$\\lambda_{\\psi^+}$", "$\\lambda_{\\psi^-}$"],
)
plt.show()
