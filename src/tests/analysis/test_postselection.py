"""Scientific controls for interpreting the global-policy study."""
from copy import deepcopy

import numpy as np
import pytest

from src.analysis.postselection import POLICIES, bb84_score, evaluate
from src.analysis.postselection_example import prepared_config
from src.core.pauli_adc.protocol_units import accepted_bell_swap
from src.visualization.config import DEFAULT
from src.visualization.evaluate import EvaluationService
from src.visualization.metrics import binary_entropy, time_quantities


def test_two_link_ad_has_no_immediate_pooled_quality_gain_from_rejection():
    # Only the older elementary pair is damped; the younger Werner pair is fresh.
    # This differs from prescribed already-biased inputs to both sides of a swap.
    base = deepcopy(DEFAULT)
    base["links"] = 2
    base["protocol"]["tree"] = [0, 1]
    base["initial_state"]["fidelity"] = 0.97
    base["hardware"]["p_swap"] = 1
    base["memory"].update(t1_slots=200, t2_slots=400)
    base["numerics"]["t_trunc"] = 512
    service = EvaluationService()
    rows = {}
    for p in POLICIES:
        base["protocol"]["bell_outcomes"] = p
        rows[p] = evaluate(service, base)
    all_row = rows["all_corrected"]
    assert all_row["pooled_secret_fraction"] > 0
    for policy in POLICIES[1:]:
        row = rows[policy]
        assert row["pooled_secret_fraction"] == pytest.approx(all_row["pooled_secret_fraction"], abs=1e-10)
        assert row["rate_bits_per_slot"] == pytest.approx(all_row["rate_bits_per_slot"]/2, rel=1e-8)
        assert row["mean_wait_repository_slots"] == pytest.approx(2*all_row["mean_wait_repository_slots"], rel=1e-8)


def test_no_noise_four_link_quality_matches_independent_werner_control():
    base = deepcopy(DEFAULT)
    base["initial_state"]["fidelity"] = 0.97
    base["hardware"]["p_swap"] = 1
    base["memory"].update(t1_slots=None, t2_slots=None)
    w = (4*0.97-1)/3
    expected_error = (1-w**4)/2
    expected_score = 1-2*float(binary_entropy(expected_error))
    service = EvaluationService()
    rows = []
    for p in POLICIES:
        base["protocol"]["bell_outcomes"] = p
        rows.append(evaluate(service, base))
    for row in rows:
        assert row["eX"] == pytest.approx(expected_error, abs=1e-10)
        assert row["eZ"] == pytest.approx(expected_error, abs=1e-10)
        assert bb84_score(row) == pytest.approx(expected_score, abs=1e-10)
    assert rows[1]["rate_bits_per_slot"] < rows[0]["rate_bits_per_slot"]
    assert rows[2]["rate_bits_per_slot"] < rows[0]["rate_bits_per_slot"]


def test_short_horizon_cannot_be_misreported_as_a_converged_boundary():
    base = deepcopy(DEFAULT)
    base["hardware"]["p_swap"] = 1
    base["protocol"]["bell_outcomes"] = "psi_pair"
    base["numerics"]["t_trunc"] = 8
    with pytest.raises(ValueError, match="horizon check failed"):
        evaluate(EvaluationService(), base)


def test_prepared_relaxed_pairs_extend_positive_key_in_psi_branch_with_retries():
    service = EvaluationService()
    results = {}
    for policy in POLICIES:
        config = prepared_config(0.9145, policy)
        summary = evaluate(service, config)
        initial = config["initial_state"]["values"]
        state, probability = accepted_bell_swap(initial, initial, outcomes=policy, conditioned=True)
        fraction = float(time_quantities(state)["secret_fraction"])
        assert summary["pooled_secret_fraction"] == pytest.approx(fraction, abs=1e-12)
        # Deterministic one-slot input batches plus destructive geometric retries.
        assert summary["mean_wait_repository_slots"] == pytest.approx(1/probability, abs=1e-10)
        assert summary["rate_bits_per_slot"] == pytest.approx(probability*fraction, abs=1e-12)
        results[policy] = summary
    assert results["all_corrected"]["bb84_score"] < 0
    assert results["phi_pair"]["bb84_score"] < 0
    assert results["psi_pair"]["bb84_score"] > 0
    assert results["psi_pair"]["rate_bits_per_slot"] == pytest.approx(0.002210459822068, abs=1e-12)
    assert results["psi_pair"]["mean_wait_repository_slots"] > results["phi_pair"]["mean_wait_repository_slots"]
