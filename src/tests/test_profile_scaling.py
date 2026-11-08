"""Check profiling accounting without running a benchmark sweep."""

import pytest

from src.core import repeater_algorithm as algorithm
from src.scripts import profile_scaling as profile


def test_nested_phase_times_are_disjoint_and_preserve_return_values(monkeypatch):
    ticks = iter([0.0, 1.0, 3.0, 5.0])
    monkeypatch.setattr(profile.time, "perf_counter", lambda: next(ticks))
    timer = profile.PhaseTimer()
    child = timer.wrap(lambda value: value + 1, "child")
    parent = timer.wrap(lambda value: child(value) * 2, "parent")
    assert parent(10) == 22
    assert timer.seconds == {"child": 2.0, "parent": 3.0}
    assert sum(timer.seconds.values()) == 5.0
    assert timer.calls == {"child": 1, "parent": 1}
    assert timer.stack == []


def test_instrumentation_restores_functions_after_an_exception():
    original = algorithm.werner_join_efficient
    timer = profile.PhaseTimer()
    with pytest.raises(RuntimeError, match="interrupted"):
        with timer.instrument():
            assert algorithm.werner_join_efficient is not original
            raise RuntimeError("interrupted")
    assert algorithm.werner_join_efficient is original


def test_profile_signature_ignores_reporting_but_tracks_timed_code(monkeypatch):
    original = profile.implementation_digest()
    monkeypatch.setattr(profile, "save_report", lambda *args: None)
    assert profile.implementation_digest() == original

    def replacement_evaluation(*args):
        return None

    monkeypatch.setattr(profile.benchmark, "evaluate", replacement_evaluation)
    assert profile.implementation_digest() != original
