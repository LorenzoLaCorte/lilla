"""Version-1 input validation and physical-to-backend translation, in slots."""
from copy import deepcopy
import json
import numbers

import numpy as np

from src.core.pauli_adc.state import RealXState
from src.types.repeater_types import validate_heterogeneous_parameters
from src.visualization.protocols import compile_protocol

MODELS = ("full_real_x", "local_channel_twirl")
OUTCOMES = ("all_corrected", "phi_pair", "psi_pair", "phi_plus")
DEFAULT = {
    "schema_version": 1, "name": "Four-link relaxation and dephasing", "links": 4,
    "slot_duration_seconds": None,
    "hardware": {"p_gen": 0.2, "p_swap": 0.9, "p_distillation": 1.0},
    "initial_state": {"kind": "werner", "fidelity": 0.98},
    "memory": {"kind": "t1_t2", "t1_slots": 2000.0, "t2_slots": 1000.0},
    "protocol": {"tree": [[0, 1], [2, 3]], "distillation_rounds": {},
                 "memory_cutoff_slots": None, "cutoff_overrides": {},
                 "bell_outcomes": "all_corrected"},
    "prediction_model": "full_real_x",
    "numerics": {"t_trunc": 1024, "coverage_target": 0.999},
}


def canonical(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)


def load_config(text):
    value = json.loads(text, parse_constant=lambda v: (_ for _ in ()).throw(
        ValueError(f"nonstandard JSON value {v}")))
    expand_config(value)
    return value


def fields(obj, expected, name):
    if not isinstance(obj, dict) or set(obj) != set(expected):
        raise ValueError(f"{name} requires exactly these fields: {', '.join(expected)}")


def scalar(value, name, low=0.0, high=None):
    if isinstance(value, bool) or not isinstance(value, numbers.Real):
        raise ValueError(f"{name} must be a finite number")
    result = float(value)
    if not np.isfinite(result) or result < low or (high is not None and result > high):
        raise ValueError(f"{name} outside permitted range")
    return result


def broadcast(value, count, name, width=None, high=None):
    shared = (width,) if width else ()
    shape = (count, width) if width else (count,)
    arr = np.asarray(value, dtype=object)
    if arr.shape == shared:
        arr = np.broadcast_to(arr, shape)
    if arr.shape != shape:
        raise ValueError(f"{name} requires shared shape {shared} or shape {shape}")
    return np.array([scalar(v, name, high=high) for v in arr.flat]).reshape(shape)


def lifetime_rates(value, count, name):
    values = value if isinstance(value, list) else [value] * count
    if len(values) != count:
        raise ValueError(f"{name} requires a scalar or {count} node values")
    rates = []
    for v in values:
        if v is None:
            rates.append(0.0)
        else:
            t = scalar(v, name)
            if t <= 0:
                raise ValueError(f"{name} must be positive or null (infinite)")
            rates.append(1.0 / t)
    return np.asarray(rates)


def expand_config(config):
    """Return canonical physical config, fresh backend parameters and operation table."""
    c = deepcopy(config)
    fields(c, DEFAULT, "configuration")
    if type(c["schema_version"]) is not int or c["schema_version"] != 1:
        raise ValueError("unsupported schema_version")
    if not isinstance(c["name"], str):
        raise ValueError("name must be a string")
    s = c["links"]
    if type(s) is not int or not 1 <= s <= 8:
        raise ValueError("MVP links must be an integer from 1 to 8")
    if c["slot_duration_seconds"] is not None:
        if scalar(c["slot_duration_seconds"], "slot duration") <= 0:
            raise ValueError("slot duration must be positive")
    fields(c["hardware"], DEFAULT["hardware"], "hardware")
    h = c["hardware"]
    p = {"p_gen": broadcast(h["p_gen"], s, "p_gen", high=1).tolist(),
         "p_swap": scalar(h["p_swap"], "p_swap", high=1),
         "p_distillation": scalar(h["p_distillation"], "p_distillation", high=1)}
    initial = c["initial_state"]
    if not isinstance(initial, dict):
        raise ValueError("initial_state must be an object")
    kind = initial.get("kind")
    if kind == "werner":
        fields(initial, ("kind", "fidelity"), "initial_state")
        f = broadcast(initial["fidelity"], s, "fidelity", high=1)
        values = np.column_stack([f, (1-f)/3, (1-f)/3, (1-f)/3])
        p["lambdas"] = values.tolist()
    elif kind in ("bell_weights", "real_x"):
        fields(initial, ("kind", "values"), "initial_state")
        if kind == "bell_weights":
            values = broadcast(initial["values"], s, "Bell weights", width=4, high=1)
            if not np.allclose(values.sum(axis=1), 1, atol=1e-12, rtol=0):
                raise ValueError("Bell weights must sum to one")
            p["lambdas"] = values.tolist()
        else:
            values = np.asarray(initial["values"], dtype=object)
            if values.shape == (6,):
                values = np.broadcast_to(values, (s, 6))
            if values.shape != (s, 6):
                raise ValueError(f"real-X requires shape (6,) or ({s},6)")
            values = np.array([scalar(v, "real-X", low=-1, high=1)
                               for v in values.flat]).reshape(s, 6)
            for row in values:
                RealXState(coordinates=row)
            p["real_x_coordinates"] = values.tolist()
    else:
        raise ValueError("initial_state kind must be werner, bell_weights or real_x")
    memory = c["memory"]
    if not isinstance(memory, dict):
        raise ValueError("memory must be an object")
    if memory.get("kind") == "t1_t2":
        fields(memory, ("kind", "t1_slots", "t2_slots"), "memory")
        k = lifetime_rates(memory["t1_slots"], s+1, "T1")
        phi = lifetime_rates(memory["t2_slots"], s+1, "T2") - k/2
        if np.any(phi < -1e-14):
            raise ValueError("physical lifetimes require T2 <= 2*T1")
        gamma = np.column_stack([np.maximum(phi, 0), np.maximum(phi, 0), np.zeros(s+1)])
    elif memory.get("kind") == "pauli_adc":
        fields(memory, ("kind", "pauli_mode_decay_rates", "amplitude_damping_rate"), "memory")
        gamma = broadcast(memory["pauli_mode_decay_rates"], s+1, "Pauli rates", width=3)
        k = broadcast(memory["amplitude_damping_rate"], s+1, "damping rates")
    else:
        raise ValueError("memory kind must be t1_t2 or pauli_adc")
    p.update(pauli_mode_decay_rates=gamma.tolist(), amplitude_damping_rate=k.tolist())
    validate_heterogeneous_parameters(p, s, RealXState)
    if c["prediction_model"] not in MODELS:
        raise ValueError("unsupported prediction_model")
    if c["prediction_model"] == "local_channel_twirl":
        p["pauli_mode_decay_rates"] = (gamma + k[:, None] * [0.5, 0.5, 1]).tolist()
        p["amplitude_damping_rate"] = np.zeros(s+1).tolist()
    fields(c["numerics"], DEFAULT["numerics"], "numerics")
    t = c["numerics"]["t_trunc"]
    if type(t) is not int or t < 2:
        raise ValueError("t_trunc must be an integer >= 2")
    scalar(c["numerics"]["coverage_target"], "coverage target", high=1)
    fields(c["protocol"], DEFAULT["protocol"], "protocol")
    if c["protocol"]["bell_outcomes"] not in OUTCOMES:
        raise ValueError("invalid Bell-outcome policy")
    sequence, operations = compile_protocol(c["protocol"], s, t)
    p.update(protocol=sequence, cutoff=[o["backend_cutoff"] for o in operations],
             bell_outcomes=c["protocol"]["bell_outcomes"], t_trunc=t,
             cut_type="memory_time", real_x_werner_twirling=False)
    canonical(c)  # also rejects nonstandard numeric JSON values
    return c, p, operations


def converted(config, model=None, double_horizon=False):
    result = deepcopy(config)
    if model is not None:
        result["prediction_model"] = model
    if double_horizon:
        result["numerics"]["t_trunc"] *= 2
    expand_config(result)
    return result
