"""Symbolic and numerical verification for Pauli noise plus amplitude damping."""
 # PATCH GUIDE — tests
 # Tests are arranged as a ladder: symbolic identities, dense independent channel
 # and circuit oracles, protocol-level metamorphic checks, then efficient/direct
 # parity.  A speed claim is meaningful only after weighted PMF/state equality is
 # asserted.
