"""Review of the local 1.0 integration (integrate/1.0 f1b21a1): semantic interactions of the merged lines.

INT-1: the freeze spec marks the 0.7 capability ``distributed_keygen`` of the stable
scheme ``sife/cp_ab_pairing`` experimental, in ``stability.EXPERIMENTAL_CAPABILITIES``; the merge left the table
empty, so the capability (and the key-issuance records) read as stable.

This is the INT-1 test of the review; its INT-2 test (a key server forked while another thread is inside
``issue``) went to the 0.7 line.
"""

from pyfe4ai.core import registry, stability


def test_int1_experimental_capabilities_of_stable_schemes_are_marked():
    assert "distributed_keygen" in stability.EXPERIMENTAL_CAPABILITIES.get("sife/cp_ab_pairing", frozenset())
    for scheme in registry.SCHEMES:
        if stability.tier(scheme.id) != stability.STABLE:
            continue
        for cap in scheme.capabilities:
            if "experimental" in registry.CAPABILITIES.get(cap, ""):
                assert cap in stability.EXPERIMENTAL_CAPABILITIES.get(scheme.id, frozenset()), (scheme.id, cap)
