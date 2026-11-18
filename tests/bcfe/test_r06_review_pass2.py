"""R-0.6 pass 2 review (feat/0.6-p1 481450d..0173603): tests of the scheme class ``bcfe/gc_ristretto255``.

See records/2026-11-perf/sessions/r-0.6/report.md of the project's records, section "Pass 2".

These tests need the scheme class (c035b42 and later); this branch sits on 481450d, where they are skipped.
Run them on 0173603:

    git show review/0.6-p1-security:tests/bcfe/test_r06_review_pass2.py > tests/bcfe/test_r06_review_pass2.py
    PYFE4AI_CACHE_DIR=<scratch> python -m pytest -q tests/bcfe/test_r06_review_pass2.py

On 0173603 the first test fails (finding R06-P2-01); the others pass and record checks of the review.

Copied from bc48e34 with two adaptations to the fix of R06-P2-01 (the slot record and its digest no longer carry
the attribute-set digest), each marked "adapted": the first test reads the field with ``get`` (absent: nothing to
test a guess against), the second reseals a slot with the new ``garbled.slot_digest`` signature.
"""

from __future__ import annotations

import copy
import itertools
import random

import pytest

scheme = pytest.importorskip("pyfe4ai.schemes.bcfe.gc_ristretto255")

import pyfe4ai  # noqa: E402
from pyfe4ai import circuits  # noqa: E402
from pyfe4ai.core.exceptions import FEKeyError  # noqa: E402
from pyfe4ai.core.formats import garbled  # noqa: E402
from pyfe4ai.policy import attribute_set_digest  # noqa: E402

SCHEME = "bcfe/gc_ristretto255"
SMALL = circuits.linear_threshold(n=2, bw=2, bx=2)  # m = 8 key bits
W_SMALL = {"weights": [1, -1], "bias": 2}


def make(tmp_path, circuit=SMALL, **options):
    kg, crypto = pyfe4ai.create(SCHEME, circuit=circuit, cache_dir=str(tmp_path), **options)
    kg.setup()
    return kg, crypto


def test_r06_p2_01_the_slot_list_does_not_let_anyone_test_a_guess_of_a_requesters_attributes(tmp_path):
    """R06-P2-01 (Low): every public slot record carries ``attribute_set_digest``, an unsalted hash of the
    requester's attribute set. Anyone who holds the public parameters -- every encryptor -- can test guesses
    offline; attribute vocabularies are small, so the set is in effect public, and slots of requesters with the
    same attributes link. The module documentation says what the slot list hides (``w``) but not that it shows
    the attributes. This test states the property the documentation suggests; it fails on 0173603."""
    kg, _ = make(tmp_path)
    kg.issue_key(W_SMALL, requester="alice", attributes=["role:doctor", "dept:oncology"])
    pp = kg.get_public_parameters()  # what every encryptor holds
    published = pp["slots"][0].get("attribute_set_digest")  # adapted: absent since the fix of R06-P2-01
    vocabulary = ["role:doctor", "role:nurse", "role:admin", "dept:oncology", "dept:cardiology", "dept:icu"]
    found = [guess for r in range(len(vocabulary) + 1) for guess in itertools.combinations(vocabulary, r)
             if attribute_set_digest(guess) == published]
    assert not found, "the public slot list gives away the requester's attributes: {}".format(found)


def test_r06_p2_checkauth_refuses_the_holders_key_also_at_the_unselected_position(tmp_path):
    """I13 with h': a grant that holds the holder's key at (j, w_j) as asked and also at (j, 1 - w_j) (an
    authority that would let the holder open both labels of wire j) fails CheckAuth, because the unselected
    positions must be H_P of k. Passes on 0173603."""
    kg, BCFE = make(tmp_path, key_mode="h2")
    pp = kg.get_public_parameters()
    setup = scheme._parse_pp(pp).setup
    holder = BCFE.key_holder(pp)
    request = holder.request(W_SMALL)
    grant = kg.authorize(request, requester="alice")
    bits = scheme.unpack_bits(scheme.pack_bits(SMALL.key_layout.encode(W_SMALL), setup.m), setup.m)
    pks = garbled.from_b64(grant["slot"]["pks"], "pks")
    table = [pks[i:i + 32] for i in range(0, len(pks), 32)]
    table[1 - bits[0]] = table[bits[0]]                                   # the holder's key at both positions of wire 0
    bad = copy.deepcopy(grant)
    slot = bad["slot"]
    slot["pks"] = garbled.b64(b"".join(table))
    policy = None if slot["policy_digest"] is None else bytes.fromhex(slot["policy_digest"])
    slot["slot_digest"] = garbled.slot_digest(setup.setup_digest, slot["slot_id"], slot["label"], policy,
                                              b"".join(table)).hex()  # adapted: no attribute-set digest (R06-P2-01)
    with pytest.raises(FEKeyError, match="CheckAuth failed: the unselected positions"):
        holder.finish(bad)
    holder.finish(grant)                                                  # the honest grant still checks


def test_r06_p2_a_key_edited_to_another_w_does_not_compute_c_of_that_w(tmp_path):
    """A key holder who rewrites the ``w`` of its key (one bit flipped) opens the other label ciphertext of that
    wire with the scalar of its own position: the label is unrelated, and the result is not ``C(w', x)`` (it may
    agree by chance on some ciphertexts, not on all). Passes on 0173603."""
    circuit = circuits.linear_score(n=2, bw=3, bx=3)
    kg, BCFE = make(tmp_path, circuit=circuit)
    w = {"weights": [3, -2], "bias": 5}
    key = kg.issue_key(w, requester="alice")
    crypto = BCFE({"keys": {"pp": kg.get_public_parameters()}})
    m = circuit.key_layout.n_bits
    bits = circuit.key_layout.encode(w)
    flipped = list(bits)
    flipped[0] ^= 1
    w_prime = circuit.key_layout.decode(flipped)
    edited = dict(key, w=garbled.b64(scheme.pack_bits(flipped, m)))
    rng = random.Random(20261119)
    trials, agree = 12, 0
    for _ in range(trials):
        x = {"x": [rng.randint(-4, 3), rng.randint(-4, 3)]}
        ct = crypto.encrypt(x)
        assert crypto.decrypt(ct, key) == circuit.evaluate(w, x)
        try:
            agree += crypto.decrypt(ct, edited) == circuit.evaluate(w_prime, x)
        except Exception:  # noqa: BLE001 -- a refusal is also not C(w', x)
            pass
    assert agree < trials
