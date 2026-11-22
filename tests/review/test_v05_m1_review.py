"""Review R1 of milestone M1 of 0.5 (main ad13755 to 5316b73): one failing test per finding.

Every test here fails on 5316b73 for the reason its finding states; the fix turns it green, and the re-check of the
fixes runs this file. The test names carry the finding ids of the review report. No finding is Critical or High:
R1-F01 is Medium, R1-F02 to R1-F04 are Low, R1-F05 to R1-F09 are Info.
"""

import json
import warnings

import pytest

import pyfe4ai
from pyfe4ai.core.exceptions import FEConfigError, FEValidationError
from pyfe4ai.policy import attribute_to_zq, validate_label
from pyfe4ai.primitives.pairing import bls12_381 as bls
from pyfe4ai.primitives.pairing.pairing_backend import G1, PairingGroup, pairing_available

pytestmark = pytest.mark.skipif(not bls.available(), reason="BLS12-381 backend not built")

UNIVERSE = ["l:a", "l:b", "l:c"]
X, Y = [3, -5, 7], [2, 4, 8]
VALUE = sum(a * b for a, b in zip(X, Y))


@pytest.fixture(autouse=True)
def own_parameter_cache(tmp_path, monkeypatch):
    monkeypatch.setenv("PYFE4AI_CACHE_DIR", str(tmp_path))


def _setup(group="BLS12-381"):
    from pyfe4ai.schemes.sife.cp_ab_pairing import SIFECPABPairing, SIFECPABPairingKeyGenerator

    kg = SIFECPABPairingKeyGenerator({"eta": 3, "bound_x": 10, "bound_y": 10, "universe": UNIVERSE,
                                      "pairing_group_param": group})
    kg.setup()
    return kg, SIFECPABPairing


def _key(kg, labels):
    return kg.get_decryption_keys("s", credentials={"fusion_weight": Y, "attributes": labels})


# ---------------------------------------------------------------------------
# Medium


def test_r1_f01_misspelt_security_profile_does_not_silently_give_toy():
    """R1-F01: an unknown option is dropped without a word, so ``security_prfile="standard"`` runs the toy profile:
    the refusal of weak groups (and, on the other schemes, the standard parameters) goes away silently."""
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        try:
            created = pyfe4ai.create("sife/cp_ab_pairing", universe=UNIVERSE, security_prfile="standard")
        except (FEConfigError, FEValidationError):
            return                                      # refused: fixed
    assert any("security_prfile" in str(w.message) for w in caught), (
        "the misspelt option was ignored without a warning: the key generator runs the {!r} profile".format(
            created.key_generator.security_profile))


# ---------------------------------------------------------------------------
# Low


def test_r1_f02_attribute_to_zq_is_uniform_on_every_modulus_it_accepts():
    """R1-F02: the 64 bytes of SHAKE cover ``q < 2^384`` only (its docstring says so), yet any larger ``q`` is
    accepted and answers with a value below ``2^512``: for a 3072-bit modulus not uniform at all."""
    modulus = (1 << 3072) - 1
    try:
        values = [attribute_to_zq("l:{}".format(i), modulus) for i in range(256)]
    except FEValidationError:
        return                                          # refused: fixed
    assert max(value.bit_length() for value in values) > 3000, (
        "a modulus of 3072 bits was accepted, and every output has at most 512 bits")


def test_r1_f03_parameters_changed_after_construction_are_not_mixed_with_the_kept_ones():
    """R1-F03: the object checks the digest of the dict it was given once, keeps a reference to that dict, and since
    5316b73 keeps the decoded ``a``, ``ua`` and ``w_a``, while ``g1`` and ``pp_binding`` are read from the dict on each
    call. A dict changed after construction gives a ciphertext made of both, with no error."""
    kg, crypto_class = _setup()
    pp = kg.get_public_parameters()
    mine = json.loads(json.dumps(pp))
    crypto = crypto_class({"keys": {"pp": mine}})
    crypto.encrypt(X, policy="l:a")
    other, _ = _setup()
    mine["g1"] = other.get_public_parameters()["g1"]
    try:
        ciphertext = crypto.encrypt(X, policy="l:a")
    except FEValidationError:
        return                                          # refused: fixed
    try:
        value = crypto_class({"keys": {"pp": pp}}).decrypt(ciphertext, _key(kg, ["l:a"]))
    except ValueError:
        pytest.fail("encrypt mixed the g1 now in the dict with the kept a and ua: the ciphertext does not decrypt")
    assert value == VALUE


def test_r1_f04_the_documents_a_user_reads_state_the_limits_of_the_proof():
    """R1-F04: selective security, one challenge and no authenticity of the ciphertext are in the module docstring
    (the first two) or nowhere (the last); the status row and the guide that recommend the scheme are silent, unlike
    the rows of the other selective schemes. Nothing claims adaptive security."""
    from pathlib import Path

    from pyfe4ai.schemes.sife import cp_ab_pairing

    root = Path(__file__).resolve().parents[2]
    status = pyfe4ai.get_scheme_status("sife/cp_ab_pairing")
    row = status.reference + " " + status.note
    assert "selective" in row.lower(), "the status row of the scheme does not say selective: {!r}".format(row)
    choosing = (root / "docs" / "choosing.rst").read_text(encoding="utf-8")
    row = choosing[choosing.index("Decrypt only for keys whose attributes satisfy"):]
    assert "selective" in row[:row.index("The registered schemes")].lower(), "docs/choosing.rst does not say selective"
    text = (cp_ab_pairing.__doc__ or "").lower()
    assert "malleable" in text or "not authenticated" in text, "the docstring is silent on the malleable ciphertext"


# ---------------------------------------------------------------------------
# Info


@pytest.mark.parametrize("character", ["͏", "️", "ᅟ", "ㅤ", "ﾠ", "\U000e0100"])
def test_r1_f05_labels_refuse_every_invisible_character(character):
    """R1-F05: the label rule refuses Cf so that two labels cannot look the same, but default-ignorable code points of
    other categories (variation selectors, the grapheme joiner, Hangul fillers) pass and render as nothing."""
    with pytest.raises(FEValidationError):
        validate_label("role:auditor" + character)


def test_r1_f06_the_deviations_listed_are_the_deviations():
    """R1-F06: Fig. 1 of ACGU20 is the k = 1 scheme, so ``k = 1`` is no deviation, whereas ``(1, a2)`` for the paper's
    ``(a, 1)``, the random generators and the nonzero ``s``, ``r``, ``a2``, ``b2`` are not listed."""
    from pyfe4ai.schemes.sife import cp_ab_pairing

    deviations = cp_ab_pairing.__doc__[cp_ab_pairing.__doc__.index("Deviations:"):]
    note = pyfe4ai.get_scheme_status("sife/cp_ab_pairing").note
    assert "k = 1" not in deviations and "k = 1" not in note, "k = 1 is listed as a deviation of a k = 1 figure"
    assert "(a, 1)" in deviations, "the paper's a = (a, 1) against the code's (1, a2) is not listed"


def test_r1_f07_a_malformed_ciphertext_is_refused_whichever_key_opens_it():
    """R1-F07: decrypt reads only the rows its coefficients use, so a ciphertext with junk in a row is malformed for
    one key and decrypts for another; its docstring promises ``FEValidationError`` for a malformed object."""
    kg, crypto_class = _setup()
    crypto = crypto_class({"keys": {"pp": kg.get_public_parameters()}})
    ciphertext = crypto.encrypt(X, policy="or(l:a,l:b)")
    ciphertext["c2"][1][0] = "1:junk"
    with pytest.raises(FEValidationError):
        crypto.decrypt(ciphertext, _key(kg, ["l:b"]))           # this key reads the row
    with pytest.raises(FEValidationError):
        crypto.decrypt(ciphertext, _key(kg, ["l:a"]))           # this one does not


@pytest.mark.skipif(not pairing_available("MNT224"), reason="needs charm")
def test_r1_f08_a_ciphertext_on_a_charm_group_has_one_spelling():
    """R1-F08: on the charm groups bytes after the element are ignored, so a ciphertext element has many spellings and
    its junk is accepted. docs/security.rst names keys and public parameters as the inputs to trust, not ciphertexts."""
    import base64

    kg, crypto_class = _setup("MNT224")
    crypto = crypto_class({"keys": {"pp": kg.get_public_parameters()}})
    ciphertext = crypto.encrypt(X, policy="l:a")
    dk = _key(kg, ["l:a"])
    assert crypto.decrypt(ciphertext, dk) == VALUE
    raw = base64.b64decode(ciphertext["c1"][0][2:])
    ciphertext["c1"][0] = "1:" + base64.b64encode(raw + b"junk").decode()
    with pytest.raises(FEValidationError):
        crypto.decrypt(ciphertext, dk)


def test_r1_f09_dual_bases_refuse_a_zero_scalar():
    """R1-F09: ``mu = 0`` modulo the order gives ``B* = 0``, while the module defines a dual pair as two invertible
    matrices. DOT18 draws a nonzero ``mu`` and M2 will pass 1, so no caller reaches it today."""
    import gmpy2 as gp

    from pyfe4ai.primitives.pairing.dpvs import random_dual_bases

    group = PairingGroup("BLS12-381")
    order = gp.mpz(int(group.order()))
    with pytest.raises(FEValidationError):
        random_dual_bases(group, 2, order, order)


# ---------------------------------------------------------------------------
# Pinned behaviour: these pass on 5316b73 and must keep passing after the fixes


def test_pinned_widening_a_policy_does_not_decrypt():
    """A key outside the true policy gets nothing by rewriting ``policy`` and ``policy_digest``, both public."""
    from pyfe4ai.policy import canonical_encoding, parse, policy_digest

    kg, crypto_class = _setup()
    crypto = crypto_class({"keys": {"pp": kg.get_public_parameters()}})
    ciphertext = crypto.encrypt(X, policy="and(l:a,l:b)")
    formula = parse("or(l:a,l:b)")
    ciphertext["policy"], ciphertext["policy_digest"] = canonical_encoding(formula), policy_digest(formula)
    with pytest.raises(ValueError):
        crypto.decrypt(ciphertext, _key(kg, ["l:a"]))


def test_pinned_keys_of_two_holders_do_not_combine():
    """The ``k_attr`` of another key does not fit: each key has its own ``r``."""
    kg, crypto_class = _setup()
    crypto = crypto_class({"keys": {"pp": kg.get_public_parameters()}})
    ciphertext = crypto.encrypt(X, policy="and(l:a,l:b)")
    first, second = _key(kg, ["l:a"]), _key(kg, ["l:b"])
    first["attributes"] = ["l:a", "l:b"]
    first["k_attr"].update(second["k_attr"])
    with pytest.raises(ValueError):
        crypto.decrypt(ciphertext, first)


def test_pinned_the_group_of_a_batch_is_checked_on_every_path():
    """An element outside G1 is refused alone, in a batch, and as a repeated entry, on any number of threads."""
    from pyfe4ai import backend

    group = PairingGroup("BLS12-381")
    # x = 5 lies on the curve outside the subgroup of order r
    bad = "1:" + __import__("base64").b64encode(_point_outside_g1()).decode()
    good = group.serialize(group.random(G1)).decode()
    with pytest.raises(ValueError):
        group.deserialize(bad.encode())
    for threads in (1, 4):
        backend.set_threads(threads)
        try:
            for items in ([bad], [good] * 20 + [bad], [bad, good] * 20):
                with pytest.raises(ValueError):
                    group.deserialize_many(items)
        finally:
            backend.set_threads(1)


def _point_outside_g1() -> bytes:
    p = 0x1a0111ea397fe69a4b1ba7b6434bacd764774b84f38512bf6730d2a0f6b0f6241eabfffeb153ffffb9feffffffffaaab
    x = 5
    while True:
        right = (x ** 3 + 4) % p
        y = pow(right, (p + 1) // 4, p)
        if y * y % p == right:
            raw = bytearray(x.to_bytes(48, "big"))
            raw[0] |= 0x80 | (0x20 if y > (p - 1) // 2 else 0)
            try:
                PairingGroup("BLS12-381").deserialize(b"1:" + __import__("base64").b64encode(bytes(raw)))
            except ValueError:
                return bytes(raw)
        x += 1
