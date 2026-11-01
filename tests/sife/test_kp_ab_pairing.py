"""The key-policy attribute-based IPFE (NPP22 Sec. 4 Fig. 1): setup, key generation, encryption and decryption.

``seal`` and ``open_sealed`` make and open a ciphertext by hand from the equations of Fig. 1, with the public
parameters and the key only. They check the keys before the scheme encrypts and stay as an independent check of
``encrypt`` and ``decrypt``: ``decrypt`` reads a ciphertext of ``seal`` and ``open_sealed`` one of ``encrypt``.
A key whose policy the attributes of a ciphertext do not satisfy gives a ``FEKeyError``; malformed input, or input over
a limit, a ``FEValidationError``; a key and a ciphertext that are well formed but were not made for each other a
``ValueError`` from the discrete log, which is not a ``FEValidationError``.
"""

import copy
import hashlib
import json
import logging
import os
import random
import threading
import warnings

import gmpy2 as gp
import pytest

from tests.pairing_group import GROUP, skip_unless_available

skip_unless_available()

from tests.sife.conftest import ETA  # noqa: E402
from pyfe4ai import backend as native_backend
from pyfe4ai.core.constants import CryptoCONST
from pyfe4ai.core.exceptions import FEKeyError, FEValidationError, FEWarning
from pyfe4ai.core.formats.pairing import KPABCiphertext, KPABFunctionalKey, KPABParameters, KPABRecord
from pyfe4ai.policy import (
    ATTR_HASH,
    POLICY_COMPILER,
    And,
    Leaf,
    Or,
    Threshold,
    attribute_to_zq,
    canonical_encoding,
    compile_formula,
    evaluate,
    from_canonical,
    parse,
    policy_digest,
    validate_formula,
)
from pyfe4ai.primitives.pairing import pairing_backend
from pyfe4ai.primitives.pairing.dpvs import vector_pair
from pyfe4ai.primitives.pairing.pairing_backend import G1, G2, PairingGroup, deserialize_many, pair, pairing_available
from pyfe4ai.schemes.common import ac_codec, kp_ab
from pyfe4ai.schemes.common.ac_binding import pp_digest
from pyfe4ai.schemes.common.ac_codec import serialize_vec
from pyfe4ai.schemes.common.kp_ab import (
    MAX_LABEL_CHARS,
    attribute_pairs,
    attribute_vector,
    check_policy_length,
    compile_policy,
    held_attributes,
    key_row,
    policy_length_limit,
    read_attributes,
    read_policy,
)
from pyfe4ai.schemes.sife import kp_ab_pairing
from pyfe4ai.schemes.sife.kp_ab_pairing import SIFEKPABPairing, SIFEKPABPairingKeyGenerator

PP_DOMAIN = b"pyfe4ai/sife/kp-ab/pp-binding/v1|"

pytestmark = pytest.mark.usefixtures("own_parameter_cache")


def public(kg):
    """The group, its order, the generators and the public bases ``f`` of *kg* as group elements."""
    group = PairingGroup(GROUP)
    pp = kg.get_public_parameters()
    return {
        "group": group,
        "order": int(group.order()),
        "g1": group.deserialize(pp["g1"].encode("utf-8")),
        "g2": group.deserialize(pp["g2"].encode("utf-8")),
        "f": [deserialize_many(group, row) for row in pp["f"]],
        "h_mask": deserialize_many(group, pp["h_mask"]),
        "h3": deserialize_many(group, pp["h3"]),
        "t": deserialize_many(group, pp["t"]),
    }


def seal(kg, x, attributes):
    """A ciphertext of *x* for *attributes*, made by hand as in Enc of Fig. 1, with its randomness."""
    p = public(kg)
    group, order = p["group"], p["order"]
    omega, psi = random.randrange(1, order), random.randrange(1, order)
    c_attr = {label: attribute_vector(p["f"], label, random.randrange(1, order), psi, order, group)
              for label in sorted(set(attributes))}
    t = [p["g1"] ** (x_i % order) * t_i ** omega for x_i, t_i in zip(x, p["t"])]
    c_ipfe = [m ** omega * h ** psi for m, h in zip(p["h_mask"], p["h3"])]
    return {"attributes": sorted(set(attributes)), "c_attr": c_attr, "t": t, "c_ipfe": c_ipfe}


def open_sealed(kg, ct, dk):
    """``e(g1, g2) ^ <x, y>`` from *ct* and the key *dk*, as in Dec of Fig. 1, or ``None`` if the key's policy is not
    satisfied."""
    p = public(kg)
    group, order = p["group"], p["order"]
    msp = compile_formula(from_canonical(dk["policy"]), order, max_rows=64)
    weights = msp.reconstruction(ct["attributes"])
    if weights is None:
        return None
    k_rows = [deserialize_many(group, row) for row in dk["k_rows"]]
    k_ipfe = deserialize_many(group, dk["k_ipfe"])
    y = [int(value) for value in dk["y"]]
    gt = pair(p["g1"], p["g2"])
    out = gt ** 0
    for row, weight in weights.items():
        out = out * vector_pair(ct["c_attr"][msp.row_labels[row]], k_rows[row]) ** weight
    for t_i, y_i in zip(ct["t"], y):
        out = out * pair(t_i, p["g2"] ** (y_i % order))
    return out / vector_pair(ct["c_ipfe"], k_ipfe)


def creds(policy, y=None):
    return {"fusion_weight": y if y is not None else [1] * ETA, "policy": policy}


def sealed_record(kg, sealed):
    """The ciphertext record of what ``seal`` made, in the form that ``decrypt`` reads."""
    group = PairingGroup(GROUP)
    return KPABCiphertext.new(
        attributes=sealed["attributes"],
        attr_hash=ATTR_HASH,
        policy_compiler=POLICY_COMPILER,
        c_attr=[serialize_vec(group, sealed["c_attr"][label]) for label in sealed["attributes"]],
        t=serialize_vec(group, sealed["t"]),
        c_ipfe=serialize_vec(group, sealed["c_ipfe"]),
        pp_binding=kg.get_public_parameters()["pp_binding"],
    ).to_dict()


def unsealed(ct):
    """The ciphertext record *ct* of ``encrypt`` in the form that ``open_sealed`` reads."""
    group = PairingGroup(GROUP)
    return {
        "attributes": ct["attributes"],
        "c_attr": {label: deserialize_many(group, row) for label, row in zip(ct["attributes"], ct["c_attr"])},
        "t": deserialize_many(group, ct["t"]),
        "c_ipfe": deserialize_many(group, ct["c_ipfe"]),
    }


def inner(x, y):
    return sum(a * b for a, b in zip(x, y))


def in_gt(kg, value):
    """``e(g1, g2) ^ value``."""
    p = public(kg)
    return pair(p["g1"], p["g2"]) ** (value % p["order"])


# ---------------------------------------------------------------------------
# setup: the bases and the public parameters


def test_bases_are_dual(make_kp):
    kg, _ = make_kp()
    p = public(kg)
    f, f_star = p["f"], kg.msk["f_star"]
    gt = pair(p["g1"], p["g2"])
    for i in range(3):
        for j in range(3):
            assert vector_pair(f[i], f_star[j]) == (gt if i == j else gt ** 0), (i, j)


def test_the_mask_bases_are_dual(make_kp):
    # h1 + mu h2 = (1, mu, 0, 0)_H against h*_1, h*_2, h*_3: gt, gt^mu, 1; h3 against h*_3: gt
    kg, _ = make_kp()
    p = public(kg)
    h_star = kg.msk["h_star"]
    gt = pair(p["g1"], p["g2"])
    assert vector_pair(p["h_mask"], h_star[0]) == gt
    assert vector_pair(p["h_mask"], h_star[2]) == gt ** 0
    assert vector_pair(p["h3"], h_star[0]) == gt ** 0
    assert vector_pair(p["h3"], h_star[1]) == gt ** 0
    assert vector_pair(p["h3"], h_star[2]) == gt


def test_the_public_key_agrees_with_the_master_key(make_kp):
    # t_i = [s_i + mu u_i]_1 and e(h1 + mu h2, h*_2) = gt^mu, so e(t_i, g2) = gt^(s_i) (gt^mu)^(u_i)
    kg, _ = make_kp()
    p = public(kg)
    gt = pair(p["g1"], p["g2"])
    gt_mu = vector_pair(p["h_mask"], kg.msk["h_star"][1])
    for t_i, s_i, u_i in zip(p["t"], kg.msk["s"], kg.msk["u"]):
        assert pair(t_i, p["g2"]) == gt ** int(s_i) * gt_mu ** int(u_i)


def test_master_key_holds_group_elements(make_kp):      # D-2
    kg, _ = make_kp()
    assert set(kg.msk) == {"z", "s", "u", "f_star", "h_star"}
    assert len(kg.msk["f_star"]) == 3 and len(kg.msk["h_star"]) == 3
    assert all(len(row) == 8 for row in kg.msk["f_star"]) and all(len(row) == 4 for row in kg.msk["h_star"])
    assert all(element.type == G2 for row in kg.msk["f_star"] + kg.msk["h_star"] for element in row)
    assert len(kg.msk["s"]) == len(kg.msk["u"]) == ETA
    order = int(PairingGroup(GROUP).order())
    for value in [kg.msk["z"], *kg.msk["s"], *kg.msk["u"]]:
        assert 0 < int(value) < order                   # drawn from Z_q^*


def reachable_scalars(root):
    """Every int, ``mpz`` and decimal string reachable from the attributes of *root* through dicts and sequences."""
    found, seen, todo = set(), set(), [vars(root)]
    while todo:
        item = todo.pop()
        if id(item) in seen:
            continue
        seen.add(id(item))
        if isinstance(item, dict):
            todo.extend(item.keys())
            todo.extend(item.values())
        elif isinstance(item, (list, tuple, set, frozenset)):
            todo.extend(item)
        elif isinstance(item, (int, type(gp.mpz(0)))):
            found.add(int(item))
        elif isinstance(item, str) and item.lstrip("-").isdigit():
            found.add(int(item))
    return found


def test_setup_keeps_no_basis_and_no_mu(make_kp, monkeypatch):
    drawn, matrices = [], []
    random_nonzero_zr, random_dual_bases = kp_ab_pairing.random_nonzero_zr, kp_ab_pairing.random_dual_bases

    def spy_zr(group):
        value = random_nonzero_zr(group)
        drawn.append(int(value))
        return value

    def spy_bases(*args, **kwargs):
        pair_of_matrices = random_dual_bases(*args, **kwargs)
        matrices.extend(pair_of_matrices)
        return pair_of_matrices

    monkeypatch.setattr(kp_ab_pairing, "random_nonzero_zr", spy_zr)
    monkeypatch.setattr(kp_ab_pairing, "random_dual_bases", spy_bases)
    kg, _ = make_kp()
    assert [len(m) for m in matrices] == [4, 4, 8, 8]       # H, H*, F, F*
    secrets = {int(kg.msk["z"]), *(int(v) for v in kg.msk["s"]), *(int(v) for v in kg.msk["u"])}
    mu = set(drawn) - secrets
    assert len(mu) == 1 and len(drawn) == 2 + 2 * ETA       # mu, z, S, U
    kept = reachable_scalars(kg)
    assert secrets <= kept
    assert not mu & kept                                    # mu is gone
    entries = {int(entry) for matrix in matrices for row in matrix for entry in row}
    assert not entries & kept                               # so are the exponents of F, F*, H, H*


def test_public_parameters_hold_no_secret(make_kp):
    kg, _ = make_kp()
    pp = kg.get_public_parameters()
    assert set(pp) == set(KPABParameters.FIELDS)
    assert list(pp) == list(KPABParameters.FIELDS)
    text = json.dumps(pp)
    for value in [kg.msk["z"], *kg.msk["s"], *kg.msk["u"]]:
        assert str(value) not in text


def test_public_parameters_layout(make_kp):
    kg, _ = make_kp()
    pp = kg.get_public_parameters()
    group = PairingGroup(GROUP)
    assert (pp["eta"], pp["bound_x"], pp["bound_y"]) == (ETA, "10", "10")
    assert (pp["max_rows"], pp["max_attributes"]) == (64, 64)
    assert pp["pairing_group_param"] == GROUP
    assert pp["policy_compiler"] == POLICY_COMPILER and pp["attr_hash"] == ATTR_HASH
    assert len(pp["h_mask"]) == len(pp["h3"]) == 4
    assert len(pp["f"]) == 3 and all(len(row) == 8 for row in pp["f"])
    assert len(pp["t"]) == ETA
    assert pp["pp_binding"] == pp_digest(pp, PP_DOMAIN)
    for field, kind in (("g1", G1), ("g2", G2)):
        assert group.deserialize(pp[field].encode("utf-8")).type == kind
    for field, kind in (("h_mask", G1), ("h3", G1), ("t", G1)):
        assert all(group.deserialize(item.encode("utf-8")).type == kind for item in pp[field])
    assert all(group.deserialize(item.encode("utf-8")).type == G1 for row in pp["f"] for item in row)


def test_setups_differ(make_kp):
    first, _ = make_kp()
    second, _ = make_kp()
    assert first.get_public_parameters()["pp_binding"] != second.get_public_parameters()["pp_binding"]
    assert first.msk["z"] != second.msk["z"]


def test_public_key_scheme(make_kp):
    kg, _ = make_kp()
    assert kg.get_private_keys("a") is None


# ---------------------------------------------------------------------------
# the configuration and its record


def test_limits_have_defaults():
    kg = SIFEKPABPairingKeyGenerator({"pairing_group_param": GROUP})
    assert (kg.max_rows, kg.max_attributes) == (64, 64)
    assert kg.eta == CryptoCONST.SIFE_DEFAULT_ETA


@pytest.mark.parametrize("option", ["eta", "bound_x", "bound_y", "max_rows", "max_attributes"])
@pytest.mark.parametrize("value", [0, -1, True, 2.0, "5", None])
def test_limits_must_be_positive_ints(option, value):
    config = {"eta": ETA, "pairing_group_param": GROUP, option: value}
    with pytest.raises(FEValidationError, match=option):
        SIFEKPABPairingKeyGenerator(config)


def test_dlog_limit():
    config = {"eta": 2, "bound_x": 2 ** 24, "bound_y": 2 ** 24, "pairing_group_param": GROUP}
    with pytest.raises(FEValidationError, match=r"2\*\*48"):
        SIFEKPABPairingKeyGenerator(config)
    assert SIFEKPABPairingKeyGenerator(dict(config, eta=1)).eta == 1        # 2**48 itself is allowed


def test_a_symmetric_group_is_refused():
    if not pairing_available("SS512"):
        pytest.skip("no charm: SS512 is not built")
    with pytest.raises(FEValidationError, match="symmetric"):
        SIFEKPABPairingKeyGenerator({"eta": ETA, "pairing_group_param": "SS512"})


def record_file(cache):
    return os.path.join(cache, "authority", CryptoCONST.TYPE_SIFE_KP_AB, "param.json")


def test_the_record_holds_the_configuration_only(make_kp, tmp_path):
    cache = str(tmp_path)
    make_kp(cache_dir=cache)
    with open(record_file(cache), encoding="utf-8") as f:
        record = json.load(f)
    assert list(record) == list(KPABRecord.FIELDS)
    assert record["bound_x"] == "10" and record["max_attributes"] == 64
    assert record["policy_compiler"] == POLICY_COMPILER and record["attr_hash"] == ATTR_HASH


def test_the_scheme_says_its_record_is_the_configuration():
    assert SIFEKPABPairingKeyGenerator._config_only_record is True


@pytest.mark.parametrize("override", [{"max_attributes": 8}, {"max_rows": 8}, {"eta": 3}, {"bound_x": 11}])
def test_a_new_configuration_replaces_the_record_without_a_warning(make_kp, tmp_path, recwarn, caplog, override):
    cache = str(tmp_path)
    make_kp(cache_dir=cache)
    with caplog.at_level(logging.INFO, logger="pyfe4ai.core.base"):
        make_kp(cache_dir=cache, **override)
    assert not [w for w in recwarn if issubclass(w.category, FEWarning)], [str(w.message) for w in recwarn]
    assert [r.levelno for r in caplog.records if "does not match" in r.getMessage()] == [logging.INFO]
    with open(record_file(cache), encoding="utf-8") as f:
        record = json.load(f)
    assert all(record[name] == str(value) if name.startswith("bound") else record[name] == value
               for name, value in override.items())


def test_same_configuration_keeps_the_cached_record(make_kp, tmp_path):
    cache = str(tmp_path)
    make_kp(cache_dir=cache)
    with warnings.catch_warnings():
        warnings.simplefilter("error", FEWarning)
        make_kp(cache_dir=cache)


# ---------------------------------------------------------------------------
# key generation


def test_key_shape(make_kp):
    kg, _ = make_kp()
    dk = kg.get_decryption_keys("s", credentials=creds("and(org:A, org:B)"))
    assert list(dk) == list(KPABFunctionalKey.FIELDS)
    assert len(dk["k_rows"]) == 2 and all(len(row) == 8 for row in dk["k_rows"])
    assert len(dk["k_ipfe"]) == 4 and dk["policy"] == canonical_encoding(parse("and(org:A, org:B)"))
    assert dk["policy_digest"] == policy_digest(parse("and(org:A, org:B)"))
    assert dk["policy_compiler"] == POLICY_COMPILER
    assert dk["y"] == ["1"] * ETA
    assert dk["pp_binding"] == kg.get_public_parameters()["pp_binding"]
    assert json.loads(json.dumps(dk)) == dk                         # JSON data, as the other keys


def test_key_elements_are_of_g2(make_kp):
    kg, _ = make_kp()
    group = PairingGroup(GROUP)
    dk = kg.get_decryption_keys("s", credentials=creds("or(org:A, org:B)"))
    assert all(group.deserialize(item.encode("utf-8")).type == G2 for row in dk["k_rows"] for item in row)
    assert all(group.deserialize(item.encode("utf-8")).type == G2 for item in dk["k_ipfe"])


def test_one_row_for_each_leaf_of_the_policy(make_kp):
    kg, _ = make_kp()
    for policy, rows in (("org:A", 1), ("or(org:A, org:B, org:C)", 3), ("or(and(org:A, org:B), and(org:A, org:C))", 4)):
        assert len(kg.get_decryption_keys("s", credentials=creds(policy))["k_rows"]) == rows


def test_formula_objects_are_accepted_as_policies(make_kp):
    kg, _ = make_kp()
    formula = And(Leaf("org:A"), Or(Leaf("org:B"), Threshold(2, Leaf("org:C"), Leaf("role:auditor"), Leaf("org:B"))))
    dk = kg.get_decryption_keys("s", credentials=creds(formula))
    assert dk["policy"] == canonical_encoding(formula) and dk["policy_digest"] == policy_digest(formula)


def test_the_policy_text_is_stored_in_its_canonical_form(make_kp):
    kg, _ = make_kp()
    spaced = kg.get_decryption_keys("s", credentials=creds("and( org:A ,org:B )"))
    assert spaced["policy"] == canonical_encoding(parse("and(org:A, org:B)"))


@pytest.mark.parametrize("policy,held", [
    ("org:A", ["org:A"]),
    ("and(org:A, role:auditor)", ["org:A", "role:auditor"]),
    ("or(org:A, role:auditor)", ["role:auditor"]),
    ("2of(org:A, org:B, role:auditor)", ["org:B", "role:auditor"]),                 # weights other than 1
    ("and(org:A, 2of(org:B, org:C, role:auditor))", ["org:A", "org:C", "role:auditor"]),
    ("or(and(org:A, org:B), and(org:A, org:C))", ["org:A", "org:C"]),               # org:A on two rows
    ("org:A", ["org:A", "org:Z"]),                                                  # an attribute no row names
])
def test_a_key_opens_a_ciphertext_of_a_satisfying_set(policy, held, make_kp):
    kg, _ = make_kp()
    x, y = [3, -1, 0, 7, 2], [1, 2, 3, -4, 5]
    dk = kg.get_decryption_keys("s", credentials=creds(policy, y))
    expected = pair(public(kg)["g1"], public(kg)["g2"]) ** (sum(a * b for a, b in zip(x, y)) % public(kg)["order"])
    assert open_sealed(kg, seal(kg, x, held), dk) == expected


@pytest.mark.parametrize("policy,held", [
    ("and(org:A, role:auditor)", ["org:A"]),
    ("2of(org:A, org:B, org:C)", ["org:C"]),
    ("org:A", []),
])
def test_a_key_does_not_open_an_unsatisfying_set(policy, held, make_kp):
    kg, _ = make_kp()
    dk = kg.get_decryption_keys("s", credentials=creds(policy))
    assert open_sealed(kg, seal(kg, [1, 2, 3, 4, 5], held), dk) is None


def test_a_row_paired_with_the_vector_of_another_label_gives_a_wrong_value(make_kp):
    # (1, -j_B) . (j_A, 1) does not cancel, so the vector of org:B in the place of that of org:A does not open
    kg, _ = make_kp()
    p = public(kg)
    x, y = [3, -1, 0, 7, 2], [1, 2, 3, -4, 5]
    dk = kg.get_decryption_keys("s", credentials=creds("org:A", y))
    forged = seal(kg, x, ["org:B"])
    forged["c_attr"]["org:A"] = forged["c_attr"]["org:B"]
    expected = pair(p["g1"], p["g2"]) ** (sum(a * b for a, b in zip(x, y)) % p["order"])
    assert open_sealed(kg, dict(forged, attributes=["org:A"]), dk) != expected


def test_negative_weights_are_accepted(make_kp):
    kg, _ = make_kp()
    p = public(kg)
    x, y = [3, -1, 0, 7, 2], [-10, 10, -1, 0, -7]
    dk = kg.get_decryption_keys("s", credentials=creds("org:A", y))
    assert dk["y"] == ["-10", "10", "-1", "0", "-7"]
    expected = pair(p["g1"], p["g2"]) ** (sum(a * b for a, b in zip(x, y)) % p["order"])
    assert open_sealed(kg, seal(kg, x, ["org:A"]), dk) == expected


def test_keys_of_one_master_key_all_open_one_ciphertext(make_kp):
    # z is fixed in msk and a0, pi are fresh for each key
    kg, _ = make_kp()
    p = public(kg)
    x = [3, -1, 0, 7, 2]
    ct = seal(kg, x, ["org:A", "org:B"])
    for y in ([1, 1, 1, 1, 1], [0, 0, 0, 0, 1], [10, -10, 10, -10, 10]):
        dk = kg.get_decryption_keys("s", credentials=creds("and(org:A, org:B)", y))
        expected = pair(p["g1"], p["g2"]) ** (sum(a * b for a, b in zip(x, y)) % p["order"])
        assert open_sealed(kg, ct, dk) == expected


def test_a_key_of_another_setup_does_not_open(make_kp):
    kg, _ = make_kp()
    other, _ = make_kp()
    p = public(kg)
    x, y = [3, -1, 0, 7, 2], [1, 2, 3, -4, 5]
    dk = other.get_decryption_keys("s", credentials=creds("org:A", y))
    expected = pair(p["g1"], p["g2"]) ** (sum(a * b for a, b in zip(x, y)) % p["order"])
    assert open_sealed(kg, seal(kg, x, ["org:A"]), dk) != expected


def test_attribute_vector_and_key_row_pair_as_the_paper_has_it(make_kp):
    # (sigma (1, -j), psi) . (pi (j', 1), a z) = sigma pi (j' - j) + psi a z, zero for the first term iff j' = j
    kg, _ = make_kp()
    p = public(kg)
    group, order = p["group"], p["order"]
    gt = pair(p["g1"], p["g2"])
    sigma, psi, pi, share_z = 12345, 678, 91011, 1213141516
    j = {label: attribute_to_zq(label, order) for label in ("org:A", "org:B")}
    c = attribute_vector(p["f"], "org:A", sigma, psi, order, group)
    assert len(c) == 8
    for label in ("org:A", "org:B"):
        k = key_row(kg.msk["f_star"], label, pi, share_z, order, group)
        assert len(k) == 8
        exponent = (sigma * pi * (j[label] - j["org:A"]) + psi * share_z) % order
        assert vector_pair(c, k) == gt ** exponent, label


def test_two_keys_use_fresh_randomness(make_kp):
    kg, _ = make_kp()
    request = creds("and(org:A, org:B)", [1, -2, 3, 0, 5])
    first = kg.get_decryption_keys("s", credentials=request)
    second = kg.get_decryption_keys("s", credentials=request)
    assert first["k_rows"] != second["k_rows"] and first["k_ipfe"] != second["k_ipfe"]
    assert all(a != b for a, b in zip(first["k_rows"], second["k_rows"]))      # pi is fresh for every row
    assert (first["policy"], first["y"]) == (second["policy"], second["y"])


def test_a_label_on_two_rows_takes_a_fresh_pi_for_each(make_kp):
    kg, _ = make_kp()
    dk = kg.get_decryption_keys("s", credentials=creds("or(org:A, org:A)"))
    assert dk["k_rows"][0] != dk["k_rows"][1]


def test_key_before_setup():
    kg = SIFEKPABPairingKeyGenerator({"eta": ETA, "pairing_group_param": GROUP})
    with pytest.raises(FEKeyError, match="setup"):
        kg.get_decryption_keys("s", credentials=creds("org:A"))


@pytest.mark.parametrize("credentials", [
    None, {}, {"policy": "org:A"}, {"fusion_weight": [1] * 5},
    {"fusion_weight": [1] * 4, "policy": "org:A"},
    {"fusion_weight": [99] * 5, "policy": "org:A"},
    {"fusion_weight": [1] * 5, "policy": None},
    {"fusion_weight": [1] * 5, "policy": "org:A and org:B"},
    {"fusion_weight": [1] * 5, "policy": "or(org:A,"},
    {"fusion_weight": [1] * 5, "policy": b"org:A"},
    {"fusion_weight": [1] * 5, "policy": ["org:A"]},
    {"fusion_weight": [1.5] * 5, "policy": "org:A"},
    {"fusion_weight": "11111", "policy": "org:A"},
    {"fusion_weight": [1] * 5, "attributes": ["org:A"]},            # the entry of the ciphertext-policy scheme
    ["fusion_weight", "policy"],
])
def test_key_requests_are_checked(make_kp, credentials):
    kg, _ = make_kp()
    with pytest.raises((FEKeyError, FEValidationError)):
        kg.get_decryption_keys("s", credentials=credentials)


def test_missing_credentials_are_a_key_error_and_bad_ones_a_validation_error(make_kp):
    kg, _ = make_kp()
    for credentials in (None, {}):
        with pytest.raises(FEKeyError, match="credentials"):
            kg.get_decryption_keys("s", credentials=credentials)
    for credentials, match in (({"policy": "org:A"}, "fusion_weight"), ({"fusion_weight": [1] * 5}, "policy"),
                               ({"fusion_weight": [11] * 5, "policy": "org:A"}, "fusion weight"),
                               ({"fusion_weight": [1] * 5, "policy": 7}, "policy")):
        with pytest.raises(FEValidationError, match=match):
            kg.get_decryption_keys("s", credentials=credentials)


def test_a_policy_of_more_rows_than_the_limit_is_refused(make_kp):
    kg, _ = make_kp(max_rows=3)
    kg.get_decryption_keys("s", credentials=creds("or(org:A, org:B, org:C)"))
    with pytest.raises(FEValidationError, match="max_rows|rows"):
        kg.get_decryption_keys("s", credentials=creds("or(org:A, org:B, org:C, org:D)"))


def test_a_policy_over_the_default_limit_is_refused(make_kp):
    kg, _ = make_kp()
    leaves = ["org:L{}".format(i) for i in range(65)]
    with pytest.raises(FEValidationError, match="64"):
        kg.get_decryption_keys("s", credentials=creds("or({})".format(", ".join(leaves))))


def test_a_refused_request_does_not_change_the_key_generator(make_kp):
    kg, _ = make_kp()
    before = (dict(kg.msk), json.dumps(kg.get_public_parameters()))
    with pytest.raises(FEValidationError):
        kg.get_decryption_keys("s", credentials=creds("or(org:A,"))
    assert (dict(kg.msk), json.dumps(kg.get_public_parameters())) == before


# ---------------------------------------------------------------------------
# the shared helpers


def test_read_policy():
    formula = Leaf("org:A")
    assert read_policy(formula) is formula
    assert read_policy("and(org:A, org:B)") == parse("and(org:A, org:B)")
    for bad in (None, 7, b"org:A", ["org:A"], {"org:A"}):
        with pytest.raises(FEValidationError, match="policy"):
            read_policy(bad)
    with pytest.raises(FEValidationError):
        read_policy("not a policy")


def test_compile_policy_works_modulo_the_order():
    order = gp.mpz(int(PairingGroup(GROUP).order()))
    msp = compile_policy(parse("2of(org:A, org:B, org:C)"), order, 64)
    assert msp.modulus == int(order) and msp.num_rows == 3
    with pytest.raises(FEValidationError):
        compile_policy(parse("or(org:A, org:B, org:C)"), order, 2)


def test_held_attributes_sorts_and_deduplicates():
    assert held_attributes({"b:1", "a:1"}, 64) == ["a:1", "b:1"]
    assert held_attributes(["b:1", "a:1", "b:1"], 64) == ["a:1", "b:1"]
    assert held_attributes(("a:1",), 64) == ["a:1"]
    assert held_attributes(iter(["b:1", "a:1"]), 64) == ["a:1", "b:1"]
    assert held_attributes([], 64) == []
    assert held_attributes(["a:1"] * 100, 2) == ["a:1"]                 # repeats are not counted


@pytest.mark.parametrize("bad", ["a:1", b"a:1", bytearray(b"a:1"), 5, None, ["a"], ["a:1", 7], [b"a:1"], [["a:1"]],
                                 ["a:1 b"], ["a:()"], ["a:\u200b1"]])
def test_held_attributes_refuses_what_is_not_a_collection_of_labels(bad):
    with pytest.raises(FEValidationError):
        held_attributes(bad, 64)


def test_held_attributes_limit():
    assert held_attributes(["a:1", "b:1"], 2) == ["a:1", "b:1"]
    with pytest.raises(FEValidationError, match="max_attributes = 2"):
        held_attributes(["a:1", "b:1", "c:1"], 2)
    assert len(held_attributes(["a:{}".format(i) for i in range(64)], 64)) == 64
    with pytest.raises(FEValidationError, match="64"):
        held_attributes(["a:{}".format(i) for i in range(65)], 64)


def test_read_attributes_takes_a_sorted_list_of_distinct_labels():
    assert read_attributes(["a:1", "b:1"], 64) == ["a:1", "b:1"]
    assert read_attributes([], 64) == []
    assert read_attributes(["a:1"], 1) == ["a:1"]


@pytest.mark.parametrize("bad", [
    ("a:1", "b:1"), {"a:1"}, "a:1", None, 5,                        # not a list
    ["b:1", "a:1"], ["a:1", "a:1"],                                  # not strictly increasing
    ["a:1", 7], [b"a:1"], ["a"], ["a:1", "b:1 c"],                   # not labels
])
def test_read_attributes_refuses_what_a_ciphertext_does_not_hold(bad):
    with pytest.raises(FEValidationError):
        read_attributes(bad, 64)


def test_read_attributes_limit_is_checked():
    with pytest.raises(FEValidationError, match="max_attributes = 2"):
        read_attributes(["a:1", "b:1", "c:1"], 2)


def test_read_attributes_names_the_labels_that_are_out_of_order():
    with pytest.raises(FEValidationError, match=r"'b:1' is followed by 'a:1'"):
        read_attributes(["a:0", "b:1", "a:1"], 64)
    with pytest.raises(FEValidationError, match=r"'a:1' is followed by 'a:1'"):
        read_attributes(["a:0", "a:1", "a:1"], 64)
    label = "a:" + "x" * 5000
    with pytest.raises(FEValidationError) as caught:
        read_attributes([label + "y", label], 64)
    assert len(str(caught.value)) < 300


def label_of(length):
    return "a:" + "x" * (length - 2)


def test_labels_have_a_length_limit():
    assert MAX_LABEL_CHARS == 1024
    assert held_attributes([label_of(1024)], 64) == [label_of(1024)]
    assert read_attributes([label_of(1024)], 64) == [label_of(1024)]
    for function in (held_attributes, read_attributes):
        with pytest.raises(FEValidationError, match="MAX_LABEL_CHARS = 1024") as caught:
            function(["a:1", label_of(1025)], 64)
        assert "1025 characters" in str(caught.value) and len(str(caught.value)) < 300


def test_the_length_of_a_label_is_checked_before_the_label_is_read():
    # a label of a million characters is refused by its length, whatever else is wrong with it
    for function in (held_attributes, read_attributes):
        with pytest.raises(FEValidationError, match="MAX_LABEL_CHARS"):
            function(["a: " + "x" * 10 ** 6], 64)


def test_policy_length_limit():
    assert policy_length_limit(64) == 64 * (1024 + 400)
    assert policy_length_limit(3) == 3 * 1424
    check_policy_length("x" * 1424, 1424)
    with pytest.raises(FEValidationError, match="1425 characters, more than the limit of 1424"):
        check_policy_length("x" * 1425, 1424)


def test_the_constants_of_the_schemes():
    assert (kp_ab.F_DIM, kp_ab.H_DIM, kp_ab.DEFAULT_MAX_ATTRIBUTES) == (8, 4, 64)


def test_ciphertext_record_fields():
    assert KPABCiphertext.FIELDS == (
        "attributes", "attr_hash", "policy_compiler", "c_attr", "t", "c_ipfe", "pp_binding")


def test_attribute_pairs_puts_the_coefficient_on_the_g1_side(make_kp):
    kg, _ = make_kp()
    p = public(kg)
    group, order = p["group"], p["order"]
    c_attr = [p["f"][0], p["f"][1]]                         # the vectors of "a:1" and "b:1"
    k_rows = {0: kg.msk["f_star"][0], 3: kg.msk["f_star"][1]}
    rows = ["a:1", "c:1", "c:1", "b:1"]
    lhs, rhs = attribute_pairs({0: 1, 3: 5}, rows, ["a:1", "b:1"], c_attr, k_rows, group, order)
    assert len(lhs) == len(rhs) == 16
    assert lhs[:8] == c_attr[0] and rhs[:8] == k_rows[0]                    # the coefficient 1: no exponentiation
    assert lhs[8:] == [element ** 5 for element in c_attr[1]] and rhs[8:] == k_rows[3]
    assert vector_pair(lhs, rhs) == vector_pair(c_attr[0], k_rows[0]) * vector_pair(c_attr[1], k_rows[3]) ** 5


def test_attribute_pairs_reduces_the_coefficient_modulo_the_order(make_kp):
    kg, _ = make_kp()
    p = public(kg)
    order = p["order"]
    lhs, _ = attribute_pairs({0: order + 1, 1: order - 1}, ["a:1", "a:1"], ["a:1"], [p["f"][0]],
                             {0: kg.msk["f_star"][0], 1: kg.msk["f_star"][1]}, p["group"], order)
    assert lhs[:8] == p["f"][0]
    assert lhs[8:] == [element ** (order - 1) for element in p["f"][0]]


def test_attribute_pairs_takes_the_vector_of_the_label_of_the_row(make_kp):
    kg, _ = make_kp()
    p = public(kg)
    lhs, _ = attribute_pairs({1: 1}, ["a:1", "b:1"], ["a:1", "b:1", "c:1"], [p["f"][0], p["f"][1], p["f"][2]],
                             {1: kg.msk["f_star"][0]}, p["group"], p["order"])
    assert lhs == p["f"][1]


def test_attribute_pairs_refuses_a_label_the_ciphertext_does_not_carry(make_kp):
    kg, _ = make_kp()
    p = public(kg)
    with pytest.raises(FEValidationError, match="b:1"):
        attribute_pairs({1: 1}, ["a:1", "b:1"], ["a:1"], [p["f"][0]], {1: kg.msk["f_star"][0]}, p["group"], p["order"])


@pytest.mark.parametrize("c_length,k_length", [(7, 8), (8, 7), (9, 8)])
def test_attribute_pairs_refuses_vectors_of_another_length(c_length, k_length, make_kp):
    kg, _ = make_kp()
    p = public(kg)
    c = (p["f"][0] * 2)[:c_length]
    k = (kg.msk["f_star"][0] * 2)[:k_length]
    with pytest.raises(FEValidationError, match="8 elements"):
        attribute_pairs({0: 1}, ["a:1"], ["a:1"], [c], {0: k}, p["group"], p["order"])


# ---------------------------------------------------------------------------
# encryption and decryption


X = [3, -1, 0, 7, 2]
Y = [1, 2, 3, -4, 5]
SATISFIED = [
    ("org:A", ["org:A"]),
    ("and(org:A, role:auditor)", ["org:A", "role:auditor"]),
    ("or(org:A, role:auditor)", ["role:auditor"]),
    ("2of(org:A, org:B, role:auditor)", ["org:B", "role:auditor"]),                 # weights other than 1
    ("and(org:A, 2of(org:B, org:C, role:auditor))", ["org:A", "org:C", "role:auditor"]),
    ("or(and(org:A, org:B), and(org:A, org:C))", ["org:A", "org:C"]),               # org:A on two rows
    ("org:A", ["org:A", "org:Z"]),                                                  # an attribute no row names
    ("2of(org:A, or(org:A, org:B))", ["org:A"]),                                    # one label meets both children
]
POLICY = "and(org:A, role:auditor)"
HELD = ["org:A", "org:B", "role:auditor"]


@pytest.fixture
def kp_fixed(make_kp):
    """A key generator, a crypto object, a ciphertext of ``X`` for ``HELD`` (``org:B`` is no row of the policy), a key
    for ``POLICY`` and ``Y`` that opens it, and their inner product."""
    kg, crypto = make_kp()
    ct = crypto.encrypt(X, attributes=HELD)
    dk = kg.get_decryption_keys("s", credentials=creds(POLICY, Y))
    return kg, crypto, ct, dk, inner(X, Y)


@pytest.mark.parametrize("policy,attributes", SATISFIED)
def test_decrypts(policy, attributes, make_kp):
    kg, crypto = make_kp()
    ct = crypto.encrypt(X, attributes=attributes)
    dk = kg.get_decryption_keys("s", credentials=creds(policy, Y))
    assert crypto.decrypt(ct, dk) == inner(X, Y)


@pytest.mark.parametrize("policy,attributes", SATISFIED)
def test_open_sealed_reads_a_ciphertext_of_encrypt(policy, attributes, make_kp):
    # the hand-made Dec of Fig. 1 against the Enc of the scheme
    kg, crypto = make_kp()
    ct = crypto.encrypt(X, attributes=attributes)
    dk = kg.get_decryption_keys("s", credentials=creds(policy, Y))
    assert open_sealed(kg, unsealed(ct), dk) == in_gt(kg, inner(X, Y))


@pytest.mark.parametrize("policy,attributes", SATISFIED)
def test_decrypt_reads_a_ciphertext_of_seal(policy, attributes, make_kp):
    # the Dec of the scheme against the hand-made Enc of Fig. 1
    kg, crypto = make_kp()
    ct = sealed_record(kg, seal(kg, X, attributes))
    dk = kg.get_decryption_keys("s", credentials=creds(policy, Y))
    assert crypto.decrypt(ct, dk) == inner(X, Y)


def test_the_two_independent_pairs_agree_on_what_is_not_opened(make_kp):
    kg, crypto = make_kp()
    dk = kg.get_decryption_keys("s", credentials=creds(POLICY, Y))
    assert open_sealed(kg, unsealed(crypto.encrypt(X, attributes=["org:A"])), dk) is None
    with pytest.raises(FEKeyError):
        crypto.decrypt(sealed_record(kg, seal(kg, X, ["org:A"])), dk)


@pytest.mark.parametrize("policy,attributes", [("and(org:A, role:auditor)", ["org:A"]),
                                               ("2of(org:A, org:B, org:C)", ["org:C"]),
                                               ("org:A", [])])
def test_unsatisfied_is_refused_before_any_pairing(policy, attributes, make_kp, monkeypatch):
    kg, crypto = make_kp()
    ct = crypto.encrypt(X, attributes=attributes)
    dk = kg.get_decryption_keys("s", credentials=creds(policy, Y))

    def never(*args, **kwargs):
        pytest.fail("a pairing or a discrete log was computed for attributes that do not satisfy the policy")

    monkeypatch.setattr(kp_ab_pairing, "multi_pair", never)
    monkeypatch.setattr(kp_ab_pairing, "bounded_discrete_log_gt", never)
    with pytest.raises(FEKeyError, match="do not satisfy"):
        crypto.decrypt(ct, dk)


def test_formula_objects_are_accepted(make_kp):
    kg, crypto = make_kp()
    formula = And(Leaf("org:A"), Or(Leaf("org:B"), Threshold(2, Leaf("org:C"), Leaf("role:auditor"), Leaf("org:B"))))
    dk = kg.get_decryption_keys("s", credentials=creds(formula, [1] * ETA))
    assert crypto.decrypt(crypto.encrypt([1, 2, 3, 4, 5], attributes=["org:A", "org:B"]), dk) == 15


@pytest.mark.parametrize("x,y", [([0] * ETA, Y), (X, [0] * ETA), ([0] * ETA, [0] * ETA)])
def test_zero_vectors(x, y, make_kp):
    kg, crypto = make_kp()
    dk = kg.get_decryption_keys("s", credentials=creds("org:A", y))
    assert crypto.decrypt(crypto.encrypt(x, attributes=["org:A"]), dk) == 0


@pytest.mark.parametrize("sign", [1, -1])
def test_extreme_inner_products(sign, make_kp):
    kg, crypto = make_kp()
    dk = kg.get_decryption_keys("s", credentials=creds("org:A", [sign * 10] * ETA))
    assert crypto.decrypt(crypto.encrypt([10] * ETA, attributes=["org:A"]), dk) == sign * ETA * 100


def test_negative_plaintexts_and_weights(make_kp):
    kg, crypto = make_kp()
    x, y = [-10, 10, -3, 0, -7], [-10, -1, 4, 9, 10]
    dk = kg.get_decryption_keys("s", credentials=creds("org:A", y))
    assert crypto.decrypt(crypto.encrypt(x, attributes=["org:A"]), dk) == inner(x, y)


@pytest.mark.parametrize("attributes", [
    lambda: ["org:B", "org:A"],
    lambda: ["org:B", "org:A", "org:B", "org:A", "org:A"],
    lambda: ("org:A", "org:B"),
    lambda: {"org:B", "org:A"},
    lambda: frozenset({"org:A", "org:B"}),
    lambda: iter(["org:B", "org:A"]),
], ids=["reversed", "repeats", "tuple", "set", "frozenset", "iterator"])
def test_attributes_in_any_order(attributes, make_kp):
    kg, crypto = make_kp()
    ct = crypto.encrypt(X, attributes=attributes())
    assert ct["attributes"] == ["org:A", "org:B"] and len(ct["c_attr"]) == 2       # sorted, each once
    dk = kg.get_decryption_keys("s", credentials=creds("and(org:A, org:B)", Y))
    assert crypto.decrypt(ct, dk) == inner(X, Y)


def test_an_empty_set_of_attributes_encrypts_and_opens_nothing(make_kp):
    kg, crypto = make_kp()
    ct = crypto.encrypt(X, attributes=[])
    assert ct["attributes"] == [] and ct["c_attr"] == [] and len(ct["t"]) == ETA
    with pytest.raises(FEKeyError):
        crypto.decrypt(ct, kg.get_decryption_keys("s", credentials=creds("org:A", Y)))


def test_many_keys_one_ciphertext(make_kp):
    # z is fixed in msk, and every key has a fresh a0 and fresh pi
    kg, crypto = make_kp()
    ct = crypto.encrypt(X, attributes=["org:A", "org:B", "role:auditor"])
    for policy, y in (("org:A", Y), ("and(org:A, org:B)", [0, 0, 0, 0, 1]), ("2of(org:A, org:C, role:auditor)", Y),
                      ("org:A", Y), ("or(org:B, org:C)", [10, -10, 10, -10, 10])):
        dk = kg.get_decryption_keys("s", credentials=creds(policy, y))
        assert crypto.decrypt(ct, dk) == inner(X, y)


def test_many_ciphertexts_one_key(make_kp):
    kg, crypto = make_kp()
    dk = kg.get_decryption_keys("s", credentials=creds("or(org:A, org:B)", Y))
    for x, attributes in ((X, ["org:A"]), ([1, 1, 1, 1, 1], ["org:B"]), ([-10, 10, -10, 10, -10], ["org:A", "org:B"])):
        assert crypto.decrypt(crypto.encrypt(x, attributes=attributes), dk) == inner(x, Y)


def test_random_formulas(make_kp):
    rng = random.Random(8)
    labels = ["a:{}".format(i) for i in range(6)]
    kg, crypto = make_kp(max_rows=12)

    def formula(depth):
        if depth == 0 or rng.random() < 0.3:
            return Leaf(rng.choice(labels))
        children = [formula(depth - 1) for _ in range(rng.randint(2, 3))]
        kind = rng.randrange(3)
        if kind == 0:
            return And(*children)
        if kind == 1:
            return Or(*children)
        return Threshold(rng.randint(1, len(children)), *children)

    tested = 0
    while tested < 10:
        policy = formula(2)
        if len(policy.leaves()) > 12:
            continue
        try:
            validate_formula(policy)    # two leaf children of a threshold gate must not share a label
        except FEValidationError:
            continue
        held = [label for label in labels if rng.random() < 0.6]
        x = [rng.randint(-10, 10) for _ in range(ETA)]
        y = [rng.randint(-10, 10) for _ in range(ETA)]
        ct = crypto.encrypt(x, attributes=held)
        dk = kg.get_decryption_keys("s", credentials=creds(policy, y))
        if evaluate(policy, held):
            assert crypto.decrypt(ct, dk) == inner(x, y)
        else:
            with pytest.raises(FEKeyError):
                crypto.decrypt(ct, dk)
        tested += 1


def test_labels_outside_ascii(make_kp):
    label = "org:café"
    kg, crypto = make_kp()
    dk = kg.get_decryption_keys("s", credentials=creds(label, [2, 0, 0, 0, 0]))
    assert crypto.decrypt(crypto.encrypt([1, 0, 0, 0, 0], attributes=[label]), dk) == 2


def test_the_longest_label_encrypts_and_decrypts(make_kp):
    label = label_of(MAX_LABEL_CHARS)
    kg, crypto = make_kp()
    dk = kg.get_decryption_keys("s", credentials=creds("or({}, org:A)".format(label), Y))
    assert crypto.decrypt(crypto.encrypt(X, attributes=[label]), dk) == inner(X, Y)


def test_objects_survive_json(kp_fixed):
    kg, _, ct, dk, expected = kp_fixed
    crypto = SIFEKPABPairing({"keys": {"pp": json.loads(json.dumps(kg.get_public_parameters()))}})
    assert crypto.decrypt(json.loads(json.dumps(ct)), json.loads(json.dumps(dk))) == expected


def test_the_fusion_weight_must_be_the_weights_of_the_key(kp_fixed):
    _, crypto, ct, dk, expected = kp_fixed
    assert crypto.decrypt(ct, dk, fusion_weight=Y) == expected
    for wrong in ([1, 2, 3, -4, 4], [1, 2, 3, -4], [1, 2, 3, -4, 5, 0]):
        with pytest.raises(FEValidationError, match="fusion"):
            crypto.decrypt(ct, dk, fusion_weight=wrong)


def test_the_ciphertext_layout(make_kp):
    _, crypto = make_kp()
    group = PairingGroup(GROUP)
    ct = crypto.encrypt(X, attributes={"org:B", "org:A"})
    assert list(ct) == list(KPABCiphertext.FIELDS)
    assert (ct["attr_hash"], ct["policy_compiler"]) == (ATTR_HASH, POLICY_COMPILER)
    assert ct["pp_binding"] == crypto.pp["pp_binding"]
    assert len(ct["c_attr"]) == 2 and all(len(row) == 8 for row in ct["c_attr"])
    assert len(ct["t"]) == ETA and len(ct["c_ipfe"]) == 4
    for items in (*ct["c_attr"], ct["t"], ct["c_ipfe"]):
        assert all(group.deserialize(item.encode("utf-8")).type == G1 for item in items)
    assert json.loads(json.dumps(ct)) == ct


def test_the_ciphertext_in_the_exponent(make_kp):
    # Enc of Fig. 1 against the master key: c_j = (sigma_j (1, -j), psi)_F, t_i = [omega (s_i + mu u_i) + x_i]_1 and
    # c_ipfe = (omega, mu omega, psi)_H, with the same psi in all of them
    kg, crypto = make_kp()
    p = public(kg)
    group, order = p["group"], p["order"]
    gt = pair(p["g1"], p["g2"])
    ct = crypto.encrypt(X, attributes=["org:A", "org:B", "role:auditor"])
    h_star, f_star = kg.msk["h_star"], kg.msk["f_star"]
    c_ipfe = deserialize_many(group, ct["c_ipfe"])
    gt_omega, gt_mu_omega, gt_psi = (vector_pair(c_ipfe, h_star[i]) for i in range(3))
    for t_i, s_i, u_i, x_i in zip(deserialize_many(group, ct["t"]), kg.msk["s"], kg.msk["u"], X):
        assert pair(t_i, p["g2"]) == gt_omega ** int(s_i) * gt_mu_omega ** int(u_i) * gt ** (x_i % order)
    for label, row in zip(ct["attributes"], ct["c_attr"]):
        c = deserialize_many(group, row)
        gt_sigma = vector_pair(c, f_star[0])
        assert gt_sigma != gt ** 0
        assert vector_pair(c, f_star[1]) == gt_sigma ** (-attribute_to_zq(label, order) % order)
        assert vector_pair(c, f_star[2]) == gt_psi


def test_two_encryptions_use_fresh_randomness(make_kp):
    _, crypto = make_kp()
    first = crypto.encrypt(X, attributes=["org:A", "org:B"])
    second = crypto.encrypt(X, attributes=["org:A", "org:B"])
    assert all(a != b for a, b in zip(first["c_attr"], second["c_attr"]))
    assert not set(first["t"]) & set(second["t"])
    assert not set(first["c_ipfe"]) & set(second["c_ipfe"])
    assert not set(first["c_attr"][0]) & set(first["c_attr"][1])                 # sigma is fresh for every label


def test_encryption_is_public_key(make_kp):
    kg, crypto = make_kp()
    assert crypto.keys == {"pp": kg.get_public_parameters()}
    assert not {"z", "s", "u", "f_star", "h_star"} & set(crypto.pp)
    assert crypto.max_policy_length == 64 * (MAX_LABEL_CHARS + 400)


@pytest.mark.parametrize("name", ["encrypt_lst_ndarray", "compute_lst_ndarray_ct", "decrypt_lst_ndarray_ct"])
def test_ndarray_helpers_are_not_implemented(name, make_kp):
    _, crypto = make_kp()
    with pytest.raises(NotImplementedError):
        getattr(crypto, name)([])


# ---------------------------------------------------------------------------
# encryption: the input


@pytest.mark.parametrize("vector", [[11] * ETA, [0] * (ETA - 1) + [-11], [1] * (ETA + 1), [1] * (ETA - 1), [], None,
                                    [1.5] * ETA, "12345", [True] * ETA], ids=str)
def test_encrypt_checks_the_plaintext(vector, make_kp):
    _, crypto = make_kp()
    with pytest.raises(FEValidationError, match="plaintext"):
        crypto.encrypt(vector, attributes=["org:A"])


@pytest.mark.parametrize("attributes", [None, "org:A", b"org:A", 7, ["org:A", 7], [b"org:A"], ["org"], ["org:A b"],
                                        [["org:A"]], ["org:​A"]], ids=str)
def test_encrypt_checks_the_attributes(attributes, make_kp):
    _, crypto = make_kp()
    with pytest.raises(FEValidationError, match="attribute"):
        crypto.encrypt(X, attributes=attributes)


def test_encrypt_without_attributes_is_refused(make_kp):
    _, crypto = make_kp()
    with pytest.raises(FEValidationError, match="attributes"):
        crypto.encrypt(X)


def test_encrypt_limits_the_attributes(make_kp):
    _, crypto = make_kp(max_attributes=3)
    labels = ["org:{}".format(i) for i in range(4)]
    assert len(crypto.encrypt(X, attributes=labels[:3])["c_attr"]) == 3
    assert len(crypto.encrypt(X, attributes=labels[:3] * 5)["c_attr"]) == 3         # repeats are not counted
    with pytest.raises(FEValidationError, match="max_attributes = 3"):
        crypto.encrypt(X, attributes=labels)


def test_encrypt_limits_the_length_of_a_label(make_kp):
    _, crypto = make_kp()
    with pytest.raises(FEValidationError, match="MAX_LABEL_CHARS = 1024"):
        crypto.encrypt(X, attributes=[label_of(1025)])


def test_a_refused_encryption_computes_nothing(make_kp, monkeypatch):
    _, crypto = make_kp()
    monkeypatch.setattr(kp_ab_pairing, "attribute_vector", lambda *a, **k: pytest.fail("computed for refused input"))
    monkeypatch.setattr(kp_ab_pairing, "multiexp_many", lambda *a, **k: pytest.fail("computed for refused input"))
    with pytest.raises(FEValidationError):
        crypto.encrypt(X, attributes="org:A")
    with pytest.raises(FEValidationError):
        crypto.encrypt([11] * ETA, attributes=["org:A"])


def test_encrypt_decodes_the_public_parameters_once(make_kp, monkeypatch):
    _, crypto = make_kp()
    decoded = []
    real = ac_codec.deserialize_many
    monkeypatch.setattr(ac_codec, "deserialize_many",
                        lambda group, items: decoded.append(list(items)) or real(group, items))
    for _ in range(3):
        crypto.encrypt(X, attributes=["org:A", "org:B"])
    pp = crypto.pp
    assert [decoded.count(pp[name]) for name in ("h_mask", "h3", "t")] == [1, 1, 1]
    assert [decoded.count(row) for row in pp["f"]] == [1, 1, 1]
    assert decoded.count([pp["g1"]]) == 1


def test_kept_public_parameters_give_correct_ciphertexts(kp_fixed):
    _, crypto, _, dk, expected = kp_fixed
    for _ in range(3):
        assert crypto.decrypt(crypto.encrypt(X, attributes=HELD), dk) == expected


@pytest.mark.parametrize("threads", [1, 4])
def test_threads_do_not_change_the_result(threads, make_kp):
    # above the size from which the native batches use threads
    eta = 40
    native_backend.set_threads(threads)
    try:
        kg, crypto = make_kp(eta=eta)
        x, y = [(-1) ** i * (i % 7) for i in range(eta)], [i % 5 for i in range(eta)]
        ct = crypto.encrypt(x, attributes=["org:A", "role:auditor"])
        dk = kg.get_decryption_keys("s", credentials=creds(POLICY, y))
        assert crypto.decrypt(ct, dk) == inner(x, y)
    finally:
        native_backend.set_threads(1)


# ---------------------------------------------------------------------------
# keys and ciphertexts that are put together, or made under other parameters


def swap_policy(dk, policy):
    """*dk* with the policy *policy* (text) and its digest in place of its own, as a forger writes them."""
    formula = parse(policy)
    return dict(dk, policy=canonical_encoding(formula), policy_digest=policy_digest(formula))


def digest_of_text(text):
    """The digest that ``policy_digest`` would give *text* if it were a formula, for a hostile text that is not one."""
    return hashlib.shake_256(b"PYFE4AI-V05-POLICY" + text.encode("utf-8")).hexdigest(32)


def without(record, field):
    return {name: value for name, value in record.items() if name != field}


@pytest.mark.parametrize("donor", [0, 1], ids=["k-ipfe-of-key1", "k-ipfe-of-key2"])
def test_collusion_splice(donor, make_kp):
    # two keys that each hold one row of a policy are put together into a key for the policy of both: the rows have
    # different a0, so the shares do not add up to the a0 of k_ipfe
    kg, crypto = make_kp()
    ct = crypto.encrypt(X, attributes=["org:A", "role:auditor"])
    keys = [kg.get_decryption_keys("s", credentials=creds("org:A", Y)),
            kg.get_decryption_keys("s", credentials=creds("role:auditor", Y))]
    for alone in keys:
        assert crypto.decrypt(ct, alone) == inner(X, Y)         # each opens the ciphertext for its own policy
    forged = dict(swap_policy(keys[donor], POLICY), k_rows=[keys[0]["k_rows"][0], keys[1]["k_rows"][0]])
    with pytest.raises(ValueError, match="discrete log") as caught:
        crypto.decrypt(ct, forged)
    assert not isinstance(caught.value, FEValidationError)


@pytest.mark.parametrize("policy,swapped,held", [
    ("and(org:A, org:B)", "or(org:A, org:B)", ["org:A"]),
    ("2of(org:A, org:B, org:C)", "or(org:A, org:B, org:C)", ["org:C"]),
], ids=["and-to-or", "threshold-to-or"])
def test_policy_swap(policy, swapped, held, make_kp):
    kg, crypto = make_kp()
    ct = crypto.encrypt(X, attributes=held)
    dk = kg.get_decryption_keys("s", credentials=creds(policy, Y))
    with pytest.raises(FEKeyError):
        crypto.decrypt(ct, dk)
    # `or` has as many rows as the policy it replaces, so the length checks pass
    forged = swap_policy(dk, swapped)
    assert len(forged["k_rows"]) == len(parse(swapped).leaves())
    with pytest.raises(ValueError, match="discrete log") as caught:
        crypto.decrypt(ct, forged)
    assert not isinstance(caught.value, FEValidationError)


def test_a_ciphertext_of_other_attributes_gives_nothing(make_kp):
    # the attributes of a ciphertext are public data, and the vectors are bound to the labels by their j: the
    # vector of org:B written under the name org:A does not open the row of org:A
    kg, crypto = make_kp()
    ct = crypto.encrypt(X, attributes=["org:B"])
    dk = kg.get_decryption_keys("s", credentials=creds("org:A", Y))
    with pytest.raises(FEKeyError):
        crypto.decrypt(ct, dk)
    with pytest.raises(ValueError, match="discrete log"):
        crypto.decrypt(dict(ct, attributes=["org:A"]), dk)


def test_a_key_of_another_setup_does_not_open_even_when_rebound(make_kp):
    kg, crypto = make_kp()
    other, _ = make_kp()
    ct = crypto.encrypt(X, attributes=["org:A"])
    dk = other.get_decryption_keys("s", credentials=creds("org:A", Y))
    with pytest.raises(FEValidationError, match="pp_binding"):
        crypto.decrypt(ct, dk)
    with pytest.raises(ValueError, match="discrete log"):
        crypto.decrypt(ct, dict(dk, pp_binding=ct["pp_binding"]))


def test_two_setups(make_kp, monkeypatch):
    kg1, crypto1 = make_kp()
    kg2, crypto2 = make_kp()
    assert kg1.get_public_parameters()["pp_binding"] != kg2.get_public_parameters()["pp_binding"]
    ct1, ct2 = crypto1.encrypt(X, attributes=["org:A"]), crypto2.encrypt(X, attributes=["org:A"])
    dk1 = kg1.get_decryption_keys("s", credentials=creds("org:A", Y))
    dk2 = kg2.get_decryption_keys("s", credentials=creds("org:A", Y))
    assert crypto1.decrypt(ct1, dk1) == crypto2.decrypt(ct2, dk2) == inner(X, Y)

    def never(*args):
        pytest.fail("multi_pair was called for objects of two setups")

    monkeypatch.setattr(kp_ab_pairing, "multi_pair", never)
    for crypto in (crypto1, crypto2):
        for ct, dk in ((ct1, dk2), (ct2, dk1)):         # the key of one setup with the ciphertext of the other
            with pytest.raises(FEValidationError, match="pp_binding"):
                crypto.decrypt(ct, dk)
    with pytest.raises(FEValidationError, match="pp_binding"):
        crypto1.decrypt(ct2, dk2)                       # and both of the other setup


def test_decrypt_checks_the_binding_before_anything_else(kp_fixed):
    _, crypto, ct, dk, _ = kp_fixed
    bad = dict(without(ct, "t"), pp_binding="0" * 64, attr_hash="x", attributes=5)
    with pytest.raises(FEValidationError, match="pp_binding"):
        crypto.decrypt(bad, dk)
    bad_key = dict(without(dk, "k_ipfe"), pp_binding="0" * 64, policy="[" * 100000, policy_compiler="c9")
    with pytest.raises(FEValidationError, match="pp_binding"):
        crypto.decrypt(ct, bad_key)


def test_decrypt_checks_in_a_fixed_order(make_kp):
    # one defect for each stage, put right one stage after the other: binding, fields, version markers, policy and
    # its digest, lengths, the weights, then whether the attributes satisfy the policy, and only then the elements
    kg, crypto = make_kp()
    ct = crypto.encrypt(X, attributes=["org:A"])                    # does not satisfy POLICY
    dk = kg.get_decryption_keys("s", credentials=creds(POLICY, Y))
    ct["c_attr"][0][0] = "junk"                                     # the last stage: an element that is no element
    defects = [
        ("pp_binding", lambda c, k: (dict(c, pp_binding="0" * 64), k)),
        ("no field k_ipfe", lambda c, k: (c, without(k, "k_ipfe"))),
        ("attr_hash", lambda c, k: (dict(c, attr_hash="x"), k)),
        ("policy_digest", lambda c, k: (c, dict(k, policy_digest="0" * 64))),
        ("t must be", lambda c, k: (dict(c, t=c["t"][:-1]), k)),
        ("y must hold", lambda c, k: (c, dict(k, y=["x"] * ETA))),
    ]
    for stage, (match, _) in enumerate(defects):
        c, k = ct, dk
        for _, defect in defects[stage:]:
            c, k = defect(c, k)
        with pytest.raises(FEValidationError, match=match):
            crypto.decrypt(c, k)
    with pytest.raises(FEKeyError, match="do not satisfy"):         # the junk element is not read for a key that
        crypto.decrypt(ct, dk)                                       # cannot open the ciphertext


# ---------------------------------------------------------------------------
# the public parameters


def rebound(pp):
    """*pp* with its ``pp_binding`` recomputed: public parameters that are consistent, but not honest."""
    return dict(pp, pp_binding=pp_digest(pp, PP_DOMAIN))


def tampered(pp, **fields):
    return rebound(dict(pp, **fields))


def test_tampered_pp(make_kp):
    kg, _ = make_kp()
    pp = copy.deepcopy(kg.get_public_parameters())
    pp["t"][0] = pp["t"][1]
    with pytest.raises(FEValidationError, match="pp_binding"):
        SIFEKPABPairing({"keys": {"pp": pp}})


@pytest.mark.parametrize("field", ["g1", "pp_binding", "t"])
def test_parameters_changed_after_construction_do_not_reach_the_object(field, make_kp):
    kg, _ = make_kp()
    original = kg.get_public_parameters()
    mine = copy.deepcopy(original)
    crypto = SIFEKPABPairing({"keys": {"pp": mine}})
    mine[field] = make_kp()[0].get_public_parameters()[field]
    ct = crypto.encrypt(X, attributes=["org:A"])
    assert ct["pp_binding"] == original["pp_binding"]
    dk = kg.get_decryption_keys("s", credentials=creds("org:A", Y))
    assert SIFEKPABPairing({"keys": {"pp": original}}).decrypt(ct, dk) == inner(X, Y)


def _nested(depth):
    value = []
    for _ in range(depth):
        value = [value]
    return value


@pytest.mark.parametrize("extra", [_nested(5000), threading.Lock()], ids=["too-deep", "not-data"])
def test_parameters_that_cannot_be_copied_are_refused(extra, make_kp):
    kg, _ = make_kp()
    with pytest.raises(FEValidationError, match="JSON data"):
        SIFEKPABPairing({"keys": {"pp": dict(kg.get_public_parameters(), extra=extra)}})


@pytest.mark.parametrize("field", list(KPABParameters.FIELDS))
def test_crypto_refuses_public_parameters_without_a_field(field, make_kp):
    kg, _ = make_kp()
    pp = without(kg.get_public_parameters(), field)
    with pytest.raises(FEValidationError, match=field):
        SIFEKPABPairing({"keys": {"pp": pp if field == "pp_binding" else rebound(pp)}})


def test_crypto_needs_public_parameters():
    with pytest.raises(FEKeyError):
        SIFEKPABPairing({"keys": {"sk": {}}})
    with pytest.raises(FEValidationError):
        SIFEKPABPairing({"keys": {"pp": "not a dict"}})


@pytest.mark.parametrize("field", ["policy_compiler", "attr_hash"])
def test_crypto_refuses_an_unknown_version_marker(field, make_kp):
    kg, _ = make_kp()
    with pytest.raises(FEValidationError, match=field) as caught:
        SIFEKPABPairing({"keys": {"pp": tampered(kg.get_public_parameters(), **{field: "v" * 100000})}})
    assert len(str(caught.value)) < 300


@pytest.mark.parametrize("value", [5, None, ["BLS12-381"], {"name": "BLS12-381"}, True])
def test_crypto_refuses_a_pairing_group_that_is_no_name(value, make_kp):
    kg, _ = make_kp()
    with pytest.raises(FEValidationError, match="pairing_group_param"):
        SIFEKPABPairing({"keys": {"pp": tampered(kg.get_public_parameters(), pairing_group_param=value)}})


@pytest.mark.parametrize("name", ["nonsense", "", "x" * 100000], ids=["unknown", "empty", "huge"])
def test_crypto_refuses_a_pairing_group_that_does_not_exist(name, make_kp):
    if not pairing_available("MNT224"):
        pytest.skip("no charm: other groups than BLS12-381 are not built")
    kg, _ = make_kp()
    with pytest.raises(FEValidationError, match="pairing_group_param") as caught:
        SIFEKPABPairing({"keys": {"pp": tampered(kg.get_public_parameters(), pairing_group_param=name)}})
    assert len(str(caught.value)) < 300


def test_a_symmetric_group_is_refused_by_the_crypto_object(make_kp):
    if not pairing_available("SS512"):
        pytest.skip("no charm: SS512 is not built")
    kg, _ = make_kp()
    with pytest.raises(FEValidationError, match="symmetric"):
        SIFEKPABPairing({"keys": {"pp": tampered(kg.get_public_parameters(), pairing_group_param="SS512")}})


@pytest.mark.parametrize("field,value", [
    ("h_mask", ["x"] * 3), ("h_mask", ["x"] * 5), ("h_mask", "abcd"), ("h3", ["x"]), ("h3", None),
    ("f", [["x"] * 8] * 2), ("f", [["x"] * 8] * 4), ("f", [["x"] * 7] * 3), ("f", None), ("f", "abc"),
    ("t", ["x"] * (ETA - 1)), ("t", ["x"] * (ETA + 1)), ("t", None),
])
def test_crypto_refuses_public_parameters_of_the_wrong_length(field, value, make_kp):
    kg, _ = make_kp()
    with pytest.raises(FEValidationError, match="^{} ".format(field)):
        SIFEKPABPairing({"keys": {"pp": tampered(kg.get_public_parameters(), **{field: value})}})


@pytest.mark.parametrize("field,value", [
    ("eta", 0), ("eta", "5"), ("eta", True), ("max_rows", -1), ("max_rows", None), ("max_attributes", 0),
    ("max_attributes", 64.0), ("max_attributes", "64"), ("bound_x", "x"), ("bound_x", 10), ("bound_y", "-1"),
    ("bound_y", "007"), ("bound_x", str(2 ** 49)),
])
def test_crypto_refuses_limits_that_are_not_limits(field, value, make_kp):
    kg, _ = make_kp()
    with pytest.raises(FEValidationError):
        SIFEKPABPairing({"keys": {"pp": tampered(kg.get_public_parameters(), **{field: value})}})


def test_the_limits_of_the_public_parameters_reach_the_object(make_kp):
    _, crypto = make_kp(max_rows=7, max_attributes=9, eta=3, bound_x=4, bound_y=5)
    assert (crypto.eta, crypto.bound_x, crypto.bound_y, crypto.max_rows, crypto.max_attributes) == (3, 4, 5, 7, 9)
    assert crypto.max_policy_length == 7 * (MAX_LABEL_CHARS + 400)


def test_a_tampered_element_of_a_read_field_is_refused_when_it_is_read(make_kp):
    kg, _ = make_kp()
    pp = copy.deepcopy(kg.get_public_parameters())
    pp["t"][0] = pp["g2"]          # an element of G2: charm reads almost any text as some element of G1
    crypto = SIFEKPABPairing({"keys": {"pp": rebound(pp)}})
    for _ in range(2):      # a refusal is not kept as a result
        with pytest.raises(FEValidationError, match="^t "):
            crypto.encrypt(X, attributes=["org:A"])


# ---------------------------------------------------------------------------
# decrypt: the key's policy


@pytest.mark.parametrize("text,reason", [
    (lambda limit: "[" * 100000, "length"),              # over the length limit: refused before from_canonical
    (lambda limit: "[" * (limit + 1), "length"),
    (lambda limit: "[" * (limit - 1), "nesting"),       # just under it: from_canonical's own check of the nesting
    (lambda limit: '["and",' * (limit // 7), "nesting"),
], ids=["100000-brackets", "one-over-the-limit", "just-under-the-limit", "nested-gates"])
def test_hostile_policy(text, reason, kp_fixed):
    _, crypto, ct, dk, _ = kp_fixed
    limit = 64 * (MAX_LABEL_CHARS + 400)
    hostile = text(limit)
    assert (len(hostile) > limit) == (reason == "length")
    message = "more than the limit of {}".format(limit) if reason == "length" else "nests brackets"
    with pytest.raises(FEValidationError, match=message):
        crypto.decrypt(ct, dict(dk, policy=hostile, policy_digest=digest_of_text(hostile)))


def test_a_policy_at_the_length_limit_reaches_the_parser(kp_fixed, monkeypatch):
    _, crypto, ct, dk, _ = kp_fixed
    limit = 64 * (MAX_LABEL_CHARS + 400)
    seen = []

    def spy(data):
        seen.append(len(data))
        raise FEValidationError("stop here")

    monkeypatch.setattr(kp_ab_pairing, "from_canonical", spy)
    with pytest.raises(FEValidationError, match="stop here"):
        crypto.decrypt(ct, dict(dk, policy="[" * limit))
    assert seen == [limit]
    with pytest.raises(FEValidationError, match="more than the limit"):
        crypto.decrypt(ct, dict(dk, policy="[" * (limit + 1)))
    assert seen == [limit]                                          # the longer text did not reach it


def test_the_policy_length_limit_follows_max_rows(make_kp):
    kg, crypto = make_kp(max_rows=2)
    limit = 2 * (MAX_LABEL_CHARS + 400)
    assert crypto.max_policy_length == limit
    ct = crypto.encrypt(X, attributes=["org:A"])
    dk = kg.get_decryption_keys("s", credentials=creds("org:A", Y))
    with pytest.raises(FEValidationError, match="more than the limit of {}".format(limit)):
        crypto.decrypt(ct, dict(dk, policy="[" * (limit + 1)))


def test_the_key_generator_refuses_a_policy_that_decryption_would_refuse(make_kp):
    # '"' takes two characters in the canonical encoding and one in a label
    label = "a:" + '"' * 800
    assert len(label) <= MAX_LABEL_CHARS
    kg, crypto = make_kp(max_rows=1)
    assert kg.get_decryption_keys("s", credentials=creds("a:1", Y))
    with pytest.raises(FEValidationError, match="characters, more than the limit of 1424"):
        kg.get_decryption_keys("s", credentials=creds(label, Y))
    with pytest.raises(FEValidationError, match="characters, more than the limit of 1424"):
        kg.get_decryption_keys("s", credentials=creds("a:" + "x" * 1500, Y))


def test_a_label_of_the_limit_length_fits_every_policy_of_the_limit_rows(make_kp):
    # the 64 rows of the default, each with a label of 1024 characters
    labels = [label_of(MAX_LABEL_CHARS - 5) + "{:05d}".format(i) for i in range(64)]
    kg, crypto = make_kp()
    dk = kg.get_decryption_keys("s", credentials=creds("or({})".format(", ".join(labels)), Y))
    assert len(dk["k_rows"]) == 64
    assert len(dk["policy"]) <= crypto.max_policy_length
    assert crypto.decrypt(crypto.encrypt(X, attributes=labels[-1:]), dk) == inner(X, Y)


@pytest.mark.parametrize("policy", [None, 5, ["and"]])
def test_decrypt_refuses_a_policy_that_is_no_text(policy, kp_fixed):
    _, crypto, ct, dk, _ = kp_fixed
    with pytest.raises(FEValidationError, match="policy"):
        crypto.decrypt(ct, dict(dk, policy=policy))


def test_digest_mismatch(kp_fixed):
    _, crypto, ct, dk, _ = kp_fixed
    with pytest.raises(FEValidationError, match="policy_digest"):
        crypto.decrypt(ct, dict(dk, policy=canonical_encoding(parse("or(org:A, role:auditor)"))))


@pytest.mark.parametrize("spelling", [
    lambda text: text.replace(",", ", "),                                           # another spelling of the formula
    lambda text: " " + text,
    lambda text: text + "\n",
])
def test_a_policy_in_another_spelling_is_refused(spelling, kp_fixed):
    _, crypto, ct, dk, _ = kp_fixed
    bad = spelling(dk["policy"])
    assert bad != dk["policy"]
    with pytest.raises(FEValidationError):
        crypto.decrypt(ct, dict(dk, policy=bad, policy_digest=digest_of_text(bad)))


def test_too_many_rows(kp_fixed):
    _, crypto, ct, dk, _ = kp_fixed
    wide = "or({})".format(", ".join(["org:A"] * 65))
    with pytest.raises(FEValidationError, match="more than the limit of 64 rows"):
        crypto.decrypt(ct, swap_policy(dk, wide))


def test_the_rows_limit_of_the_parameters_is_the_one_that_counts(make_kp):
    kg, crypto = make_kp(max_rows=2)
    ct = crypto.encrypt(X, attributes=["org:A"])
    dk = kg.get_decryption_keys("s", credentials=creds("or(org:A, org:B)", Y))
    assert crypto.decrypt(ct, dk) == inner(X, Y)
    with pytest.raises(FEValidationError, match="more than the limit of 2 rows"):
        crypto.decrypt(ct, swap_policy(dk, "or(org:A, org:B, org:C)"))
    with pytest.raises(FEValidationError, match="more than the limit of 2 rows"):
        kg.get_decryption_keys("s", credentials=creds("or(org:A, org:B, org:C)", Y))


# ---------------------------------------------------------------------------
# decrypt: the version markers, fields and shapes


@pytest.mark.parametrize("field", list(KPABCiphertext.FIELDS))
def test_decrypt_refuses_a_ciphertext_without_a_field(field, kp_fixed):
    _, crypto, ct, dk, _ = kp_fixed
    with pytest.raises(FEValidationError, match=field):
        crypto.decrypt(without(ct, field), dk)


@pytest.mark.parametrize("field", list(KPABFunctionalKey.FIELDS))
def test_decrypt_refuses_a_key_without_a_field(field, kp_fixed):
    _, crypto, ct, dk, _ = kp_fixed
    with pytest.raises(FEValidationError, match=field):
        crypto.decrypt(ct, without(dk, field))


@pytest.mark.parametrize("ct_bad", ["not a dict", None, ["c_attr"]])
def test_decrypt_refuses_what_is_no_ciphertext(ct_bad, kp_fixed):
    _, crypto, _, dk, _ = kp_fixed
    with pytest.raises(FEValidationError, match="ciphertext"):
        crypto.decrypt(ct_bad, dk)


@pytest.mark.parametrize("dk_bad", ["not a dict", None, ["k_rows"]])
def test_decrypt_refuses_what_is_no_key(dk_bad, kp_fixed):
    _, crypto, ct, _, _ = kp_fixed
    with pytest.raises(FEValidationError, match="functional key"):
        crypto.decrypt(ct, dk_bad)


@pytest.mark.parametrize("field,record", [("policy_compiler", "ct"), ("attr_hash", "ct"), ("policy_compiler", "dk")])
def test_decrypt_refuses_an_unknown_version_marker(field, record, kp_fixed):
    _, crypto, ct, dk, _ = kp_fixed
    with pytest.raises(FEValidationError, match=field) as caught:
        if record == "ct":
            crypto.decrypt(dict(ct, **{field: "v" * 100000}), dk)
        else:
            crypto.decrypt(ct, dict(dk, **{field: "v" * 100000}))
    assert len(str(caught.value)) < 300


@pytest.mark.parametrize("field,shape", [
    ("attributes", lambda v: "org:A"), ("attributes", lambda v: None), ("attributes", lambda v: tuple(v)),
    ("attributes", lambda v: [5]), ("attributes", lambda v: ["org:A", "org:A"]),
    ("attributes", lambda v: ["role:auditor", "org:A", "org:B"]), ("attributes", lambda v: ["org"]),
    ("attributes", lambda v: v + ["zzz:1"] * 70),
    ("c_attr", lambda v: v[:-1]), ("c_attr", lambda v: v + v[:1]), ("c_attr", lambda v: [v[0], v[1][:7], v[2]]),
    ("c_attr", lambda v: [v[0], v[1] + v[1][:1], v[2]]), ("c_attr", lambda v: [v[0], "ab", v[2]]),
    ("c_attr", lambda v: None), ("c_attr", lambda v: "abc"),
    ("t", lambda v: v[:-1]), ("t", lambda v: v + v[:1]), ("t", lambda v: None), ("t", lambda v: "ab"),
    ("c_ipfe", lambda v: v[:-1]), ("c_ipfe", lambda v: v + v[:1]), ("c_ipfe", lambda v: None),
])
def test_decrypt_refuses_a_ciphertext_of_the_wrong_shape(field, shape, kp_fixed):
    _, crypto, ct, dk, _ = kp_fixed
    with pytest.raises(FEValidationError, match="attribute" if field == "attributes" else "^{} ".format(field)):
        crypto.decrypt(dict(ct, **{field: shape(ct[field])}), dk)


@pytest.mark.parametrize("field,shape", [
    ("k_rows", lambda v: v[:1]), ("k_rows", lambda v: v + v[:1]), ("k_rows", lambda v: [v[0], v[1][:7]]),
    ("k_rows", lambda v: [v[0], v[1] + v[1][:1]]), ("k_rows", lambda v: [v[0], "ab"]), ("k_rows", lambda v: None),
    ("k_rows", lambda v: {"a": 1}),
    ("k_ipfe", lambda v: v[:-1]), ("k_ipfe", lambda v: v + v[:1]), ("k_ipfe", lambda v: None),
    ("y", lambda v: v[:-1]), ("y", lambda v: v + v[:1]), ("y", lambda v: None), ("y", lambda v: "1"),
    ("y", lambda v: [1] + v[1:]), ("y", lambda v: ["x"] + v[1:]), ("y", lambda v: ["1.5"] + v[1:]),
    ("y", lambda v: ["11"] + v[1:]), ("y", lambda v: ["-11"] + v[1:]), ("y", lambda v: ["007"] + v[1:]),
    ("y", lambda v: ["1" * 5000] + v[1:]), ("y", lambda v: [" 1"] + v[1:]), ("y", lambda v: ["1_0"] + v[1:]),
    ("y", lambda v: ["١"] + v[1:]),
])
def test_decrypt_refuses_a_key_of_the_wrong_shape(field, shape, kp_fixed):
    _, crypto, ct, dk, _ = kp_fixed
    with pytest.raises(FEValidationError, match="^{} ".format(field)):
        crypto.decrypt(ct, dict(dk, **{field: shape(dk[field])}))


def test_decrypt_refuses_attributes_over_the_limit_before_it_reads_the_labels(make_kp, monkeypatch):
    kg, crypto = make_kp(max_attributes=2)
    ct = crypto.encrypt(X, attributes=["org:A", "org:B"])
    dk = kg.get_decryption_keys("s", credentials=creds("org:A", Y))
    monkeypatch.setattr(kp_ab, "validate_label", lambda label: pytest.fail("a label was read"))
    with pytest.raises(FEValidationError, match="max_attributes = 2"):
        crypto.decrypt(dict(ct, attributes=["a:1", "b:1", "c:1"]), dk)


def test_decrypt_refuses_a_label_over_the_length_limit(kp_fixed):
    _, crypto, ct, dk, _ = kp_fixed
    long = label_of(MAX_LABEL_CHARS + 1)
    with pytest.raises(FEValidationError, match="MAX_LABEL_CHARS = 1024"):
        crypto.decrypt(dict(ct, attributes=["org:A", "org:B", long]), dk)


def test_decrypt_names_the_labels_that_are_out_of_order(kp_fixed):
    _, crypto, ct, dk, _ = kp_fixed
    with pytest.raises(FEValidationError, match="'role:auditor' is followed by 'org:B'"):
        crypto.decrypt(dict(ct, attributes=["org:A", "role:auditor", "org:B"]), dk)


# ---------------------------------------------------------------------------
# decrypt: the group elements


def test_decrypt_refuses_a_g2_string_in_a_field_of_g1_before_reading_it(kp_fixed, monkeypatch):
    _, crypto, ct, dk, _ = kp_fixed
    read = []
    real = ac_codec.deserialize_many

    def spy(group, items):
        read.append(items)
        return real(group, items)

    monkeypatch.setattr(ac_codec, "deserialize_many", spy)
    with pytest.raises(FEValidationError, match="^t "):
        crypto.decrypt(dict(ct, t=dk["k_ipfe"][:1] + ct["t"][1:]), dk)
    assert not any(dk["k_ipfe"][0] in items for items in read)


@pytest.mark.parametrize("item", ["junk", "", 5, None])
@pytest.mark.parametrize("field", ["t", "c_ipfe"])
def test_decrypt_refuses_undecodable_ciphertext_elements(field, item, kp_fixed):
    _, crypto, ct, dk, _ = kp_fixed
    with pytest.raises(FEValidationError, match="^{} ".format(field)):
        crypto.decrypt(dict(ct, **{field: [item] + ct[field][1:]}), dk)


@pytest.mark.parametrize("row", [0, 1, 2], ids=["org:A", "org:B-no-row-of-the-key", "role:auditor"])
@pytest.mark.parametrize("item", ["junk", "", 5, None])
def test_decrypt_decodes_every_vector_of_the_ciphertext_also_the_unused(row, item, kp_fixed):
    # R1-F07: a ciphertext is refused whichever key opens it
    _, crypto, ct, dk, _ = kp_fixed
    c_attr = [list(entry) for entry in ct["c_attr"]]
    c_attr[row][3] = item
    with pytest.raises(FEValidationError, match="c_attr"):
        crypto.decrypt(dict(ct, c_attr=c_attr), dk)


def test_decrypt_refuses_an_element_of_g2_among_the_vectors_of_g1(kp_fixed):
    _, crypto, ct, dk, _ = kp_fixed
    c_attr = [list(entry) for entry in ct["c_attr"]]
    c_attr[1][0] = dk["k_ipfe"][0]
    with pytest.raises(FEValidationError, match="c_attr"):
        crypto.decrypt(dict(ct, c_attr=c_attr), dk)


@pytest.mark.parametrize("field,shape", [
    ("k_ipfe", lambda v: ["junk"] + v[1:]),
    ("k_rows", lambda v: [["junk"] + v[0][1:], v[1]]),
    ("k_rows", lambda v: [v[0], v[1][:3] + [5] + v[1][4:]]),
])
def test_decrypt_refuses_undecodable_key_elements(field, shape, kp_fixed):
    _, crypto, ct, dk, _ = kp_fixed
    with pytest.raises(FEValidationError, match=field):
        crypto.decrypt(ct, dict(dk, **{field: shape(dk[field])}))


def test_decrypt_refuses_elements_of_the_wrong_group(kp_fixed):
    _, crypto, ct, dk, _ = kp_fixed
    with pytest.raises(FEValidationError, match="c_ipfe"):
        crypto.decrypt(dict(ct, c_ipfe=dk["k_ipfe"]), dk)
    with pytest.raises(FEValidationError, match="k_ipfe"):
        crypto.decrypt(ct, dict(dk, k_ipfe=ct["c_ipfe"]))
    with pytest.raises(FEValidationError, match="k_rows"):
        crypto.decrypt(ct, dict(dk, k_rows=ct["c_attr"][:2]))


def test_a_failed_discrete_log_is_a_value_error(kp_fixed):
    _, crypto, ct, dk, _ = kp_fixed
    bad = dict(ct, t=list(reversed(ct["t"])))
    with pytest.raises(ValueError, match="discrete log") as caught:
        crypto.decrypt(bad, dk)
    assert not isinstance(caught.value, FEValidationError)


def test_a_ciphertext_with_a_changed_vector_gives_no_value(kp_fixed):
    # the vector of org:B is not read by the key and changing it changes nothing; that of org:A is, and gives nothing
    _, crypto, ct, dk, expected = kp_fixed
    other = crypto.encrypt(X, attributes=HELD)
    assert crypto.decrypt(dict(ct, c_attr=[ct["c_attr"][0], other["c_attr"][1], ct["c_attr"][2]]), dk) == expected
    with pytest.raises(ValueError, match="discrete log"):
        crypto.decrypt(dict(ct, c_attr=[other["c_attr"][0], ct["c_attr"][1], ct["c_attr"][2]]), dk)


def test_multi_pair_is_called_once_with_8_pairs_for_each_used_row_and_5_more(kp_fixed, monkeypatch):
    kg, crypto, ct, dk, expected = kp_fixed
    calls = []
    real = pairing_backend.multi_pair

    def spy(lhs, rhs):
        calls.append(len(lhs))
        return real(lhs, rhs)

    monkeypatch.setattr(kp_ab_pairing, "multi_pair", spy)
    assert crypto.decrypt(ct, dk) == expected                       # two used rows
    assert calls == [8 * 2 + 5]
    one = kg.get_decryption_keys("s", credentials=creds("or(org:A, role:auditor, org:C)", Y))
    assert crypto.decrypt(ct, one) == expected
    assert calls == [21, 8 * 1 + 5]                                 # the first satisfied child only
    three = kg.get_decryption_keys("s", credentials=creds("and(org:A, org:B, role:auditor)", Y))
    assert crypto.decrypt(ct, three) == expected
    assert calls[-1] == 8 * 3 + 5


def test_a_coefficient_of_1_costs_no_exponentiation(make_kp, monkeypatch):
    kg, crypto = make_kp()
    ct = crypto.encrypt(X, attributes=["org:A", "org:B", "org:C"])
    calls = []
    real = kp_ab.multiexp_many
    monkeypatch.setattr(kp_ab, "multiexp_many",
                        lambda rows, exps, group=None: calls.append(1) or real(rows, exps, group))
    for policy in ("and(org:A, org:B)", "or(org:A, org:B)"):        # the coefficients of both rows are 1
        dk = kg.get_decryption_keys("s", credentials=creds(policy, Y))
        assert crypto.decrypt(ct, dk) == inner(X, Y)
    assert calls == []
    dk = kg.get_decryption_keys("s", credentials=creds("2of(org:A, org:B, org:C)", Y))
    assert crypto.decrypt(ct, dk) == inner(X, Y)
    assert calls == [1]                                             # one batch, for the coefficients that are not 1


def test_decrypt_uses_the_bound_of_the_parameters(make_kp):
    kg, crypto = make_kp(eta=2, bound_x=3, bound_y=4)
    dk = kg.get_decryption_keys("s", credentials=creds("org:A", [4, -4]))
    assert crypto.decrypt(crypto.encrypt([3, -3], attributes=["org:A"]), dk) == 24
    assert crypto.decrypt(crypto.encrypt([-3, 3], attributes=["org:A"]), dk) == -24


def test_the_generator_pair_is_computed_once(kp_fixed, monkeypatch):
    _, crypto, ct, dk, expected = kp_fixed
    calls = []
    real = pairing_backend.pair
    monkeypatch.setattr(kp_ab_pairing, "pair", lambda a, b: calls.append(1) or real(a, b))
    for _ in range(3):
        assert crypto.decrypt(ct, dk) == expected
    assert calls == [1]


# ---------------------------------------------------------------------------
# hostile text in error messages


HUGE = 100000


def test_huge_attributes_are_clipped_in_the_messages(kp_fixed):
    _, crypto, ct, dk, _ = kp_fixed
    for attributes in (["org:A", "org:" + "x" * HUGE], ["org:{}".format(i) for i in range(HUGE // 10)]):
        with pytest.raises(FEValidationError) as caught:
            crypto.decrypt(dict(ct, attributes=attributes), dk)
        assert len(str(caught.value)) < 300


def test_the_message_of_an_unsatisfied_policy_names_the_attributes_and_is_clipped(kp_fixed):
    _, crypto, _, dk, _ = kp_fixed
    with pytest.raises(FEKeyError, match="org:A"):
        crypto.decrypt(crypto.encrypt(X, attributes=["org:A"]), dk)
    with pytest.raises(FEKeyError) as caught:
        crypto.decrypt(crypto.encrypt(X, attributes=["org:{}".format(i) for i in range(64)]), dk)
    assert len(str(caught.value)) < 300
