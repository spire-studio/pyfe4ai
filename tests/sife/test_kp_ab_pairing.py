"""The key-policy attribute-based IPFE (NPP22 Sec. 4 Fig. 1): setup, key generation, encryption and decryption.

``seal`` and ``open_sealed`` make and open a ciphertext by hand from the equations of Fig. 1, with the public
parameters and the key only. They check the keys before the scheme encrypts and stay as an independent check of
``encrypt`` and ``decrypt``: ``decrypt`` reads a ciphertext of ``seal`` and ``open_sealed`` one of ``encrypt``.
This file is what the scheme does when it is used as it is meant to be; test_kp_ab_pairing_hardening.py is what it
refuses. A key whose policy the attributes of a ciphertext do not satisfy gives a ``FEKeyError``; malformed input, or
input over a limit, a ``FEValidationError``; a key and a ciphertext that are well formed but were not made for each
other a ``ValueError`` from the discrete log, which is not a ``FEValidationError``.
"""

import json
import logging
import os
import random
import warnings

import gmpy2 as gp
import pytest

import pyfe4ai
from tests.pairing_group import GROUP, skip_unless_available

skip_unless_available()

from tests.sife.conftest import ETA, KP_HELD as HELD, KP_POLICY as POLICY, KP_X as X, KP_Y as Y  # noqa: E402
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
from pyfe4ai.primitives.pairing.pairing_backend import G1, G2, PairingGroup, deserialize_many, pair
from pyfe4ai.schemes.common import ac_codec, ac_decoded, kp_ab
from pyfe4ai.schemes.common.ac_binding import pp_digest
from pyfe4ai.schemes.common.ac_codec import serialize_vec
from pyfe4ai.schemes.common.kp_ab import (
    MAX_LABEL_CHARS,
    attribute_pairs,
    attribute_vector,
    compile_policy,
    held_attributes,
    key_row,
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


def label_of(length):
    return "a:" + "x" * (length - 2)


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


def test_create_and_list():
    kg, crypto_cls = pyfe4ai.create("sife/kp_ab_pairing", eta=2, pairing_group_param=GROUP)
    assert (type(kg).__name__, crypto_cls.__name__) == ("SIFEKPABPairingKeyGenerator", "SIFEKPABPairing")
    # `in`, not `==`: each attribute-based scheme that is added joins the list
    assert {"sife/cp_ab_pairing", "sife/kp_ab_pairing"} <= set(
        pyfe4ai.list_schemes("sife", capabilities="access_control"))
    assert pyfe4ai.get_scheme("sife/kp_ab_pairing").capabilities == {"access_control", "pairing"}
    kg.setup()
    crypto = crypto_cls({"keys": {"pp": kg.get_public_parameters()}})
    dk = kg.get_decryption_keys("s", credentials={"fusion_weight": [2, -1], "policy": "org:A"})
    assert crypto.decrypt(crypto.encrypt([3, 4], attributes=["org:A"]), dk) == 2


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


def test_read_attributes_takes_a_sorted_list_of_distinct_labels():
    assert read_attributes(["a:1", "b:1"], 64) == ["a:1", "b:1"]
    assert read_attributes([], 64) == []
    assert read_attributes(["a:1"], 1) == ["a:1"]


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


# ---------------------------------------------------------------------------
# encryption and decryption


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


def test_the_limits_of_the_public_parameters_reach_the_object(make_kp):
    _, crypto = make_kp(max_rows=7, max_attributes=9, eta=3, bound_x=4, bound_y=5)
    assert (crypto.eta, crypto.bound_x, crypto.bound_y, crypto.max_rows, crypto.max_attributes) == (3, 4, 5, 7, 9)
    assert crypto.max_policy_length == 7 * (MAX_LABEL_CHARS + 400)


def test_decrypt_uses_the_bound_of_the_parameters(make_kp):
    kg, crypto = make_kp(eta=2, bound_x=3, bound_y=4)
    dk = kg.get_decryption_keys("s", credentials=creds("org:A", [4, -4]))
    assert crypto.decrypt(crypto.encrypt([3, -3], attributes=["org:A"]), dk) == 24
    assert crypto.decrypt(crypto.encrypt([-3, 3], attributes=["org:A"]), dk) == -24


# ---------------------------------------------------------------------------
# encryption and decryption: what is kept, and what is computed


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


def test_the_generator_pair_is_computed_once(kp_fixed, monkeypatch):
    _, crypto, ct, dk, expected = kp_fixed
    calls = []
    real = pairing_backend.pair
    monkeypatch.setattr(ac_decoded, "pair", lambda a, b: calls.append(1) or real(a, b))
    for _ in range(3):
        assert crypto.decrypt(ct, dk) == expected
    assert calls == [1]
