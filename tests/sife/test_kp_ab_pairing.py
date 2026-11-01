"""Setup and key generation of the key-policy attribute-based IPFE (NPP22 Sec. 4 Fig. 1).

Encryption and decryption are not part of this module yet: the keys are checked against a ciphertext made and opened
by hand from the equations of Fig. 1 (``seal`` and ``open_sealed``).
"""

import json
import logging
import os
import random
import warnings

import gmpy2 as gp
import pytest

from tests.pairing_group import GROUP, skip_unless_available

skip_unless_available()

from tests.sife.conftest import ETA  # noqa: E402
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
    from_canonical,
    parse,
    policy_digest,
)
from pyfe4ai.primitives.pairing.dpvs import vector_pair
from pyfe4ai.primitives.pairing.pairing_backend import G1, G2, PairingGroup, deserialize_many, pair, pairing_available
from pyfe4ai.schemes.common import kp_ab
from pyfe4ai.schemes.common.ac_binding import pp_digest
from pyfe4ai.schemes.common.kp_ab import (
    attribute_vector,
    compile_policy,
    held_attributes,
    key_row,
    read_attributes,
    read_policy,
)
from pyfe4ai.schemes.sife import kp_ab_pairing
from pyfe4ai.schemes.sife.kp_ab_pairing import SIFEKPABPairingKeyGenerator

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


def test_the_constants_of_the_schemes():
    assert (kp_ab.F_DIM, kp_ab.H_DIM, kp_ab.DEFAULT_MAX_ATTRIBUTES) == (8, 4, 64)


def test_ciphertext_record_fields():
    assert KPABCiphertext.FIELDS == (
        "attributes", "attr_hash", "policy_compiler", "c_attr", "t", "c_ipfe", "pp_binding")
