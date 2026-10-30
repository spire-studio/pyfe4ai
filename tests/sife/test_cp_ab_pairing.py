import ast
import copy
import hashlib
import json
import os
import random
import sys
import warnings

import pytest

import pyfe4ai
from tests.pairing_group import GROUP, skip_unless_available  # noqa: E402

skip_unless_available()

from pyfe4ai.core.constants import CryptoCONST
from pyfe4ai.core.exceptions import FEConfigError, FEKeyError, FEValidationError, FEWarning
from pyfe4ai.core.formats.pairing import CPABCiphertext, CPABFunctionalKey, CPABParameters, CPABRecord
from pyfe4ai.policy import (
    POLICY_COMPILER,
    And,
    Leaf,
    Or,
    Threshold,
    Universe,
    canonical_encoding,
    compile_formula,
    evaluate,
    parse,
    policy_digest,
    read_once_rows,
    validate_formula,
)
from pyfe4ai.primitives.pairing import pairing_backend
from pyfe4ai.primitives.pairing.pairing_backend import G1, G2
from pyfe4ai.primitives.pairing.pairing_backend import PairingGroup
from pyfe4ai.schemes.common import ac_binding
from pyfe4ai.schemes.common.ac_binding import check_binding, pp_digest
from pyfe4ai.schemes.sife import cp_ab_pairing
from pyfe4ai.schemes.sife.cp_ab_pairing import SIFECPABPairing, SIFECPABPairingKeyGenerator

UNIVERSE = ["org:A", "org:B", "org:C", "role:auditor"]
PP_DOMAIN = b"pyfe4ai/sife/cp-ab/pp-binding/v1|"
ETA = 5

SATISFIED = [
    ("org:A", ["org:A"]),                                                      # ACGU20 App. C.1 special case
    ("and(org:A, role:auditor)", ["org:A", "role:auditor"]),
    ("or(org:A, role:auditor)", ["role:auditor"]),
    ("2of(org:A, org:B, role:auditor)", ["org:B", "role:auditor"]),
    ("and(org:A, 2of(org:B, org:C, role:auditor))", ["org:A", "org:C", "role:auditor"]),
    ("or(and(org:A, org:B), and(org:A, org:C))", ["org:A", "org:C"]),          # a repeated attribute takes copies
]


@pytest.fixture(autouse=True)
def own_parameter_cache(tmp_path, monkeypatch):
    """Tests with other limits would otherwise replace each other's parameter record, with a warning each."""
    monkeypatch.setenv("PYFE4AI_CACHE_DIR", str(tmp_path))


@pytest.fixture
def make():
    def build(**overrides):
        config = {"eta": ETA, "bound_x": 10, "bound_y": 10, "universe": UNIVERSE, "pairing_group_param": GROUP}
        config.update(overrides)
        kg = SIFECPABPairingKeyGenerator(config)
        kg.setup()
        return kg, SIFECPABPairing({"keys": {"pp": kg.get_public_parameters()}})
    return build


@pytest.fixture
def fixed(make):
    """A key generator, a crypto object, a ciphertext of ``[3, -1, 0, 7, 2]`` under
    ``and(org:A, role:auditor)`` and a key that opens it, with their inner product."""
    kg, crypto = make()
    x, y = [3, -1, 0, 7, 2], [1, 2, 3, -4, 5]
    ct = crypto.encrypt(x, policy="and(org:A, role:auditor)")
    dk = kg.get_decryption_keys("s", credentials={"fusion_weight": y, "attributes": ["org:A", "role:auditor"]})
    return kg, crypto, ct, dk, sum(a * b for a, b in zip(x, y))


def rebound(pp):
    """*pp* with its ``pp_binding`` recomputed: public parameters that are consistent, but not honest."""
    return dict(pp, pp_binding=pp_digest(pp, PP_DOMAIN))


# ---------------------------------------------------------------------------
# correctness


@pytest.mark.parametrize("policy,held", SATISFIED)
def test_decrypts(policy, held, make):
    kg, crypto = make()
    x, y = [3, -1, 0, 7, 2], [1, 2, 3, -4, 5]
    ct = crypto.encrypt(x, policy=policy)
    dk = kg.get_decryption_keys("s", credentials={"fusion_weight": y, "attributes": held})
    assert crypto.decrypt(ct, dk) == sum(a * b for a, b in zip(x, y))


def test_formula_objects_are_accepted_as_policies(make):
    kg, crypto = make()
    policy = And(Leaf("org:A"), Or(Leaf("org:B"), Threshold(2, Leaf("org:C"), Leaf("role:auditor"), Leaf("org:B"))))
    ct = crypto.encrypt([1, 2, 3, 4, 5], policy=policy)
    assert ct["policy"] == canonical_encoding(policy)
    dk = kg.get_decryption_keys("s", credentials={"fusion_weight": [1, 1, 1, 1, 1], "attributes": ["org:A", "org:B"]})
    assert crypto.decrypt(ct, dk) == 15


@pytest.mark.parametrize("x,y", [([0] * ETA, [1, 2, 3, -4, 5]), ([3, -1, 0, 7, 2], [0] * ETA), ([0] * ETA, [0] * ETA)])
def test_zero_vectors(x, y, make):
    kg, crypto = make()
    ct = crypto.encrypt(x, policy="org:A")
    dk = kg.get_decryption_keys("s", credentials={"fusion_weight": y, "attributes": ["org:A"]})
    assert crypto.decrypt(ct, dk) == 0


@pytest.mark.parametrize("sign", [1, -1])
def test_extreme_inner_products(sign, make):
    kg, crypto = make()
    ct = crypto.encrypt([10] * ETA, policy="org:A")
    dk = kg.get_decryption_keys("s", credentials={"fusion_weight": [sign * 10] * ETA, "attributes": ["org:A"]})
    assert crypto.decrypt(ct, dk) == sign * ETA * 100


def test_random_formulas(make):
    rng = random.Random(8)
    labels = ["a:{}".format(i) for i in range(6)]
    kg, crypto = make(universe=labels, max_rows=12, max_columns=16)

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
    while tested < 8:
        policy = formula(2)
        leaves = policy.leaves()
        if max(leaves.count(label) for label in leaves) > 4 or len(leaves) > 12:
            continue
        try:
            validate_formula(policy)    # the children of a threshold gate must not share a label
        except FEValidationError:
            continue
        held = [label for label in labels if rng.random() < 0.6]
        x = [rng.randint(-10, 10) for _ in range(ETA)]
        y = [rng.randint(-10, 10) for _ in range(ETA)]
        ct = crypto.encrypt(x, policy=policy)
        dk = kg.get_decryption_keys("s", credentials={"fusion_weight": y, "attributes": held})
        if evaluate(policy, held):
            assert crypto.decrypt(ct, dk) == sum(a * b for a, b in zip(x, y))
        else:
            with pytest.raises(FEKeyError):
                crypto.decrypt(ct, dk)
        tested += 1


def test_labels_outside_ascii(make):
    label = "org:caf\u00e9"
    kg, crypto = make(universe=[label, "org:A"])
    ct = crypto.encrypt([1, 0, 0, 0, 0], policy=label)
    dk = kg.get_decryption_keys("s", credentials={"fusion_weight": [2, 0, 0, 0, 0], "attributes": [label]})
    assert crypto.decrypt(ct, dk) == 2


def test_objects_survive_json(fixed):
    kg, _, ct, dk, expected = fixed
    pp = json.loads(json.dumps(kg.get_public_parameters()))
    crypto = SIFECPABPairing({"keys": {"pp": pp}})
    assert crypto.decrypt(json.loads(json.dumps(ct)), json.loads(json.dumps(dk))) == expected


# ---------------------------------------------------------------------------
# access control


@pytest.mark.parametrize("policy,held", [("and(org:A, role:auditor)", ["org:A"]),
                                         ("2of(org:A, org:B, org:C)", ["org:C"]),
                                         ("org:A", [])])
def test_unsatisfied_is_refused(policy, held, make):
    kg, crypto = make()
    ct = crypto.encrypt([1] * 5, policy=policy)
    dk = kg.get_decryption_keys("s", credentials={"fusion_weight": [1] * 5, "attributes": held})
    with pytest.raises(FEKeyError):
        crypto.decrypt(ct, dk)


def test_public_key_scheme(make):
    kg, crypto = make()
    assert kg.get_private_keys("nid_0") is None
    assert set(kg.msk) == {"u"}
    assert len(kg.msk["u"]) == 2 and all(len(row) == ETA for row in kg.msk["u"])
    assert all(isinstance(value, str) for row in kg.msk["u"] for value in row)
    pp = kg.get_public_parameters()
    assert not {"u", "U", "W", "w"} & set(pp)
    # the encryption key is the public parameters
    assert crypto.keys == {"pp": pp}


def test_setup_keeps_nothing_new_on_the_key_generator():
    kg = SIFECPABPairingKeyGenerator({"eta": ETA, "universe": UNIVERSE, "pairing_group_param": GROUP})
    before = set(vars(kg))
    kg.setup()
    assert set(vars(kg)) == before      # pp and msk only change value: W_i, a2 and b2 stay local to setup


def test_the_construction_in_the_exponent(make, monkeypatch):
    """Every element of the public parameters, a ciphertext and a key against the formulas of
    the module docstring, the randomness captured from the scheme."""
    scalars, matrices = [], []
    real_scalar, real_matrix = cp_ab_pairing.random_nonzero_zr, cp_ab_pairing._rand_matrix

    def scalar(group):
        value = real_scalar(group)
        scalars.append(int(value))
        return value

    def matrix(group, rows, cols):
        value = real_matrix(group, rows, cols)
        matrices.append([[int(entry) for entry in row] for row in value])
        return value

    monkeypatch.setattr(cp_ab_pairing, "random_nonzero_zr", scalar)
    monkeypatch.setattr(cp_ab_pairing, "_rand_matrix", matrix)
    kg, crypto = make(max_copies=2, max_columns=3)
    pp = kg.get_public_parameters()
    a2, b2 = scalars
    u, *w = matrices
    a, b = (1, a2), (1, b2)

    group = PairingGroup(GROUP)
    order = int(group.order())
    universe = Universe(UNIVERSE, 2)
    assert len(w) == universe.size + 3

    def element(text):
        return group.deserialize(text.encode("utf-8"))

    g1, g2 = element(pp["g1"]), element(pp["g2"])

    def is_g1(texts, exponents):
        assert [element(text) for text in texts] == pp_many(g1, exponents)

    def is_g2(texts, exponents):
        assert [element(text) for text in texts] == pp_many(g2, exponents)

    def pp_many(base, exponents):
        return pairing_backend.pow_many(base, [value % order for value in exponents], group)

    def wt_a(i):
        return [w[i][0][t] + a2 * w[i][1][t] for t in (0, 1)]

    def w_b(i):
        return [w[i][t][0] + b2 * w[i][t][1] for t in (0, 1)]

    is_g1(pp["a"], a)
    is_g2(pp["b"], b)
    is_g1(pp["ua"], [u[0][i] + a2 * u[1][i] for i in range(ETA)])
    for i in range(len(w)):
        is_g1(pp["w_a"][i], wt_a(i))
        is_g2(pp["w_b"][i], w_b(i))

    policy = "or(and(org:A, org:B), and(org:A, org:C))"
    x, y = [3, -1, 0, 7, 2], [1, 2, 3, -4, 5]

    scalars.clear()
    ct = crypto.encrypt(x, policy=policy)
    s = scalars.pop()
    msp = compile_formula(parse(policy), order, max_rows=64)
    assert msp.num_cols == 3
    rows = read_once_rows(msp, 2)
    assert rows[2] == ("org:A", 2)
    is_g1(ct["c1"], [s * value for value in a])
    is_g1(ct["c3"], [x[i] + s * (u[0][i] + a2 * u[1][i]) for i in range(ETA)])
    for j, (label, copy_number) in enumerate(rows):
        slot = wt_a(universe.index(label, copy_number))
        columns = [wt_a(universe.size + k) for k in range(3)]
        is_g1(ct["c2"][j], [s * (slot[t] + sum(msp.matrix[j][k] * columns[k][t] for k in range(3))) for t in (0, 1)])

    scalars.clear()
    dk = kg.get_decryption_keys("s", credentials={"fusion_weight": y, "attributes": ["org:A", "org:C"]})
    r = scalars.pop()
    is_g2(dk["k1"], [r * value for value in b])
    u_y = [sum(u[t][i] * y[i] for i in range(ETA)) for t in (0, 1)]
    is_g2(dk["k0"], [u_y[t] + r * w_b(universe.size)[t] for t in (0, 1)])
    for label in ("org:A", "org:C"):
        for copy_number in (1, 2):
            entry = w_b(universe.index(label, copy_number))
            is_g2(dk["k_attr"]["{}#{}".format(label, copy_number)], [r * entry[t] for t in (0, 1)])


def test_no_key_before_setup():
    kg = SIFECPABPairingKeyGenerator({"eta": ETA, "universe": UNIVERSE, "pairing_group_param": GROUP})
    with pytest.raises(FEKeyError):
        kg.get_decryption_keys("s", credentials={"fusion_weight": [1] * ETA, "attributes": ["org:A"]})


def test_key_needs_credentials(make):
    kg, _ = make()
    with pytest.raises(FEKeyError):
        kg.get_decryption_keys("s")


@pytest.mark.parametrize("attributes", [None, "org:A", 7])
def test_key_attributes_must_be_a_collection_of_labels(attributes, make):
    kg, _ = make()
    with pytest.raises(FEValidationError, match="attributes"):
        kg.get_decryption_keys("s", credentials={"fusion_weight": [1] * ETA, "attributes": attributes})


def test_key_refuses_a_label_outside_the_universe(make):
    kg, _ = make()
    with pytest.raises(FEValidationError, match="org:Z"):
        kg.get_decryption_keys("s", credentials={"fusion_weight": [1] * ETA, "attributes": ["org:A", "org:Z"]})


def test_key_refuses_weights_above_the_bound(make):
    kg, _ = make()
    with pytest.raises(FEValidationError):
        kg.get_decryption_keys("s", credentials={"fusion_weight": [11] * ETA, "attributes": ["org:A"]})
    with pytest.raises(FEValidationError):
        kg.get_decryption_keys("s", credentials={"fusion_weight": [1] * (ETA - 1), "attributes": ["org:A"]})


def test_key_layout(make):
    kg, _ = make(max_copies=2)
    dk = kg.get_decryption_keys("s", credentials={"fusion_weight": [1, -2, 3, 0, 5],
                                                  "attributes": ["role:auditor", "org:A", "org:A"]})
    assert list(dk) == list(CPABFunctionalKey.FIELDS)
    assert dk["attributes"] == ["org:A", "role:auditor"]
    assert dk["y"] == ["1", "-2", "3", "0", "5"]
    assert set(dk["k_attr"]) == {"org:A#1", "org:A#2", "role:auditor#1", "role:auditor#2"}
    assert len(dk["k1"]) == len(dk["k0"]) == 2
    assert all(len(value) == 2 for value in dk["k_attr"].values())
    assert dk["pp_binding"] == kg.get_public_parameters()["pp_binding"]


def test_ciphertext_layout(make):
    _, crypto = make()
    policy = "or(and(org:A, org:B), and(org:A, org:C))"
    ct = crypto.encrypt([1, 2, 3, 4, 5], policy=policy)
    assert list(ct) == list(CPABCiphertext.FIELDS)
    assert len(ct["c1"]) == 2
    assert len(ct["c2"]) == 4 and all(len(row) == 2 for row in ct["c2"])
    assert len(ct["c3"]) == ETA
    assert ct["policy_compiler"] == POLICY_COMPILER


def test_digest_ignores_copies(make):
    _, crypto = make()
    policy = "or(and(org:A, org:B), and(org:A, org:C))"
    ct = crypto.encrypt([1, 2, 3, 4, 5], policy=policy)
    assert ct["policy_digest"] == policy_digest(parse(policy))
    assert ct["policy"] == canonical_encoding(parse(policy))


# ---------------------------------------------------------------------------
# the parameters and their record


def test_public_parameters_layout(make):
    kg, _ = make()
    pp = kg.get_public_parameters()
    universe = Universe(UNIVERSE, 4)
    assert list(pp) == list(CPABParameters.FIELDS)
    assert (pp["bound_x"], pp["bound_y"]) == ("10", "10")
    assert (pp["max_columns"], pp["max_rows"]) == (16, 64)
    assert pp["universe"] == universe.encoding()
    assert pp["universe_digest"] == universe.digest()
    assert pp["policy_compiler"] == POLICY_COMPILER
    assert len(pp["a"]) == len(pp["b"]) == 2
    assert len(pp["ua"]) == ETA
    assert len(pp["w_a"]) == len(pp["w_b"]) == universe.size + 16
    assert all(len(pair) == 2 for pair in pp["w_a"] + pp["w_b"])
    assert pp["pp_binding"] == pp_digest(pp, PP_DOMAIN)


def test_public_parameters_are_of_their_groups(make):
    kg, _ = make()
    pp = kg.get_public_parameters()
    group = PairingGroup(GROUP)
    for field, kind in (("a", G1), ("b", G2), ("ua", G1)):
        assert all(group.deserialize(item.encode("utf-8")).type == kind for item in pp[field])
    for field, kind in (("w_a", G1), ("w_b", G2)):
        assert all(group.deserialize(item.encode("utf-8")).type == kind for row in pp[field] for item in row)


def test_a_new_universe_replaces_the_cached_record(make, tmp_path):
    cache = str(tmp_path)
    make(cache_dir=cache)
    record_file = os.path.join(cache, "authority", CryptoCONST.TYPE_SIFE_CP_AB, "param.json")
    with open(record_file, encoding="utf-8") as f:
        record = json.load(f)
    assert list(record) == list(CPABRecord.FIELDS)
    assert record["universe"] == Universe(UNIVERSE, 4).encoding()

    other = ["org:A", "org:Z"]
    with pytest.warns(FEWarning, match="does not match this configuration"):
        make(universe=other, cache_dir=cache)
    with open(record_file, encoding="utf-8") as f:
        assert json.load(f)["universe"] == Universe(other, 4).encoding()


def test_same_configuration_keeps_the_cached_record(make, tmp_path):
    cache = str(tmp_path)
    make(cache_dir=cache)
    with warnings.catch_warnings():
        warnings.simplefilter("error", FEWarning)
        make(cache_dir=cache)


def test_missing_universe_is_a_config_error():
    with pytest.raises(FEConfigError, match="universe"):
        SIFECPABPairingKeyGenerator({"eta": ETA, "pairing_group_param": GROUP})


@pytest.mark.parametrize("option", ["eta", "bound_x", "bound_y", "max_copies", "max_columns", "max_rows"])
@pytest.mark.parametrize("value", [0, -1, True, 2.0, "5", None])
def test_limits_must_be_positive_ints(option, value):
    config = {"eta": ETA, "universe": UNIVERSE, "pairing_group_param": GROUP, option: value}
    with pytest.raises(FEValidationError, match=option):
        SIFECPABPairingKeyGenerator(config)


def test_the_discrete_log_limit_is_refused_at_construction():
    config = {"eta": 2, "bound_x": 2 ** 24, "bound_y": 2 ** 24, "universe": UNIVERSE, "pairing_group_param": GROUP}
    with pytest.raises(FEValidationError, match=r"2\*\*48"):
        SIFECPABPairingKeyGenerator(config)
    assert SIFECPABPairingKeyGenerator(dict(config, eta=1)).eta == 1        # 2**48 itself is allowed


def test_invalid_universe_is_refused():
    with pytest.raises(FEValidationError):
        SIFECPABPairingKeyGenerator({"eta": ETA, "universe": "org:A", "pairing_group_param": GROUP})


@pytest.mark.parametrize("name", ["encrypt_lst_ndarray", "compute_lst_ndarray_ct", "decrypt_lst_ndarray_ct"])
def test_ndarray_helpers_are_not_implemented(name, make):
    _, crypto = make()
    with pytest.raises(NotImplementedError):
        getattr(crypto, name)([])


# ---------------------------------------------------------------------------
# the binding of objects to their public parameters


def test_pp_digest_covers_every_field_but_the_binding():
    pp = {"a": 1, "b": ["x"], "pp_binding": "old"}
    assert pp_digest(pp, b"d|") == pp_digest(dict(pp, pp_binding="new"), b"d|")
    assert pp_digest(pp, b"d|") != pp_digest(dict(pp, a=2), b"d|")
    assert pp_digest(pp, b"d|") != pp_digest(dict(pp, c=None), b"d|")
    assert pp_digest(pp, b"d|") != pp_digest(pp, b"e|")
    assert pp_digest({"b": ["x"], "a": 1}, b"d|") == pp_digest(pp, b"d|")   # the order of the fields does not count


def test_pp_digest_is_the_sha256_of_the_canonical_json():
    pp = {"b": ["\u00e9"], "a": 1, "pp_binding": "old"}
    text = '{"a":1,"b":["\u00e9"]}'
    assert pp_digest(pp, b"d|") == hashlib.sha256(b"d|" + text.encode("utf-8")).hexdigest()


@pytest.mark.parametrize("pp", [["not a dict"], {"a": object()}, {"a": "\ud800"}, {1: 2, "a": 3}])
def test_pp_digest_refuses_what_is_not_json_data(pp):
    with pytest.raises(FEValidationError):
        pp_digest(pp, b"d|")


def test_check_binding():
    pp = {"pp_binding": "abc"}
    check_binding(pp, {"pp_binding": "abc"}, "ciphertext")
    with pytest.raises(FEValidationError, match="ciphertext.*pp_binding"):
        check_binding(pp, {"pp_binding": "abd"}, "ciphertext")
    with pytest.raises(FEValidationError, match="functional key.*pp_binding"):
        check_binding(pp, {}, "functional key")


def test_ac_binding_depends_on_the_standard_library_and_exceptions_only():
    with open(ac_binding.__file__, encoding="utf-8") as f:
        tree = ast.parse(f.read())
    names = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            names.update(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.level == 0:
            names.add(node.module)
    names.discard("__future__")
    assert names - {"pyfe4ai.core.exceptions"} <= set(sys.stdlib_module_names)


def tampered(pp, **fields):
    return rebound(dict(pp, **fields))


def test_crypto_refuses_public_parameters_whose_digest_differs(make):
    kg, _ = make()
    pp = copy.deepcopy(kg.get_public_parameters())
    pp["ua"][0] = pp["ua"][1]
    with pytest.raises(FEValidationError, match="pp_binding"):
        SIFECPABPairing({"keys": {"pp": pp}})


@pytest.mark.parametrize("field", list(CPABParameters.FIELDS))
def test_crypto_refuses_public_parameters_without_a_field(field, make):
    kg, _ = make()
    pp = dict(kg.get_public_parameters())
    del pp[field]
    with pytest.raises(FEValidationError, match=field):
        SIFECPABPairing({"keys": {"pp": pp if field == "pp_binding" else rebound(pp)}})


def test_crypto_needs_public_parameters():
    with pytest.raises(FEKeyError):
        SIFECPABPairing({"keys": {"sk": {}}})
    with pytest.raises(FEValidationError):
        SIFECPABPairing({"keys": {"pp": "not a dict"}})


def test_crypto_refuses_an_unknown_compiler(make):
    kg, _ = make()
    with pytest.raises(FEValidationError, match="c2"):
        SIFECPABPairing({"keys": {"pp": tampered(kg.get_public_parameters(), policy_compiler="c2")}})


def test_crypto_refuses_a_universe_of_another_spelling(make):
    kg, _ = make()
    pp = kg.get_public_parameters()
    with pytest.raises(FEValidationError):
        SIFECPABPairing({"keys": {"pp": tampered(pp, universe=pp["universe"].replace(",", ", ", 1))}})


def test_crypto_refuses_a_universe_digest_that_differs(make):
    kg, _ = make()
    other = Universe(["org:A"], 4)
    with pytest.raises(FEValidationError, match="universe_digest"):
        SIFECPABPairing({"keys": {"pp": tampered(kg.get_public_parameters(), universe=other.encoding())}})


def test_crypto_refuses_a_huge_max_copies_by_the_length_check(make):
    kg, _ = make()
    universe = Universe(UNIVERSE, 10 ** 30)
    pp = tampered(kg.get_public_parameters(), universe=universe.encoding(), universe_digest=universe.digest())
    with pytest.raises(FEValidationError, match="w_a"):
        SIFECPABPairing({"keys": {"pp": pp}})


@pytest.mark.parametrize("field,value", [
    ("a", ["x"]), ("b", ["x", "y", "z"]), ("ua", ["x"]), ("w_a", [["x", "y"]]), ("w_b", [["x", "y"]]),
    ("a", "ab"), ("ua", None), ("w_a", 5),
])
def test_crypto_refuses_public_parameters_of_the_wrong_length(field, value, make):
    kg, _ = make()
    with pytest.raises(FEValidationError, match=field):
        SIFECPABPairing({"keys": {"pp": tampered(kg.get_public_parameters(), **{field: value})}})


@pytest.mark.parametrize("field", ["w_a", "w_b"])
def test_crypto_refuses_a_table_entry_of_the_wrong_length(field, make):
    kg, _ = make()
    pp = copy.deepcopy(kg.get_public_parameters())
    pp[field][3] = pp[field][3][:1]
    with pytest.raises(FEValidationError, match=field):
        SIFECPABPairing({"keys": {"pp": rebound(pp)}})


@pytest.mark.parametrize("field,value", [
    ("eta", 0), ("eta", "5"), ("eta", True), ("max_columns", None), ("max_rows", -1),
    ("bound_x", "x"), ("bound_x", 10), ("bound_y", "-1"), ("bound_y", "007"), ("bound_x", str(2 ** 49)),
])
def test_crypto_refuses_limits_that_are_not_limits(field, value, make):
    kg, _ = make()
    with pytest.raises(FEValidationError):
        SIFECPABPairing({"keys": {"pp": tampered(kg.get_public_parameters(), **{field: value})}})


# ---------------------------------------------------------------------------
# encrypt


@pytest.mark.parametrize("policy", [None, 5, b"org:A", ["org:A"]])
def test_encrypt_refuses_what_is_not_a_policy(policy, make):
    _, crypto = make()
    with pytest.raises(FEValidationError, match="policy"):
        crypto.encrypt([1] * ETA, policy=policy)


def test_encrypt_without_a_policy_is_refused(make):
    _, crypto = make()
    with pytest.raises(FEValidationError, match="policy"):
        crypto.encrypt([1] * ETA)


def test_encrypt_refuses_a_plaintext_above_the_bound(make):
    _, crypto = make()
    with pytest.raises(FEValidationError):
        crypto.encrypt([11] * ETA, policy="org:A")
    with pytest.raises(FEValidationError):
        crypto.encrypt([1] * (ETA + 1), policy="org:A")


def test_encrypt_checks_the_policy_in_order(make):
    _, crypto = make(max_rows=2, max_columns=1)
    with pytest.raises(FEValidationError, match="2 rows"):
        crypto.encrypt([1] * ETA, policy="and(org:A, org:B, org:C)")
    with pytest.raises(FEValidationError, match="max_columns"):
        crypto.encrypt([1] * ETA, policy="and(org:A, org:B)")
    with pytest.raises(FEValidationError, match="org:Z"):
        crypto.encrypt([1] * ETA, policy="or(org:A, org:Z)")


def test_encrypt_refuses_more_rows_of_a_label_than_copies(make):
    _, crypto = make(max_copies=1)
    with pytest.raises(FEValidationError, match="org:A"):
        crypto.encrypt([1] * ETA, policy="or(and(org:A, org:B), and(org:A, org:C))")


def test_encrypt_refuses_a_shared_formula_with_too_many_leaves_cheaply(make):
    _, crypto = make()
    node = Leaf("org:A")
    for _ in range(31):
        node = And(node, node)
    with pytest.raises(FEValidationError, match="rows"):
        crypto.encrypt([1] * ETA, policy=node)


def test_encrypt_reads_only_the_table_entries_of_its_policy(make):
    kg, _ = make()
    pp = copy.deepcopy(kg.get_public_parameters())
    universe = Universe(UNIVERSE, 4)
    used = {universe.index("org:A", 1), universe.size}
    for index in set(range(len(pp["w_a"]))) - used:
        pp["w_a"][index] = ["junk", "junk"]
    crypto = SIFECPABPairing({"keys": {"pp": rebound(pp)}})
    assert len(crypto.encrypt([1] * ETA, policy="org:A")["c2"]) == 1


def test_encrypt_refuses_a_policy_decryption_would_refuse(make):
    # '"' takes two characters in the canonical encoding and one in the limit
    label = "a:" + '"' * 500
    _, crypto = make(universe=[label, "a:1"], max_rows=2)
    other = "a:1"
    assert crypto.encrypt([1] * ETA, policy="and({}, {})".format(other, other))
    with pytest.raises(FEValidationError, match="characters"):
        crypto.encrypt([1] * ETA, policy="and({}, {})".format(label, label))


# ---------------------------------------------------------------------------
# decrypt


def test_decrypt_checks_the_binding_before_the_policy(fixed, make):
    kg, crypto, ct, dk, _ = fixed
    bad = dict(ct, pp_binding="0" * 64, policy="[" * 100000, policy_compiler="c9")
    with pytest.raises(FEValidationError, match="pp_binding"):
        crypto.decrypt(bad, dk)
    with pytest.raises(FEValidationError, match="pp_binding"):
        crypto.decrypt(ct, dict(dk, pp_binding="0" * 64))


def test_decrypt_refuses_keys_of_another_setup_before_pairing(fixed, monkeypatch):
    kg, crypto, ct, dk, _ = fixed
    kg.setup()
    credentials = {"fusion_weight": [1] * ETA, "attributes": ["org:A", "role:auditor"]}
    other = kg.get_decryption_keys("s", credentials=credentials)
    called = []
    monkeypatch.setattr(cp_ab_pairing, "multi_pair", lambda *args: called.append(args))
    with pytest.raises(FEValidationError, match="pp_binding"):
        crypto.decrypt(ct, other)
    assert not called


def test_decrypt_refuses_an_unknown_compiler(fixed):
    _, crypto, ct, dk, _ = fixed
    with pytest.raises(FEValidationError, match="c2"):
        crypto.decrypt(dict(ct, policy_compiler="c2"), dk)


def test_decrypt_refuses_a_policy_above_the_length_limit(fixed):
    _, crypto, ct, dk, _ = fixed
    limit = 64 * (len("role:auditor") + 400)
    with pytest.raises(FEValidationError, match=str(limit)):
        crypto.decrypt(dict(ct, policy="[" * (limit + 1)), dk)


def test_a_policy_at_the_length_limit_reaches_the_parser(fixed, monkeypatch):
    _, crypto, ct, dk, _ = fixed
    limit = 64 * (len("role:auditor") + 400)
    seen = []

    def spy(data):
        seen.append(len(data))
        raise FEValidationError("stop here")

    monkeypatch.setattr(cp_ab_pairing, "from_canonical", spy)
    with pytest.raises(FEValidationError, match="stop here"):
        crypto.decrypt(dict(ct, policy="[" * limit), dk)
    assert seen == [limit]


@pytest.mark.parametrize("policy", [None, 5, ["and"]])
def test_decrypt_refuses_a_policy_that_is_no_text(policy, fixed):
    _, crypto, ct, dk, _ = fixed
    with pytest.raises(FEValidationError, match="policy"):
        crypto.decrypt(dict(ct, policy=policy), dk)


def test_decrypt_refuses_a_digest_that_differs(fixed):
    _, crypto, ct, dk, _ = fixed
    with pytest.raises(FEValidationError, match="policy_digest"):
        crypto.decrypt(dict(ct, policy=canonical_encoding(parse("or(org:A, role:auditor)"))), dk)


def test_decrypt_refuses_a_policy_with_too_many_rows(fixed):
    _, crypto, ct, dk, _ = fixed
    wide = Or(*[Leaf("org:A")] * 65)
    bad = dict(ct, policy=canonical_encoding(wide), policy_digest=policy_digest(wide))
    with pytest.raises(FEValidationError, match="rows"):
        crypto.decrypt(bad, dk)


def test_decrypt_refuses_more_rows_of_a_label_than_copies(fixed):
    _, crypto, ct, dk, _ = fixed
    wide = Or(*[Leaf("org:A")] * 5)
    bad = dict(ct, policy=canonical_encoding(wide), policy_digest=policy_digest(wide))
    with pytest.raises(FEValidationError, match="org:A"):
        crypto.decrypt(bad, dk)


@pytest.mark.parametrize("field", list(CPABCiphertext.FIELDS))
def test_decrypt_refuses_a_ciphertext_without_a_field(field, fixed):
    _, crypto, ct, dk, _ = fixed
    bad = {name: value for name, value in ct.items() if name != field}
    with pytest.raises(FEValidationError, match=field):
        crypto.decrypt(bad, dk)


@pytest.mark.parametrize("field", list(CPABFunctionalKey.FIELDS))
def test_decrypt_refuses_a_key_without_a_field(field, fixed):
    _, crypto, ct, dk, _ = fixed
    bad = {name: value for name, value in dk.items() if name != field}
    with pytest.raises(FEValidationError, match=field):
        crypto.decrypt(ct, bad)


@pytest.mark.parametrize("ct_bad", ["not a dict", None, ["c1"]])
def test_decrypt_refuses_what_is_no_ciphertext(ct_bad, fixed):
    _, crypto, _, dk, _ = fixed
    with pytest.raises(FEValidationError, match="ciphertext"):
        crypto.decrypt(ct_bad, dk)


@pytest.mark.parametrize("dk_bad", ["not a dict", None, ["k1"]])
def test_decrypt_refuses_what_is_no_key(dk_bad, fixed):
    _, crypto, ct, _, _ = fixed
    with pytest.raises(FEValidationError, match="functional key"):
        crypto.decrypt(ct, dk_bad)


@pytest.mark.parametrize("field,shape", [
    ("c1", lambda v: v[:1]), ("c1", lambda v: v + v[:1]), ("c1", lambda v: "ab"), ("c1", lambda v: None),
    ("c2", lambda v: v[:1]), ("c2", lambda v: v + v[:1]), ("c2", lambda v: [v[0], v[1][:1]]),
    ("c2", lambda v: [v[0], "ab"]), ("c2", lambda v: None),
    ("c3", lambda v: v[:-1]), ("c3", lambda v: v + v[:1]), ("c3", lambda v: None),
])
def test_decrypt_refuses_a_ciphertext_of_the_wrong_shape(field, shape, fixed):
    _, crypto, ct, dk, _ = fixed
    with pytest.raises(FEValidationError, match=field):
        crypto.decrypt(dict(ct, **{field: shape(ct[field])}), dk)


@pytest.mark.parametrize("field,shape", [
    ("k1", lambda v: v[:1]), ("k1", lambda v: None), ("k0", lambda v: v + v[:1]), ("k0", lambda v: "ab"),
    ("k_attr", lambda v: dict(v, **{"org:A#1": v["org:A#1"][:1]})),
    ("k_attr", lambda v: dict(v, **{"org:C#1": None})),
    ("k_attr", lambda v: None), ("k_attr", lambda v: list(v)),
    ("attributes", lambda v: "org:A"), ("attributes", lambda v: None), ("attributes", lambda v: [5]),
    ("y", lambda v: v[:-1]), ("y", lambda v: v + v[:1]), ("y", lambda v: None), ("y", lambda v: "1"),
    ("y", lambda v: [1] + v[1:]), ("y", lambda v: ["x"] + v[1:]), ("y", lambda v: ["1.5"] + v[1:]),
    ("y", lambda v: ["11"] + v[1:]), ("y", lambda v: ["-11"] + v[1:]), ("y", lambda v: ["007"] + v[1:]),
    ("y", lambda v: ["1" * 5000] + v[1:]), ("y", lambda v: [" 1"] + v[1:]), ("y", lambda v: ["1_0"] + v[1:]),
    ("y", lambda v: ["\u0661"] + v[1:]),
])
def test_decrypt_refuses_a_key_of_the_wrong_shape(field, shape, fixed):
    _, crypto, ct, dk, _ = fixed
    with pytest.raises(FEValidationError, match=field):
        crypto.decrypt(ct, dict(dk, **{field: shape(dk[field])}))


def test_decrypt_refuses_a_g2_string_in_a_field_of_g1_before_reading_it(fixed, monkeypatch):
    _, crypto, ct, dk, _ = fixed
    read = []
    real = cp_ab_pairing.deserialize_many

    def spy(group, items):
        read.append(items)
        return real(group, items)

    monkeypatch.setattr(cp_ab_pairing, "deserialize_many", spy)
    with pytest.raises(FEValidationError, match="c3"):
        crypto.decrypt(dict(ct, c3=dk["k1"][:1] + ct["c3"][1:]), dk)
    assert not any(dk["k1"][0] in items for items in read)


@pytest.mark.parametrize("item", ["junk", "", 5, None])
@pytest.mark.parametrize("field", ["c1", "c3"])
def test_decrypt_refuses_undecodable_ciphertext_elements(field, item, fixed):
    _, crypto, ct, dk, _ = fixed
    with pytest.raises(FEValidationError, match=field):
        crypto.decrypt(dict(ct, **{field: [item] + ct[field][1:]}), dk)


def test_decrypt_refuses_undecodable_elements_of_c2(fixed):
    _, crypto, ct, dk, _ = fixed
    with pytest.raises(FEValidationError, match="c2"):
        crypto.decrypt(dict(ct, c2=[["junk", ct["c2"][0][1]], ct["c2"][1]]), dk)


@pytest.mark.parametrize("field", ["k1", "k0"])
def test_decrypt_refuses_undecodable_key_elements(field, fixed):
    _, crypto, ct, dk, _ = fixed
    with pytest.raises(FEValidationError, match=field):
        crypto.decrypt(ct, dict(dk, **{field: ["junk", dk[field][1]]}))


def test_decrypt_refuses_undecodable_elements_of_k_attr(fixed):
    _, crypto, ct, dk, _ = fixed
    k_attr = dict(dk["k_attr"], **{"org:A#1": ["junk", dk["k_attr"]["org:A#1"][1]]})
    with pytest.raises(FEValidationError, match="k_attr"):
        crypto.decrypt(ct, dict(dk, k_attr=k_attr))


def test_decrypt_refuses_elements_of_the_wrong_group(fixed):
    _, crypto, ct, dk, _ = fixed
    with pytest.raises(FEValidationError, match="c1"):
        crypto.decrypt(dict(ct, c1=dk["k1"]), dk)
    with pytest.raises(FEValidationError, match="k1"):
        crypto.decrypt(ct, dict(dk, k1=ct["c1"]))


def test_fusion_weight_must_be_the_weights_of_the_key(fixed):
    _, crypto, ct, dk, expected = fixed
    assert crypto.decrypt(ct, dk, fusion_weight=[1, 2, 3, -4, 5]) == expected
    with pytest.raises(FEValidationError, match="fusion"):
        crypto.decrypt(ct, dk, fusion_weight=[1, 2, 3, -4, 4])
    with pytest.raises(FEValidationError, match="fusion"):
        crypto.decrypt(ct, dk, fusion_weight=[1, 2, 3, -4])


def test_a_used_row_missing_from_k_attr_is_refused(fixed):
    _, crypto, ct, dk, _ = fixed
    k_attr = {name: value for name, value in dk["k_attr"].items() if name != "role:auditor#1"}
    with pytest.raises(FEValidationError, match="role:auditor#1"):
        crypto.decrypt(ct, dict(dk, k_attr=k_attr))


def test_an_unused_row_may_be_missing_from_k_attr(fixed):
    _, crypto, ct, dk, expected = fixed
    k_attr = {name: value for name, value in dk["k_attr"].items() if name != "role:auditor#4"}
    assert crypto.decrypt(ct, dict(dk, k_attr=k_attr)) == expected


def test_decrypt_needs_the_keys_attributes_to_satisfy_the_policy_even_with_k_attr(fixed):
    _, crypto, ct, dk, _ = fixed
    with pytest.raises(FEKeyError):
        crypto.decrypt(ct, dict(dk, attributes=["org:A"]))


def test_a_failed_discrete_log_is_a_value_error(fixed):
    _, crypto, ct, dk, _ = fixed
    bad = dict(ct, c3=list(reversed(ct["c3"])))
    with pytest.raises(ValueError, match="discrete log") as caught:
        crypto.decrypt(bad, dk)
    assert not isinstance(caught.value, FEValidationError)


def test_multi_pair_is_called_once(fixed, monkeypatch):
    _, crypto, ct, dk, expected = fixed
    calls = []
    real = pairing_backend.multi_pair

    def spy(lhs, rhs):
        calls.append(len(lhs))
        return real(lhs, rhs)

    monkeypatch.setattr(cp_ab_pairing, "multi_pair", spy)
    assert crypto.decrypt(ct, dk) == expected
    assert calls == [5]


def test_create_and_list():
    kg, crypto_cls = pyfe4ai.create("sife/cp_ab_pairing", eta=2, universe=["a:1"], pairing_group_param=GROUP)
    assert (type(kg).__name__, crypto_cls.__name__) == ("SIFECPABPairingKeyGenerator", "SIFECPABPairing")
    assert pyfe4ai.list_schemes("sife", capabilities="access_control") == ["sife/cp_ab_pairing"]


def test_missing_universe():
    with pytest.raises(FEConfigError, match="universe"):
        pyfe4ai.create("sife/cp_ab_pairing", eta=2, pairing_group_param=GROUP)
