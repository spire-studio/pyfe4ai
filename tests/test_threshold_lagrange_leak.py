"""Regression tests: honest partial decryptions of the threshold LWE layers
must be simulatable from the result and ``t - 1`` corrupted key shares.

With ``t - 1`` servers ``C`` corrupted, an honest partial of a Shamir sharing
is ``lambda_k^E F(k) + e_k`` with ``F(k) = L_{k,0} F(0) + sum_{j in C} L_{k,j}
F(j)`` and ``F(0) = c_y - Delta mu - nu``: computable from the result and
``C``'s shares up to ``e_k - M nu``, ``M = lambda_k^E L_{k,0}``. For ``s = 3,
t = 2``, ``C = {2}``, ``E = {1, 3}``, ``k = 1``: ``M = 3/4``, and the residue
mod 4 of four times the difference is ``nu mod 4``, whatever the flooding.
Replicated sharing, the default for ``t < s``, makes ``M`` 0 or 1. The
tests check the simulation itself (difference within the flooding range) and
the residue attack, for the default and for ``sharing = "replicated"``; the
explicit ``"shamir"`` cases (the sharing of v0.3.0, which now warns under
``toy`` and is refused under ``standard``) are expected failures.

Only names that exist in v0.3.0 / ``main`` 810a4a5 are imported, so that there
these tests fail on the leak itself (``sharing`` is then an ignored config
entry).
"""

import random
from fractions import Fraction
from itertools import combinations

import gmpy2 as gp
import pytest

from pyfe4ai.schemes.mcfe.lwe_threshold import ThresholdMCFELWE, ThresholdMCFELWEKeyGenerator
from pyfe4ai.schemes.mcfe.ring_lwe import MCFERingLWEKeyGenerator
from pyfe4ai.schemes.mcfe.ring_lwe_threshold import ThresholdMCFERingLWE, ThresholdMCFERingLWEKeyGenerator
from pyfe4ai.schemes.mife.lwe import MIFELWEKeyGenerator
from pyfe4ai.schemes.mife.lwe_threshold import ThresholdMIFELWE, ThresholdMIFELWEKeyGenerator
from pyfe4ai.utils.matrix_utils import digits_to_vector
from pyfe4ai.utils.ring_lwe_utils import poly_add, poly_mul

SIDS = ["s1", "s2", "s3"]  # points 1, 2, 3
SHARINGS = [
    pytest.param(None, id="default"),
    "replicated",
    pytest.param("shamir", marks=pytest.mark.xfail(strict=True, reason="Shamir with t < s: rational multipliers")),
]


def _with_sharing(cfg, sharing):
    return cfg if sharing is None else dict(cfg, sharing=sharing)


def _center(v, q):
    v = gp.mpz(v) % q
    return v - q if v > q // 2 else v


def _rational_mod(r: Fraction, q):
    return gp.mpz(r.numerator) * gp.invert(gp.mpz(r.denominator) % q, q) % q


def _sim(dks, corrupted, enrolled, k, f0, key_part, q, t):
    """Server *k*'s partial for *enrolled* minus its noise, as computed by the
    coalition *corrupted* (its key material) and a simulator knowing only
    ``f0 = c_y - Delta mu`` in place of ``F(0) = f0 - nu`` (vectors mod q)."""
    pts = {sid: i + 1 for i, sid in enumerate(SIDS)}
    xs, xk = [pts[sid] for sid in enrolled], pts[k]
    if "pieces" in dks[corrupted[0]]:
        # replicated: the coalition holds every piece but r_C; r_C = F(0) - the others
        known = {}
        for sid in corrupted:
            for piece in dks[sid]["pieces"]:
                known[tuple(piece["set"])] = key_part(piece["share"])
        sets = list(combinations(sorted(pts.values()), t - 1))
        missing = [T for T in sets if T not in known]
        assert len(missing) == 1
        rest = [known[T] for T in sets if T != missing[0]]
        known[missing[0]] = [(a - sum(col)) % q for a, *col in zip(f0, *rest)]
        assigned = [T for T in sets if min(x for x in xs if x not in T) == xk]
        return [sum(col) % q for col in zip([0] * len(f0), *[known[T] for T in assigned])]
    # Shamir: F(k) interpolated at k from F(0) and the coalition's shares
    lam = Fraction(1)
    for x in xs:
        if x != xk:
            lam *= Fraction(x, x - xk)
    cpts = [pts[sid] for sid in corrupted]
    l0 = Fraction(1)
    for j in cpts:
        l0 *= Fraction(xk - j, 0 - j)
    out = [_rational_mod(lam * l0, q) * v for v in f0]
    for sid, j in zip(corrupted, cpts):
        lj = Fraction(xk - 0, j - 0)
        for i in cpts:
            if i != j:
                lj *= Fraction(xk - i, j - i)
        out = [(a + _rational_mod(lam * lj, q) * b) % q for a, b in zip(out, key_part(dks[sid]))]
    return out


def _configs():
    """Every (C, E, k): |C| = t - 1 = 1, |E| >= 2, k in E outside C (s = 3, t = 2)."""
    for c in SIDS:
        for size in (2, 3):
            for enrolled in combinations(SIDS, size):
                for k in enrolled:
                    if k != c:
                        yield [c], list(enrolled), k


# ------------------------------------------------------------------ tMIFE-LWE

def _tmife(sharing):
    cfg = {"sec_param": 64, "eta": 1, "n": 2, "s": 3, "t": 2, "lst_nid": ["c0", "c1"],
           "lst_sid": SIDS, "lwe_n": 16, "bound_x": 6, "bound_y": 6}
    kg = ThresholdMIFELWEKeyGenerator(_with_sharing(cfg, sharing))
    kg.setup()
    return cfg, kg, kg.get_public_parameters()


def _tmife_round(cfg, kg, pp, enrolled_sets):
    y = {nid: [random.randint(-6, 6)] for nid in cfg["lst_nid"]}
    x = {nid: [random.randint(-6, 6)] for nid in cfg["lst_nid"]}
    cts = {nid: ThresholdMIFELWE({"id": nid, "keys": {"pp": pp, "sk": kg.get_private_keys(nid)}}).encrypt(x[nid])
           for nid in cfg["lst_nid"]}
    dks = {sid: kg.get_decryption_keys(sid, credentials={"fusion_weight": y}) for sid in SIDS}
    parts = {}
    for enrolled in enrolled_sets:
        group = {sid: ThresholdMIFELWE({"id": sid, "keys": {"pp": pp, "dk": dks[sid]}}).share_decrypt(
            cts, y, dks[sid], enrolled) for sid in enrolled}
        assert ThresholdMIFELWE({"id": "c", "keys": {"pp": pp}}).combine_decrypt(group) == sum(
            x[n][0] * y[n][0] for n in x)
        parts[tuple(enrolled)] = group
    q = gp.mpz(pp["q"])
    delta = q // gp.mpz(pp["p"])
    mu = sum(x[n][0] * y[n][0] for n in x)

    def key_part(material):
        v = delta * gp.mpz(material["z"])
        for nid, ct in cts.items():
            v += sum(gp.mpz(a) * gp.mpz(b) for a, b in zip(digits_to_vector(ct["ct0"]), material["sk_y"][nid]))
        return [v % q]

    c_y = gp.mpz(next(iter(next(iter(parts.values())).values()))["c_y"])
    f0 = [(c_y - delta * mu) % q]
    full = MIFELWEKeyGenerator.get_decryption_keys(kg, SIDS[0], credentials={"fusion_weight": y})
    nu = _center(f0[0] - key_part(full)[0], q)
    return dks, parts, key_part, f0, nu, q


@pytest.mark.parametrize("sharing", SHARINGS)
def test_tmife_lwe_residue_attack_reads_nothing(sharing):
    """``scratchpad/scripts/lagrange_leak.py``: server at point 2 corrupted,
    ``E = {1, 3}``, four times the honest partial of server 1 minus its
    simulation; on Shamir its residue mod 4 is ``nu mod 4`` in every trial."""
    cfg, kg, pp = _tmife(sharing)
    sigma_f = int(pp["sigma_f"])
    leaked, residues, diffs = [], [], []
    for _ in range(20):
        dks, parts, key_part, f0, nu, q = _tmife_round(cfg, kg, pp, [["s1", "s3"]])
        diff = _center(gp.mpz(parts[("s1", "s3")]["s1"]["w"]) - _sim(dks, ["s2"], ["s1", "s3"], "s1", f0, key_part, q, 2)[0], q)
        diffs.append(diff)
        leaked.append(int(_center(4 * diff, q)) % 4)
        residues.append(int(nu) % 4)
    hits = sum(a == b for a, b in zip(leaked, residues))
    # the attack reads a constant, not nu mod 4 (which does vary)
    assert leaked == [0] * len(leaked), "nu mod 4 read off in {} of {} trials".format(hits, len(leaked))
    assert len(set(residues)) > 1
    assert max(abs(d) for d in diffs) <= 13 * sigma_f


@pytest.mark.parametrize("sharing", SHARINGS)
def test_tmife_lwe_honest_partials_are_simulatable(sharing):
    """For every corrupted server, enrolled set and honest member, the honest
    partial is its simulation plus ``e_k - M nu``, ``M`` 0 or 1."""
    cfg, kg, pp = _tmife(sharing)
    sigma_f = int(pp["sigma_f"])
    configs = list(_configs())
    sets = sorted({tuple(e) for _, e, _ in configs})
    for _ in range(4):
        dks, parts, key_part, f0, nu, q = _tmife_round(cfg, kg, pp, [list(e) for e in sets])
        for corrupted, enrolled, k in configs:
            diff = _center(gp.mpz(parts[tuple(enrolled)][k]["w"]) - _sim(dks, corrupted, enrolled, k, f0, key_part, q, 2)[0], q)
            assert abs(diff) <= 13 * sigma_f, (corrupted, enrolled, k)
            # and the shift is nu or nothing: no fraction of nu
            assert any(abs(diff + m * nu) <= 12 * sigma_f for m in (0, 1))


# ------------------------------------------------------------- tMCFE-Ring-LWE

def _tmcfe_ring(sharing):
    cfg = {"sec_param": 64, "eta": 1, "n": 3, "s": 3, "t": 2, "lst_nid": ["a", "b", "c"],
           "lst_sid": SIDS, "ring_n": 16, "bound_x": 4, "bound_y": 4}
    kg = ThresholdMCFERingLWEKeyGenerator(_with_sharing(cfg, sharing))
    kg.setup()
    return cfg, kg, kg.get_public_parameters()


def _tmcfe_ring_round(cfg, kg, pp, enrolled_sets, label):
    k_len = 5
    x = {nid: [[random.randint(-4, 4) for _ in range(k_len)]] for nid in cfg["lst_nid"]}
    y = {nid: [random.randint(-4, 4)] for nid in cfg["lst_nid"]}
    cts = {nid: ThresholdMCFERingLWE({"id": nid, "keys": {"pp": pp, "sk": kg.get_private_keys(nid)}}).encrypt(x[nid], label)
           for nid in cfg["lst_nid"]}
    cred = {"fusion_weight": y, "label": label}
    dks = {sid: kg.get_decryption_keys(sid, credentials=cred) for sid in SIDS}
    expected = [sum(x[nid][0][j] * y[nid][0] for nid in x) for j in range(k_len)]
    parts = {}
    for enrolled in enrolled_sets:
        group = {sid: ThresholdMCFERingLWE({"id": sid, "keys": {"pp": pp, "dk": dks[sid]}}).share_decrypt(
            cts, cred, dks[sid], enrolled) for sid in enrolled}
        assert ThresholdMCFERingLWE({"id": "c", "keys": {"pp": pp}}).combine_decrypt(group) == expected
        parts[tuple(enrolled)] = group
    q = gp.mpz(pp["q"])
    n = pp["ring_n"]
    delta = q // gp.mpz(pp["p"])
    mu = [gp.mpz(v) for v in expected] + [gp.mpz(0)] * (n - k_len)

    def key_part(material):
        acc = [delta * gp.mpz(v) % q for v in material["z"]]
        for nid, ct in cts.items():
            acc = poly_add(acc, poly_mul([gp.mpz(v) for v in ct["ct1"]], [gp.mpz(v) for v in material["sk_y"][nid]], q), q)
        return acc

    c_y = [gp.mpz(v) for v in next(iter(next(iter(parts.values())).values()))["c_y"]]
    f0 = [(a - delta * b) % q for a, b in zip(c_y, mu)]
    full = MCFERingLWEKeyGenerator.get_decryption_keys(kg, SIDS[0], credentials=cred)
    nu = [_center(a - b, q) for a, b in zip(f0, key_part(full))]
    return dks, parts, key_part, f0, nu, q


@pytest.mark.parametrize("sharing", SHARINGS)
def test_tmcfe_ring_lwe_residue_attack_reads_nothing(sharing):
    """``scratchpad/scripts/lagrange_leak_rlwe.py``: as the tMIFE test, per
    coefficient (on Shamir: ``nu mod 4`` for every coefficient)."""
    cfg, kg, pp = _tmcfe_ring(sharing)
    sigma_f = int(pp["sigma_f"])
    leaked, residues, diffs = [], [], []
    for trial in range(10):
        dks, parts, key_part, f0, nu, q = _tmcfe_ring_round(cfg, kg, pp, [["s1", "s3"]], "L%d" % trial)
        sim = _sim(dks, ["s2"], ["s1", "s3"], "s1", f0, key_part, q, 2)
        for w, s_, n_ in zip(parts[("s1", "s3")]["s1"]["w"], sim, nu):
            diff = _center(gp.mpz(w) - s_, q)
            diffs.append(diff)
            leaked.append(int(_center(4 * diff, q)) % 4)
            residues.append(int(n_) % 4)
    hits = sum(a == b for a, b in zip(leaked, residues))
    assert leaked == [0] * len(leaked), "nu mod 4 read off for {} of {} coefficients".format(hits, len(leaked))
    assert len(set(residues)) > 1
    assert max(abs(d) for d in diffs) <= 13 * sigma_f


@pytest.mark.parametrize("sharing", SHARINGS)
def test_tmcfe_ring_lwe_honest_partials_are_simulatable(sharing):
    cfg, kg, pp = _tmcfe_ring(sharing)
    sigma_f = int(pp["sigma_f"])
    configs = list(_configs())
    sets = sorted({tuple(e) for _, e, _ in configs})
    for trial in range(2):
        dks, parts, key_part, f0, nu, q = _tmcfe_ring_round(cfg, kg, pp, [list(e) for e in sets], "S%d" % trial)
        for corrupted, enrolled, k in configs:
            sim = _sim(dks, corrupted, enrolled, k, f0, key_part, q, 2)
            diffs = [_center(gp.mpz(w) - s_, q) for w, s_ in zip(parts[tuple(enrolled)][k]["w"], sim)]
            assert max(abs(d) for d in diffs) <= 13 * sigma_f, (corrupted, enrolled, k)
            assert any(all(abs(d + m * n_) <= 12 * sigma_f for d, n_ in zip(diffs, nu)) for m in (0, 1))


# ------------------------------------------------------------------ tMCFE-LWE

@pytest.mark.parametrize("sharing", SHARINGS)
def test_tmcfe_lwe_honest_answers_carry_the_key_term_once(sharing):
    """tMCFE-LWE (computational argument): given the coalition's key material,
    an honest answer is ``known + M <sk_y, H(l)> + e_k``. The argument needs
    ``M`` 0 or 1 and one answer with ``M = 1`` per enrolled set; on Shamir
    ``M`` takes the values 3/4, 1/4, ... and the answers of one enrolled set
    fix ``sum_E e_k`` modulo their common denominator."""
    cfg = {"sec_param": 64, "eta": 1, "n": 3, "s": 3, "t": 2, "lst_nid": ["c0", "c1", "c2"],
           "lst_sid": SIDS, "lwe_n": 16, "bound_x": 6, "bound_y": 6, "bound_u": 2,
           "label_modulus": 8}
    kg = ThresholdMCFELWEKeyGenerator(_with_sharing(cfg, sharing))
    kg.setup()
    pp = kg.get_public_parameters()
    q = gp.mpz(pp["q"])
    sigma_f = int(pp["sigma_f"])
    y = {nid: [random.randint(-6, 6)] for nid in cfg["lst_nid"]}
    x = {nid: [random.randint(-6, 6)] for nid in cfg["lst_nid"]}
    configs = list(_configs())
    for trial in range(3):
        label = "T%d" % trial
        cts = {nid: ThresholdMCFELWE({"id": nid, "keys": {"pp": pp, "sk": kg.get_private_keys(nid)}}).encrypt(x[nid], label)
               for nid in cfg["lst_nid"]}
        cred = {"fusion_weight": y, "label": label}
        dks = {sid: kg.get_decryption_keys(sid, credentials=cred) for sid in SIDS}
        server = ThresholdMCFELWE({"id": "c", "keys": {"pp": pp}})
        h = server._hash_label(label)

        def key_part(material):
            return [sum(gp.mpz(a) * gp.mpz(b) for a, b in zip(material["sk_y"], h)) % q]

        key_term = key_part({"sk_y": kg._aggregate_key(y)})[0]
        for enrolled in sorted({tuple(e) for _, e, _ in configs}):
            group = {sid: ThresholdMCFELWE({"id": sid, "keys": {"pp": pp, "dk": dks[sid]}}).share_decrypt(
                cts, cred, dks[sid], list(enrolled)) for sid in enrolled}
            assert server.combine_decrypt(group) is not None
            for corrupted in SIDS:
                ones = 0
                for k in enrolled:
                    if k == corrupted:
                        continue
                    rest = _center(gp.mpz(group[k]["w"]) - _sim(dks, [corrupted], list(enrolled), k, [0], key_part, q, 2)[0], q)
                    m = [m for m in (0, 1) if abs(_center(rest - m * key_term, q)) <= 12 * sigma_f]
                    assert m, (corrupted, enrolled, k)
                    ones += m[0]
                assert ones == 1, (corrupted, enrolled)
