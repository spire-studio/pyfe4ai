"""The opt-in exact discrete Gaussian sampler (FACCT-style double sampling)."""

import math
import os
import statistics
import subprocess
import sys
import threading
from pathlib import Path

import gmpy2 as gp
import numpy as np
import pytest

from pyfe4ai.primitives import sampling as su
from pyfe4ai.core.exceptions import FEValidationError
from pyfe4ai.ml import FESchemeWrapper, aggregate_gradients, encrypt_gradient

ROOT = Path(__file__).resolve().parents[2]


def _rng(tag: str) -> su.XofRandom:
    return su.XofRandom(b"pyfe4ai/facct-test/" + tag.encode() + b"\x00" * 16)


@pytest.fixture(autouse=True)
def _rounded_default(monkeypatch):
    """Start every test from the shipped default, whatever the environment says."""
    monkeypatch.setattr(su, "_default_sampler", "rounded")


# ------------------------------------------------------------------ selection

def test_default_is_rounded_and_context_selects_facct(monkeypatch):
    calls = []
    original = su.discrete_gaussian_facct
    monkeypatch.setattr(su, "discrete_gaussian_facct", lambda *a, **k: calls.append(a) or original(*a, **k))
    assert su.get_gaussian_sampler() == "rounded"
    su.discrete_gaussian_matrix(2, 3, 3.2)
    assert not calls
    with su.gaussian_sampler("facct"):
        assert su.get_gaussian_sampler() == "facct"
        m = su.discrete_gaussian_matrix(2, 3, 3.2)
        su.discrete_gaussian_bigint(2 ** 80, 2)
    assert su.get_gaussian_sampler() == "rounded"
    assert len(calls) == 2 and [len(row) for row in m] == [3, 3]
    assert su.discrete_gaussian_matrix(1, 4, 3.2, sampler="facct")[0][0] is not None
    assert len(calls) == 3


def test_set_default_and_reject_unknown_names():
    su.set_gaussian_sampler("facct")
    assert su.get_gaussian_sampler() == "facct"
    with su.gaussian_sampler("rounded"):
        assert su.get_gaussian_sampler() == "rounded"
    for bad in (lambda: su.set_gaussian_sampler("exact"),
                lambda: su.gaussian_sampler("FACCT").__enter__(),
                lambda: su.discrete_gaussian_matrix(1, 1, 3.2, sampler="box-muller")):
        with pytest.raises(FEValidationError):
            bad()


def test_environment_variable_sets_the_default():
    code = "from pyfe4ai.primitives.sampling import get_gaussian_sampler as g; print(g())"
    env = dict(os.environ, PYFE4AI_GAUSSIAN_SAMPLER="facct", PYTHONPATH=str(ROOT))
    out = subprocess.run([sys.executable, "-c", code], env=env, capture_output=True, text=True, check=True)
    assert out.stdout.strip() == "facct"


# ------------------------------------------------------------------ exactness

def _exact_output_distribution(sigma):
    """Probability of every output of discrete_gaussian_facct, from its own tables."""
    p = su.FACCT_PRECISION
    params = su._facct_params(gp.mpq(sigma))
    k = params.k
    bounds = (0,) + params.cdt + (1 << p,)
    with gp.context(precision=p + 64, round=gp.RoundToNearest):
        thresholds = {
            (x, y): (1 << p) if y == 0 else int(gp.floor(gp.exp(params.neg_c * (y * (y + 2 * k * x))) * (1 << p)))
            for x in range(len(bounds) - 1) for y in range(k)
        }
    with gp.context(precision=800):
        half = {}
        for x in range(len(bounds) - 1):
            p_x = gp.mpfr(bounds[x + 1] - bounds[x]) / (1 << p)
            for y in range(k):
                half[k * x + y] = p_x * thresholds[x, y] / (gp.mpfr(k) * (1 << p))
        out = {0: half[0] / 2}
        for z, w in half.items():
            if z:
                out[z] = out[-z] = w / 2  # sign bit; z = 0 with sign 1 restarts
        total = sum(out.values())
        return {z: w / total for z, w in out.items()}


@pytest.mark.parametrize("sigma", [0.3, 1.0, 3.2, 7.5])
def test_tables_give_the_discrete_gaussian(sigma):
    """Statistical distance of the sampler's exact output law to D_{Z,sigma}."""
    law = _exact_output_distribution(sigma)
    with gp.context(precision=800):
        s = gp.mpfr(sigma)
        span = int(40 * sigma) + 40
        rho = {z: gp.exp(-gp.mpfr(z * z) / (2 * s * s)) for z in range(-span, span + 1)}
        norm = sum(rho.values())
        dist = sum(abs(law.get(z, 0) - r / norm) for z, r in rho.items())
        dist += sum(w for z, w in law.items() if z not in rho)
        assert dist / 2 < gp.mpfr(2) ** -180


def test_width_is_exact_and_base_width_small():
    for sigma in (0.3, 1.0, 3.2, 4030.5, 4.97e9, 2 ** 300 + 7):
        k, base = su.facct_base_width(sigma)
        assert k >= 1 and base <= 1 / math.sqrt(2 * math.log(2)) + 1e-12
        assert abs(gp.mpfr(k) * base / gp.mpfr(gp.mpq(sigma)) - 1) < 1e-12


# ------------------------------------------------------------------ known answers

# Bit source: XofRandom(b"pyfe4ai/facct-kat"); 12 samples per width. The tables
# and thresholds are correctly rounded MPFR values, so these are the same on
# every platform; another implementation consuming getrandbits in the
# documented order must reproduce them.
KAT = {
    3.2: [1, 4, -3, 3, -1, 2, -7, -1, 3, 3, -2, 1],
    4030.5: [2942, 1457, 1249, -3075, 4103, -5067, 2025, -4900, -4051, -2138, 822, -1568],
    4.97e9: [
        3085787851, 1527969198, 1310312754, -3224513143, 4303184549, -6188400009,
        1349982059, 7975165662, -6014219869, 5706443305, 2734212516, 6357263064,
    ],
    2 ** 100 + 1: [
        910763820332433964543431243192, 450976908057451478613641509872,
        386736064532375229120881263690, -951708299576399242717959480008,
        1270075905519499994230018304961, -1591912615652740139815272300354,
        626728233439438577842292085860, -1540503712145949095876785146091,
        -1253832672520891840862765721208, -661746640258382582253948330779,
        254685613136526993496674043516, -485496054380200519804711133740,
    ],
}


@pytest.mark.parametrize("sigma", list(KAT))
def test_known_answers(sigma):
    got = [int(v) for v in su.discrete_gaussian_facct(sigma, 12, rng=su.XofRandom(b"pyfe4ai/facct-kat"))]
    assert got == KAT[sigma]


def test_seeded_noise_is_reproducible_and_sampler_specific():
    a = su.discrete_gaussian_bigint(2 ** 90, 5, rng=_rng("seed"), sampler="facct")
    b = su.discrete_gaussian_bigint(2 ** 90, 5, rng=_rng("seed"), sampler="facct")
    c = su.discrete_gaussian_bigint(2 ** 90, 5, rng=_rng("seed"), sampler="rounded")
    assert a == b and a != c


# ------------------------------------------------------------------ statistics

def test_chi_square_at_small_width():
    sigma, n = 3.2, 30000
    samples = [int(v) for v in su.discrete_gaussian_facct(sigma, n, rng=_rng("chi2"))]
    edge = 10
    observed = {}
    for v in samples:
        key = max(-edge, min(edge, v))
        observed[key] = observed.get(key, 0) + 1
    rho = {z: math.exp(-z * z / (2 * sigma ** 2)) for z in range(-60, 61)}
    norm = sum(rho.values())
    expected = {}
    for z, r in rho.items():
        key = max(-edge, min(edge, z))
        expected[key] = expected.get(key, 0) + n * r / norm
    chi2 = sum((observed.get(z, 0) - e) ** 2 / e for z, e in expected.items())
    df = len(expected) - 1
    assert chi2 < df + 6 * math.sqrt(2 * df)


@pytest.mark.parametrize("sigma,n", [(4030.5, 20000), (4.97e9, 20000), (2 ** 200, 3000)])
def test_moments_and_low_bits(sigma, n):
    samples = [int(v) for v in su.discrete_gaussian_facct(sigma, n, rng=_rng(str(sigma)))]
    s = float(sigma)
    assert abs(statistics.fmean(samples)) < 5 * s / math.sqrt(n)
    assert abs(statistics.pvariance([v / s for v in samples]) - 1) < 5 * math.sqrt(2 / n)
    assert abs(sum(v & 1 for v in samples) / n - 0.5) < 5 * 0.5 / math.sqrt(n)
    assert max(abs(v) for v in samples) < 10 * s


# ------------------------------------------------------------------ randomness

def test_buffered_source_never_hands_out_the_same_bytes_twice():
    draws, lock = [], threading.Lock()

    def worker():
        local = [su._OS_RANDOM.getrandbits(128) for _ in range(2000)]
        with lock:
            draws.extend(local)

    threads = [threading.Thread(target=worker) for _ in range(8)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    assert len(set(draws)) == len(draws) == 16000


@pytest.mark.skipif(not hasattr(os, "fork"), reason="needs fork")
def test_forked_child_does_not_reuse_the_parent_buffer():
    su._OS_RANDOM.getrandbits(8)  # fill the buffer before forking
    read, write = os.pipe()
    pid = os.fork()
    if pid == 0:  # child
        os.close(read)
        os.write(write, su._OS_RANDOM.getrandbits(256).to_bytes(32, "big"))
        os._exit(0)
    os.close(write)
    child = int.from_bytes(os.read(read, 32), "big")
    os.close(read)
    os.waitpid(pid, 0)
    assert child != su._OS_RANDOM.getrandbits(256)


# ------------------------------------------------------------------ schemes

@pytest.mark.parametrize("scheme,config", [
    (("mcfe", "ring_lwe"), {"n": 3, "eta": 1, "bound_x": 100, "bound_y": 1}),  # discrete_gaussian_matrix
    (("mcfe", "paillier"), {"n": 3, "eta": 1, "bound_x": 100, "bound_y": 1}),  # discrete_gaussian_bigint
])
def test_schemes_decrypt_with_facct(tmp_path, monkeypatch, scheme, config):
    monkeypatch.chdir(tmp_path)  # parameter caches go to tmp_path/config
    monkeypatch.setenv("PYFE4AI_CACHE_DIR", os.path.join(tmp_path, "config"))
    used = []
    original = su.discrete_gaussian_facct
    monkeypatch.setattr(su, "discrete_gaussian_facct", lambda *a, **k: used.append(a) or original(*a, **k))
    clients = ["a", "b", "c"]
    with su.gaussian_sampler("facct"):
        wrapper = FESchemeWrapper(*scheme, {**config, "lst_nid": clients}, precision=2)
        rng = np.random.default_rng(3)
        grads = {c: rng.uniform(-1, 1, 40).round(2) for c in clients}
        enc = {c: encrypt_gradient(g, wrapper, nid=c, label="r0") for c, g in grads.items()}
        total = aggregate_gradients(enc, wrapper, {c: 1 for c in clients}, label="r0")
    assert used  # the scheme drew its Gaussians from the exact sampler
    assert np.allclose(total, sum(grads.values()))
