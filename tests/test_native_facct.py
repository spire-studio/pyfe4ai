"""The native exact sampler (FACCT) against the Python reference of sampling_utils."""

import os
import random

import gmpy2 as gp
import numpy as np
import pytest

from pyfe4ai.schemes.mcfe import ring_lwe as mcfe_rlwe
from pyfe4ai.schemes.mife import ring_lwe as mife_rlwe
from pyfe4ai import backend as native_backend
from pyfe4ai.primitives import sampling as su

pytestmark = pytest.mark.skipif(
    native_backend.ext() is None or not hasattr(native_backend.ext(), "FacctTable"),
    reason="Rust extension with FACCT not built",
)

KAT_SEED = b"pyfe4ai/facct-kat"
KAT = {  # tests/test_sampling_facct.py, the widths whose samples fit an i64
    3.2: [1, 4, -3, 3, -1, 2, -7, -1, 3, 3, -2, 1],
    4030.5: [2942, 1457, 1249, -3075, 4103, -5067, 2025, -4900, -4051, -2138, 822, -1568],
    4.97e9: [
        3085787851, 1527969198, 1310312754, -3224513143, 4303184549, -6188400009,
        1349982059, 7975165662, -6014219869, 5706443305, 2734212516, 6357263064,
    ],
}
WIDTHS = [0.3, 1.0, 3.2, 17.25, 4030.5, 4.97e9, 2.0 ** 40]


@pytest.fixture(autouse=True)
def _cwd_auto(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("PYFE4AI_CACHE_DIR", os.path.join(tmp_path, "config"))
    with native_backend.backend("auto"):
        yield


@pytest.mark.parametrize("sigma", list(KAT))
def test_known_answers(sigma):
    assert native_backend.ext().facct_sample(native_backend.facct_table(sigma), 12, xof_seed=KAT_SEED) == KAT[sigma]


@pytest.mark.parametrize("sigma", WIDTHS)
def test_same_samples_as_python(sigma):
    seed = b"native-facct/" + str(sigma).encode()
    py = [int(v) for v in su.discrete_gaussian_facct(sigma, 4000, rng=su.XofRandom(seed))]
    assert native_backend.ext().facct_sample(native_backend.facct_table(sigma), 4000, xof_seed=seed) == py


def _table(sigma, counter, c_exp_shift=0):
    """facct_table with a counting exact threshold (and optionally a useless c);
    the counter is cleared after the table's own check at construction."""
    p = su._facct_params(su._exact(sigma))
    mant, exp = p.neg_c.as_mantissa_exp()
    mant, exp = -int(mant), int(exp)
    shift = mant.bit_length() - 128
    mant = mant >> shift if shift > 0 else mant << -shift

    def exact(x, y):
        counter.append((x, y))
        with gp.context(precision=su.FACCT_PRECISION + 64, round=gp.RoundToNearest):
            return int(gp.floor(gp.exp(p.neg_c * (y * (y + 2 * p.k * x))) * (1 << su.FACCT_PRECISION)))

    table = native_backend.ext().FacctTable(p.k, p.y_bits, list(p.cdt), mant, exp + shift + c_exp_shift, exact)
    counter.clear()
    return table


@pytest.mark.parametrize("sigma", [3.2, 4030.5, 4.97e9])
def test_fixed_point_test_decides_alone(sigma):
    """The exact Python threshold is for about one test in 2^99: never here."""
    calls = []
    native_backend.ext().facct_sample(_table(sigma, calls), 50000)
    assert calls == []


@pytest.mark.parametrize("sigma", [3.2, 4030.5])
def test_exact_fallback_gives_the_same_samples(sigma):
    """A table whose fixed-point test always defers: every decision is exact."""
    calls = []
    seed = b"fallback/" + str(sigma).encode()
    nat = native_backend.ext().facct_sample(_table(sigma, calls, c_exp_shift=200), 500, xof_seed=seed)
    assert calls  # deferred
    assert nat == [int(v) for v in su.discrete_gaussian_facct(sigma, 500, rng=su.XofRandom(seed))]


def test_python_errors_of_the_exact_threshold_propagate():
    p = su._facct_params(su._exact(3.2))

    def broken(x, y):
        raise RuntimeError("boom")

    table = native_backend.ext().FacctTable(p.k, p.y_bits, list(p.cdt), 1, 10 ** 6, broken)
    with pytest.raises(RuntimeError, match="boom"):
        native_backend.ext().facct_sample(table, 100)


# ------------------------------------------------------------- encryption

class _StreamBits:
    """``getrandbits`` of _OSRandomBuffer over a fixed byte string."""

    def __init__(self, data: bytes) -> None:
        self.data, self.pos = data, 0

    def getrandbits(self, k: int) -> int:
        if k <= 0:
            return 0
        n = (k + 7) // 8
        block = self.data[self.pos:self.pos + n]
        assert len(block) == n, "byte stream exhausted"
        self.pos += n
        return int.from_bytes(block, "big") >> (8 * n - k)


@pytest.fixture
def same_bytes_facct(monkeypatch):
    """``run(fn, seed)``: *fn* under sampler="facct" with the Python reference
    (its _OS_RANDOM) and the native path fed the same bytes (one native call)."""

    def run(fn, seed=1):
        data = random.Random(seed).randbytes(1 << 23)
        with su.gaussian_sampler("facct"):
            with native_backend.backend("python"):
                monkeypatch.setattr(su, "_OS_RANDOM", _StreamBits(data))
                py = fn()
            used = []
            original = native_backend.noise_kwargs
            monkeypatch.setattr(native_backend, "noise_kwargs", lambda *s, **kw: used.append(original(*s, **kw)) or used[-1])
            with native_backend.backend("native"):
                monkeypatch.setattr(native_backend, "_random_override", lambda n: data)
                nat = fn()
                monkeypatch.setattr(native_backend, "_random_override", None)
            monkeypatch.setattr(native_backend, "noise_kwargs", original)
            assert used and all("facct" in kw for kw in used)  # the native calls drew FACCT noise
        return py, nat

    return run


def _setup(mod, eta, ring_n, bound_x, mcfe):
    nids = ["c0", "c1"]
    kg_cls = mod.MCFERingLWEKeyGenerator if mcfe else mod.MIFERingLWEKeyGenerator
    cls = mod.MCFERingLWE if mcfe else mod.MIFERingLWE
    kg = kg_cls({"n": 2, "lst_nid": nids, "eta": eta, "ring_n": ring_n, "sec_param": 64,
                 "bound_x": bound_x, "bound_y": 1})
    kg.setup()
    pp = kg.get_public_parameters()
    return kg, {nid: cls({"id": nid, "precision": 0, "keys": {"pp": pp, "sk": kg.get_private_keys(nid)}})
                for nid in nids}


@pytest.mark.parametrize("eta,ring_n", [(1, 16), (2, 32), (1, 256)])
def test_mife_encrypt_identical(same_bytes_facct, eta, ring_n):
    _, clients = _setup(mife_rlwe, eta, ring_n, 3, mcfe=False)
    rng = random.Random(eta)
    cl = clients["c0"]
    x = [[rng.randint(-3, 3) for _ in range(ring_n)] for _ in range(eta)]
    py, nat = same_bytes_facct(lambda: cl.encrypt(x))
    assert py == nat


@pytest.mark.parametrize("bound_x", [3, 70, 1000])  # 64-bit words (two kinds) and 128-bit words
def test_mcfe_tensor_and_precompute_identical(same_bytes_facct, bound_x):
    _, clients = _setup(mcfe_rlwe, 1, 64, bound_x, mcfe=True)
    cl = clients["c0"]
    cl.precision = 0
    x = np.random.default_rng(bound_x).integers(-bound_x, bound_x + 1, 150)
    py, nat = same_bytes_facct(lambda: cl.encrypt_lst_ndarray([x], label="r"))
    assert py == nat
    py, nat = same_bytes_facct(lambda: cl.encrypt_lst_ndarray([x], label="r", precomputed=cl.precompute("r", [x.shape])))
    assert py == nat


def test_facct_encryption_decrypts():
    kg, clients = _setup(mcfe_rlwe, 1, 64, 100, mcfe=True)
    agg = mcfe_rlwe.MCFERingLWE({"id": "agg", "precision": 0, "keys": {"pp": kg.get_public_parameters()}})
    rng = np.random.default_rng(5)
    xs = {nid: rng.integers(-100, 101, 200) for nid in clients}
    with su.gaussian_sampler("facct"), native_backend.backend("native"):
        native_backend.set_threads(3)
        try:
            enc = {nid: cl.encrypt_lst_ndarray([xs[nid]], label="L") for nid, cl in clients.items()}
        finally:
            native_backend.set_threads(1)
    weights = {nid: [1] for nid in clients}
    dk = kg.get_decryption_keys("agg", credentials={"fusion_weight": weights, "label": "L",
                                                    "n_chunks": agg.key_chunks(enc["c0"])})
    out = agg.decrypt_lst_ndarray_ct(enc, dk=dk, fusion_weight=weights, label="L")
    assert np.array_equal(np.asarray(out[0]), sum(xs.values()))
