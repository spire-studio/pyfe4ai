"""Per-scheme Gaussian samplers: FACCT by default where the proofs assume D_{Z,sigma}.

The Ring-LWE MIFE / MCFE family (sife / mife / mcfe ring_lwe, the decentralized
and threshold MCFE) and the dense-LWE MCFE keys t (mcfe lwe, fullysec_lwe,
lwe_threshold, lwe_decentralized) sample exactly by default and record their
sampler in the public parameters; "rounded" stays selectable.
"""

import numpy as np
import pytest

from pyfe4ai.schemes.mcfe import lwe as mcfe_lwe
from pyfe4ai.schemes.mcfe import lwe_decentralized, lwe_threshold
from pyfe4ai.schemes.mcfe import ring_lwe as mcfe_rlwe
from pyfe4ai.schemes.mcfe import ring_lwe_decentralized, ring_lwe_threshold
from pyfe4ai.schemes.mcfe.fullysec_lwe import MCFEFullySecLWEKeyGenerator
from pyfe4ai.schemes.mife import ring_lwe as mife_rlwe
from pyfe4ai.schemes.sife import ring_lwe as sife_rlwe
from pyfe4ai.utils import native_backend
from pyfe4ai.utils import sampling_utils as su
from pyfe4ai.utils.exceptions import FEValidationError

NIDS = ["a", "b"]
SIDS = ["s0", "s1", "s2"]
RLWE = {"sec_param": 64, "eta": 1, "ring_n": 32, "bound_x": 4, "bound_y": 4}
KEYGENS = {
    "sife/ring_lwe": lambda c: sife_rlwe.SIFERingLWEKeyGenerator({**RLWE, **c}),
    "mife/ring_lwe": lambda c: mife_rlwe.MIFERingLWEKeyGenerator({**RLWE, "n": 2, "lst_nid": NIDS, **c}),
    "mcfe/ring_lwe": lambda c: mcfe_rlwe.MCFERingLWEKeyGenerator({**RLWE, "n": 2, "lst_nid": NIDS, **c}),
    "mcfe/ring_lwe_decentralized": lambda c: ring_lwe_decentralized.DecentralizedMCFERingLWEKeyGenerator(
        {**RLWE, "n": 2, "lst_nid": NIDS, **c}),
    "mcfe/ring_lwe_threshold": lambda c: ring_lwe_threshold.ThresholdMCFERingLWEKeyGenerator(
        {**RLWE, "n": 2, "lst_nid": NIDS, "s": 3, "t": 2, "lst_sid": SIDS, **c}),
    "mcfe/lwe": lambda c: mcfe_lwe.MCFELWEKeyGenerator({"n": 2, "lst_nid": NIDS, "eta": 1, "lwe_n": 16, **c}),
    "mcfe/fullysec_lwe": lambda c: MCFEFullySecLWEKeyGenerator({"n": 2, "lst_nid": NIDS, "eta": 1, "lwe_n": 16, **c}),
    "mcfe/lwe_threshold": lambda c: lwe_threshold.ThresholdMCFELWEKeyGenerator(
        {"n": 2, "lst_nid": NIDS, "eta": 1, "lwe_n": 16, "s": 3, "t": 2, "lst_sid": SIDS, **c}),
    "mcfe/lwe_decentralized": lambda c: lwe_decentralized.DecentralizedMCFELWEKeyGenerator(
        {"n": 2, "lst_nid": NIDS, "eta": 1, "lwe_n": 16, **c}),
}


@pytest.fixture(autouse=True)
def _cwd(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)


class _Draws:
    """discrete_gaussian_matrix, recording the sampler of every call."""

    def __init__(self) -> None:
        self.samplers = []
        self.original = su.discrete_gaussian_matrix

    def __call__(self, rows, cols, sigma, sampler=None):
        self.samplers.append(su._resolve_sampler(sampler))
        return self.original(rows, cols, sigma, sampler=sampler)


@pytest.fixture
def draws(monkeypatch):
    d = _Draws()
    for mod in (sife_rlwe, mife_rlwe, mcfe_lwe):
        monkeypatch.setattr(mod, "discrete_gaussian_matrix", d)
    return d


@pytest.mark.parametrize("name", list(KEYGENS))
def test_facct_by_default_and_recorded(name, draws, monkeypatch):
    # the process-wide choice does not apply to these schemes
    monkeypatch.setattr(su, "_default_sampler", "rounded")
    kg = KEYGENS[name]({})
    kg.setup()
    assert kg.get_public_parameters()["gaussian_sampler"] == "facct"
    assert draws.samplers and set(draws.samplers) == {"facct"}


@pytest.mark.parametrize("name", list(KEYGENS))
def test_rounded_stays_selectable(name, draws):
    kg = KEYGENS[name]({"gaussian_sampler": "rounded"})
    kg.setup()
    assert kg.get_public_parameters()["gaussian_sampler"] == "rounded"
    assert set(draws.samplers) == {"rounded"}
    with su.gaussian_sampler("rounded"):  # an explicit block also selects it
        assert KEYGENS[name]({}).gaussian_sampler == "rounded"
    with pytest.raises(FEValidationError, match="unknown Gaussian sampler"):
        KEYGENS[name]({"gaussian_sampler": "box-muller"})


def _mcfe(cfg=None):
    kg = KEYGENS["mcfe/ring_lwe"](cfg or {})
    kg.setup()
    pp = kg.get_public_parameters()
    return kg, pp


@pytest.mark.parametrize("backend", ["python", "native"])
def test_encryption_noise_follows_the_public_parameters(backend, draws, monkeypatch):
    if backend == "native" and not native_backend.available():
        pytest.skip("native extension not built")
    tables = []
    original = native_backend.noise_kwargs
    monkeypatch.setattr(native_backend, "noise_kwargs",
                        lambda *s, **kw: tables.append(original(*s, **kw)) or tables[-1])
    kg, pp = _mcfe()
    x = np.arange(-20, 20) % 9 - 4
    for sampler_pp, expect in ((pp, "facct"), ({k: v for k, v in pp.items() if k != "gaussian_sampler"}, "facct"),
                               (dict(pp, gaussian_sampler="rounded"), "rounded")):
        draws.samplers.clear()
        tables.clear()
        with native_backend.backend(backend):
            enc = mcfe_rlwe.MCFERingLWE({"id": "a", "precision": 0, "keys": {"pp": sampler_pp,
                                                                              "sk": kg.get_private_keys("a")}})
            assert enc.gaussian_sampler == expect
            enc.encrypt_lst_ndarray([x], label="L")
        if backend == "python":
            assert set(draws.samplers) == {expect}
        else:
            assert tables and all(("facct" in t) == (expect == "facct") for t in tables)
    # the crypto config overrides pp
    enc = mcfe_rlwe.MCFERingLWE({"id": "a", "keys": {"pp": pp, "sk": kg.get_private_keys("a")},
                                 "gaussian_sampler": "rounded"})
    assert enc.gaussian_sampler == "rounded"


@pytest.mark.parametrize("sampler", ["facct", "rounded"])
def test_decryption_does_not_depend_on_the_sampler(sampler):
    kg, pp = _mcfe({"gaussian_sampler": sampler})
    rng = np.random.default_rng(1)
    xs = {nid: rng.integers(-4, 5, 40) for nid in NIDS}
    weights = {nid: [1] for nid in NIDS}
    enc = {nid: mcfe_rlwe.MCFERingLWE({"id": nid, "precision": 0, "keys": {"pp": pp, "sk": kg.get_private_keys(nid)}})
           .encrypt_lst_ndarray([xs[nid]], label="L") for nid in NIDS}
    n_chunks = mcfe_rlwe.MCFERingLWE.key_chunks(enc[NIDS[0]])
    dk = kg.get_decryption_keys("agg", credentials={"fusion_weight": weights, "label": "L", "n_chunks": n_chunks})
    old_pp = {k: v for k, v in pp.items() if k != "gaussian_sampler"}  # public parameters of earlier versions
    for p in (pp, old_pp):
        agg = mcfe_rlwe.MCFERingLWE({"id": "agg", "precision": 0, "keys": {"pp": p}})
        out = agg.decrypt_lst_ndarray_ct(enc, dk=dk, fusion_weight=weights, label="L")
        assert np.array_equal(np.asarray(out[0]), xs["a"] + xs["b"])
