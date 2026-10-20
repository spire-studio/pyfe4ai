"""Integration of the October 2026 performance work (code review).

No merge dropped or weakened a change, and the new code of the integration
(flooding binding, construction ids, sponge wipes, KL pinning, RNS replay with
FACCT and label hashes) behaved as specified. Before the fixes:

* dense tMCFE-LWE key shares did not record the label hash, and the flooding
  noise did not depend on it: one key share served under public parameters
  that differ only in ``label_hash`` answered a label with the same noise and
  another ``H(l)``, so ``w_A - w_B = lambda <share, H_A(l) - H_B(l)>`` exactly
  (about ``n0 + m0`` labels give the share). tMCFE-Ring binds the mode in its
  key shares (``_check_key_label_hash``); the dense layer did not;
* the tMIFE-LWE key generator accepted ``flooding_accounting = "kl"`` with a
  replicated sharing and issued public parameters whose construction names
  Shamir (servers then refused them; tMCFE-Ring refuses at key generation);
* the Python rejection-sampling label hash never returned for ``length = 0``
  (its digest doubled without bound), while the native kernel and the
  ``shake256-wide`` mode return ``[]``.
"""

import os
import signal
import warnings

import gmpy2 as gp
import pytest

from pyfe4ai.primitives import hashing
from pyfe4ai.core.exceptions import FEKeyError, FESchemeError, FEValidationError

NIDS, SIDS = ["a", "b"], ["s0", "s1", "s2"]


@pytest.fixture(autouse=True)
def _cwd(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("PYFE4AI_CACHE_DIR", os.path.join(tmp_path, "config"))


def test_dense_threshold_key_share_binds_the_label_hash():
    from pyfe4ai.schemes.mcfe import lwe_threshold as dense
    from pyfe4ai.schemes.mcfe.lwe import dot_mod
    from pyfe4ai.schemes.threshold.sharing import lagrange_coefficient_at_zero

    kg = dense.ThresholdMCFELWEKeyGenerator({"sec_param": 64, "n": 2, "lst_nid": NIDS, "eta": 1, "bound_x": 4,
                                             "bound_y": 4, "s": 3, "t": 3, "lst_sid": SIDS})
    kg.setup()
    pp_a = kg.get_public_parameters()
    other = "shake256-wide" if pp_a["label_hash"] != "shake256-wide" else "shake128-rejection"
    pp_b = dict(pp_a, label_hash=other)
    fw = {n: [1] for n in NIDS}
    dk = kg.get_decryption_keys("s0", credentials={"fusion_weight": fw})
    label = "round-0"
    cts = {n: dense.ThresholdMCFELWE({"id": n, "keys": {"pp": pp_a, "sk": kg.get_private_keys(n)}}).encrypt([1], label)
           for n in NIDS}
    srv_a = dense.ThresholdMCFELWE({"id": "s0", "keys": {"pp": pp_a, "dk": dk}})
    w_a = gp.mpz(srv_a.share_decrypt(cts, {"fusion_weight": fw, "label": label}, dk, SIDS)["w"])
    try:
        srv_b = dense.ThresholdMCFELWE({"id": "s0", "keys": {"pp": pp_b, "dk": dk}})
        cts_b = {n: {"c": ct["c"]} for n, ct in cts.items()}  # the code field is not authenticated
        w_b = gp.mpz(srv_b.share_decrypt(cts_b, {"fusion_weight": fw, "label": label}, dk, SIDS)["w"])
    except (FEKeyError, FESchemeError, FEValidationError):
        return  # the key share is refused under another label hash
    q = gp.mpz(pp_a["q"])
    share = [gp.mpz(v) for v in dk["sk_y"]]
    lam = lagrange_coefficient_at_zero(1, [1, 2, 3], q)
    exact = lam * (dot_mod(share, srv_a._hash_label(label), q) - dot_mod(share, srv_b._hash_label(label), q)) % q
    assert (w_a - w_b) % q != exact, (
        "one key share answered a label under two label hashes with the same noise: "
        "w_A - w_B = lambda <share, H_A(l) - H_B(l)> mod q exactly"
    )


@pytest.mark.parametrize("sharing", ["replicated", "replicated-seeded"])
def test_tmife_kl_accounting_refuses_a_replicated_sharing(sharing):
    from pyfe4ai.schemes.mife.lwe_threshold import ThresholdMIFELWEKeyGenerator

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        with pytest.raises(FEValidationError):
            ThresholdMIFELWEKeyGenerator({
                "sec_param": 64, "eta": 1, "n": 2, "lst_nid": NIDS, "lwe_n": 16, "bound_x": 4, "bound_y": 4,
                "s": 3, "t": 3, "lst_sid": SIDS, "sharing": sharing,
                "flooding_accounting": "kl", "max_partial_decryptions": 4,
            })


@pytest.mark.parametrize("mode", ["shake128-rejection", "shake256-rejection", "shake256-wide"])
def test_hash_to_zq_vector_of_length_zero_returns(mode):
    def stop(*_):
        raise TimeoutError("hash_to_zq_vector(length=0) did not return")

    previous = signal.signal(signal.SIGALRM, stop)
    signal.alarm(3)
    try:
        assert hashing.hash_to_zq_vector("label", gp.next_prime(gp.mpz(2) ** 80), 0, mode=mode) == []
    finally:
        signal.alarm(0)
        signal.signal(signal.SIGALRM, previous)
