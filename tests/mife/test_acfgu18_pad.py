"""ACFGU18 one-time pad: one slot's partial decryption reveals nothing.

Versions <= 0.2.0 decoded each slot separately with the functional key, so
the key holder learned every client's <x_i, y_i>. With the pad, a slot's
Dec1 output encodes <x_i + u_i, y_i> for a uniform u_i.
"""

import gmpy2 as gp
import pytest

from pyfe4ai.utils.exceptions import FESchemeError
from pyfe4ai.utils.lwe_utils import decode_scaled
from pyfe4ai.utils.matrix_utils import digits_to_vector, vecdot_mod
from pyfe4ai.utils.modular_utils import pow_signed

NIDS = ["nid_0", "nid_1", "nid_2"]


def _damgard():
    from pyfe4ai.schemes.mife.damgard_ddh import MIFEDamgard, MIFEDamgardKeyGenerator

    cfg = {"sec_param": 64, "eta": 2, "n": 3, "lst_nid": NIDS, "modulus_length": 64, "bound": 10}

    def slot_value(pp, ct, dk, nid, y):
        p = gp.mpz(pp["p"])
        acc = gp.mpz(1)
        for e_j, y_j in zip(ct["e"], y):
            acc = acc * pow_signed(gp.mpz(e_j), gp.mpz(y_j), p) % p
        den = gp.powmod(gp.mpz(ct["c"]), gp.mpz(dk["k1"][nid]), p)
        den = den * gp.powmod(gp.mpz(ct["d"]), gp.mpz(dk["k2"][nid]), p) % p
        return gp.divm(acc, den, p)

    def expected(pp, value):
        return pow_signed(gp.mpz(pp["g"]), gp.mpz(value), gp.mpz(pp["p"]))

    return MIFEDamgardKeyGenerator, MIFEDamgard, cfg, slot_value, expected


def _paillier():
    from pyfe4ai.schemes.mife.paillier import MIFEPaillier, MIFEPaillierKeyGenerator

    cfg = {"sec_param": 64, "eta": 2, "n": 3, "lst_nid": NIDS, "bit_length": 64,
           "bound_x": 10, "bound_y": 10}

    def slot_value(pp, ct, dk, nid, y):
        n_mod = gp.mpz(pp["n_mod"])
        n2 = n_mod * n_mod
        acc = pow_signed(gp.mpz(ct["ct0"]), -gp.mpz(dk["k"][nid]), n2)
        for c_j, y_j in zip(ct["ct1"], y):
            acc = acc * pow_signed(gp.mpz(c_j), gp.mpz(y_j), n2) % n2
        return ((acc - 1) % n2) // n_mod

    def expected(pp, value):
        return gp.mpz(value) % gp.mpz(pp["n_mod"])

    return MIFEPaillierKeyGenerator, MIFEPaillier, cfg, slot_value, expected


def _lwe(module, kg_name, crypto_name, delta_key):
    import importlib

    mod = importlib.import_module(module)
    cfg = {"sec_param": 64, "eta": 2, "n": 3, "lst_nid": NIDS, "lwe_n": 16,
           "bound_x": 6, "bound_y": 6}

    def slot_value(pp, ct, dk, nid, y):
        q = gp.mpz(pp["q"])
        d = vecdot_mod([gp.mpz(v) for v in y], digits_to_vector(ct["ct1"]), q)
        d -= vecdot_mod(digits_to_vector(ct["ct0"]), digits_to_vector(dk["sk_y"][nid]), q)
        return decode_scaled(d % q, q // gp.mpz(pp[delta_key]), q, 10 ** 9)

    def expected(pp, value):
        return value

    return getattr(mod, kg_name), getattr(mod, crypto_name), cfg, slot_value, expected


SCHEMES = {
    "damgard": _damgard,
    "paillier": _paillier,
    "lwe": lambda: _lwe("pyfe4ai.schemes.mife.lwe", "MIFELWEKeyGenerator", "MIFELWE", "p"),
    "fullysec_lwe": lambda: _lwe(
        "pyfe4ai.schemes.mife.fullysec_lwe", "MIFEFullySecLWEKeyGenerator", "MIFEFullySecLWE", "k"
    ),
}


@pytest.mark.parametrize("name", sorted(SCHEMES))
def test_slot_partials_hide_individual_inner_products(name):
    kg_cls, crypto_cls, cfg, slot_value, expected = SCHEMES[name]()
    kg = kg_cls(cfg)
    kg.setup()
    pp = kg.get_public_parameters()
    dct_x = {nid: [3, -2] for nid in NIDS}
    dct_y = {nid: [1, 2] for nid in NIDS}
    dct_ct = {
        nid: crypto_cls({"id": nid, "keys": {"pp": pp, "sk": kg.get_private_keys(nid)}}).encrypt(dct_x[nid])
        for nid in NIDS
    }
    dk = kg.get_decryption_keys("sid_0", credentials={"fusion_weight": dct_y})
    assert "z" in dk and "u" in kg.get_private_keys("nid_0")
    decryptor = crypto_cls({"id": "sid_0", "keys": {"pp": pp}})
    assert decryptor.decrypt(dct_ct, dk, dct_y) == -3

    # the per-slot value the v0.2.0 decryptor decoded no longer equals <x_i, y_i> = -1
    leaks = [
        slot_value(pp, dct_ct[nid], dk, nid, dct_y[nid]) == expected(pp, -1) for nid in NIDS
    ]
    assert not any(leaks)

    with pytest.raises(FESchemeError):
        decryptor.decrypt({nid: dct_ct[nid] for nid in NIDS[:2]}, dk, dct_y)
