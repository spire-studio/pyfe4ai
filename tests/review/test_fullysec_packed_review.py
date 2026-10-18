"""Review 2026-10 (exp/fullysec-packing: 671d3ab, fix b40746a): packed layout of sife/fullysec_ring_lwe.

* The G5* step of the packing argument needs ``(2X sqrt(D))^D < q`` (the
  denominators of the differences in a basis of their span are invertible mod
  ``q``). 671d3ab accepted standard packed parameters below it (e.g. eta 32,
  X 1: 110-bit q against a 112-bit bound); b40746a enforces it, and the first
  test checks that for every packed parameter set the generators accept.
* Fail on b40746a: the ``ndarray_layout`` flag of the public parameters is
  not bound to them (an elementwise record relabelled ``"packed"`` encrypts
  matrices with widths 64x too small); the packed-layer parser accepts
  shapes such as ``[True, 70]`` / ``["7", 10.9]``, ignores ``first_chunk``,
  decrypts reordered chunks into a permuted array and lets raw ``ValueError``
  escape; the status table labels the packed layout ``reviewed`` although
  its leveraging argument is the project's own.
"""

import copy

import numpy as np
import pytest

from pyfe4ai.schemes.sife import fullysec_ring_lwe as fs
from pyfe4ai.utils import ring_lwe_utils as rlu
from pyfe4ai.utils.exceptions import FEValidationError


@pytest.fixture(autouse=True)
def _cwd(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)


def test_packed_moduli_exceed_the_minor_bound():
    checked = 0
    for std in (True, False):
        for eta in (2, 4, 16, 24, 32, 64):
            for bound_x in (1, 4, 50, 1000):
                for ring_n in (["auto"] if std else [1024, 4096]):
                    try:
                        p = rlu.bmms21_adaptive_choice(128, eta, bound_x, 1, ring_n, "auto", std, packed=True)
                    except Exception:  # noqa: BLE001 - no parameters for this config
                        continue
                    d = min(eta, p["ring_n"])
                    checked += 1
                    assert (2 * bound_x) ** (2 * d) * d ** d < int(p["q"]) ** 2, (std, eta, bound_x, p["ring_n"])
    assert checked > 20


def test_layout_flag_is_bound_to_the_parameters():
    kg = fs.SIFEFullySecRingLWEKeyGenerator({"eta": 3, "bound_x": 4, "bound_y": 4})
    kg.setup()
    pp = dict(kg.get_public_parameters(), ndarray_layout="packed")  # an elementwise record, relabelled
    enc = fs.SIFEFullySecRingLWE({"keys": {"pp": pp, "sk": kg.get_private_keys()}})
    with pytest.raises(FEValidationError):
        enc.encrypt([[4] * 32, [-4] * 32, [3] * 32])


@pytest.fixture
def packed_round():
    kg = fs.SIFEFullySecRingLWEKeyGenerator({"eta": 1, "bound_x": 100, "bound_y": 1, "ndarray_layout": "packed"})
    kg.setup()
    pp = kg.get_public_parameters()
    enc = fs.SIFEFullySecRingLWE({"keys": {"pp": pp, "sk": kg.get_private_keys()}})
    dec = fs.SIFEFullySecRingLWE({"keys": {"pp": pp}})
    dk = kg.get_decryption_keys("s", credentials={"fusion_weight": [1]})
    layers = enc.encrypt_lst_ndarray([np.arange(70).reshape(7, 10) - 35])
    assert len(layers[0]["chunks"]) > 1
    return layers, lambda lst: dec.decrypt_lst_ndarray_ct(lst, dk=dk, fusion_weight=[1])


@pytest.mark.parametrize("mutate", [
    lambda layer: layer.update(shape=[True, 70]),
    lambda layer: layer.update(shape=["7", 10.9]),
    lambda layer: layer.update(shape=[-7, -10]),
    lambda layer: layer.update(shape=[2 ** 62, 4], chunks=[]),
    lambda layer: layer.update(first_chunk="garbage"),
    lambda layer: layer.update(chunks=layer["chunks"][::-1]),
], ids=["bool-dim", "str-float-dims", "negative-dims", "int64-wrap", "first-chunk", "reversed-chunks"])
def test_malformed_packed_layers_are_refused(packed_round, mutate):
    layers, decrypt = packed_round
    bad = copy.deepcopy(layers)
    mutate(bad[0])
    with pytest.raises(FEValidationError):
        decrypt(bad)


def test_status_marks_the_packed_layout_as_unreviewed():
    from pyfe4ai.schemes.status import SCHEME_STATUS

    status = SCHEME_STATUS[("sife", "fullysec_ring_lwe")]
    text = " ".join(str(v) for v in vars(status).values()) if hasattr(status, "__dict__") else str(status)
    assert "experimental" in text.lower() or "unreviewed" in text.lower(), text
