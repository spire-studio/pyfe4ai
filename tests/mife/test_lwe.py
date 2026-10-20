import logging
import random

import numpy as np
import pytest

from pyfe4ai.logger import init_logging_config
from pyfe4ai.schemes.mife.lwe import MIFELWE
from pyfe4ai.schemes.mife.lwe import MIFELWEKeyGenerator

init_logging_config()
logger = logging.getLogger(__name__)


@pytest.fixture
def kg_config():
    return {
        "sec_param": 64,
        "eta": 2,
        "n": 3,
        "s": 1,
        "lst_nid": ["nid_0", "nid_1", "nid_2"],
        "lwe_n": 16,
        "bound_x": 6,
        "bound_y": 6,
    }


@pytest.mark.parametrize("mode", ["memory", "stream"])
def test_entire_process(kg_config, mode):
    kg = MIFELWEKeyGenerator({**kg_config, "matrix_mode": mode})
    kg.setup()

    dct_x = {
        nid: [random.randint(-kg_config["bound_x"], kg_config["bound_x"]) for _ in range(2)]
        for nid in kg_config["lst_nid"]
    }
    dct_y = {
        nid: [random.randint(-kg_config["bound_y"], kg_config["bound_y"]) for _ in range(2)]
        for nid in kg_config["lst_nid"]
    }
    expected = sum(
        x_i * y_i
        for nid in kg_config["lst_nid"]
        for x_i, y_i in zip(dct_x[nid], dct_y[nid])
    )

    pp = kg.get_public_parameters()
    dct_ct = {}
    for nid in kg_config["lst_nid"]:
        crypto_nid = MIFELWE(
            {
                "id": nid,
                "keys": {"pp": pp, "sk": kg.get_private_keys(nid)},
                "matrix_mode": mode,
            }
        )
        dct_ct[nid] = crypto_nid.encrypt(dct_x[nid])

    decryptor = MIFELWE({"id": "sid_0", "keys": {"pp": pp}})
    dk = kg.get_decryption_keys("sid_0", credentials={"fusion_weight": dct_y})
    computed = decryptor.decrypt(dct_ct, dk, dct_y)
    assert computed == expected


def test_entire_process_lst_ndarray(kg_config):
    cfg = dict(kg_config)
    cfg["eta"] = 1
    cfg["bound_x"] = 50

    kg = MIFELWEKeyGenerator(cfg)
    kg.setup()

    precision = 1
    dct_x = {
        nid: [np.random.random_sample((2, 3)), np.random.random_sample((2, 2, 2))]
        for nid in cfg["lst_nid"]
    }
    dct_y = {"nid_0": [1], "nid_1": [2], "nid_2": [-1]}

    expected = []
    for layer_idx in range(len(dct_x[cfg["lst_nid"][0]])):
        expected.append(
            sum(
                (
                    np.rint(dct_x[nid][layer_idx] * pow(10, precision))
                    / pow(10, precision)
                )
                * dct_y[nid][0]
                for nid in cfg["lst_nid"]
            )
        )

    pp = kg.get_public_parameters()
    dct_ct = {}
    for nid in cfg["lst_nid"]:
        crypto_nid = MIFELWE(
            {
                "id": nid,
                "precision": precision,
                "keys": {"pp": pp, "sk": kg.get_private_keys(nid)},
            }
        )
        dct_ct[nid] = crypto_nid.encrypt_lst_ndarray(dct_x[nid])

    decryptor = MIFELWE(
        {"id": "sid_0", "precision": precision, "keys": {"pp": pp}}
    )
    dk = kg.get_decryption_keys("sid_0", credentials={"fusion_weight": dct_y})
    computed = decryptor.compute_lst_ndarray_ct(dct_ct, dk=dk, fusion_weight=dct_y)

    for idx in range(len(expected)):
        assert np.allclose(computed[idx], expected[idx], atol=1e-2)


def test_clients_share_one_expanded_matrix(kg_config):
    kg = MIFELWEKeyGenerator(kg_config)
    kg.setup()
    pp = kg.get_public_parameters()
    a, b = (
        MIFELWE({"id": nid, "keys": {"pp": pp, "sk": kg.get_private_keys(nid)}})
        for nid in ("nid_0", "nid_1")
    )
    a.encrypt([1, 2])
    b.encrypt([3, 4])
    assert a.A is b.A


def _noise_margin(n_slots, tmp_path, monkeypatch):
    """Standard deviation of the decryption noise of a full key (sum over the
    slots of y_i^T E_i^T r_i, r binary) in units of Delta = floor(q / p)."""
    import math

    monkeypatch.chdir(tmp_path / str(n_slots))
    nids = ["nid_{}".format(i) for i in range(n_slots)]
    kg = MIFELWEKeyGenerator({"n": n_slots, "lst_nid": nids, "eta": 1, "lwe_n": 64,
                              "bound_x": 20, "bound_y": 20})
    sd = 20 * kg.sigma_q * math.sqrt(n_slots * kg.eta * kg.m / 2)
    return sd / float(kg.q // kg.p)


def test_noise_is_sized_for_all_slots(tmp_path, monkeypatch):
    """ACFGU18 (footnote 14) picks the ALS16 parameters as if the vectors had
    dimension n * eta: decryption adds the noise of all n slots. Sized for eta
    alone, the margin shrank with sqrt(n)."""
    (tmp_path / "1").mkdir()
    (tmp_path / "16").mkdir()
    assert _noise_margin(16, tmp_path, monkeypatch) < 1.5 * _noise_margin(1, tmp_path, monkeypatch)


def test_parameters_are_deterministic():
    from pyfe4ai.primitives.lattice.lwe_utils import abdp15_parameters

    first = abdp15_parameters(20, 20, 3, 64)
    assert abdp15_parameters(20, 20, 3, 64) == first
