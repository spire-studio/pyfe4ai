"""Server and client ids (:mod:`pyfe4ai.core.ids`).

An id is a ``str`` or an ``int`` (not a ``bool``). A key generator refuses a
``lst_sid`` or ``lst_nid`` whose ids cannot key a dict (an unhashable entry,
or a value that is not a sized collection) with
:class:`~pyfe4ai.core.exceptions.FEValidationError` and a constant message,
where it raised a bare ``TypeError`` (``unhashable type``); messages that list
ids order ids of both types by ``repr``.
"""

from __future__ import annotations

import pytest

import pyfe4ai
from pyfe4ai import backend as native_backend
from pyfe4ai.core.exceptions import FEValidationError
from pyfe4ai.core.ids import DUPLICATE_IDS, ID_LISTS, id_list, is_id

needs_bls = pytest.mark.skipif(not native_backend.available(), reason="BLS12-381 needs the native extension")

THRESHOLD = pyfe4ai.list_schemes(capabilities="threshold")
#: every scheme whose key generator takes ``lst_nid``
WITH_CLIENTS = [s for s in pyfe4ai.list_schemes() if s.split("/")[0] in ("mife", "mcfe")] + ["quadratic/miqfe"]


def _marks(scheme):
    return pytest.param(scheme, marks=needs_bls) if "pairing" in scheme else scheme


@pytest.fixture(autouse=True)
def _cwd(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("PYFE4AI_CACHE_DIR", str(tmp_path / "config"))


def _config(scheme, **ids):
    group = any(token in scheme for token in ("ddh", "damgard", "paillier"))
    config = {"eta": 1, "n": 2, "lst_nid": ["c0", "c1"], "s": 2, "t": 2, "lst_sid": ["s0", "s1"], "ring_n": 16,
              "lwe_n": 16, "sec_param": 128 if group else 64, "bound_x": 4, "bound_y": 4, "bound_u": 2,
              "label_modulus": 8}
    config.update(ids)
    return config


def test_ids_are_str_or_int():
    assert all(is_id(v) for v in ("s0", "", 0, 1, -3, 2 ** 70))
    assert not any(is_id(v) for v in (True, False, 1.0, None, b"s0", ("s", 0), ["s0"], {"s": 0}))


def test_messages_order_ids_of_both_types():
    assert id_list({"b", "a"}) == ["a", "b"] and id_list([3, 1]) == [1, 3]
    assert id_list(["s0", 1]) == ["s0", 1]  # by repr: "'s0'" < "1"
    assert id_list([None, 1]) == [1, None]


@pytest.mark.parametrize("ids", [[["s", 0], ["s", 1]], [{"s": 0}, {"s": 1}], 5, iter(["s0", "s1"])],
                         ids=["lists", "dicts", "an int", "an iterator"])
@pytest.mark.parametrize("scheme", [_marks(s) for s in THRESHOLD])
def test_server_ids_that_cannot_key_a_dict_are_refused_at_setup(scheme, ids):
    with pytest.raises(FEValidationError) as info:
        kg, _ = pyfe4ai.create(scheme, _config(scheme, lst_sid=ids))
        kg.setup()
    assert str(info.value) == ID_LISTS["lst_sid"]


@pytest.mark.parametrize("ids", [[["c", 0], ["c", 1]], [{"c": 0}, {"c": 1}], iter(["c0", "c1"])],
                         ids=["lists", "dicts", "an iterator"])
@pytest.mark.parametrize("scheme", [_marks(s) for s in WITH_CLIENTS])
def test_client_ids_that_cannot_key_a_dict_are_refused_at_setup(scheme, ids):
    with pytest.raises(FEValidationError) as info:
        kg, _ = pyfe4ai.create(scheme, _config(scheme, lst_nid=ids))
        kg.setup()
    assert str(info.value) == ID_LISTS["lst_nid"]


@pytest.mark.parametrize("scheme", [_marks(s) for s in THRESHOLD])
def test_repeated_ids_keep_their_message(scheme):
    with pytest.raises(FEValidationError) as info:
        pyfe4ai.create(scheme, _config(scheme, lst_sid=["s0", "s0"]))[0].setup()
    assert str(info.value) == DUPLICATE_IDS["lst_sid"]
    with pytest.raises(FEValidationError) as info:
        pyfe4ai.create(scheme, _config(scheme, lst_nid=[1, 1]))[0].setup()
    assert str(info.value) == DUPLICATE_IDS["lst_nid"]
