"""Server ids of partial decryptions that are not server ids.

The checks of a set of partial decryptions
(:func:`~pyfe4ai.schemes.threshold.roles.validate_partial_decryptions`)
build their messages from the server ids, sorted: a ``bytes`` (or any other
value that is not an id) ``sid`` or entry of ``lst_sid_enrolled`` made that
fail with a bare ``TypeError`` (``'<' not supported``, ``unhashable type``,
``object is not iterable``), whose text depended on the hash seed. The
combiners now refuse such a partial decryption with
:class:`~pyfe4ai.core.exceptions.FEValidationError` and a constant message
that holds none of the caller's values. A server id is a ``str`` or an
``int`` (:func:`~pyfe4ai.core.ids.is_id`): integer ids combine as before, and
a message listing ids of both types orders them by ``repr``.
"""

from __future__ import annotations

import pytest

from pyfe4ai.core.exceptions import FESchemeError, FEValidationError
from pyfe4ai.schemes.threshold.roles import validate_partial_decryptions

MESSAGE = "malformed partial decryption: sid and lst_sid_enrolled must be server ids (str or int)"
NIDS, SIDS = ["nid_0", "nid_1", "nid_2"], ["sid_0", "sid_1", "sid_2"]
MARK = "\n<caller data>"
#: (field, value): the malformed value given to the second server's partial decryption
MALFORMED = [
    ("sid", b"sid_1"),
    ("sid", ("sid_1" + MARK).encode()),
    ("sid", True),
    ("sid", 1.0),
    ("sid", None),
    ("sid", ["sid_1"]),
    ("lst_sid_enrolled", [b"sid_0", "sid_1"]),
    ("lst_sid_enrolled", ["sid_0", False]),
    ("lst_sid_enrolled", ["sid_0", 1.5]),
    ("lst_sid_enrolled", 5),
    ("lst_sid_enrolled", "sid_0" + MARK),
    ("lst_sid_enrolled", {"sid_0": 1, "sid_1": 2}),
]
IDS = ["{}={!r}".format(field, value)[:40] for field, value in MALFORMED]


@pytest.fixture(autouse=True)
def _cwd(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("PYFE4AI_CACHE_DIR", str(tmp_path / "config"))


def _mife_ddh():
    from pyfe4ai.schemes.mife.ddh_threshold import ThresholdMIFEDDH, ThresholdMIFEDDHKeyGenerator

    kg = ThresholdMIFEDDHKeyGenerator({"sec_param": 256, "lst_nid": NIDS, "lst_sid": SIDS, "eta": 1, "n": 3,
                                       "s": 3, "t": 2})
    kg.setup()
    pp = kg.get_public_parameters()
    y = {n: [1] for n in NIDS}
    cts = {n: ThresholdMIFEDDH({"id": n, "keys": {"pp": pp, "sk": kg.get_private_keys(n)}}).encrypt([2]) for n in NIDS}
    parts = {}
    for sid in SIDS[:2]:
        dk = kg.get_decryption_keys(sid, credentials={"fusion_weight": y})
        parts[sid] = ThresholdMIFEDDH({"id": sid, "keys": {"pp": pp, "dk": dk}}).share_decrypt(cts, y, dk, SIDS[:2])
    return ThresholdMIFEDDH({"id": "c", "keys": {"pp": pp}}), parts, 6


def _mcfe_lwe():
    from pyfe4ai.schemes.mcfe.lwe_threshold import ThresholdMCFELWE, ThresholdMCFELWEKeyGenerator

    kg = ThresholdMCFELWEKeyGenerator({"sec_param": 64, "eta": 1, "n": 3, "s": 3, "t": 2, "lst_nid": NIDS,
                                       "lst_sid": SIDS, "lwe_n": 16, "bound_x": 6, "bound_y": 6, "bound_u": 2,
                                       "label_modulus": 8})
    kg.setup()
    pp = kg.get_public_parameters()
    y = {n: [1] for n in NIDS}
    creds = {"fusion_weight": y, "label": "L"}
    cts = {n: ThresholdMCFELWE({"id": n, "keys": {"pp": pp, "sk": kg.get_private_keys(n)}}).encrypt([2], "L")
           for n in NIDS}
    parts = {}
    for sid in SIDS[:2]:
        dk = kg.get_decryption_keys(sid, credentials=creds)
        parts[sid] = ThresholdMCFELWE({"id": sid, "keys": {"pp": pp, "dk": dk}}).share_decrypt(cts, creds, dk, SIDS[:2])
    return ThresholdMCFELWE({"id": "c", "keys": {"pp": pp}}), parts, 6


def _mcfe_ring_lwe():
    from pyfe4ai.schemes.mcfe.ring_lwe_threshold import ThresholdMCFERingLWE, ThresholdMCFERingLWEKeyGenerator

    kg = ThresholdMCFERingLWEKeyGenerator({"n": 3, "lst_nid": NIDS, "s": 3, "t": 2, "lst_sid": SIDS, "eta": 1,
                                           "ring_n": 16, "sec_param": 64, "bound_x": 10, "bound_y": 3})
    kg.setup()
    pp = kg.get_public_parameters()
    creds = {"fusion_weight": {n: [1] for n in NIDS}, "label": "L", "key_part": "full"}
    cts = {n: ThresholdMCFERingLWE({"id": n, "precision": 0, "keys": {"pp": pp, "sk": kg.get_private_keys(n)}})
           .encrypt([[2]], "L") for n in NIDS}
    parts = {}
    for sid in SIDS[:2]:
        dk = kg.get_decryption_keys(sid, credentials=creds)
        server = ThresholdMCFERingLWE({"id": sid, "precision": 0, "keys": {"pp": pp}})
        parts[sid] = server.share_decrypt(cts, creds, dk, SIDS[:2])
    return ThresholdMCFERingLWE({"id": "c", "precision": 0, "keys": {"pp": pp}}), parts, [6]


@pytest.mark.parametrize("field,value", MALFORMED, ids=IDS)
def test_the_checks_refuse_values_that_are_not_ids(field, value):
    parts = {sid: {"w": "1", "sid": sid, "lst_sid_enrolled": SIDS[:2]} for sid in SIDS[:2]}
    parts["sid_1"] = dict(parts["sid_1"], **{field: value})
    with pytest.raises(FEValidationError) as info:
        validate_partial_decryptions(parts, 2)
    assert str(info.value) == MESSAGE


@pytest.mark.parametrize("field,value", MALFORMED, ids=IDS)
@pytest.mark.parametrize("build", [_mife_ddh, _mcfe_lwe, _mcfe_ring_lwe], ids=["mife/ddh_threshold",
                                                                            "mcfe/lwe_threshold",
                                                                            "mcfe/ring_lwe_threshold"])
def test_combiners_refuse_values_that_are_not_ids(build, field, value):
    combiner, parts, right = build()
    assert combiner.combine_decrypt(parts) == right
    bad = dict(parts, sid_1=dict(parts["sid_1"], **{field: value}))
    with pytest.raises(FEValidationError) as info:
        combiner.combine_decrypt(bad)
    assert str(info.value) == MESSAGE and "caller data" not in str(info.value)
    with pytest.raises(FEValidationError, match="malformed partial decryption"):
        combiner.combine_decrypt([parts, bad])  # in a second enrolled set too


def test_integer_server_ids_pass_the_checks():
    parts = {sid: {"w": "1", "sid": sid, "lst_sid_enrolled": [1, 2]} for sid in (1, 2)}
    validate_partial_decryptions(parts, 2)
    validate_partial_decryptions({sid: dict(p, lst_sid_enrolled=(2, 1)) for sid, p in parts.items()}, 2)


def test_messages_list_ids_of_both_types():
    # a server id 2 ("sid_1" enrolled as 2): the contributors are not the enrolled set; the message lists
    # str and int ids (sorting them by value raised TypeError: '<' not supported between 'int' and 'str')
    parts = {sid: {"w": "1", "sid": sid, "lst_sid_enrolled": ["sid_0", 2]} for sid in SIDS[:2]}
    with pytest.raises(FESchemeError) as info:
        validate_partial_decryptions(parts, 2)
    assert str(info.value) == ("partial decryptions from ['sid_0', 'sid_1'] do not match the enrolled set "
                               "['sid_0', 2]")
