"""R-0.6 pass 1 review (481450d): failing tests for the findings that can be tests.

See records/2026-11-perf/sessions/r-0.6/report.md of the project's records.
"""

from __future__ import annotations

import hashlib
import os

import pytest

from pyfe4ai.schemes.common import slots
from pyfe4ai.schemes.common.slots import SlotStore


def _dkey(i) -> bytes:
    return hashlib.sha3_256("r06 dedup key {}".format(i).encode()).digest()


def _h2_build(slot_id):
    """An h2 slot: the public record holds the holder's own public keys at the selected positions, so the holder
    can decrypt the slot's components as soon as the record is listed; the private record is only the center's
    ``k`` (for CheckAuth), not a key."""
    return ({"pks": [os.urandom(32) for _ in range(5)], "issuance": "h2"}, {"k": os.urandom(16)})


def test_r06_01_a_listed_h2_slot_is_returned_to_a_retry_not_duplicated(tmp_path, monkeypatch):
    """R06-01: an error (or crash) between the public write and the private write leaves the slot listed without a
    dedup entry. For h2 the holder already holds that slot's key (its own scalars, its keys in the listed record),
    so the slot is delivered in effect. A retry of the same request then gets a second slot: the same requester
    holds two slots, and the store's dedup index (and an issuance policy built on it) knows of one. A listed slot
    should stay reachable by its dedup key (status "hit", or a refusal), never be shadowed by a new one."""
    store = SlotStore(tmp_path)
    store.issue(_dkey(0), 5, _h2_build)
    count, real = [0], slots._atomic_write

    def write(*args):
        count[0] += 1
        if count[0] == 3:  # 1 reservation, 2 public record, 3 private record + dedup entry
            raise OSError("disk full")
        real(*args)

    monkeypatch.setattr(slots, "_atomic_write", write)
    with pytest.raises(OSError):
        store.issue(_dkey(1), 5, _h2_build)
    monkeypatch.setattr(slots, "_atomic_write", real)
    listed = [s["slot_id"] for s in store.public_slots()]
    assert listed == [0, 1]                      # slot 1 is public: its holder can decrypt it
    retry = store.issue(_dkey(1), 5, _h2_build)
    assert retry.slot_id == 1, "the retry made slot {} beside the listed slot 1".format(retry.slot_id)
