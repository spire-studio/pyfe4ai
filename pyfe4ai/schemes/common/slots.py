"""The slot list of a bounded-key scheme, kept on disk by its authority.

| "Worry-Free Encryption: Functional Encryption with Public Keys"
| By Amit Sahai and Hakan Seyalioglu, ACM CCS 2010 -- SS10 Sec. 3
| "Functional Encryption with Bounded Collusions via Multi-Party Computation"
| By Sergey Gorbunov, Vinod Vaikuntanathan and Hoeteck Wee, CRYPTO 2012

A bounded-key scheme issues at most ``max_keys`` functional keys, each in a
*slot* of its own: a public record (the slot's public keys, its label, ...)
that encrypters encrypt to, and private material that only the authority
keeps. The security argument needs each slot to hold exactly one key, made
when the slot is made: a slot id is never handed out twice, the public list
only grows, and a repeated request returns the stored key, never a new one.
:class:`SlotStore` keeps that state for one authority instance in one
directory (the caller picks it, e.g. ``<cache_dir>/authority/bcfe/<setup
id>``). It knows nothing about keys or groups: the records are opaque
JSON values (``None``, booleans, integers, finite floats, strings, lists,
string-keyed dicts) and ``bytes``.

Files of the directory:

``slots.json`` (public, mode ``0o644``)
    The slot list: the records in issue order, each with its ``slot_id``,
    and ``slot_list_version`` = the number of slots. Slot ids increase but
    may skip a number (an id burned by a failed or crashed issue).
``slots.private.json`` (private, mode ``0o600``)
    ``next_slot_id`` (no id below it is ever handed out again), the private
    record of each delivered slot and the deduplication index (dedup key ->
    slot id). Neither the dedup keys (hashes of the requested functions) nor
    the private records ever reach the public file, a ``repr``, an
    exception message or a log line: a public dedup key is an offline test
    of guesses of a function.
``slots.lock``
    The file lock (``fcntl.flock``); never removed.

:meth:`SlotStore.issue` runs under one exclusive lock (the store object's
lock for its threads, the file lock for every other process and store
object): look the dedup key up; refuse with the same :class:`FEKeyError`
and the same text, hit or miss, once the list holds ``max_keys`` slots (so
that a requester cannot test functions through the deduplication once the
cap is reached); return a hit without building; otherwise reserve the next
slot id (persisted before ``build`` runs), call ``build(slot_id)``, append
the public record, then store the private record with its dedup key. Every
write goes to a temporary file in the directory that is fsynced, renamed
over the old file and followed by an fsync of the directory.

A crash (or an exception) anywhere leaves a consistent store:

* before the reservation: nothing changed;
* after the reservation, before the public record (``build`` raised, or a
  crash): the id is burned, no slot appears, a retry gets a new id;
* after the public record, before the private one: the slot is listed but
  was never delivered (no private record, no dedup entry); it counts
  against ``max_keys`` and a retry makes a new slot. The public record goes
  first so that the dedup index never points to a slot missing from the
  list, and a slot's records are written by the one call that reserved its
  id: no slot can end up with two keys.

Temporary files of a crashed writer are removed by the next locked call.
Unreadable, truncated or inconsistent files raise :class:`FEConfigError`
(fail closed): the store never starts empty while its files exist.

Locks and processes: the store object's lock is renewed in a forked child
(:func:`pyfe4ai.core.forksafe.track`), which also drops its copy of a held
file-lock descriptor; the file lock itself belongs to the parent's open
file and is released by the parent. A copy or a pickle of a store is a new
store object on the same directory.
"""

from __future__ import annotations

import base64
import binascii
import contextlib
import json
import logging
import math
import os
import tempfile
import threading
from typing import Any, Callable, Dict, Iterator, List, NamedTuple, Optional, Tuple

from pyfe4ai.core import forksafe
from pyfe4ai.core.exceptions import FEConfigError, FEKeyError, FESchemeError, FEValidationError

logger = logging.getLogger(__name__)

#: The public slot list.
PUBLIC_FILE = "slots.json"
#: The private side: reservations, private records, the dedup index.
PRIVATE_FILE = "slots.private.json"
#: The file lock.
LOCK_FILE = "slots.lock"
#: The ``format`` entries of the two files and their version.
PUBLIC_FORMAT = "pyfe4ai-slot-list"
PRIVATE_FORMAT = "pyfe4ai-slot-private"
FORMAT_VERSION = 1
#: Length of a dedup key (a SHAKE-256 or SHA3-256 output).
DEDUP_KEY_BYTES = 32
#: Modes of the two files.
PUBLIC_MODE = 0o644
PRIVATE_MODE = 0o600

#: Temporary files of the writes (removed under the lock if a writer crashed).
_TEMP_PREFIX = ".slots-"
_TEMP_SUFFIX = ".tmp"
#: How ``bytes`` sit in a record on disk: ``{"__bytes__": "<base64>"}``.
_BYTES_TAG = "__bytes__"


class IssueResult(NamedTuple):
    """What :meth:`SlotStore.issue` returns: ``("hit" or "new", slot id,
    private record)``. Its ``repr`` leaves the private record out."""

    status: str
    slot_id: int
    private: Any

    def __repr__(self) -> str:
        return "IssueResult(status={!r}, slot_id={}, private=<hidden>)".format(self.status, self.slot_id)


class _PrivateSide:
    """The private file, loaded: ``next_slot_id``, the private records
    (slot id -> record as stored) and the dedup index (key -> slot id)."""

    __slots__ = ("next_slot_id", "records", "dedup")

    def __init__(self, next_slot_id: int, records: Dict[int, Any], dedup: Dict[bytes, int]) -> None:
        self.next_slot_id, self.records, self.dedup = next_slot_id, records, dedup

    def __repr__(self) -> str:
        return "<private slot side: next id {}, {} records>".format(self.next_slot_id, len(self.records))


# -- records <-> JSON


def _encode(value: Any) -> Any:
    """*value* as JSON (``bytes`` tagged); never echoes a value in an error."""
    if value is None or isinstance(value, (bool, str)):
        return value
    if isinstance(value, int):
        return int(value)
    if isinstance(value, float):
        if not math.isfinite(value):
            raise FEValidationError("a slot record holds only finite floats")
        return value
    if isinstance(value, (bytes, bytearray, memoryview)):
        return {_BYTES_TAG: base64.b64encode(bytes(value)).decode("ascii")}
    if isinstance(value, (list, tuple)):
        return [_encode(item) for item in value]
    if isinstance(value, dict):
        out = {}
        for name, item in value.items():
            if not isinstance(name, str):
                raise FEValidationError("the keys of a slot record are strings, got {}".format(type(name).__name__))
            if name == _BYTES_TAG:
                raise FEValidationError("{!r} is reserved in slot records".format(_BYTES_TAG))
            out[name] = _encode(item)
        return out
    raise FEValidationError("a slot record holds JSON values and bytes, not {}".format(type(value).__name__))


def _decode(value: Any, path: str) -> Any:
    """The record *value* (as stored in *path*) with its ``bytes`` back; a new
    object (the caller may change it)."""
    if isinstance(value, list):
        return [_decode(item, path) for item in value]
    if isinstance(value, dict):
        if _BYTES_TAG in value:
            data = value[_BYTES_TAG]
            if len(value) != 1 or not isinstance(data, str):
                raise _corrupt(path, "a malformed bytes value")
            try:
                return base64.b64decode(data.encode("ascii"), validate=True)
            except (binascii.Error, UnicodeEncodeError):
                raise _corrupt(path, "a malformed bytes value") from None
        return {name: _decode(item, path) for name, item in value.items()}
    return value


# -- files


def _corrupt(path: str, what: str) -> FEConfigError:
    return FEConfigError(
        "the slot store file {} is unusable ({}); it is not replaced: restore it or move the directory away "
        "(a new store issues keys from slot 0 again)".format(path, what))


def _read_json(path: str) -> Optional[dict]:
    """The JSON object in *path*, ``None`` if the file does not exist."""
    try:
        with open(path, "rb") as f:
            raw = f.read()
    except FileNotFoundError:
        return None
    except OSError as exc:
        raise FEConfigError("the slot store file {} is unreadable ({})".format(
            path, exc.strerror or type(exc).__name__)) from None
    try:
        data = json.loads(raw.decode("utf-8"))
    except (UnicodeDecodeError, ValueError):  # no excerpt of the file in the message
        raise _corrupt(path, "not JSON, or truncated") from None
    if not isinstance(data, dict):
        raise _corrupt(path, "not a JSON object")
    return data


def _check_header(data: dict, path: str, fmt: str) -> None:
    if data.get("format") != fmt:
        raise _corrupt(path, "not a {} file".format(fmt))
    if data.get("format_version") != FORMAT_VERSION:
        raise _corrupt(path, "not format version {} (another version of the library?)".format(FORMAT_VERSION))


def _is_id(value: Any) -> bool:
    return isinstance(value, int) and not isinstance(value, bool) and value >= 0


def _sync(fd: int) -> None:
    """Flush *fd* to the device (``F_FULLFSYNC`` where the OS has it: macOS's
    ``fsync`` leaves the drive's cache unflushed)."""
    import fcntl  # POSIX only; checked by SlotStore._locked first

    if hasattr(fcntl, "F_FULLFSYNC"):
        try:
            fcntl.fcntl(fd, fcntl.F_FULLFSYNC)
            return
        except OSError:
            pass  # not supported by this file system: fsync
    os.fsync(fd)


def _sync_dir(directory: str) -> None:
    """Make a rename in *directory* durable (best effort where a directory
    cannot be opened or synced)."""
    try:
        fd = os.open(directory, os.O_RDONLY | getattr(os, "O_DIRECTORY", 0))
    except OSError:
        return
    try:
        _sync(fd)
    except OSError:
        pass
    finally:
        os.close(fd)


def _atomic_write(directory: str, name: str, data: dict, mode: int) -> None:
    """Replace ``directory/name`` with *data* (JSON): a temporary file in the
    same directory, fsynced, renamed over the old file, then the directory
    fsynced. A crash leaves the old file or the new one, never a mix."""
    payload = json.dumps(data, separators=(",", ":"), allow_nan=False).encode("utf-8")
    fd, tmp = tempfile.mkstemp(dir=directory, prefix=_TEMP_PREFIX, suffix=_TEMP_SUFFIX)  # mode 0o600
    try:
        try:
            os.fchmod(fd, mode)
            view = memoryview(payload)
            while view:
                view = view[os.write(fd, view):]
            _sync(fd)
        finally:
            os.close(fd)
        os.replace(tmp, os.path.join(directory, name))
    except BaseException:
        with contextlib.suppress(OSError):
            os.unlink(tmp)
        raise
    _sync_dir(directory)


# -- arguments


def _dedup_key(value: Any) -> bytes:
    if not isinstance(value, (bytes, bytearray)) or len(value) != DEDUP_KEY_BYTES:
        raise FEValidationError("a dedup key is {} bytes (got {})".format(
            DEDUP_KEY_BYTES, type(value).__name__ if not isinstance(value, (bytes, bytearray)) else
            "{} bytes".format(len(value))))
    return bytes(value)


def _slot_id(value: Any) -> int:
    if not _is_id(value):
        raise FEValidationError("a slot id is an integer >= 0, got {!r}".format(value))
    return value


class SlotStore:
    """The append-only slot list, private records and dedup index of one
    authority instance, in *directory* (created on first use, mode
    ``0o700``). See the module documentation for the files and the crash
    behaviour.

    Several store objects, threads and processes may use one directory: every
    call holds the store object's lock and the directory's file lock
    (``fcntl.flock``; POSIX only). Every call reads the files again, so a
    store object holds no state of the slots.

    Args:
        directory: The directory of this authority instance (path-like).

    Raises:
        FEValidationError: If *directory* is not a non-empty path.
    """

    def __init__(self, directory) -> None:
        if not isinstance(directory, (str, os.PathLike)) or not os.fspath(directory):
            raise FEValidationError("a slot store needs a non-empty directory path, got {}".format(
                type(directory).__name__))
        self._dir = os.path.abspath(os.fspath(directory))
        self._public_path = os.path.join(self._dir, PUBLIC_FILE)
        self._private_path = os.path.join(self._dir, PRIVATE_FILE)
        self._lock = threading.Lock()
        self._holder: Optional[int] = None  # the thread inside _locked() (refuses re-entry from build)
        self._lock_fd: Optional[int] = None  # the file-lock descriptor while held
        forksafe.track(self)  # a forked child: a new lock, the inherited descriptor dropped

    def _after_fork_in_child(self) -> None:
        """In a forked child: a new lock (another thread of the parent may hold
        the old one), and the child's copy of a held file-lock descriptor
        closed (closing a copy does not release the parent's lock; keeping it
        would keep the lock alive after the parent let go of it)."""
        self._lock = threading.Lock()
        self._holder = None
        if self._lock_fd is not None:
            with contextlib.suppress(OSError):
                os.close(self._lock_fd)
            self._lock_fd = None

    def __reduce__(self):
        return (type(self), (self._dir,))  # copies and pickles: a new store object, its own lock

    def __repr__(self) -> str:
        return "{}({!r})".format(type(self).__name__, self._dir)

    @property
    def directory(self) -> str:
        """The directory of the store (absolute)."""
        return self._dir

    # -- locking and loading

    @contextlib.contextmanager
    def _locked(self) -> Iterator[None]:
        """Hold the store object's lock and the directory's exclusive file
        lock; remove temporary files a crashed writer left."""
        if self._holder == threading.get_ident():
            raise FESchemeError("the slot store was used from inside its own issue(): build() must not call it")
        try:
            import fcntl
        except ImportError:  # pragma: no cover - Windows
            raise FESchemeError("the slot store needs POSIX file locks (fcntl), which this platform lacks") from None
        with self._lock:
            self._holder = threading.get_ident()
            try:
                os.makedirs(self._dir, mode=0o700, exist_ok=True)
                fd = os.open(os.path.join(self._dir, LOCK_FILE), os.O_RDWR | os.O_CREAT, PRIVATE_MODE)
                pid = os.getpid()
                self._lock_fd = fd
                try:
                    fcntl.flock(fd, fcntl.LOCK_EX)
                    try:
                        self._remove_temporary_files()
                        yield
                    finally:
                        if os.getpid() == pid:  # a child forked meanwhile must not release the parent's lock
                            fcntl.flock(fd, fcntl.LOCK_UN)
                finally:
                    if os.getpid() == pid:
                        self._lock_fd = None
                        os.close(fd)
            finally:
                self._holder = None

    def _remove_temporary_files(self) -> None:
        for name in os.listdir(self._dir):
            if name.startswith(_TEMP_PREFIX) and name.endswith(_TEMP_SUFFIX):
                with contextlib.suppress(OSError):
                    os.unlink(os.path.join(self._dir, name))

    def _load(self) -> Tuple[List[dict], _PrivateSide]:
        """Both files, checked against each other (under the lock).

        Raises:
            FEConfigError: If a file is unreadable, truncated, of another
                format or version, or the two do not fit together.
        """
        public = self._load_public()
        ids = [slot["slot_id"] for slot in public]
        return public, self._load_private(set(ids), ids[-1] if ids else -1)

    def _load_public(self) -> List[dict]:
        path = self._public_path
        data = _read_json(path)
        if data is None:
            return []
        _check_header(data, path, PUBLIC_FORMAT)
        slots = data.get("slots")
        if not isinstance(slots, list) or data.get("slot_list_version") != len(slots):
            raise _corrupt(path, "the slot list and its version do not match")
        last = -1
        for slot in slots:
            if not isinstance(slot, dict) or not _is_id(slot.get("slot_id")) or slot["slot_id"] <= last:
                raise _corrupt(path, "the slot ids are not increasing")
            last = slot["slot_id"]
        _decode(slots, path)  # every bytes value well formed
        return slots

    def _load_private(self, public_ids: set, last_id: int) -> _PrivateSide:
        path = self._private_path
        data = _read_json(path)
        if data is None:
            if public_ids:  # written before the first public record: missing means lost
                raise _corrupt(path, "missing while the slot list exists")
            return _PrivateSide(0, {}, {})
        _check_header(data, path, PRIVATE_FORMAT)
        next_id, records, dedup = data.get("next_slot_id"), data.get("private"), data.get("dedup")
        if not _is_id(next_id) or next_id <= last_id:
            raise _corrupt(path, "next_slot_id does not exceed the listed slot ids: an id would be reused")
        if not isinstance(records, dict) or not isinstance(dedup, dict):
            raise _corrupt(path, "no private records or dedup index")
        loaded: Dict[int, Any] = {}
        for name, record in records.items():
            if not name.isdigit() or str(int(name)) != name or int(name) not in public_ids:
                raise _corrupt(path, "a private record of a slot that is not listed")
            _decode(record, path)  # every bytes value well formed
            loaded[int(name)] = record
        index: Dict[bytes, int] = {}
        for name, slot_id in dedup.items():
            try:
                key = bytes.fromhex(name)
            except ValueError:
                key = b""
            if len(key) != DEDUP_KEY_BYTES or name != key.hex() or not _is_id(slot_id) or slot_id not in loaded:
                raise _corrupt(path, "a malformed dedup entry")
            index[key] = slot_id
        if len(set(index.values())) != len(index):
            raise _corrupt(path, "two dedup keys of one slot")
        return _PrivateSide(next_id, loaded, index)

    def _write_public(self, public: List[dict]) -> None:
        _atomic_write(self._dir, PUBLIC_FILE, {
            "format": PUBLIC_FORMAT, "format_version": FORMAT_VERSION,
            "slot_list_version": len(public), "slots": public}, PUBLIC_MODE)

    def _write_private(self, side: _PrivateSide) -> None:
        _atomic_write(self._dir, PRIVATE_FILE, {
            "format": PRIVATE_FORMAT, "format_version": FORMAT_VERSION,
            "next_slot_id": side.next_slot_id,
            "private": {str(slot_id): record for slot_id, record in sorted(side.records.items())},
            "dedup": {key.hex(): slot_id for key, slot_id in side.dedup.items()}}, PRIVATE_MODE)

    def _snapshot(self) -> Tuple[List[dict], _PrivateSide]:
        if not os.path.isdir(self._dir):  # nothing issued yet: no directory made by a read
            return [], _PrivateSide(0, {}, {})
        with self._locked():
            return self._load()

    # -- issuing

    def issue(self, dedup_key: bytes, max_keys: int,
              build: Callable[[int], Tuple[dict, Any]]) -> IssueResult:
        """The slot of *dedup_key*: the stored one, or a new one made by
        *build*, under one exclusive lock.

        1. Look *dedup_key* up.
        2. If the list holds ``max_keys`` slots or more: raise
           :class:`FEKeyError`, with the same text whether step 1 found a slot
           or not; nothing is written.
        3. Found: return ``IssueResult("hit", slot_id, private)`` without
           calling *build*.
        4. Otherwise reserve the next slot id (persisted: never handed out
           again), call ``build(slot_id)`` -> ``(public_record,
           private_record)``, append the public record (``slot_id`` added),
           store the private record with *dedup_key* and return
           ``IssueResult("new", slot_id, private)``.

        If *build* raises, or returns records that cannot be stored, the
        exception propagates and only the reserved id is burned. *build* must
        not call the store.

        Args:
            dedup_key: 32 bytes; private (never written to the public file).
            max_keys: The cap on the number of slots (an integer >= 1; it may
                grow between calls).
            build: Makes the slot's records from its id: the public record a
                dict (its ``slot_id``, if present, must be the one given), the
                private record a dict or ``bytes``; JSON values and bytes.

        Returns:
            ``(status, slot_id, private)``; *private* is a new object each
            call, equal for a hit to what the call that made the slot
            returned.

        Raises:
            FEKeyError: If the cap is reached.
            FEValidationError: For a malformed argument or record.
            FEConfigError: If the files are unusable (nothing is written).
            FESchemeError: If *build* uses the store.
        """
        key = _dedup_key(dedup_key)
        if not isinstance(max_keys, int) or isinstance(max_keys, bool) or max_keys < 1:
            raise FEValidationError("max_keys is an integer >= 1, got {!r}".format(max_keys))
        if not callable(build):
            raise FEValidationError("build must be callable, got {}".format(type(build).__name__))
        with self._locked():
            public, side = self._load()
            hit = side.dedup.get(key)
            if len(public) >= max_keys:  # the same error, hit or miss
                raise FEKeyError(
                    "the key cap max_keys = {} of this setup is reached: no key is issued or returned".format(
                        max_keys))
            if hit is not None:
                return IssueResult("hit", hit, _decode(side.records[hit], self._private_path))
            slot_id = side.next_slot_id
            side.next_slot_id = slot_id + 1
            self._write_private(side)  # the reservation: slot_id is burned from here on
            pid = os.getpid()
            try:
                public_record, private_record = self._build(build, slot_id)
            except BaseException as exc:
                logger.info("slot id %d is burned: build() raised %s", slot_id, type(exc).__name__)
                raise
            if os.getpid() != pid:
                raise FESchemeError("build() forked: the child does not hold the slot store's lock")
            public.append(public_record)
            self._write_public(public)  # listed (not delivered yet)
            side.records[slot_id] = private_record
            side.dedup[key] = slot_id
            self._write_private(side)  # delivered
            logger.debug("slot %d issued (slot list version %d)", slot_id, len(public))
            return IssueResult("new", slot_id, _decode(private_record, self._private_path))

    @staticmethod
    def _build(build, slot_id: int) -> Tuple[dict, Any]:
        """``build(slot_id)``, its records checked and encoded for the files."""
        out = build(slot_id)
        if not isinstance(out, tuple) or len(out) != 2:
            raise FEValidationError("build() returns (public_record, private_record)")
        public_record, private_record = out
        if not isinstance(public_record, dict):
            raise FEValidationError("a public slot record is a dict, got {}".format(type(public_record).__name__))
        if "slot_id" in public_record and (not _is_id(public_record["slot_id"])
                                           or public_record["slot_id"] != slot_id):
            raise FEValidationError("the public record of slot {} names another slot_id".format(slot_id))
        if not isinstance(private_record, (dict, bytes, bytearray)):
            raise FEValidationError("a private slot record is a dict or bytes, got {}".format(
                type(private_record).__name__))
        encoded = _encode({name: item for name, item in public_record.items() if name != "slot_id"})
        return {"slot_id": slot_id, **encoded}, _encode(private_record)

    # -- reading

    def public_slots(self) -> List[dict]:
        """The slot list: the public records in issue order, each with its
        ``slot_id`` (new objects)."""
        public, _ = self._snapshot()
        return _decode(public, self._public_path)

    def version(self) -> int:
        """``slot_list_version``: the number of slots listed."""
        public, _ = self._snapshot()
        return len(public)

    def private(self, slot_id: int) -> Any:
        """The private record of slot *slot_id* (a new object), ``None`` if
        the slot is not listed or was never delivered."""
        slot_id = _slot_id(slot_id)
        _, side = self._snapshot()
        record = side.records.get(slot_id)
        return None if record is None else _decode(record, self._private_path)

    def lookup(self, dedup_key: bytes) -> Optional[int]:
        """The slot id stored for *dedup_key*, ``None`` if there is none."""
        key = _dedup_key(dedup_key)
        _, side = self._snapshot()
        return side.dedup.get(key)
