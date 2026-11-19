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
    record of each slot, the deduplication index (dedup key -> slot id) and
    the *pending* public records: a write-ahead copy of the public record of
    the slot being issued, until a later write of this file finds it
    listed. Neither the dedup keys (hashes of the requested functions) nor
    the private records ever reach the public file, a ``repr``, an
    exception message or a log line: a public dedup key is an offline test
    of guesses of a function.
``slots.lock``
    The file lock (``fcntl.flock``); never removed.

:meth:`SlotStore.issue` runs under one exclusive lock (the store object's
lock for its threads, the file lock for every other process and store
object): look the dedup key up; refuse with the same :class:`~pyfe4ai.core.exceptions.FEKeyError`
and the same text, hit or miss, once the list holds ``max_keys`` slots (so
that a requester cannot test functions through the deduplication once the
cap is reached); return a hit without building; otherwise

1. reserve the next slot id (the private file, before ``build`` runs);
2. call ``build(slot_id)``;
3. write the private file with the slot's private record, its dedup entry
   and the pending copy of its public record (the commit point);
4. append the public record to the slot list.

Every write goes to a temporary file in the directory that is fsynced,
renamed over the old file and followed by an fsync of the directory.

**Recovery (roll forward).** Every locked call, reads included, first lists
the pending public records whose slots the slot list lacks (in slot-id
order, with the same write discipline) and only then does its work. A crash
(or an exception) anywhere therefore leaves one of two states:

* before step 3 (before the reservation, in ``build``, or in a write of
  step 1 or 3 that did not reach the rename): nothing is visible, except a
  burned id when step 1 completed; a retry gets a new id;
* from step 3 on: the slot is issued. It is listed (by step 4 or by the
  recovery), its dedup entry and private record exist, and a retry of the
  same request is a hit that returns the stored private record (the key
  under central issuance, the grant when the holder made its own keys).

The order matters when the holder made its own keys: it can use its slot as
soon as the slot is listed, so a listed slot must always be reachable
through its dedup key, or a retry would give the same requester a second
slot that the dedup index does not know. The public file is therefore never
written ahead of the private one. Rejected alternatives: (a) persist the
dedup key with the reservation (a retry would hit a slot whose private
record, e.g. its secret ``k``, was never stored); (b) count slots by the
public list only (bounds the slots, but a retry still makes a second one);
(c) the private file first without rolling forward (a dedup entry of an
unlisted slot, refused as inconsistent). With central issuance and
``keep_issued_keys`` off, a crash after step 3 leaves an issued slot whose
key no one holds: its retry is a hit the scheme refuses (a lost key is not
issued again).

Temporary files of a crashed writer are removed by the next locked call.
Unreadable, truncated or inconsistent files raise :class:`~pyfe4ai.core.exceptions.FEConfigError`
(fail closed): the store never starts empty while its files exist. The two
files are consistent when every listed slot has its private record and its
one dedup entry, a pending copy of a listed slot equals the listed record,
``next_slot_id`` exceeds every slot id, and a private record of a slot that
is not listed carries a pending public record with a slot id above every
listed one (the one state the recovery completes); anything else is
refused.

**Trust in the directory.** Whoever can write the directory can reset the
store (slot ids handed out again) or replace a listed slot's public keys by
keys whose secrets they know, which the authority would then publish. So
every locked call, reads included, first checks the directory: not a
symbolic link (``lstat``), a directory, owned by this process's effective
user and not writable by group or others (``mode & 0o022 == 0``; a missing
directory is created with mode ``0o700``, set again after the creation as
the umask may clear bits). Each store file is opened with ``O_NOFOLLOW``
(and ``O_NONBLOCK``, so that a FIFO cannot block the call) and must be a
regular file of that user that no other user can write; the private file
and the lock file must not be readable by others either. Anything else
raises :class:`~pyfe4ai.core.exceptions.FEConfigError` naming the path and the reason (never the
content). The directories above the store are not checked (a cache
directory reached through a symbolic link works), nor are the files other
code keeps in the directory. Alternative not taken: trust the caller's
directory (the first version did), which leaves the integrity of the store
to the permissions of a path the store does not control.

Every locked call opens the directory once (``O_DIRECTORY | O_NOFOLLOW``),
checks it through that descriptor (``fstat``), and opens, writes, renames and
removes the store files relative to it (``dir_fd``), so all the files of one
call are those of the directory it checked: a directory renamed away during
the call keeps receiving its writes, and one put in its place is not touched.
:func:`read_json_file` and :func:`write_json_file` do the same for one file
each. Where the platform lacks the ``*at`` calls (``os.supports_dir_fd``) the
files are opened by path after the check, as before. Alternative not taken:
check by path and open by path (the first version), which leaves a window in
which a directory swapped in after the check is used.

Known limits of these checks: they read the mode bits only. On Linux a POSIX
ACL entry that lets another user write shows in the group bits (the ACL
mask), so it is refused; a macOS ACL (``chmod +a``) does not change the mode
bits and is not seen: do not put ACLs on the directory (``ls -led`` shows
them). Alternative not taken: refuse a directory with an extended macOS ACL
(``acl_get_fd_np`` through ``ctypes``), platform-specific code for a
directory that only its owner can give an ACL. Two calls are two opens: a
directory swapped between two calls is checked again by the second, and
passes only if it is a directory of the same user that no other user can
write -- for instance a copy of the store (see :class:`SlotStore`: never
issue from two copies of one directory).

The owner of a store may keep files of its own in the directory (the
bounded-key scheme keeps its setup and its options there):
:func:`read_json_file` and :func:`write_json_file` read and write them with
the same checks and the same durable writes, and
:meth:`SlotStore.check_directory` checks the directory before anything in
it is read.

**A cap the caller keeps.** The cap of :meth:`SlotStore.issue` is the
caller's (the bounded-key scheme keeps it in a file of its own, and it may
be changed after setup). Read before the lock, it could be lowered by
another process between the read and the issue, which would then list more
slots than the cap. So :meth:`SlotStore.issue` also takes the cap as a
callable that it evaluates under its lock, and :meth:`SlotStore.run_locked`
runs a caller's check-and-write (lowering the cap not below the slots
listed) under the same lock: no issuance runs between that check and that
write. Alternative not taken: keep the cap in the store's private file
(one file and one lock as now, but the store would then own an option of
the scheme and its changes).

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
import secrets
import stat
import tempfile
import threading
from typing import Any, Callable, Dict, Iterator, List, NamedTuple, Optional, Tuple, Union

from pyfe4ai.core import forksafe
from pyfe4ai.core.exceptions import FEConfigError, FEKeyError, FESchemeError, FEValidationError

logger = logging.getLogger(__name__)

#: The public slot list.
PUBLIC_FILE = "slots.json"
#: The private side (reservations, private records, the dedup index).
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
#: The files of a checked directory are opened relative to its descriptor (``dir_fd``) where the platform can;
#: ``os.replace`` takes ``src_dir_fd`` / ``dst_dir_fd`` where ``os.rename`` does.
_DIR_FD = (all(f in os.supports_dir_fd for f in (os.open, os.stat, os.unlink, os.rename))
           and os.listdir in os.supports_fd)
_O_DIRECTORY = getattr(os, "O_DIRECTORY", 0)


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
    (slot id -> record as stored), the dedup index (key -> slot id) and the
    pending public records (slot id -> record as stored)."""

    __slots__ = ("next_slot_id", "records", "dedup", "pending")

    def __init__(self, next_slot_id: int, records: Dict[int, Any], dedup: Dict[bytes, int],
                 pending: Dict[int, dict]) -> None:
        self.next_slot_id, self.records, self.dedup, self.pending = next_slot_id, records, dedup, pending

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


def _refused(path: str, reason: str) -> FEConfigError:
    return FEConfigError(
        "the slot store refuses {}: {} (it trusts only a directory and files of its own user that no other user "
        "can write)".format(path, reason))


def _check_owner_and_mode(st: os.stat_result, path: str, private: bool) -> None:
    """Refuse *path* (its ``stat`` *st*) unless this process's effective user
    owns it and no other user can write it (*private*: nor read it)."""
    euid = os.geteuid()
    if st.st_uid != euid:
        raise _refused(path, "it is owned by uid {}, not by this process's uid {}".format(st.st_uid, euid))
    mode = stat.S_IMODE(st.st_mode)
    if mode & (0o077 if private else 0o022):
        raise _refused(path, "its mode {:04o} lets other users {} it".format(
            mode, "read or write" if private else "write"))


def _open_directory(path: str) -> int:
    """A descriptor of the directory *path*, opened without following a
    symbolic link (``O_DIRECTORY | O_NOFOLLOW``) and checked through it
    (``fstat``): a directory of this process's effective user that no other
    user can write.

    Raises:
        FileNotFoundError: If *path* does not exist.
        FEConfigError: If it is refused or cannot be opened.
    """
    try:
        fd = os.open(path, os.O_RDONLY | _O_DIRECTORY | os.O_NOFOLLOW)
    except FileNotFoundError:
        raise
    except OSError as exc:  # ELOOP / ENOTDIR for a symbolic link or a file, EACCES, ...
        try:
            st = os.lstat(path)
        except OSError:
            st = None
        if st is not None and stat.S_ISLNK(st.st_mode):
            raise _refused(path, "it is a symbolic link") from None
        if st is not None and not stat.S_ISDIR(st.st_mode):
            raise _refused(path, "it is not a directory") from None
        raise FEConfigError("the slot store directory {} cannot be opened ({})".format(
            path, exc.strerror or type(exc).__name__)) from None
    try:
        st = os.fstat(fd)
        if not stat.S_ISDIR(st.st_mode):  # O_DIRECTORY missing on this platform
            raise _refused(path, "it is not a directory")
        _check_owner_and_mode(st, path, private=False)
    except BaseException:
        os.close(fd)
        raise
    return fd


def _os_open(path: str, target: str, flags: int, rel: dict) -> int:
    """``os.open(target, flags, 0o600, **rel)``. Relative to a directory
    descriptor and with ``O_CREAT``, the file is created with ``O_EXCL`` and,
    if it exists, opened without ``O_CREAT``: on macOS, of two threads that
    ``openat`` one new name with ``O_CREAT`` at once, one can fail with
    ``ENOENT`` (an open by path retries in the kernel)."""
    if not (rel and flags & os.O_CREAT):
        return os.open(target, flags, PRIVATE_MODE, **rel)
    for _ in range(100):
        try:
            return os.open(target, flags | os.O_EXCL, PRIVATE_MODE, **rel)
        except FileExistsError:  # also a symbolic link or another kind of file: refused by the open below
            pass
        try:
            return os.open(target, flags & ~os.O_CREAT, PRIVATE_MODE, **rel)
        except FileNotFoundError:  # removed in between: create it again
            continue
    raise FEConfigError("the slot store file {} keeps disappearing while it is opened".format(path))


def _open_file(path: str, flags: int, private: bool, dir_fd: Optional[int] = None) -> Optional[int]:
    """A descriptor of the store file *path*, opened without following a
    symbolic link and without blocking on a FIFO, once it is checked to be a
    regular file of this user (:func:`_check_owner_and_mode`); ``None`` if it
    does not exist (and *flags* do not create it, with mode ``0o600``). With
    *dir_fd* (a checked directory, :func:`_open_directory`), the file named
    by the last component of *path* in that directory; *path* names it in
    messages."""
    target, rel = (os.path.basename(path), {"dir_fd": dir_fd}) if dir_fd is not None else (path, {})
    try:
        fd = _os_open(path, target, flags | os.O_NOFOLLOW | os.O_NONBLOCK, rel)
    except FileNotFoundError:
        return None
    except OSError as exc:  # ELOOP for a symbolic link, EISDIR for a directory opened for writing, ...
        try:
            st = os.lstat(target, **rel)
        except OSError:
            st = None
        if st is not None and stat.S_ISLNK(st.st_mode):
            raise _refused(path, "it is a symbolic link") from None
        if st is not None and not stat.S_ISREG(st.st_mode):
            raise _refused(path, "it is not a regular file") from None
        raise FEConfigError("the slot store file {} is unreadable ({})".format(
            path, exc.strerror or type(exc).__name__)) from None
    try:
        st = os.fstat(fd)
        if not stat.S_ISREG(st.st_mode):
            raise _refused(path, "it is not a regular file")
        _check_owner_and_mode(st, path, private)
        os.set_blocking(fd, True)
    except BaseException:
        os.close(fd)
        raise
    return fd


def _read_json(path: str, private: bool, dir_fd: Optional[int] = None) -> Optional[dict]:
    """The JSON object in the store file *path* (checked by
    :func:`_open_file`; in the directory *dir_fd* if given), ``None`` if the
    file does not exist."""
    fd = _open_file(path, os.O_RDONLY, private, dir_fd)
    if fd is None:
        return None
    try:
        with os.fdopen(fd, "rb") as f:
            raw = f.read()
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


def _parse_id(name: str) -> Optional[int]:
    """The slot id written as *name* (canonical decimal), ``None`` otherwise."""
    if name.isascii() and name.isdigit() and str(int(name)) == name:
        return int(name)
    return None


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


def _make_temp(dir_fd: int) -> Tuple[int, str]:
    """A new temporary file (mode ``0o600``, ``O_EXCL``, no symbolic link
    followed) in the directory *dir_fd*: its descriptor and its name."""
    for _ in range(100):
        name = "{}{}{}".format(_TEMP_PREFIX, secrets.token_hex(8), _TEMP_SUFFIX)
        try:
            fd = os.open(name, os.O_RDWR | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW, PRIVATE_MODE, dir_fd=dir_fd)
        except FileExistsError:
            continue
        return fd, name
    raise FileExistsError("no free temporary file name in the slot store directory")  # pragma: no cover


def _atomic_write(directory: str, name: str, data: dict, mode: int, dir_fd: Optional[int] = None) -> None:
    """Replace ``directory/name`` with *data* (JSON): a temporary file in the
    same directory, fsynced, renamed over the old file, then the directory
    fsynced. A crash leaves the old file or the new one, never a mix. With
    *dir_fd* (the checked descriptor of *directory*), every step runs
    relative to it."""
    payload = json.dumps(data, separators=(",", ":"), allow_nan=False).encode("utf-8")
    if dir_fd is not None:
        fd, tmp = _make_temp(dir_fd)
        rel: dict = {"dir_fd": dir_fd}
    else:
        fd, tmp = tempfile.mkstemp(dir=directory, prefix=_TEMP_PREFIX, suffix=_TEMP_SUFFIX)  # mode 0o600
        rel = {}
    try:
        try:
            os.fchmod(fd, mode)
            view = memoryview(payload)
            while view:
                view = view[os.write(fd, view):]
            _sync(fd)
        finally:
            os.close(fd)
        if dir_fd is not None:
            os.replace(tmp, name, src_dir_fd=dir_fd, dst_dir_fd=dir_fd)
        else:
            os.replace(tmp, os.path.join(directory, name))
    except BaseException:
        with contextlib.suppress(OSError):
            os.unlink(tmp, **rel)
        raise
    if dir_fd is not None:
        with contextlib.suppress(OSError):  # best effort, as for a directory opened by path
            _sync(dir_fd)
    else:
        _sync_dir(directory)


# -- arguments


def _dedup_key(value: Any) -> bytes:
    if not isinstance(value, (bytes, bytearray)) or len(value) != DEDUP_KEY_BYTES:
        raise FEValidationError("a dedup key is {} bytes (got {})".format(
            DEDUP_KEY_BYTES, type(value).__name__ if not isinstance(value, (bytes, bytearray)) else
            "{} bytes".format(len(value))))
    return bytes(value)


def _check_cap(value: Any) -> int:
    if not isinstance(value, int) or isinstance(value, bool) or value < 1:
        raise FEValidationError("max_keys is an integer >= 1, got {!r}".format(value))
    return value


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

    One directory, one store: the locks order the calls on one directory,
    not on its copies. Two copies of a directory (a second host, a backup
    restored while the original goes on) each hand out the same slot ids
    again, to different slots, and each counts its own cap. Issue from one
    copy only; after restoring a backup, issue from the restored directory
    alone and treat the slots issued after the backup as lost. Alternative
    not taken: record the directory's device and inode in the private file
    and refuse a store that has moved, which would also refuse every restore,
    move or file-system migration until an operator confirmed it.

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
        self._dir_fd: Optional[int] = None  # the checked directory's descriptor while held
        forksafe.track(self)  # a forked child: a new lock, the inherited descriptors dropped

    def _after_fork_in_child(self) -> None:
        """In a forked child: a new lock (another thread of the parent may hold
        the old one), and the child's copies of a held file-lock descriptor and
        directory descriptor closed (closing a copy does not release the
        parent's lock; keeping it would keep the lock alive after the parent
        let go of it)."""
        self._lock = threading.Lock()
        self._holder = None
        for fd in (self._lock_fd, self._dir_fd):
            if fd is not None:
                with contextlib.suppress(OSError):
                    os.close(fd)
        self._lock_fd = self._dir_fd = None

    def _rel(self) -> Optional[int]:
        """The descriptor the store files are opened relative to, under the
        lock; ``None`` where the platform lacks the ``*at`` calls (by path)."""
        return self._dir_fd if _DIR_FD else None

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
            raise FESchemeError("the slot store was used from inside its own issue() or run_locked(): build() and "
                                "the callables they run must not call it")
        try:
            import fcntl
        except ImportError:  # pragma: no cover - Windows
            raise FESchemeError("the slot store needs POSIX file locks (fcntl), which this platform lacks") from None
        with self._lock:
            self._holder = threading.get_ident()
            try:
                dfd = self._open_checked_directory()
                pid = os.getpid()
                self._dir_fd = dfd
                try:
                    fd = _open_file(os.path.join(self._dir, LOCK_FILE), os.O_RDWR | os.O_CREAT, private=True,
                                    dir_fd=self._rel())
                    if fd is None:  # O_CREAT: only if the directory went away meanwhile
                        raise FEConfigError("the slot store's lock file {} cannot be created".format(
                            os.path.join(self._dir, LOCK_FILE)))
                    self._lock_fd = fd
                    try:
                        if stat.S_IMODE(os.fstat(fd).st_mode) != PRIVATE_MODE:  # a umask took 0o600 bits
                            os.fchmod(fd, PRIVATE_MODE)
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
                    if os.getpid() == pid:
                        self._dir_fd = None
                        os.close(dfd)
            finally:
                self._holder = None

    def _open_checked_directory(self) -> int:
        """Create the store directory (mode ``0o700``) if it is missing, then
        open it and check it through its descriptor (:func:`_open_directory`):
        a directory, not a symbolic link, owned by this process's user and not
        writable by other users."""
        try:
            return _open_directory(self._dir)
        except FileNotFoundError:
            self._make_directory()
        try:
            return _open_directory(self._dir)
        except FileNotFoundError:
            raise FEConfigError("the slot store directory {} was removed while it was made".format(self._dir)) from None

    def _make_directory(self) -> os.stat_result:
        os.makedirs(os.path.dirname(self._dir), exist_ok=True)
        try:
            os.mkdir(self._dir, 0o700)
        except FileExistsError:  # made by another process meanwhile: checked as found
            return os.lstat(self._dir)
        fd = os.open(self._dir, os.O_RDONLY | getattr(os, "O_DIRECTORY", 0) | os.O_NOFOLLOW)
        try:
            os.fchmod(fd, 0o700)  # the umask may have cleared bits of 0o700
        finally:
            os.close(fd)
        return os.lstat(self._dir)

    def _remove_temporary_files(self) -> None:
        rel = self._rel()
        for name in os.listdir(self._dir if rel is None else rel):
            if name.startswith(_TEMP_PREFIX) and name.endswith(_TEMP_SUFFIX):
                with contextlib.suppress(OSError):
                    if rel is None:
                        os.unlink(os.path.join(self._dir, name))
                    else:
                        os.unlink(name, dir_fd=rel)

    def _load(self) -> Tuple[List[dict], _PrivateSide]:
        """Both files, checked against each other, after the recovery: the
        pending public records of slots the slot list lacks are appended to
        it first, in slot-id order (under the lock).

        Raises:
            FEConfigError: If a file is unreadable, truncated, of another
                format or version, or the two do not fit together.
        """
        public = self._load_public()
        side = self._load_private(public)
        listed = {slot["slot_id"] for slot in public}
        missing = sorted(slot_id for slot_id in side.pending if slot_id not in listed)
        if missing:  # an issue stopped after its commit point: roll it forward
            public = public + [side.pending[slot_id] for slot_id in missing]
            self._write_public(public)
            logger.info("slot ids %s listed from their write-ahead records (an interrupted issue)", missing)
        return public, side

    def _load_public(self) -> List[dict]:
        path = self._public_path
        data = _read_json(path, private=False, dir_fd=self._rel())
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

    def _load_private(self, public: List[dict]) -> _PrivateSide:
        path = self._private_path
        listed = {slot["slot_id"]: slot for slot in public}
        last_id = public[-1]["slot_id"] if public else -1
        data = _read_json(path, private=True, dir_fd=self._rel())
        if data is None:
            if public:  # written before the first public record: missing means lost
                raise _corrupt(path, "missing while the slot list exists")
            return _PrivateSide(0, {}, {}, {})
        _check_header(data, path, PRIVATE_FORMAT)
        next_id = data.get("next_slot_id")
        records, dedup, pending = data.get("private"), data.get("dedup"), data.get("pending")
        if not all(isinstance(part, dict) for part in (records, dedup, pending)):
            raise _corrupt(path, "no private records, dedup index or pending records")
        loaded: Dict[int, Any] = {}
        for name, record in records.items():
            slot_id = _parse_id(name)
            if slot_id is None:
                raise _corrupt(path, "a malformed slot id")
            _decode(record, path)  # every bytes value well formed
            loaded[slot_id] = record
        waiting: Dict[int, dict] = {}
        for name, record in pending.items():
            slot_id = _parse_id(name)
            if slot_id is None or slot_id not in loaded:
                raise _corrupt(path, "a pending public record of a slot without a private record")
            if not isinstance(record, dict) or not _is_id(record.get("slot_id")) or record["slot_id"] != slot_id:
                raise _corrupt(path, "a malformed pending public record")
            if slot_id in listed:
                if listed[slot_id] != record:
                    raise _corrupt(path, "a pending public record differs from the listed one")
            elif slot_id <= last_id:  # the recovery appends: never below a listed id
                raise _corrupt(path, "a pending public record below a listed slot id")
            _decode(record, path)
            waiting[slot_id] = record
        if any(slot_id not in listed and slot_id not in waiting for slot_id in loaded):
            raise _corrupt(path, "a private record of a slot that is neither listed nor pending")
        if any(slot_id not in loaded for slot_id in listed):
            raise _corrupt(path, "a listed slot without its private record")
        if not _is_id(next_id) or next_id <= max([last_id, *loaded]):
            raise _corrupt(path, "next_slot_id does not exceed the slot ids: an id would be reused")
        index: Dict[bytes, int] = {}
        for name, slot_id in dedup.items():
            try:
                key = bytes.fromhex(name)
            except ValueError:
                key = b""
            if len(key) != DEDUP_KEY_BYTES or name != key.hex() or not _is_id(slot_id) or slot_id not in loaded:
                raise _corrupt(path, "a malformed dedup entry")
            index[key] = slot_id
        if len(set(index.values())) != len(index) or set(index.values()) != set(loaded):
            raise _corrupt(path, "not exactly one dedup entry per slot")
        return _PrivateSide(next_id, loaded, index, waiting)

    def _write_public(self, public: List[dict]) -> None:
        _atomic_write(self._dir, PUBLIC_FILE, {
            "format": PUBLIC_FORMAT, "format_version": FORMAT_VERSION,
            "slot_list_version": len(public), "slots": public}, PUBLIC_MODE, self._rel())

    def _write_private(self, side: _PrivateSide, public: List[dict]) -> None:
        """Write *side*; the pending public records of slots *public* lists
        are dropped (the slot list holds them)."""
        listed = {slot["slot_id"] for slot in public}
        _atomic_write(self._dir, PRIVATE_FILE, {
            "format": PRIVATE_FORMAT, "format_version": FORMAT_VERSION,
            "next_slot_id": side.next_slot_id,
            "private": {str(slot_id): record for slot_id, record in sorted(side.records.items())},
            "dedup": {key.hex(): slot_id for key, slot_id in side.dedup.items()},
            "pending": {str(slot_id): record for slot_id, record in sorted(side.pending.items())
                        if slot_id not in listed}}, PRIVATE_MODE, self._rel())

    def _snapshot(self) -> Tuple[List[dict], _PrivateSide]:
        try:
            os.lstat(self._dir)  # a symbolic link or a file is refused by _locked
        except FileNotFoundError:  # nothing issued yet: no directory made by a read
            return [], _PrivateSide(0, {}, {}, {})
        with self._locked():
            return self._load()

    # -- issuing

    def issue(self, dedup_key: bytes, max_keys: Union[int, Callable[[], int]],
              build: Callable[[int], Tuple[dict, Any]]) -> IssueResult:
        """The slot of *dedup_key*: the stored one, or a new one made by
        *build*, under one exclusive lock.

        1. Look *dedup_key* up.
        2. If the list holds ``max_keys`` slots or more: raise
           :class:`~pyfe4ai.core.exceptions.FEKeyError`, with the same text whether step 1 found a slot
           or not; nothing is written.
        3. Found: return ``IssueResult("hit", slot_id, private)`` without
           calling *build*.
        4. Otherwise reserve the next slot id (persisted: never handed out
           again), call ``build(slot_id)`` -> ``(public_record,
           private_record)``, store the private record with *dedup_key* and
           a pending copy of the public record (``slot_id`` added), append
           the public record and return ``IssueResult("new", slot_id,
           private)``.

        If *build* raises, or returns records that cannot be stored, the
        exception propagates and only the reserved id is burned. An error
        after the private record is stored (the public write) propagates
        too, but the slot is issued: the next call of any method lists it,
        and a retry of *dedup_key* is a hit. *build* must not call the store.

        Args:
            dedup_key: 32 bytes; private (never written to the public file).
            max_keys: The cap on the number of slots: an integer >= 1, or
                a callable without arguments that returns it, called once
                under the lock before step 1 (where the caller keeps the cap
                elsewhere and another process may change it; it must not
                call the store). It may change between calls.
            build: Makes the slot's records from its id: the public record a
                dict (its ``slot_id``, if present, must be the one given), the
                private record a dict or ``bytes``; JSON values and bytes.

        Returns:
            ``(status, slot_id, private)``; *private* is a new object each
            call, equal for a hit to what the call that made the slot
            returned.

        Raises:
            FEKeyError: If the cap is reached.
            FEValidationError: For a malformed argument or record (also a cap
                the callable returns).
            FEConfigError: If the files are unusable (nothing is written).
            FESchemeError: If *build* (or the cap's callable) uses the store.
        """
        key = _dedup_key(dedup_key)
        if not callable(max_keys):
            _check_cap(max_keys)
        if not callable(build):
            raise FEValidationError("build must be callable, got {}".format(type(build).__name__))
        with self._locked():
            public, side = self._load()
            if callable(max_keys):  # decided under the lock: no change of the cap between this and the issue
                max_keys = _check_cap(max_keys())
            hit = side.dedup.get(key)
            if len(public) >= max_keys:  # the same error, hit or miss
                raise FEKeyError(
                    "the key cap max_keys = {} of this setup is reached: no key is issued or returned".format(
                        max_keys))
            if hit is not None:
                return IssueResult("hit", hit, _decode(side.records[hit], self._private_path))
            slot_id = side.next_slot_id
            side.next_slot_id = slot_id + 1
            self._write_private(side, public)  # 1. the reservation: slot_id is burned from here on
            pid = os.getpid()
            try:
                public_record, private_record = self._build(build, slot_id)
            except BaseException as exc:
                logger.info("slot id %d is burned: build() raised %s", slot_id, type(exc).__name__)
                raise
            if os.getpid() != pid:
                raise FESchemeError("build() forked: the child does not hold the slot store's lock")
            side.records[slot_id] = private_record
            side.dedup[key] = slot_id
            side.pending[slot_id] = public_record
            self._write_private(side, public)  # 3. the commit point: the slot is issued from here on
            public.append(public_record)
            self._write_public(public)  # 4. listed (after a failure here: by the next call's recovery)
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

    # -- the caller's own state, under the lock

    def run_locked(self, fn: Callable[[List[dict]], Any]) -> Any:
        """``fn(slots)`` under the store's exclusive lock -- the one every
        :meth:`issue` holds -- with ``slots`` the slot list at that moment
        (as :meth:`public_slots` returns it: new objects, after the recovery
        of an interrupted issue); returns what *fn* returns. A caller keeps
        state that :meth:`issue` depends on (a cap it passes as a callable)
        consistent with the slot list by changing it here: no issuance runs
        while *fn* does. *fn* must not call the store.

        Raises:
            FEValidationError: If *fn* is not callable.
            FEConfigError: If the files are unusable.
            FESchemeError: If *fn* uses the store or forks.
            Exception: Whatever *fn* raises (the store writes nothing).
        """
        if not callable(fn):
            raise FEValidationError("fn must be callable, got {}".format(type(fn).__name__))
        with self._locked():
            public, _ = self._load()
            pid = os.getpid()
            out = fn(_decode(public, self._public_path))
            if os.getpid() != pid:
                raise FESchemeError("fn forked: the child does not hold the slot store's lock")
            return out

    def check_directory(self) -> None:
        """Refuse the store's directory unless it exists and passes the checks
        of every locked call (a directory, not a symbolic link, owned by this
        process's effective user, not writable by other users); creates
        nothing. For an owner that reads files of its own there before any
        store call.

        Raises:
            FEConfigError: If the directory is missing or refused.
        """
        try:
            fd = _open_directory(self._dir)
        except FileNotFoundError:
            raise FEConfigError("the slot store directory {} does not exist".format(self._dir)) from None
        os.close(fd)


# -- the owner's own files in a store's directory (appended)


@contextlib.contextmanager
def _checked_directory(directory: str) -> Iterator[Optional[int]]:
    """The descriptor of *directory*, opened and checked as a store's
    (:func:`_open_directory`), for the files of one call to be opened
    relative to it; ``None`` (files by path, after the check) where the
    platform lacks the ``*at`` calls."""
    try:
        fd = _open_directory(directory)
    except FileNotFoundError:
        raise FEConfigError("the directory {} does not exist".format(directory)) from None
    try:
        yield fd if _DIR_FD else None
    finally:
        os.close(fd)


def read_json_file(path, private: bool) -> Optional[dict]:
    """The JSON object in *path*, a file that the owner of a store keeps in
    the store's directory, read with the checks of the store's own files: the
    directory of *path* opened once and checked through its descriptor (a
    directory of this process's effective user, not a symbolic link, that no
    other user can write), the file opened relative to it, not a symbolic
    link, no wait on a FIFO, a regular file of this user that no other user
    can write (*private*: nor read); ``None`` if it does not exist.

    Raises:
        FEConfigError: For a directory or a file that is refused or missing (the
            directory), unreadable, not JSON or not a JSON object (the message
            names the path, never the content).
    """
    path = os.fspath(path)
    with _checked_directory(os.path.dirname(os.path.abspath(path))) as dir_fd:
        fd = _open_file(path, os.O_RDONLY, private, dir_fd)
    if fd is None:
        return None
    try:
        with os.fdopen(fd, "rb") as f:
            raw = f.read()
    except OSError as exc:
        raise FEConfigError("the file {} is unreadable ({})".format(path, exc.strerror or type(exc).__name__)) from None
    try:
        data = json.loads(raw.decode("utf-8"))
    except (UnicodeDecodeError, ValueError, RecursionError):  # no excerpt of the file in the message
        raise FEConfigError("the file {} is not JSON, or truncated".format(path)) from None
    if not isinstance(data, dict):
        raise FEConfigError("the file {} is not a JSON object".format(path))
    return data


def sync_directory(directory) -> None:
    """Flush *directory* to the device (``F_FULLFSYNC`` where the OS has it),
    so that an entry just made in it -- a new file, a new subdirectory --
    survives a crash; best effort where a directory cannot be opened or
    synced, as for the store's own renames."""
    _sync_dir(os.fspath(directory))


def write_json_file(directory, name: str, data: dict, mode: int) -> None:
    """Replace ``directory/name`` with *data* (JSON) as the store writes its
    own files: the directory opened once and checked through its descriptor
    (as :func:`read_json_file`), a temporary file in it, flushed to the
    device (``F_FULLFSYNC`` where the OS has it), renamed over the old file,
    then the directory flushed, each step relative to that descriptor. A
    crash leaves the old file or the new one; a temporary file left behind is
    removed by the next locked call of a store on the directory.

    Raises:
        FEConfigError: For a directory that is missing or refused.
        FEValidationError: For a *name* that is not one file name.
    """
    directory = os.fspath(directory)
    if not isinstance(name, str) or not name or os.path.basename(name) != name or name in (".", ".."):
        raise FEValidationError("write_json_file takes the name of one file in the directory, got {!r}".format(name))
    with _checked_directory(directory) as dir_fd:
        _atomic_write(directory, name, data, mode, dir_fd)
