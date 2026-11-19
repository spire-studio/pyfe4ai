"""
Threshold key issuance for ``sife/cp_ab_pairing`` (experimental).

``s`` key servers hold Shamir shares of the master key ``U`` of
:mod:`pyfe4ai.schemes.sife.cp_ab_pairing`; any ``t`` of them issue a
functional key together, and no party holds ``U`` after setup. The public
parameters, ciphertexts, functional keys and decryption are those of the
scheme, unchanged: the combined key is a
:class:`~pyfe4ai.core.formats.pairing.CPABFunctionalKey` that
:meth:`~pyfe4ai.schemes.sife.cp_ab_pairing.SIFECPABPairing.decrypt` takes.

Roles:

* :class:`SIFECPABPairingDealer` runs the scheme's setup, shares ``U`` among
  the servers, draws the PRF keys, and hands each server its share once
  (:meth:`~SIFECPABPairingDealer.server_share`); a delivered share is
  dropped, so after the last one the dealer holds neither ``U`` nor any
  share or PRF key.
* :class:`SIFECPABPairingKeyServer` answers a request
  (:meth:`~SIFECPABPairingKeyServer.issue`) with a masked partial key, after
  its checks and the approval rule of its operator.
* :class:`SIFECPABPairingKeyCombiner` -- the requester itself -- builds
  requests and multiplies the ``t`` partial keys, checking the result against
  the public parameters before it returns a key.

Protocol (KeyGen of ACGU20 Fig. 1 is linear in ``(U, r)``). For a request
``req`` with digest ``d`` and server set ``T``, server ``i`` answers
``KeyPart(lambda_i U_i, r_i)`` with ``r_i = F(k_i^r, d)``, each component
multiplied by ``g2^z_{i,c}``, ``z_{i,c} = sum_{j in T, j != i} +-F(k_ij, d, c)``
(the masks of a pair cancel in the product). The product of the ``t``
answers is ``KeyPart(U, sum r_i)``: a key of the scheme with uniform ``r``.

Security (the argument is in the project's design notes; it was reviewed,
revised and re-checked by reviewers inside the project, and is neither
published nor externally reviewed): with fewer than
``t`` servers corrupted statically, the view of the corrupted servers and
users is simulated from the scheme's key oracle alone, one query per request
that every honest server of its ``T`` accepted. The scheme's own security
(SEL-SIM, one challenge ciphertext) is inherited, not improved. This holds
under these conditions, which the library does not provide:

* **Mutually authenticated, confidential channels** (mTLS or equivalent)
  between the requester and each server. A partial key depends on the request
  only, so whoever can replay a user's request, or read the ``t`` partial keys
  of one, gets that user's key. :meth:`~SIFECPABPairingKeyServer.issue` takes
  the identity the transport authenticated (``authenticated_requester``: the
  peer identity of the channel, never a value read from the request) and
  refuses a request whose ``sid`` is another one.
* **Partial keys are secret.** Any ``t`` partial keys of one request are that
  user's key: send them over the confidential channel to the authenticated
  requester only, and do not log or store them (the library logs none).
* **The approval rule sees the whole request.** ``approve(request,
  requester, history)`` decides on the attributes *and* the weights ``y``. A
  rule that looks at the attributes only lets an approved user take ``eta``
  keys of unit vectors and recover every plaintext its attributes open. Keys
  are as safe as the most permissive honest server's rule.
* **A trusted dealer** that deletes everything it holds after the delivery
  (Python cannot wipe memory; run the dealer in a process that ends).
* Fewer than ``t`` corrupted servers, chosen before setup; no share refresh
  (a new sharing needs new PRF keys, which :meth:`SIFECPABPairingDealer.setup`
  draws).

A corrupted server can make an issuance fail but not hand the user a wrong
key: :meth:`~SIFECPABPairingKeyCombiner.combine` checks every component of
the product with pairings, and any key that passes decrypts every ciphertext
its attributes satisfy correctly. A failed check does not tell which server
cheated; retry with another ``T``.

Records: :mod:`pyfe4ai.core.formats.keyissue`. Request digests are SHAKE-256
to 64 bytes, never truncated; the PRFs are SHAKE-256 keyed with 32 bytes,
read to 64 bytes and reduced mod ``p``.
"""

from __future__ import annotations

import copy
import hashlib
import hmac
import json
import re
import secrets
import threading

import gmpy2 as gp

from pyfe4ai._validation import int_vector
from pyfe4ai.core.exceptions import FEConfigError, FEKeyError, FEValidationError
from pyfe4ai.core.formats.keyissue import CPABKeyRequest
from pyfe4ai.core.formats.keyissue import CPABKeyServerShare
from pyfe4ai.core.formats.keyissue import CPABKeyServers
from pyfe4ai.core.formats.keyissue import CPABPartialKey
from pyfe4ai.core.formats.pairing import CPABFunctionalKey
from pyfe4ai.core.versions import CP_AB_KEYISSUE_PROTOCOL
from pyfe4ai.core.versions import CP_AB_KEYREQUEST_VERSION
from pyfe4ai.core.versions import CP_AB_KEYSERVER_SHARE_VERSION
from pyfe4ai.core.versions import CP_AB_KEYSERVERS_VERSION
from pyfe4ai.core.versions import CP_AB_PARTIAL_KEY_VERSION
from pyfe4ai.primitives.pairing.pairing_backend import G1, G2, multi_pair, multiexp
from pyfe4ai.schemes.common.ac_codec import check_length, decimal, read_elements, read_record, require_fields
from pyfe4ai.schemes.common.ac_codec import copy_parameters, scalar, serialize_vec
from pyfe4ai.schemes.sife.cp_ab_pairing import SIFECPABPairing
from pyfe4ai.schemes.sife.cp_ab_pairing import SIFECPABPairingKeyGenerator
from pyfe4ai.schemes.threshold.sharing import lagrange_coefficient_at_zero
from pyfe4ai.schemes.threshold.sharing import share_points
from pyfe4ai.schemes.threshold.sharing import share_secret

#: Domain separation tags (length-prefixed in every hash input).
DST_REQUEST = b"PYFE4AI-V07-CPAB-KEYREQ"
DST_KEYSERVERS = b"PYFE4AI-V07-CPAB-KEYSERVERS"
DST_R = b"PYFE4AI-V07-CPAB-KEYISSUE-R"
DST_MASK = b"PYFE4AI-V07-CPAB-KEYISSUE-MASK"
#: Bytes of a request digest (SHAKE-256), of a PRF output and of a PRF key.
DIGEST_BYTES = 64
PRF_BYTES = 64
KEY_BYTES = 32
_ID_BYTES = 16
_MAX_SID = 1024
#: Component kinds in the mask index ``(kind, slot, coordinate)``.
_K1, _K0, _KATTR = 1, 2, 3

_HEX = re.compile("[0-9a-f]+")


def _hex(value, nbytes: int, name: str) -> bytes:
    if type(value) is not str or len(value) != 2 * nbytes or _HEX.fullmatch(value) is None:
        raise FEValidationError("{} must be {} bytes in lowercase hex".format(name, nbytes))
    return bytes.fromhex(value)


def _hash(dst: bytes, payload: bytes, nbytes: int) -> bytes:
    h = hashlib.shake_256(len(dst).to_bytes(2, "big") + dst)
    h.update(payload)
    return h.digest(nbytes)


def _canonical(value) -> bytes:
    try:
        return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode("utf-8")
    except UnicodeEncodeError as exc:  # a lone surrogate in a label, a server id or a sid
        raise FEValidationError("a key request holds a string that does not encode to UTF-8") from exc


def _prf(key: bytes, dst: bytes, digest: bytes, index: bytes, order: int) -> int:
    """``SHAKE-256(dst || key || digest || index)`` read to 64 bytes, mod ``order``."""
    return int.from_bytes(_hash(dst, key + digest + index, PRF_BYTES), "big") % order


def _index(kind: int, slot: int, coordinate: int) -> bytes:
    """The fixed-length mask index of a component: injective over the components of a request."""
    return bytes([kind]) + slot.to_bytes(4, "big") + bytes([coordinate])


def keyservers_digest(record: dict) -> str:
    """Hex SHAKE-256 (64 bytes) of a :class:`~pyfe4ai.core.formats.keyissue.CPABKeyServers` record."""
    return _hash(DST_KEYSERVERS, _canonical(record), DIGEST_BYTES).hex()


def request_digest(request: dict) -> str:
    """Hex SHAKE-256 (64 bytes) of a request: every field but ``digest``, canonically encoded."""
    r = CPABKeyRequest(request)
    payload = [r.version, r.id, r.sid, r.y, r.attributes, r.servers, r.pp_binding, r.keyservers]
    return _hash(DST_REQUEST, _canonical(payload), DIGEST_BYTES).hex()


class _FrozenDict(dict):
    """A read-only dict: what the approval rule and its history hold (a request, frozen once when it is
    accepted). It reads, iterates and serialises (``json.dumps``) like a dict; every method that would change it
    raises ``TypeError``. ``copy.copy``, ``copy.deepcopy`` and pickling give a plain, mutable dict, so a rule
    that wants to change its own copy can."""

    __slots__ = ()

    def _refuse(self, *args, **kwargs):
        raise TypeError("the requests given to the approval rule are read-only")

    __setitem__ = __delitem__ = __ior__ = _refuse
    clear = pop = popitem = setdefault = update = _refuse

    def __copy__(self):
        return _thaw(self)

    def __deepcopy__(self, memo):
        return _thaw(self)

    def __reduce__(self):
        return (_thaw, (dict(self),))


class _FrozenList(list):
    """A read-only list (see :class:`_FrozenDict`): equal to the list it was made from, refuses changes."""

    __slots__ = ()

    def _refuse(self, *args, **kwargs):
        raise TypeError("the requests given to the approval rule are read-only")

    __setitem__ = __delitem__ = __iadd__ = __imul__ = _refuse
    append = extend = insert = pop = remove = clear = sort = reverse = _refuse

    def __copy__(self):
        return _thaw(self)

    def __deepcopy__(self, memo):
        return _thaw(self)

    def __reduce__(self):
        return (_thaw, (list(self),))


def _freeze(value):
    """*value* (JSON data) with every dict a :class:`_FrozenDict` and every list a :class:`_FrozenList`,
    built once (equal to *value*, and serialised alike)."""
    if isinstance(value, dict):
        return _FrozenDict((k, _freeze(v)) for k, v in value.items())
    if isinstance(value, (list, tuple)):
        return _FrozenList(_freeze(v) for v in value)
    return value


def _thaw(value):
    """A plain, mutable copy of frozen (or any JSON) *value*: dicts as dicts, lists as lists."""
    if isinstance(value, dict):
        return {k: _thaw(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_thaw(v) for v in value]
    return value


def _check_sid(sid) -> str:
    """*sid* if it is a ``str`` of 1 to 1024 characters that encodes to UTF-8, else ``FEValidationError``."""
    if type(sid) is not str or not 0 < len(sid) <= _MAX_SID:
        raise FEValidationError("sid must be a non-empty str of at most {} characters".format(_MAX_SID))
    try:
        sid.encode("utf-8")
    except UnicodeEncodeError as exc:
        raise FEValidationError("sid must encode to UTF-8 (no lone surrogates)") from exc
    return sid


def _check_server_ids(servers, name: str) -> list:
    if type(servers) is not list or not servers or not all(type(s) is str and s for s in servers):
        raise FEValidationError("{} must be a non-empty list of non-empty str server ids".format(name))
    if len(set(servers)) != len(servers):
        raise FEValidationError("{} holds a server id twice".format(name))
    return servers


class _Parameters:
    """What the roles read from the public parameters and the key-server record, checked."""

    def __init__(self, pp: dict, key_servers: dict) -> None:
        self.crypto = SIFECPABPairing({"keys": {"pp": pp}})  # the scheme's checks of pp
        self.pp = self.crypto.pp
        self.group, self.order = self.crypto.group, int(self.crypto.order)
        self.universe, self.eta, self.bound_y = self.crypto.universe, self.crypto.eta, self.crypto.bound_y
        ks = read_record(CPABKeyServers, key_servers, "key-server record")
        require_fields(ks, "key-server record")
        if ks.version != CP_AB_KEYSERVERS_VERSION or ks.protocol != CP_AB_KEYISSUE_PROTOCOL:
            raise FEValidationError("unknown key-server record version or protocol")
        if ks.pp_binding != self.pp["pp_binding"]:
            raise FEValidationError("the key-server record is for other public parameters")
        self.servers = _check_server_ids(ks.servers, "servers")
        if ks.points != [share_points(self.servers)[s] for s in self.servers]:
            raise FEValidationError("the share points are not 1..s in the order of the servers")
        if type(ks.threshold) is not int or not 1 <= ks.threshold <= len(self.servers):
            raise FEValidationError("threshold must be an int from 1 to the number of servers")
        if ks.sharing != "shamir":
            raise FEValidationError("unknown sharing {!r}".format(ks.sharing))
        _hex(ks.sharing_id, 16, "sharing_id")
        self.key_servers = copy.deepcopy(key_servers)
        self.threshold = ks.threshold
        self.points = dict(zip(self.servers, ks.points))
        self.digest = keyservers_digest(self.key_servers)
        self.sharing_id = ks.sharing_id
        g2 = read_elements(self.group, [self.pp["g2"]], G2, "g2")[0]
        self.g2 = g2
        self.identity_g2 = multiexp([g2], [0], self.group)
        self.b = read_elements(self.group, self.pp["b"], G2, "b")
        self.a = read_elements(self.group, self.pp["a"], G1, "a")
        self._w_b, self._w_a = {}, {}

    def w_b(self, index: int) -> list:
        if index not in self._w_b:
            self._w_b[index] = read_elements(self.group, self.pp["w_b"][index], G2, "w_b")
        return self._w_b[index]

    def w_a(self, index: int) -> list:
        if index not in self._w_a:
            self._w_a[index] = read_elements(self.group, self.pp["w_a"][index], G1, "w_a")
        return self._w_a[index]

    def slots(self, attributes: list) -> list:
        """``(name, slot)`` of every copy of every label: ``label#copy`` and ``Universe.index``."""
        return [("{}#{}".format(label, c), self.universe.index(label, c))
                for label in attributes for c in range(1, self.universe.max_copies + 1)]

    def checked_request(self, request) -> tuple:
        """``(record, y, digest bytes)`` of a request in its one canonical form, or ``FEValidationError``."""
        if type(request) is not dict or set(request) != set(CPABKeyRequest.FIELDS):
            # a field the digest does not bind must not reach the rule
            raise FEValidationError("a key request is a dict with exactly the fields {}".format(
                ", ".join(CPABKeyRequest.FIELDS)))
        r = CPABKeyRequest(request)
        if r.version != CP_AB_KEYREQUEST_VERSION:
            raise FEValidationError("unknown key request version")
        _hex(r.id, _ID_BYTES, "request id")
        _check_sid(r.sid)
        check_length(r.y, self.eta, "y")
        y = [decimal(v, "y") for v in r.y]
        if any(abs(v) > self.bound_y for v in y):
            raise FEValidationError("y has an entry above the limit bound_y = {}".format(self.bound_y))
        attributes = r.attributes
        if (type(attributes) is not list or not attributes or not all(type(a) is str for a in attributes)
                or attributes != sorted(set(attributes))):
            raise FEValidationError("attributes must be a non-empty sorted list of distinct labels")
        for label in attributes:
            if label not in self.universe:
                raise FEValidationError("an attribute of the request is not in the universe")
        servers = _check_server_ids(r.servers, "the servers of the request")
        if servers != sorted(servers) or len(servers) != self.threshold or not set(servers) <= set(self.servers):
            raise FEValidationError(
                "the servers of a request are t = {} distinct ids of the key-server record, sorted".format(
                    self.threshold))
        if r.pp_binding != self.pp["pp_binding"] or r.keyservers != self.digest:
            raise FEValidationError("the request is for other public parameters or another sharing")
        digest = _hex(r.digest, DIGEST_BYTES, "request digest")
        if not hmac.compare_digest(request_digest(request), r.digest):
            raise FEValidationError("the request does not match its digest")
        return r, y, digest


class SIFECPABPairingDealer:
    """The trusted dealer: setup of ``sife/cp_ab_pairing`` and the sharing of its master key.

    Args:
        config: The configuration of
            :class:`~pyfe4ai.schemes.sife.cp_ab_pairing.SIFECPABPairingKeyGenerator`.
        key_servers: The server ids (distinct non-empty ``str``); server
            ``i`` of the list gets the share point ``i + 1``.
        threshold: ``t``, the servers per request (``1 <= t <= s``). With
            ``t = 1`` there is no threshold: every server's share is ``U``
            itself, and any one server issues keys alone.
    """

    def __init__(self, config: dict, key_servers: list, threshold: int) -> None:
        self.servers = list(_check_server_ids(key_servers, "key_servers"))
        if type(threshold) is not int or not 1 <= threshold <= len(self.servers):
            raise FEValidationError("threshold must be an int from 1 to the number of key servers")
        self.threshold = threshold
        self._kg = SIFECPABPairingKeyGenerator(config)
        self._record = None
        self._pending = {}

    def setup(self) -> None:
        """Run the scheme's setup, share ``U`` and draw fresh PRF keys; ``U`` is dropped."""
        kg = self._kg
        kg.setup()
        order = gp.mpz(int(kg.order))
        points = share_points(self.servers)
        u = [[gp.mpz(v) for v in row] for row in kg.msk["u"]]
        kg.msk = None  # the key generator refuses keys from now on
        shares = {s: [[None] * kg.eta for _ in range(2)] for s in self.servers}
        for t in range(2):
            for k in range(kg.eta):
                values = share_secret(u[t][k], self.threshold, list(points.values()), order)
                for s in self.servers:
                    shares[s][t][k] = gp.digits(values[points[s]])
        del u
        self._record = CPABKeyServers.new(
            version=CP_AB_KEYSERVERS_VERSION,
            protocol=CP_AB_KEYISSUE_PROTOCOL,
            pp_binding=kg.pp["pp_binding"],
            servers=list(self.servers),
            points=[points[s] for s in self.servers],
            threshold=self.threshold,
            sharing="shamir",
            sharing_id=secrets.token_hex(16),
        ).to_dict()
        digest = keyservers_digest(self._record)
        pair_keys = {}
        for i, a in enumerate(self.servers):
            for b in self.servers[i + 1:]:
                pair_keys[(a, b)] = pair_keys[(b, a)] = secrets.token_hex(KEY_BYTES)
        self._pending = {
            s: CPABKeyServerShare.new(
                version=CP_AB_KEYSERVER_SHARE_VERSION,
                server=s,
                keyservers=digest,
                sharing_id=self._record["sharing_id"],
                u=shares[s],
                r_key=secrets.token_hex(KEY_BYTES),
                mask_keys={o: pair_keys[(s, o)] for o in self.servers if o != s},
            ).to_dict()
            for s in self.servers
        }
        del shares, pair_keys

    def get_public_parameters(self) -> dict:
        """The public parameters of the scheme (unchanged by the sharing)."""
        if self._record is None:
            raise FEKeyError("run setup() first")
        return self._kg.get_public_parameters()

    def key_servers(self) -> dict:
        """The public :class:`~pyfe4ai.core.formats.keyissue.CPABKeyServers` record."""
        if self._record is None:
            raise FEKeyError("run setup() first")
        return copy.deepcopy(self._record)

    def server_share(self, server: str) -> dict:
        """Server *server*'s secret share, once; the dealer drops it.

        Deliver it to that server only, over a confidential channel: it holds
        a share of ``U`` and PRF keys shared with every other server.

        Raises:
            FEKeyError: Before setup, or if that share was delivered already.
        """
        if self._record is None:
            raise FEKeyError("run setup() first")
        share = self._pending.pop(server, None)
        if share is None:
            raise FEKeyError("no share for server {!r}: unknown, or delivered already".format(server))
        return share

    def get_decryption_keys(self, *args, **kwargs):
        """Refused: the master key is shared among the key servers."""
        raise FEKeyError("the master key is shared among key servers: request keys from them")


class SIFECPABPairingKeyServer:
    """A key server.

    Args:
        pp: The public parameters.
        key_servers: The public key-server record.
        share: This server's secret share
            (:meth:`SIFECPABPairingDealer.server_share`).
        approve: The operator's rule, ``approve(request, requester, history)
            -> bool``: *request* is the whole request (``sid``, ``y``,
            ``attributes``, ``servers``, ...), *requester* the identity the
            channel authenticated, *history* the tuple of ``(requester,
            request)`` this server accepted before, oldest first (refused
            requests are not in it). The request and the history are
            read-only views, frozen once when a request is accepted: dicts
            and lists that refuse changes (``TypeError``) and otherwise
            compare, read and serialise (``json.dumps``) as the request does;
            ``copy.deepcopy`` gives a mutable copy. Only the return value ``True`` approves; anything
            else, a reason string or ``None`` among it, refuses, and so does
            any exception the rule raises (the request is then refused with
            ``FEKeyError`` and leaves no record). It must decide on ``y`` too
            (see the module docstring). It runs under the server's lock, so the
            requests to one server are approved one at a time; it should keep
            its own state (e.g. counters) rather than scan a long history on
            every call. A repeat of an accepted
            request (same digest, same content) is answered again without a
            new approval: it gets the same partial key and reveals nothing
            new.

    Raises:
        FEValidationError: For a share or a record that does not fit the
            public parameters or each other.
        FEConfigError: Without an approval rule.
    """

    def __init__(self, pp: dict, key_servers: dict, share: dict, approve) -> None:
        if not callable(approve):
            raise FEConfigError("a key server needs an approval rule approve(request, requester, history)")
        self._p = _Parameters(pp, key_servers)
        sh = read_record(CPABKeyServerShare, share, "key-server share")
        require_fields(sh, "key-server share")
        if sh.version != CP_AB_KEYSERVER_SHARE_VERSION:
            raise FEValidationError("unknown key-server share version")
        if sh.server not in self._p.servers:
            raise FEValidationError("the share is for a server outside the key-server record")
        if sh.keyservers != self._p.digest or sh.sharing_id != self._p.sharing_id:
            raise FEValidationError("the share belongs to another sharing")
        check_length(sh.u, 2, "u")
        for row in sh.u:
            check_length(row, self._p.eta, "u")
        self._u = [[gp.mpz(scalar(v, "u", self._p.order)) for v in row] for row in sh.u]
        self._r_key = _hex(sh.r_key, KEY_BYTES, "r_key")
        others = [s for s in self._p.servers if s != sh.server]
        if type(sh.mask_keys) is not dict or sorted(sh.mask_keys) != sorted(others):
            raise FEValidationError("mask_keys must hold one key for every other server")
        self._mask_keys = {s: _hex(k, KEY_BYTES, "mask_keys") for s, k in sh.mask_keys.items()}
        self.server = sh.server
        self._approve = approve
        self._lock = threading.Lock()
        self._contents = {}   # digest -> canonical content, of accepted requests
        self._ids = {}        # (sid, request id) -> digest: one sid reusing its id for other content is refused
        self._history = []    # (requester, frozen request) accepted, oldest first

    def issue(self, request: dict, authenticated_requester: str) -> dict:
        """This server's masked partial key for *request*
        (:class:`~pyfe4ai.core.formats.keyissue.CPABPartialKey`).

        Args:
            request: The request (:meth:`SIFECPABPairingKeyCombiner.new_request`).
            authenticated_requester: The identity that the transport
                authenticated for this request (e.g. the peer of a mutually
                authenticated TLS channel), never a value taken from the
                request; it must be the request's ``sid``.

        The returned partial key is secret (see the module docstring).

        Raises:
            FEValidationError: For a malformed request, one for other
                parameters or another sharing, or one whose servers are not
                ``t`` distinct ids of the record including this one.
            FEKeyError: If *authenticated_requester* is not the request's
                ``sid``, the approval rule refuses (or raises), the digest was
                used for other content, or this ``sid`` used the request's id
                for other content.
        """
        p = self._p
        snapshot = copy_parameters(request, "key request")  # what is checked, approved, recorded and answered
        r, y, digest = p.checked_request(snapshot)
        requester = authenticated_requester
        try:
            same = type(requester) is str and hmac.compare_digest(requester.encode("utf-8"), r.sid.encode("utf-8"))
        except UnicodeEncodeError:
            same = False
        if not same:
            raise FEKeyError("the request's sid is not the authenticated requester")
        if self.server not in r.servers:
            raise FEValidationError("this server is not among the servers of the request")
        content = _canonical([r.version, r.id, r.sid, r.y, r.attributes, r.servers, r.pp_binding, r.keyservers])
        with self._lock:
            seen = self._contents.get(r.digest)
            if seen is not None and seen != content:
                raise FEKeyError("the digest of this request was used for other content")
            if seen is None:  # a repeat of an accepted request is answered again, without a new approval
                # ids are kept per sid: only the requester itself can block its own id
                if self._ids.get((r.sid, r.id), r.digest) != r.digest:
                    raise FEKeyError("this sid used the id of this request for other content")
                frozen = _freeze(snapshot)  # built once: the rule's view now, the history entry once accepted
                try:
                    verdict = self._approve(frozen, requester, tuple(self._history))
                except Exception as exc:  # fail closed: a rule that raises refuses, and nothing is recorded
                    raise FEKeyError("the approval rule of server {!r} raised {}: the request is refused".format(
                        self.server, type(exc).__name__)) from exc
                if verdict is not True:  # fail closed: a reason string, None or any other value refuses
                    raise FEKeyError("the approval rule of server {!r} refuses this request".format(self.server))
                self._contents[r.digest] = content
                self._ids[(r.sid, r.id)] = r.digest
                self._history.append((requester, frozen))
        return self._partial(r, y, digest)

    def _partial(self, r, y: list, digest: bytes) -> dict:
        p, order = self._p, self._p.order
        xs = [p.points[s] for s in r.servers]
        lam = int(lagrange_coefficient_at_zero(p.points[self.server], xs, order))
        uy = [sum(int(u_tk) * (v % order) for u_tk, v in zip(row, y)) * lam % order for row in self._u]
        rand = _prf(self._r_key, DST_R, digest, b"", order)
        mine = p.points[self.server]
        signed = [(1 if mine < p.points[s] else -1, self._mask_keys[s]) for s in r.servers if s != self.server]

        def mask(kind: int, slot: int, coordinate: int) -> int:
            index = _index(kind, slot, coordinate)
            return sum(sign * _prf(key, DST_MASK, digest, index, order) for sign, key in signed) % order

        group, g2, size = p.group, p.g2, p.universe.size
        k1 = [multiexp([p.b[t], g2], [rand, mask(_K1, 0, t)], group) for t in (0, 1)]
        w_size = p.w_b(size)
        k0 = [multiexp([w_size[t], g2], [rand, (uy[t] + mask(_K0, 0, t)) % order], group) for t in (0, 1)]
        k_attr = {}
        for name, slot in p.slots(r.attributes):
            w = p.w_b(slot)
            k_attr[name] = serialize_vec(group, [multiexp([w[t], g2], [rand, mask(_KATTR, slot, t)], group)
                                                 for t in (0, 1)])
        return CPABPartialKey.new(
            version=CP_AB_PARTIAL_KEY_VERSION,
            server=self.server,
            request=r.digest,
            k1=serialize_vec(group, k1),
            k0=serialize_vec(group, k0),
            k_attr=k_attr,
        ).to_dict()


class SIFECPABPairingKeyCombiner:
    """The requester: builds requests and combines partial keys.

    Args:
        pp: The public parameters.
        key_servers: The public key-server record.
    """

    def __init__(self, pp: dict, key_servers: dict) -> None:
        self._p = _Parameters(pp, key_servers)

    def new_request(self, sid: str, fusion_weight: list, attributes, servers: list) -> dict:
        """A request (:class:`~pyfe4ai.core.formats.keyissue.CPABKeyRequest`)
        with a fresh random id.

        Args:
            sid: The requester's identity, the one its channels to the servers
                authenticate.
            fusion_weight: ``y``, ``eta`` ints with ``|y_i| <= bound_y``.
            attributes: The labels the key is to hold (a collection; sorted
                and deduplicated here).
            servers: ``t`` distinct server ids of the key-server record.
        """
        p = self._p
        _check_sid(sid)
        y = int_vector(fusion_weight, "fusion weight", length=p.eta, bound=p.bound_y)
        if isinstance(attributes, (str, bytes, bytearray)):
            raise FEValidationError("attributes must be a collection of labels, not a bare string")
        labels = sorted(set(attributes))
        request = CPABKeyRequest.new(
            version=CP_AB_KEYREQUEST_VERSION,
            id=secrets.token_hex(_ID_BYTES),
            sid=sid,
            y=[str(v) for v in y],
            attributes=labels,
            servers=sorted(_check_server_ids(list(servers), "servers")),
            pp_binding=p.pp["pp_binding"],
            keyservers=p.digest,
            digest="0" * (2 * DIGEST_BYTES),
        ).to_dict()
        request["digest"] = request_digest(request)
        p.checked_request(request)
        return request

    def combine(self, request: dict, partials: list) -> dict:
        """The functional key: the product of the partial keys, checked.

        Raises:
            FEKeyError: Without exactly one partial key from every server of
                the request, all for this request, or if the product fails the
                check against the public parameters (some server cheated or
                erred: retry with other servers).
            FEValidationError: For a malformed request or partial key.
        """
        p, group = self._p, self._p.group
        r, y, _ = p.checked_request(request)
        if type(partials) not in (list, tuple):
            raise FEValidationError("partials must be a list of partial keys")
        names = [name for name, _ in p.slots(r.attributes)]
        by_server = {}
        for item in partials:
            pk = read_record(CPABPartialKey, item, "partial key")
            require_fields(pk, "partial key")
            if pk.version != CP_AB_PARTIAL_KEY_VERSION:
                raise FEValidationError("unknown partial key version")
            if pk.request != r.digest:
                raise FEKeyError("a partial key answers another request")
            if pk.server not in r.servers or pk.server in by_server:
                raise FEKeyError("partial keys must come from the servers of the request, one each")
            check_length(pk.k1, 2, "k1")
            check_length(pk.k0, 2, "k0")
            if type(pk.k_attr) is not dict or sorted(pk.k_attr) != sorted(names):
                raise FEKeyError("a partial key does not hold every copy of every attribute of the request")
            by_server[pk.server] = pk
        if sorted(by_server) != r.servers:
            raise FEKeyError("need one partial key from each of the {} servers of the request".format(len(r.servers)))

        def product(field, name=None) -> list:
            rows = [read_elements(group, getattr(pk, field) if name is None else pk.k_attr[name], G2, field)
                    for pk in by_server.values()]
            return [multiexp([row[t] for row in rows], [1] * len(rows), group) for t in (0, 1)]

        k1, k0 = product("k1"), product("k0")
        k_attr = {name: product("k_attr", name) for name in names}
        self._verify(r, y, k1, k0, k_attr)
        return CPABFunctionalKey.new(
            attributes=list(r.attributes),
            y=[str(v) for v in y],
            k1=serialize_vec(group, k1),
            k0=serialize_vec(group, k0),
            k_attr={name: serialize_vec(group, k_attr[name]) for name in names},
            pp_binding=p.pp["pp_binding"],
        ).to_dict()

    def _verify(self, r, y: list, k1: list, k0: list, k_attr: dict) -> None:
        """The check (V): ``e(a, k0) = e(prod ua_i^y_i, g2) e(W_size^T a, k1)`` and
        ``e(a, k_attr[l#c]) = e(W_l#c^T a, k1)`` for every entry."""
        p, group, order = self._p, self._p.group, self._p.order
        if k1[0] == p.identity_g2 and k1[1] == p.identity_g2:
            raise FEKeyError("the combined key has k1 = 1: refused")
        ua = read_elements(group, p.pp["ua"], G1, "ua")
        folded = multiexp(ua, [v % order for v in y], group)
        w_size = p.w_a(p.universe.size)
        if multi_pair(p.a, k0) != multi_pair([folded, w_size[0], w_size[1]], [p.g2, k1[0], k1[1]]):
            raise FEKeyError("the combined key fails its check (k0): a server cheated or erred")
        for name, slot in p.slots(r.attributes):
            if multi_pair(p.a, k_attr[name]) != multi_pair(p.w_a(slot), k1):
                raise FEKeyError("the combined key fails its check ({}): a server cheated or erred".format(name))
