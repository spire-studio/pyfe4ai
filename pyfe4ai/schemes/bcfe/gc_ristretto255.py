"""
Bounded-key functional encryption for boolean circuits: one garbled-circuit slot per issued function
| From "Worry-Free Encryption: Functional Encryption with Public Keys"
| Published in: ACM CCS 2010
| By Amit Sahai and Hakan Seyalioglu (SS10), Sec. 3.2 (the guard bit) and Sec. 3.3 (the single-key scheme)
| With "Two Halves Make a Whole: Reducing Data Transfer in Garbled Circuits Using Half Gates"
| Published in: EUROCRYPT 2015
| By Samee Zahur, Mike Rosulek and David Evans (ZRE15), Fig. 2

* type:     public-key functional encryption for a public circuit ``C(w, x)``; a key fixes ``w``
* setting:  the prime-order group ristretto255 (RFC 9496) for the label KEM; SHAKE128 gate hash

**Construction.** The authority holds a public circuit ``C(w, x)``
(:mod:`pyfe4ai.circuits`: a library circuit with its key, data and output
layouts; ``config["circuit"]``). Every key it issues is a *slot*: one
instance of the single-key scheme of SS10 Sec. 3.3, for the guarded circuit
``guard(C)(w || g, x) = C(w, x) AND NOT g`` (SS10 Sec. 3.2). A slot is a
table of ``2m + 1`` public keys ``P[j, b]`` (``j < m``, ``b = 0, 1``, and
``P[m, 0]`` for the guard bit); its key holds the secret keys of the
selected positions ``(j, w_j)`` and ``(m, 0)``. The unselected positions are
``H_P(setup_digest, slot_id, j, k)`` (hash to the group, refinement h' of the
0.6 design), so their secret keys exist nowhere. An encryption of ``x``
garbles ``guard(C)`` with half-gates (ZRE15 Fig. 2) once per slot, with a
fresh ``Delta``, fresh labels and a fresh ephemeral key ``R`` for every
(ciphertext, slot), and encrypts each key-wire label under its public key
(hashed ElGamal: :mod:`pyfe4ai.primitives.bcfe_kem`). The holder of slot
``i``'s key opens the labels of ``w || 0`` and evaluates ``C(w, x)``. The byte
layouts are documented in :mod:`pyfe4ai.circuits` (the circuit encoding),
:mod:`pyfe4ai.primitives.garbling` (the garbling) and
:mod:`pyfe4ai.primitives.bcfe_kem` (the slot keys and labels); the records and
their digests are in :mod:`pyfe4ai.core.formats.garbled`. The parallel instances
follow AV19 Thm. 4 (Ananth and Vaikuntanathan, TCC 2019) with a
deterministic allocation; GVW12 (Gorbunov, Vaikuntanathan and Wee, CRYPTO
2012) is the bounded-collusion setting this is *not* (see below).

Issuance (:meth:`BCFEGCRistretto255KeyGenerator.issue_key`, h1, the
default ``key_mode``): the authority makes the selected key pairs, returns
the key and, with ``keep_issued_keys`` (default), keeps it. With
``key_mode = "h2"`` (SS10's Pre / Auth) the key holder makes its own key
pairs (:meth:`BCFEGCRistretto255.key_holder`, ``request``), the authority
places its public keys (:meth:`BCFEGCRistretto255KeyGenerator.authorize`) and
the holder checks the slot (``finish``, SS10's CheckAuth: its keys at ``(j,
w_j)`` and ``(m, 0)``, exactly ``2m + 1`` entries, the unselected positions
recomputed from ``k``); the authority alone then cannot decrypt the slot.
Both follow one order, and a step that fails changes nothing: (1) ``w``
through the key layout to its canonical bits; (2) the issuance policy
(:func:`~pyfe4ai.policy.formula.evaluate` of ``issuance_policy`` on the requester's
attributes) before anything is reserved; (3) the private dedup key of
(scheme, circuit, requester, ``w``); (4) the cap ``max_keys``, with the same
error whether (3) found a slot or not; (5) the slot (backend operations
``bcfe_new_slot`` / ``bcfe_place_slot``), its public record appended to the
slot list (:class:`~pyfe4ai.schemes.common.slots.SlotStore`), the private
record kept by the authority. The same requester asking again for the same
``w`` gets the same key (h1, byte for byte) or the same grant (h2); another
requester asking for the same ``w`` gets a slot of its own. Under h2 the
holder's ``m + 1`` public keys must be pairwise distinct (``FEKeyError``
before anything is reserved): one key at two selected positions, or at a
selected one and ``(m, 0)``, would show bits of ``w`` in the public slot
list (alternative not taken: accept them, as only a holder that bypasses
``request`` sends them and only its own ``w`` shows).

**What is proved, and what is not** (the project's security argument for
this scheme, reviewed internally twice; there is no external review): IND
security against adaptive key, corruption and registration queries, many
challenge ciphertexts, in the random oracle model (``H_G``, ``H_K``,
``H_P``) under the strong DH assumption in ristretto255 (for the label
encryption, with arbitrary public keys), for the half-gates garbling with
the instance salt and, in the default mode, the garbler-known data wires
(a lemma of the argument's own). The argument also states what the code
must do for it to apply: fresh randomness for every (ciphertext, slot),
hash inputs that are domain-separated and of fixed or prefixed length, no
secret key at an unselected position, one key per slot, checks of the
holder's public keys and of the grant, a private dedup key and no fixed
seed outside the tests; :mod:`pyfe4ai.primitives.bcfe_kem` and
:mod:`pyfe4ai.primitives.garbling` say where each is done. Not proved: SIM
security, the secrecy of ``w`` from its key holder, a malicious encryptor,
a forged slot list, side channels, quantum adversaries. Status
*experimental*: our own security argument, reviewed internally, not by
outside reviewers.

Read before use:

* **The key holder learns w.** A key opens the labels of ``w``, and the key
  record carries ``w``; the scheme hides ``x`` from the key holders (they
  learn ``C(w, x)`` only) and ``w`` from the encryptors (the slot list does
  not show it), never ``w`` from its holder or from the authority.
* **Keys decrypt only ciphertexts made after their slot existed.** A
  ciphertext is encrypted to the slots of the slot list it was made with;
  a key of a later slot has no component in it (``FEKeyError`` saying so).
* **Total issuance bounded by max_keys.** Every distinct (requester,
  function) takes a slot, until ``max_keys`` slots exist; a ciphertext holds
  one component per slot it is encrypted to, so its size grows with the
  keys issued. This is a bound on the keys issued in all, not the
  collusion bound of GVW12 / AV19: any number of the issued keys may
  collude.
* **Classical security only: harvest-now-decrypt-later.** The labels are
  encrypted with ristretto255 hashed ElGamal: a ciphertext recorded today
  gives up ``x`` to a future quantum computer.
* **Honest encryptor.** The construction assumes the encryptor garbles
  ``C`` correctly. A malicious one can (1) garble another circuit, so that
  a holder's result is any function of ``w`` -- harmful once the result is
  published or returned; (2) cause selective failure, a bad label at one
  position, and learn a bit of ``w`` from whether decryption "worked"; (3)
  use different ``x`` in different slots, so that the holders' results
  disagree. Decryption reports every failure inside a component as one
  error (:data:`DECRYPTION_FAILED`); do not send results or failures
  back to encryptors.
* **The library cannot authenticate requester ids: pass an authenticated
  id.** The dedup scope is the ``requester`` string; an application that
  passes an unauthenticated one lets a requester take another's key (h1
  with ``keep_issued_keys``) or test another's functions. The id is taken
  as its UTF-8 bytes, not normalized: two spellings of one name (NFC and
  NFD, say) are two requesters, and an id that UTF-8 cannot encode (a lone
  surrogate) is refused with ``FEValidationError`` before anything is
  reserved. Alternatives not taken: Unicode normalization (it would merge
  ids that the application's authentication may keep apart), or encoding
  with ``surrogatepass`` (it would accept ids that other components cannot
  carry).
* **The slot list is trusted as distributed** (as every public parameter of
  the library): an encryptor that takes a forged slot list encrypts ``x``
  to whoever forged it. Pin the ``pp_digest``, or check ``slot_list_version``
  and the slots you encrypt to (``encrypt(x, slots=...)``).
* **Slot labels are names, not identities.** The authority labels each slot
  as it likes, and nothing makes a label unique or ties it to a requester:
  ``encrypt(x, slots="credit")`` encrypts to every slot labelled
  ``"credit"``, including slots of other requesters and slots listed after
  the public parameters you checked. To encrypt to exactly the slots you
  mean, pin the ``pp_digest`` and select by slot id (alternative not taken:
  refusing a repeated label, which would let one requester block a name for
  the others).
* The issuance policy decides whom the authority builds a slot for; it is
  not a cryptographic access control (the cryptography enforces only that a
  key computes ``C(w, .)``).

**State.** An instance lives in ``<cache_dir>/authority/bcfe/<setup
digest, hex>``: ``setup.json`` (the public setup,
:class:`~pyfe4ai.core.formats.garbled.BCFESetup`), ``authority.json`` (mode
``0o600``: ``max_keys``, ``key_mode``, ``keep_issued_keys``, the issuance
policy, the profile) and the slot store's files. The private record of a
slot holds its ``k`` (16 bytes), the ``max_keys`` of its issue, its
``key_mode``, the digest of the requester's attribute set
(:func:`~pyfe4ai.policy.attributes.attribute_set_digest`, for the authority's audit)
and, for h1 with ``keep_issued_keys``, its scalars and ``w``; never an
unselected secret key (none exists) and never in the public list (with
``k`` anyone could tell the unselected positions of the slot from the
selected ones, and so read ``w``). The attribute-set digest is unsalted
and vocabularies are small: in the public slot record it would let every
holder of the public parameters test guesses of a requester's attributes
offline and link the slots of requesters with equal attributes, so it is
kept private; the public record keeps the policy digest, the authority's
own policy. Alternative not taken: publish it and document the leak; to
undo, see :mod:`pyfe4ai.core.formats.garbled` (a field of the slot record
and of its digest). :meth:`BCFEGCRistretto255KeyGenerator.open` resumes an instance
from its directory.

The two files are read with the slot store's file checks
(:func:`~pyfe4ai.schemes.common.slots.read_json_file`: not a symbolic link,
no wait on a FIFO, a regular file of the authority's user that no other user
can write; ``authority.json`` not readable by others either) and written as
the store writes its own (:func:`~pyfe4ai.schemes.common.slots.write_json_file`:
``F_FULLFSYNC`` or ``fsync``, rename, directory flush), and
:meth:`~BCFEGCRistretto255KeyGenerator.open` checks the directory
(:meth:`~pyfe4ai.schemes.common.slots.SlotStore.check_directory`) before it
reads either. Both are parsed fail closed: ``setup.json`` holds exactly the
setup fields, ``authority.json`` exactly :data:`AUTHORITY_FIELDS`, each of its
type and range and matching the setup (``FEConfigError`` otherwise).
Alternative not taken: plain reads with the constructor's defaults for a
missing option (the first version: an ``authority.json`` without its
``issuance_policy`` opened an instance that issued to every requester).
``max_keys`` is an int in ``[1, 2^63)`` wherever it is read or set (the
config, :meth:`~BCFEGCRistretto255KeyGenerator.set_max_keys`,
``authority.json``, the public parameters); alternative not taken: any
positive int (the first version: at ``2^64`` an issuance committed its slot,
then failed in the pp digest, and its key could never be returned).

**The cap is decided under the slot store's lock.** ``max_keys`` lives in
``authority.json``, and another authority object (another process) on the
same directory may change it. :meth:`~BCFEGCRistretto255KeyGenerator.issue_key`
and :meth:`~BCFEGCRistretto255KeyGenerator.authorize` read it inside the slot
store's issue, under its exclusive lock
(:meth:`~pyfe4ai.schemes.common.slots.SlotStore.issue` with the cap as a
callable), and the slot's private record keeps the value read there (the
pp digest of a key or grant "as of its issue" is computed with it);
:meth:`~BCFEGCRistretto255KeyGenerator.set_max_keys` checks the slots
listed and writes the new cap under the same lock
(:meth:`~pyfe4ai.schemes.common.slots.SlotStore.run_locked`), and
:meth:`~BCFEGCRistretto255KeyGenerator.get_public_parameters` reads the cap
and the slot list together under it. So the cap cannot be lowered between
an issuance's check and its slot: the slot list never holds more than
``max_keys`` slots, and the public parameters always verify. Alternatives
not taken: reading the cap before the lock (the first version: a lowering
in flight left more slots than ``max_keys`` and public parameters that
their own check refuses), or keeping the cap in the slot store's private
file (one file, but the store would own an option of the scheme).

**Checks of the versions.** A key or grant records the slot list version of
its slot's issue, which decides whether a ciphertext without a component
for the slot is "older than the key". It must be in ``[1, 2^63)`` and at
most ``slot_id + 1`` (slot ids start at 0 and increase, so at most
``slot_id`` slots come before a slot); a ciphertext's version is below
``2^63`` and at least its number of components. Alternatives not taken:
trust the recorded value (it only chooses an error message, but a key
claiming a huge version would call every ciphertext "older than the key");
also check it against the reader's public parameters (one more than the
slots they list before the key's slot, exactly that when they list it),
which would refuse keys that are only internally inconsistent -- a key
moved to another slot, a slot id the public parameters do not list --
whose components the hashes already turn into unrelated bits.

**Keys are checked against their slot.** Half-gates outputs are not
authenticated, so a damaged key (a flipped bit of ``w``, the scalars of
another slot) would decrypt to a wrong value without an error. When the
decryptor's public parameters list the key's slot,
:meth:`~BCFEGCRistretto255.decrypt` first checks ``s_j B = P[2j + w_j]`` for
``j < m`` and ``s_g B = P[2m]`` (``m + 1`` fixed-base multiplications, once
per key and crypto object: the verdict is cached) and raises the one
decryption error :data:`DECRYPTION_FAILED` on a mismatch, without telling
which position failed. A key of a slot the public parameters do not list yet
is not checked. Alternative not taken: trust the key file (a damaged key then
gives a wrong result silently).

**Python and native.** Every slot and label operation goes through the
backend (:func:`pyfe4ai.backend.ops`: ``bcfe_new_slot``, ``bcfe_place_slot``,
``bcfe_unselected_ok``, ``bcfe_encrypt_slot``, ``bcfe_decrypt_slot``). The
Python reference is not constant time and cannot wipe its copies of
secrets; the native implementation draws its randomness from the operating
system and wipes ``Delta``, the seeds, ``r`` and the labels.

**Test randomness.** The backend's test
hook ``pyfe4ai.backend._random_override`` hands every operation that draws
random bytes the bytes of the test: left set outside a test, every (ciphertext,
slot) of this scheme would garble with attacker-known ``Delta``, labels and
``r``, and every slot would be issued from known scalars and ``k`` -- a total
break. So the scheme refuses to draw while the hook is set, under every
profile: :meth:`~BCFEGCRistretto255KeyGenerator.setup`,
:meth:`~BCFEGCRistretto255KeyGenerator.issue_key`,
:meth:`~BCFEGCRistretto255KeyGenerator.authorize`,
:meth:`~BCFEGCRistretto255.encrypt` and the key holder's ``request`` raise
:class:`~pyfe4ai.core.exceptions.FESchemeError`
(:data:`TEST_RANDOMNESS_REFUSED`) before they draw or change anything (the
backend's ``random_override_active()``, read once at the start of the call: a
thread that sets the hook while the call runs is not detected). Decryption,
``finish``, :meth:`~BCFEGCRistretto255KeyGenerator.get_public_parameters`
and :meth:`~BCFEGCRistretto255KeyGenerator.open` draw nothing and run. The
scheme's own draws (``setup_nonce``, ``ct_nonce``, the key holder's
scalars) come from :func:`os.urandom` and never read the hook; the hook
reaches only the backend operations (``bcfe_new_slot``, ``bcfe_place_slot``,
``bcfe_encrypt_slot``). Tests of fixed randomness call those operations
directly. Alternatives not taken: refusing only under
``security_profile="standard"`` (tests could still drive the scheme through
the hook, but a hook left set in a ``toy`` application would break it
silently: weaker); an :class:`~pyfe4ai.core.exceptions.FESecurityWarning`
only (the break happens all the same); honouring the hook only when an
environment variable is set (it would change every scheme and the backend's
hooks: out of the scope of 0.6).
"""

from __future__ import annotations

import contextlib
import hashlib
import hmac
import logging
import os
import threading
import unicodedata
from collections.abc import Iterable, Mapping

from pyfe4ai import backend
from pyfe4ai.circuits import Circuit, DataLayout, KeyLayout, OutputLayout, ParamCircuit
from pyfe4ai.core import forksafe, profiles, versions
from pyfe4ai.core.circuit_base import CircuitFEAbsCrypto, CircuitFEAbsKeyGenerator
from pyfe4ai.core.exceptions import FEConfigError, FEKeyError, FESchemeError, FEValidationError
from pyfe4ai.core.formats import garbled
from pyfe4ai.core.formats.garbled import BCFECiphertext
from pyfe4ai.core.formats.garbled import BCFEComponent
from pyfe4ai.core.formats.garbled import BCFEGrant
from pyfe4ai.core.formats.garbled import BCFEKey
from pyfe4ai.core.formats.garbled import BCFEKeyRequest
from pyfe4ai.core.formats.garbled import BCFEPublicParameters
from pyfe4ai.core.formats.garbled import BCFESetup
from pyfe4ai.core.formats.garbled import BCFESlot
from pyfe4ai.core.formats.garbled import b64, from_b64, from_hex
from pyfe4ai.policy import Formula, attribute_set_digest, canonical_encoding, evaluate, from_canonical, policy_digest
from pyfe4ai.primitives import bcfe_kem
from pyfe4ai.primitives.garbling import pack_bits, unpack_bits
from pyfe4ai.primitives.groups import ec_group as eg
from pyfe4ai.schemes.common.slots import SlotStore, read_json_file, sync_directory, write_json_file

logger = logging.getLogger(__name__)

#: The scheme id.
SCHEME_ID = "bcfe/gc_ristretto255"
#: The values the records carry (:mod:`pyfe4ai.core.versions`).
BCFE_VERSION = versions.BCFE_VERSION
KEM = versions.BCFE_KEM
GC = versions.BCFE_GC
GC_HASHES = (versions.BCFE_GC_HASH,)
LABEL_BITS = (128,)
DATA_WIRES = ("hardwired", "labels")
KEY_MODES = ("h1", "h2")
DEFAULT_MAX_KEYS = 16
#: ``max_keys`` is below this bound everywhere (config, ``set_max_keys``, ``authority.json``, public parameters).
MAX_KEYS_BOUND = 1 << 63
#: The longest slot label (characters).
MAX_LABEL_CHARS = 256

#: Files of an instance directory (besides the slot store's).
SETUP_FILE = "setup.json"
AUTHORITY_FILE = "authority.json"
AUTHORITY_FORMAT = "pyfe4ai-bcfe-authority"
AUTHORITY_FORMAT_VERSION = 1
#: The fields of ``authority.json``, exactly (a file with another field set is refused).
AUTHORITY_FIELDS = ("format", "format_version", "setup_digest", "max_keys", "key_mode", "keep_issued_keys",
                    "issuance_policy", "data_wires", "gc_hash", "label_bits", "security_profile", "sec_param")
#: Key checks a crypto object remembers (verdicts only, by slot digest and a hash of the key's secrets).
_KEY_CHECKS_KEPT = 256

#: The error of the calls that draw randomness while the tests' hook is set.
TEST_RANDOMNESS_REFUSED = ("pyfe4ai.backend._random_override is set: bcfe refuses to issue, set up or encrypt with "
                           "test randomness")

#: The one error of a component that does not decrypt (malformed, or of
#: another slot, ciphertext or key); the cause is not told apart.
DECRYPTION_FAILED = ("the ciphertext component of this key's slot does not decrypt with this key (malformed, or "
                     "made for another slot, ciphertext or key)")

_DIGEST = garbled.DIGEST_BYTES
_POINT = bcfe_kem.POINT_BYTES
_SCALAR = bcfe_kem.SCALAR_BYTES
_IDENTITY = bytes(_POINT)  # the encoding of the identity of ristretto255


# ---------------------------------------------------------------------------
# argument checks


def _max_keys(value, error=FEValidationError, what: str = "max_keys") -> int:
    """*value* if it is an int in ``[1, 2^63)`` (:data:`MAX_KEYS_BOUND`), else *error*."""
    if isinstance(value, bool) or not isinstance(value, int) or not 1 <= value < MAX_KEYS_BOUND:
        raise error("{} must be an int in [1, 2^63), got {!r}".format(what, value))
    return value


@contextlib.contextmanager
def _not_too_deep(what: str):
    """Report a record nested too deeply for Python's recursion limit as :class:`~pyfe4ai.core.exceptions.FEValidationError`."""
    try:
        yield
    except RecursionError:
        raise FEValidationError("{} is nested too deeply (or the caller's stack is too deep)".format(what)) from None


def _select_bytes(bit: int, a: bytes, b: bytes) -> bytes:
    """*a* for ``bit = 0`` and *b* for ``bit = 1``, by a mask (equal lengths)."""
    ia, ib = int.from_bytes(a, "little"), int.from_bytes(b, "little")
    return (ia ^ ((ia ^ ib) & -bit)).to_bytes(len(a), "little")


def _choice(value, allowed: tuple, what: str):
    if isinstance(value, bool) or type(value) not in {type(a) for a in allowed} or value not in allowed:
        raise FEValidationError("{} must be one of {}, got {!r}".format(
            what, ", ".join(repr(a) for a in allowed), value))
    return value


def _check_label(label):
    """A slot label: ``None`` or a str of 1 to :data:`MAX_LABEL_CHARS` characters without control or format
    characters."""
    if label is None:
        return None
    if type(label) is not str or not 1 <= len(label) <= MAX_LABEL_CHARS:
        raise FEValidationError("a slot label is None or a str of 1 to {} characters".format(MAX_LABEL_CHARS))
    if any(unicodedata.category(c) in ("Cc", "Cf", "Cs") for c in label):
        raise FEValidationError("a slot label has no control, format or surrogate characters")
    return label


def _check_requester(requester) -> str:
    """The requester id: a non-empty ``str`` that UTF-8 encodes (its bytes are the dedup scope; see the module
    documentation)."""
    if type(requester) is not str or not requester:
        raise FEValidationError("requester must be a non-empty str: the authenticated id of the requester")
    try:
        requester.encode("utf-8")
    except UnicodeEncodeError:
        raise FEValidationError("requester must be encodable as UTF-8 (it holds a lone surrogate)") from None
    return requester


def _attributes(attributes) -> list:
    """The requester's attributes as a list (the labels are checked by their set digest)."""
    if isinstance(attributes, (str, bytes, bytearray)):
        raise FEValidationError("attributes must be an iterable of attribute labels, not a bare str or bytes")
    try:
        return list(attributes)
    except TypeError:
        raise FEValidationError("attributes must be an iterable of attribute labels") from None


def _key_bits(layout: KeyLayout, w) -> list:
    """Step (1): ``w`` (a dict by field name) through the key layout: its canonical bits."""
    return layout.encode(w)  # FEValidationError: names the field and the element, never the value


def _random_scalar() -> int:
    """A secret scalar of the holder: 64 bytes of the operating system's CSPRNG, little-endian, mod ``l``."""
    while True:
        s = int.from_bytes(os.urandom(bcfe_kem.SCALAR_DRAW_BYTES), "little") % bcfe_kem.ORDER
        if s:
            return s


def _refuse_test_randomness() -> None:
    """Raise while the backend's test hook for random bytes is set (see the module documentation).

    Raises:
        FESchemeError: :data:`TEST_RANDOMNESS_REFUSED`.
    """
    if backend.ops().random_override_active():
        raise FESchemeError(TEST_RANDOMNESS_REFUSED)


def _chunks(data: bytes, size: int) -> list:
    return [data[i:i + size] for i in range(0, len(data), size)]


#: The bound of every ``slot_list_version`` read (a count that fits a signed 64-bit integer).
_MAX_VERSION = 1 << 63


def _check_issue_version(version: int, slot_id: int) -> None:
    """Check the slot list version a key or grant records for slot *slot_id*'s issue: in ``[1, 2^63)`` and at
    most ``slot_id + 1`` (see the module documentation).

    Raises:
        ValueError: Otherwise.
    """
    if not 1 <= version < _MAX_VERSION:
        raise ValueError("slot_list_version must be in [1, 2^63)")
    if version > slot_id + 1:
        raise ValueError("slot_list_version exceeds the slots that can come before slot {} (ids start at 0 and "
                         "increase)".format(slot_id))


# ---------------------------------------------------------------------------
# reading the setup, the slots and the public parameters (fail closed)


class _Setup:
    """A checked setup record, with its circuit (``C`` and the encoding of ``guard(C)``)."""

    __slots__ = ("record", "circuit", "guarded", "setup_digest", "circuit_digest", "data_wires", "m", "n", "n_out",
                 "desc_json")

    def __repr__(self) -> str:
        return "<bcfe setup {}...>".format(self.setup_digest[:8].hex())


def _parse_setup(record, error=FEValidationError, what: str = "the public parameters",
                 cache: dict | None = None) -> _Setup:
    """Check the setup fields of *record* (public parameters or ``setup.json``)
    and rebuild the circuit; *cache* (a dict) keeps the circuit of an encoding
    already read.

    Raises:
        error: For a missing or unknown marker, another scheme, a field of
            another form, a circuit that does not decode or does not match
            its digest or description, or a setup digest that does not match.
    """
    versions.check_bcfe_markers(record, ("bcfe_version", "kem", "gc", "gc_hash"), error, what)
    if record.get("scheme") != SCHEME_ID:
        raise error("{} are not of {}".format(what, SCHEME_ID))
    label_bits = record.get("label_bits")
    if type(label_bits) is not int or label_bits not in LABEL_BITS:
        raise error("{}: unknown label_bits {!r}".format(what, label_bits))
    data_wires = record.get("data_wires")
    if data_wires not in DATA_WIRES:
        raise error("{}: unknown data_wires {!r}".format(what, data_wires))
    try:
        encoding = record.get("circuit_encoding")
        cached = cache.get(encoding) if cache is not None and isinstance(encoding, str) else None
        if cached is None:
            circuit = Circuit.decode(from_b64(encoding, "circuit_encoding"))
            cached = (circuit, circuit.digest(), circuit.guard().encode())
            if cache is not None:
                cache.clear()
                cache[encoding] = cached
        circuit, circuit_digest, guarded = cached
        if from_hex(record.get("circuit_digest"), "circuit_digest", _DIGEST) != circuit_digest:
            raise ValueError("circuit_digest is not the digest of circuit_encoding")
        desc = record.get("circuit")
        if not isinstance(desc, dict) or set(desc) != {"name", "params", "circuit", "key_layout", "data_layout",
                                                       "output_layout"}:
            raise ValueError("circuit is not a circuit description")
        param_circuit = ParamCircuit(desc["name"], desc["params"], circuit, KeyLayout.from_dict(desc["key_layout"]),
                                     DataLayout.from_dict(desc["data_layout"]),
                                     OutputLayout.from_dict(desc["output_layout"]))
        if param_circuit.describe() != desc:
            raise ValueError("the circuit description does not match the circuit")
        nonce = from_hex(record.get("setup_nonce"), "setup_nonce", _DIGEST)
        setup_digest = bcfe_kem.setup_digest(SCHEME_ID, BCFE_VERSION, circuit_digest, data_wires,
                                             record["gc_hash"], record["kem"], label_bits, nonce)
        if from_hex(record.get("setup_digest"), "setup_digest", _DIGEST) != setup_digest:
            raise ValueError("setup_digest does not match the setup")
    except (ValueError, TypeError, FEValidationError) as exc:
        raise error("{}: {}".format(what, exc)) from None
    except RecursionError:
        raise error("{}: nested too deeply (or the caller's stack is too deep)".format(what)) from None
    out = _Setup()
    out.record = {name: record[name] for name in BCFESetup.FIELDS}
    out.circuit, out.guarded, out.setup_digest, out.circuit_digest = param_circuit, guarded, setup_digest, circuit_digest
    out.data_wires, out.m, out.n, out.n_out = data_wires, circuit.m, circuit.n, circuit.n_out
    out.desc_json = param_circuit.describe_json()
    return out


class _Slot:
    __slots__ = ("slot_id", "label", "pks", "digest", "record")


def _parse_slot(slot, setup: _Setup, error=FEValidationError, what: str = "the public parameters") -> _Slot:
    """Check a slot record against *setup*: its fields, the length of its
    table (exactly ``2m + 1`` keys) and its digest.

    Raises:
        error: Otherwise.
    """
    if not isinstance(slot, dict) or set(slot) != set(BCFESlot.FIELDS):
        raise error("{}: a slot record has exactly the fields {}".format(what, ", ".join(BCFESlot.FIELDS)))
    out = _Slot()
    try:
        slot_id = slot["slot_id"]
        if isinstance(slot_id, bool) or not isinstance(slot_id, int) or not 0 <= slot_id < 1 << 64:
            raise ValueError("a slot id must be an int in [0, 2^64)")
        label = _check_label(slot["label"])
        policy = slot["policy_digest"]
        policy = None if policy is None else from_hex(policy, "policy_digest", _DIGEST)
        pks = from_b64(slot["pks"], "pks", _POINT * (2 * setup.m + 1))
        digest = garbled.slot_digest(setup.setup_digest, slot_id, label, policy, pks)
        if from_hex(slot["slot_digest"], "slot_digest", _DIGEST) != digest:
            raise ValueError("the slot digest of slot {} does not match the slot".format(slot_id))
    except (ValueError, TypeError, FEValidationError) as exc:
        raise error("{}: {}".format(what, exc)) from None
    out.slot_id, out.label, out.pks, out.digest, out.record = slot_id, label, pks, digest, slot
    return out


class _PP:
    __slots__ = ("setup", "max_keys", "version", "slots", "pp_digest")


def _parse_pp(pp, cache: dict | None = None) -> _PP:
    """Check public parameters, fail closed: the markers, the setup, every
    slot record and its digest, the slot ids, the pp digest.

    Raises:
        FEValidationError: Otherwise (also for a record nested too deeply).
    """
    with _not_too_deep("the public parameters"):
        return _parse_pp_checked(pp, cache)


def _parse_pp_checked(pp, cache: dict | None) -> _PP:
    what = "the public parameters"
    versions.check_bcfe_markers(pp, ("bcfe_version", "kem", "gc", "gc_hash", "slot_list_version"),
                                FEValidationError, what)
    if set(pp) != set(BCFEPublicParameters.FIELDS):
        raise FEValidationError("{} have exactly the fields {}".format(what, ", ".join(BCFEPublicParameters.FIELDS)))
    setup = _parse_setup(pp, FEValidationError, what, cache)
    max_keys = _max_keys(pp["max_keys"], FEValidationError, "{}: max_keys".format(what))
    slots = pp["slots"]
    version = pp["slot_list_version"]
    if not isinstance(slots, list) or len(slots) != version or version > max_keys:
        raise FEValidationError("{}: the slot list does not match slot_list_version (or exceeds max_keys)".format(
            what))
    parsed = [_parse_slot(slot, setup, FEValidationError, what) for slot in slots]
    for before, after in zip(parsed, parsed[1:]):
        if after.slot_id <= before.slot_id:
            raise FEValidationError("{}: the slot ids are not increasing".format(what))
    try:
        stored = from_hex(pp["pp_digest"], "pp_digest", _DIGEST)
    except ValueError as exc:
        raise FEValidationError("{}: {}".format(what, exc)) from None
    expected = garbled.pp_digest(setup.setup_digest, GC, setup.desc_json, max_keys, [s.digest for s in parsed])
    if stored != expected:
        raise FEValidationError("{}: pp_digest does not match the public parameters".format(what))
    out = _PP()
    out.setup, out.max_keys, out.version, out.slots, out.pp_digest = setup, max_keys, version, parsed, stored
    return out


# ---------------------------------------------------------------------------
# files of an instance


def _read_instance_file(directory: str, name: str, private: bool) -> dict:
    """The JSON object of the instance file *name*, read with the slot store's file checks
    (:func:`~pyfe4ai.schemes.common.slots.read_json_file`: no symbolic link, no FIFO wait, a regular file of this
    user that no other user can write -- *private*: nor read).

    Raises:
        FEConfigError: For a missing, refused or unreadable file.
    """
    data = read_json_file(os.path.join(directory, name), private)
    if data is None:
        raise FEConfigError("{} is not a bcfe instance directory (no {})".format(directory, name))
    return data


def _parse_authority(data: dict, directory: str, setup: _Setup) -> dict:
    """Check ``authority.json`` fail closed: exactly :data:`AUTHORITY_FIELDS`, each of its type and range, the
    setup's digest and the setup's ``data_wires``, ``gc_hash`` and ``label_bits``; returns *data* with
    ``issuance_policy`` read back as a formula.

    Raises:
        FEConfigError: Otherwise.
    """
    def bad(reason: str) -> FEConfigError:
        return FEConfigError("{} of {} is not the authority record of this setup: {}".format(
            AUTHORITY_FILE, directory, reason))

    if set(data) != set(AUTHORITY_FIELDS):
        raise bad("it must hold exactly the fields {}".format(", ".join(AUTHORITY_FIELDS)))
    if data["format"] != AUTHORITY_FORMAT or data["format_version"] != AUTHORITY_FORMAT_VERSION \
            or isinstance(data["format_version"], bool):
        raise bad("another format")
    if data["setup_digest"] != setup.setup_digest.hex():
        raise bad("another setup")
    _max_keys(data["max_keys"], bad, "max_keys")
    for name, allowed in (("key_mode", KEY_MODES), ("security_profile", profiles.PROFILES)):
        if not isinstance(data[name], str) or data[name] not in allowed:
            raise bad("{} must be one of {}".format(name, ", ".join(allowed)))
    if type(data["keep_issued_keys"]) is not bool:
        raise bad("keep_issued_keys must be a bool")
    if (data["data_wires"], data["gc_hash"], data["label_bits"]) != (
            setup.data_wires, setup.record["gc_hash"], setup.record["label_bits"]) or type(data["label_bits"]) is not int:
        raise bad("data_wires, gc_hash or label_bits differ from the setup's")
    sec_param = data["sec_param"]
    if isinstance(sec_param, bool) or not isinstance(sec_param, int) or sec_param < 1:
        raise bad("sec_param must be a positive int")
    policy = data["issuance_policy"]
    try:
        with _not_too_deep("the issuance policy"):
            formula = None if policy is None else from_canonical(policy) if isinstance(policy, str) else False
    except FEValidationError as exc:
        raise bad("issuance_policy: {}".format(exc)) from None
    if formula is False:
        raise bad("issuance_policy must be the canonical text of a formula, or null")
    return dict(data, issuance_policy=formula)


# ---------------------------------------------------------------------------
# the key generator (the authority)


class BCFEGCRistretto255KeyGenerator(CircuitFEAbsKeyGenerator):
    """The authority of ``bcfe/gc_ristretto255``: sets up an instance for one
    circuit and issues at most ``max_keys`` keys, one slot each (see the
    module documentation for the issuance steps and what the scheme does and
    does not protect).

    Args:
        config: ``circuit`` (required, a :class:`~pyfe4ai.circuits.library.ParamCircuit`),
            ``max_keys`` (default 16), ``data_wires`` (``"hardwired"`` or
            ``"labels"``), ``key_mode`` (``"h1"`` or ``"h2"``),
            ``keep_issued_keys`` (default ``True``), ``issuance_policy`` (a
            :mod:`pyfe4ai.policy` formula or ``None``), ``gc_hash``
            (``"shake128"``), ``label_bits`` (128), ``cache_dir``,
            ``security_profile`` (see :class:`~pyfe4ai.core.config.BCFEGCRistretto255Config`).

    Raises:
        FEConfigError: Without ``circuit``.
        FEValidationError: For an option of another type or value.
    """

    def __init__(self, config: dict, **kwargs) -> None:
        super().__init__(config, **kwargs)
        circuit = config.get("circuit")
        if circuit is None:
            raise FEConfigError("{} needs a circuit: config['circuit'], a pyfe4ai.circuits library circuit".format(
                SCHEME_ID))
        if not isinstance(circuit, ParamCircuit):
            raise FEValidationError("circuit must be a pyfe4ai.circuits.ParamCircuit, got {}".format(
                type(circuit).__name__))
        self.circuit = circuit
        if isinstance(self.sec_param, bool) or not isinstance(self.sec_param, int) or self.sec_param < 1:
            raise FEValidationError("sec_param must be a positive int (recorded only), got {!r}".format(self.sec_param))
        self.max_keys = _max_keys(config.get("max_keys", DEFAULT_MAX_KEYS))
        self.data_wires = _choice(config.get("data_wires", DATA_WIRES[0]), DATA_WIRES, "data_wires")
        self.key_mode = _choice(config.get("key_mode", KEY_MODES[0]), KEY_MODES, "key_mode")
        keep = config.get("keep_issued_keys", True)
        if type(keep) is not bool:
            raise FEValidationError("keep_issued_keys must be a bool, got {!r}".format(keep))
        self.keep_issued_keys = keep
        policy = config.get("issuance_policy")
        if policy is not None:
            if not isinstance(policy, Formula):
                raise FEValidationError("issuance_policy must be a pyfe4ai.policy formula or None, got {}".format(
                    type(policy).__name__))
            canonical_encoding(policy)  # validates the formula
        self.issuance_policy = policy
        self.gc_hash = _choice(config.get("gc_hash", GC_HASHES[0]), GC_HASHES, "gc_hash")
        self.label_bits = _choice(config.get("label_bits", LABEL_BITS[0]), LABEL_BITS, "label_bits")
        self._policy_digest = None if policy is None else bytes.fromhex(policy_digest(policy))
        self._dir: str | None = None
        self._setup: _Setup | None = None
        self._store: SlotStore | None = None

    def __repr__(self) -> str:
        return "{}(circuit={}, {})".format(type(self).__name__, self.circuit.name,
                                           "not set up" if self._setup is None else repr(self._setup))

    # -- setup and reload

    @property
    def directory(self) -> str | None:
        """The directory of the instance (``None`` before :meth:`setup`); :meth:`open` resumes it."""
        return self._dir

    def setup(self) -> None:
        """Set up a new instance: a fresh 32-byte ``setup_nonce``, the setup
        digest, the instance directory ``<cache_dir>/authority/bcfe/<setup
        digest>`` with ``setup.json``, ``authority.json`` and an empty slot list.

        Raises:
            FESchemeError: While ``pyfe4ai.backend._random_override`` is set
                (:data:`TEST_RANDOMNESS_REFUSED`).
        """
        _refuse_test_randomness()
        circuit = self.circuit.circuit
        nonce = os.urandom(_DIGEST)
        setup_digest = bcfe_kem.setup_digest(SCHEME_ID, BCFE_VERSION, circuit.digest(), self.data_wires,
                                             self.gc_hash, KEM, self.label_bits, nonce)
        record = BCFESetup.new(
            scheme=SCHEME_ID, bcfe_version=BCFE_VERSION, kem=KEM, gc=GC, gc_hash=self.gc_hash,
            label_bits=self.label_bits, data_wires=self.data_wires, circuit=self.circuit.describe(),
            circuit_encoding=b64(circuit.encode()), circuit_digest=circuit.digest().hex(), setup_nonce=nonce.hex(),
            setup_digest=setup_digest.hex(),
        ).to_dict()
        setup = _parse_setup(record, FEConfigError, "the new setup")
        parent = os.path.join(self.config_folder, "bcfe")
        os.makedirs(parent, exist_ok=True)
        directory = os.path.join(parent, setup_digest.hex())
        os.mkdir(directory, 0o700)
        sync_directory(parent)                                           # the instance's entry survives a crash
        store = SlotStore(directory)
        store.check_directory()
        write_json_file(directory, SETUP_FILE, record, 0o644)
        write_json_file(directory, AUTHORITY_FILE, self._authority_record(setup_digest, self.max_keys), 0o600)
        self._dir, self._setup, self._store = directory, setup, store
        logger.info("bcfe setup %s... in %s", setup_digest[:8].hex(), directory)

    def _authority_record(self, setup_digest: bytes, max_keys: int) -> dict:
        return {
            "format": AUTHORITY_FORMAT, "format_version": AUTHORITY_FORMAT_VERSION, "setup_digest": setup_digest.hex(),
            "max_keys": max_keys, "key_mode": self.key_mode, "keep_issued_keys": self.keep_issued_keys,
            "issuance_policy": None if self.issuance_policy is None else canonical_encoding(self.issuance_policy),
            "data_wires": self.data_wires, "gc_hash": self.gc_hash, "label_bits": self.label_bits,
            "security_profile": self.security_profile, "sec_param": self.sec_param,
        }

    @staticmethod
    def _read_authority(directory: str, setup: _Setup) -> dict:
        """``authority.json``, read with the store's file checks and parsed fail closed (:func:`_parse_authority`;
        ``issuance_policy`` as a formula)."""
        return _parse_authority(_read_instance_file(directory, AUTHORITY_FILE, private=True), directory, setup)

    @classmethod
    def open(cls, directory) -> "BCFEGCRistretto255KeyGenerator":
        """Resume the instance in *directory* (a :attr:`directory` of an
        earlier :meth:`setup`): its setup, its options and its slot list.

        Raises:
            FEConfigError: If the directory holds no instance, or its files are
                unreadable, inconsistent or do not match each other.
        """
        directory = os.path.abspath(os.fspath(directory))
        store = SlotStore(directory)
        store.check_directory()                                          # before anything in it is read
        what = "{} of {}".format(SETUP_FILE, directory)
        record = _read_instance_file(directory, SETUP_FILE, private=False)
        if set(record) != set(BCFESetup.FIELDS):
            raise FEConfigError("{} must hold exactly the fields {}".format(what, ", ".join(BCFESetup.FIELDS)))
        setup = _parse_setup(record, FEConfigError, what)
        data = cls._read_authority(directory, setup)
        config = {
            "circuit": setup.circuit, "max_keys": data["max_keys"], "data_wires": data["data_wires"],
            "key_mode": data["key_mode"], "keep_issued_keys": data["keep_issued_keys"],
            "issuance_policy": data["issuance_policy"], "gc_hash": data["gc_hash"], "label_bits": data["label_bits"],
            "security_profile": data["security_profile"], "sec_param": data["sec_param"],
            "cache_dir": os.path.dirname(os.path.dirname(os.path.dirname(directory))),
        }
        try:
            kg = cls(config)
        except (FEValidationError, FEConfigError) as exc:
            raise FEConfigError("{} of {} does not hold the options of a bcfe authority: {}".format(
                AUTHORITY_FILE, directory, exc)) from None
        kg._dir, kg._setup, kg._store = directory, setup, store
        return kg

    def _require_setup(self) -> _Setup:
        if self._setup is None:
            raise FESchemeError("call setup() (or open an instance) first")
        return self._setup

    def _read_max_keys(self) -> int:
        """``max_keys`` as ``authority.json`` records it now (another authority object on the directory may have
        changed it); called under the slot store's lock."""
        return self._read_authority(self._dir, self._setup)["max_keys"]

    def _locked_cap(self) -> tuple:
        """``(cap, decided)``: the cap for :meth:`SlotStore.issue` as a callable that reads ``max_keys`` under the
        store's lock, and the dict in which it leaves the value read (``decided["max_keys"]``)."""
        decided: dict = {}

        def cap() -> int:
            decided["max_keys"] = self._read_max_keys()
            return decided["max_keys"]

        return cap, decided

    def set_max_keys(self, max_keys: int) -> None:
        """Change the cap on the slots: raise it, or lower it, never below the slots listed. The check and the
        write run under the slot store's lock, so no issuance runs between them (see the module documentation).

        Raises:
            FEValidationError: For a value that is not an int in ``[1, 2^63)``, or below the slots listed.
            FEConfigError: For an ``authority.json`` that does not check.
        """
        setup = self._require_setup()
        max_keys = _max_keys(max_keys)

        def change(slots: list) -> None:
            if max_keys < len(slots):
                raise FEValidationError("max_keys cannot go below the {} slots listed".format(len(slots)))
            raw = _read_instance_file(self._dir, AUTHORITY_FILE, private=True)
            _parse_authority(raw, self._dir, setup)                      # the record checks before it is rewritten
            write_json_file(self._dir, AUTHORITY_FILE, dict(raw, max_keys=max_keys), 0o600)

        self._store.run_locked(change)
        self.max_keys = max_keys

    # -- the public parameters

    def _slots(self, records: list | None = None) -> list:
        """The slot list (*records*, else the store's now), every record checked (a slot store that does not
        match its setup is refused: ``FEConfigError``)."""
        setup = self._require_setup()
        records = self._store.public_slots() if records is None else records
        return [_parse_slot(slot, setup, FEConfigError, "the slot list of {}".format(self._dir)) for slot in records]

    def get_public_parameters(self) -> dict:
        """The public parameters as they are now
        (:class:`~pyfe4ai.core.formats.garbled.BCFEPublicParameters`): the
        setup, ``max_keys``, the slot list, its version and the pp digest. A
        new slot (or a new ``max_keys``) gives new public parameters: hand
        them to the encryptors."""
        setup = self._require_setup()
        max_keys, records = self._store.run_locked(lambda listed: (self._read_max_keys(), listed))  # one moment
        self.max_keys = max_keys
        slots = self._slots(records)
        digest = garbled.pp_digest(setup.setup_digest, GC, setup.desc_json, max_keys, [s.digest for s in slots])
        return BCFEPublicParameters.new(
            **setup.record, max_keys=max_keys, slot_list_version=len(slots), slots=[s.record for s in slots],
            pp_digest=digest.hex(),
        ).to_dict()

    def _at_issue(self, slot_id: int, issue_max_keys: int) -> tuple:
        """``(slot record, slot_list_version, pp_digest)`` as of slot *slot_id*'s issue."""
        setup = self._setup
        slots = self._slots()
        index = next((i for i, s in enumerate(slots) if s.slot_id == slot_id), None)
        if index is None:
            raise FESchemeError("slot {} is not in the slot list".format(slot_id))
        digest = garbled.pp_digest(setup.setup_digest, GC, setup.desc_json, issue_max_keys,
                                   [s.digest for s in slots[:index + 1]])
        return slots[index], index + 1, digest

    # -- issuance

    def _request(self, w, requester, attributes, label) -> tuple:
        """Steps (1) to (3): ``(w packed, slot label, policy digest, attribute set digest, dedup key)``."""
        setup = self._require_setup()
        bits = _key_bits(self.circuit.key_layout, w)                     # (1)
        requester = _check_requester(requester)
        label = _check_label(label)
        attributes = _attributes(attributes)
        attrs_digest = bytes.fromhex(attribute_set_digest(attributes))   # checks every label
        if self.issuance_policy is not None and not evaluate(self.issuance_policy, attributes):  # (2)
            raise FEKeyError("the requester's attributes do not satisfy the issuance policy: no key is issued")
        w_packed = pack_bits(bits, setup.m)
        dedup = garbled.dedup_key(SCHEME_ID, setup.circuit_digest, requester, w_packed)  # (3), never published
        return w_packed, label, attrs_digest, dedup

    def _public_record(self, slot_id: int, label, pks: bytes) -> dict:
        digest = garbled.slot_digest(self._setup.setup_digest, slot_id, label, self._policy_digest, pks)
        record = BCFESlot.new(
            slot_id=slot_id, label=label,
            policy_digest=None if self._policy_digest is None else self._policy_digest.hex(),
            pks=b64(pks), slot_digest=digest.hex(),
        ).to_dict()
        del record["slot_id"]  # the store adds it first
        return record

    def _key(self, slot_id: int, scalars: bytes, w_packed: bytes, issue_max_keys: int) -> dict:
        slot, version, digest = self._at_issue(slot_id, issue_max_keys)
        setup = self._setup
        return BCFEKey.new(
            scheme=SCHEME_ID, bcfe_version=BCFE_VERSION, setup_digest=setup.setup_digest.hex(),
            circuit_digest=setup.circuit_digest.hex(), slot_id=slot_id, slot_digest=slot.digest.hex(),
            pp_digest=digest.hex(), slot_list_version=version, scalars=b64(scalars), w=b64(w_packed),
        ).to_dict()

    def issue_key(self, w, requester: str, attributes=(), label: str | None = None) -> dict:
        """Issue the key of ``w`` to *requester* (h1): steps (1) to (5) of the
        module documentation. The authority makes the slot's selected key
        pairs (backend ``bcfe_new_slot``).

        Args:
            w: The key values, a dict by field name of the circuit's key layout.
            requester: The requester's **authenticated** id (a non-empty
                ``str``): the scope of the deduplication. The library cannot
                authenticate it.
            attributes: The requester's attribute labels (checked against
                ``issuance_policy``; their set digest is published with the slot).
            label: The slot's public name (``None`` or a short ``str``);
                encryptors may select slots by it. A repeated request keeps
                the first slot's label.

        Returns:
            The key (:class:`~pyfe4ai.core.formats.garbled.BCFEKey`); for the
            same requester and ``w`` again, the same key byte for byte (with
            ``keep_issued_keys``).

        Raises:
            FEValidationError: For ``w`` outside the key layout, or a
                malformed requester, label or attribute.
            FEKeyError: If the attributes do not satisfy the issuance policy
                (nothing is reserved); if the cap ``max_keys`` is reached (the
                same error whether this request was issued before or not);
                for a repeated request when the authority keeps no keys; if
                the setup issues keys with ``key_mode = "h2"``.
            FESchemeError: Before :meth:`setup`; while
                ``pyfe4ai.backend._random_override`` is set
                (:data:`TEST_RANDOMNESS_REFUSED`).
        """
        _refuse_test_randomness()
        self._require_setup()
        if self.key_mode != "h1":
            raise FEKeyError("this setup issues keys with key_mode='h2': the key holder requests a slot "
                             "(BCFEGCRistretto255.key_holder) and the authority answers with authorize()")
        w_packed, label, attrs_digest, dedup = self._request(w, requester, attributes, label)
        m, keep = self._setup.m, self.keep_issued_keys
        cap, decided = self._locked_cap()
        made = {}

        def build(slot_id: int):                                         # (5), under the store's lock
            out = backend.ops().bcfe_new_slot(self._setup.setup_digest, slot_id, w_packed, m=m)
            made["scalars"] = out["scalars"]
            private = {"key_mode": "h1", "max_keys": decided["max_keys"], "k": out["k"],
                       "attribute_set_digest": attrs_digest}
            if keep:
                private.update(w=w_packed, scalars=out["scalars"])
            return self._public_record(slot_id, label, out["pks"]), private

        try:
            result = self._store.issue(dedup, cap, build)                # (4): the cap read under the lock
        finally:
            self.max_keys = decided.get("max_keys", self.max_keys)
        if result.status == "hit":
            private = result.private
            if private.get("key_mode") != "h1" or not keep or "scalars" not in private:
                raise FEKeyError("this function was already issued to this requester, and the authority keeps no "
                                 "copy of the key (keep_issued_keys=False when it was issued, or now): a lost key "
                                 "is not issued again for the same requester")
            scalars, issue_max_keys = private["scalars"], private["max_keys"]
        else:
            scalars, issue_max_keys = made["scalars"], decided["max_keys"]
            logger.info("bcfe: slot %d issued (h1)", result.slot_id)
        return self._key(result.slot_id, scalars, w_packed, issue_max_keys)

    def authorize(self, request: dict, requester: str, attributes=(), label: str | None = None) -> dict:
        """Answer a key holder's request (h2, SS10's Auth): steps (1) to (5)
        of the module documentation, with the holder's ``m + 1`` public keys
        at the selected positions (backend ``bcfe_place_slot``; each must be a
        group element other than the identity) and the unselected ones
        derived from a fresh slot secret ``k``.

        Args:
            request: The holder's request
                (:class:`~pyfe4ai.core.formats.garbled.BCFEKeyRequest`, from
                ``BCFEGCRistretto255.key_holder(pp).request(w)``).
            requester, attributes, label: As :meth:`issue_key`.

        Returns:
            The grant (:class:`~pyfe4ai.core.formats.garbled.BCFEGrant`): the
            slot record and ``k``, for the holder's ``finish``. It holds ``k``,
            which reveals ``w``: send it to the holder only. The same
            requester asking again for the same ``w`` with the same keys gets
            the same grant.

        Raises:
            FEValidationError: For a malformed request, one of another setup,
                ``w`` outside the key layout, or a holder key that is not a
                group element or is the identity.
            FEKeyError: As :meth:`issue_key`; also for holder keys that are
                not pairwise distinct (nothing is reserved; equal keys at two
                selected positions, or at a selected one and ``(m, 0)``, would
                show bits of ``w`` in the public slot list); for a repeated
                request of the same requester and ``w`` with other public
                keys (the authority never replaces a slot's keys), or if the
                setup issues keys with ``key_mode = "h1"``.
            FESchemeError: Before :meth:`setup`; while
                ``pyfe4ai.backend._random_override`` is set
                (:data:`TEST_RANDOMNESS_REFUSED`).
        """
        _refuse_test_randomness()
        setup = self._require_setup()
        if self.key_mode != "h2":
            raise FEKeyError("this setup issues keys with key_mode='h1' (issue_key): it does not place holders' keys")
        what = "the key request"
        with _not_too_deep(what):
            versions.check_bcfe_markers(request, ("bcfe_version",), FEValidationError, what)
            if set(request) != set(BCFEKeyRequest.FIELDS) or request.get("scheme") != SCHEME_ID:
                raise FEValidationError("{} is not a request of {} (fields {})".format(
                    what, SCHEME_ID, ", ".join(BCFEKeyRequest.FIELDS)))
            try:
                if (from_hex(request["setup_digest"], "setup_digest", _DIGEST) != setup.setup_digest
                        or from_hex(request["circuit_digest"], "circuit_digest", _DIGEST) != setup.circuit_digest):
                    raise ValueError("it is for another setup")
                holder_pks = from_b64(request["pks"], "pks", _POINT * (setup.m + 1))
            except ValueError as exc:
                raise FEValidationError("{}: {}".format(what, exc)) from None
            holder = _chunks(holder_pks, _POINT)
            for i, pk in enumerate(holder):                              # no identity, checked before reserving
                if pk == _IDENTITY:
                    raise FEValidationError("{}: holder public key {} is the identity".format(what, i))
                eg.from_hex(pk.hex(), "holder public key {}".format(i))
            if len(set(holder)) != len(holder):                          # equal keys would show w in the slot list
                raise FEKeyError("{}: the holder's {} public keys are not pairwise distinct: no slot is issued".format(
                    what, len(holder)))
            w_packed, label, attrs_digest, dedup = self._request(request["w"], requester, attributes, label)
        m = setup.m
        cap, decided = self._locked_cap()

        def build(slot_id: int):
            out = backend.ops().bcfe_place_slot(setup.setup_digest, slot_id, w_packed, holder_pks, m=m)
            private = {"key_mode": "h2", "max_keys": decided["max_keys"], "k": out["k"],
                       "attribute_set_digest": attrs_digest}
            return self._public_record(slot_id, label, out["pks"]), private

        try:
            result = self._store.issue(dedup, cap, build)                # the cap read under the lock
        finally:
            self.max_keys = decided.get("max_keys", self.max_keys)
        private = result.private
        if private.get("key_mode") != "h2":
            raise FEKeyError("this function was already issued to this requester by the authority (h1)")
        slot, version, digest = self._at_issue(result.slot_id, private["max_keys"])
        if result.status == "hit":
            bits = unpack_bits(w_packed, m)
            table = _chunks(slot.pks, _POINT)
            placed = b"".join(table[2 * j + bits[j]] for j in range(m)) + table[2 * m]
            if placed != holder_pks:
                raise FEKeyError("this function was already issued to this requester for other public keys; the "
                                 "authority does not replace a slot's keys")
        else:
            logger.info("bcfe: slot %d issued (h2)", result.slot_id)
        return BCFEGrant.new(
            scheme=SCHEME_ID, bcfe_version=BCFE_VERSION, setup_digest=setup.setup_digest.hex(),
            circuit_digest=setup.circuit_digest.hex(), slot=slot.record, k=private["k"].hex(),
            pp_digest=digest.hex(), slot_list_version=version,
        ).to_dict()


# ---------------------------------------------------------------------------
# the key holder of h2


class BCFEGCRistretto255KeyHolder:
    """The key holder's side of the holder-generated issuance (h2, SS10's Pre
    and CheckAuth): :meth:`request` makes the ``m + 1`` key pairs of the
    selected positions and returns the request; :meth:`finish` checks the
    authority's grant and returns the key.

    The secret keys stay in this object between the two calls (several
    requests may be open). They are secret: the object has no secrets in its
    ``repr``, refuses to be pickled or copied, and cannot wipe them (Python);
    if it is lost, request again (the authority keeps no key of an h2 slot,
    and does not replace a slot's keys: a new slot needs another
    function or requester id).

    Threads may share a holder: a lock of the object covers its open
    requests (renewed in a forked child, :func:`pyfe4ai.core.forksafe.track`),
    so two :meth:`finish` calls of one grant give one key and one
    :class:`~pyfe4ai.core.exceptions.FEKeyError`.

    Args:
        pp: The public parameters (checked, :class:`~pyfe4ai.core.exceptions.FEValidationError` otherwise).
    """

    def __init__(self, pp: dict) -> None:
        self._pp = _parse_pp(pp)
        self._pending: dict = {}  # holder pks -> (w packed, scalars)
        self._lock = threading.Lock()
        forksafe.track(self)  # a forked child: a new lock (another thread may hold this one at the fork)

    def _after_fork_in_child(self) -> None:
        self._lock = threading.Lock()

    def __repr__(self) -> str:
        return "{}(setup={}..., open requests={})".format(
            type(self).__name__, self._pp.setup.setup_digest[:8].hex(), len(self._pending))

    def __reduce__(self):
        raise TypeError("a {} holds secret keys: it is not pickled or copied".format(type(self).__name__))

    def request(self, w) -> dict:
        """The request for ``w`` (SS10's Pre): ``m + 1`` fresh secret keys from
        the operating system's CSPRNG, kept here; the request carries ``w``
        and their public keys.

        Returns:
            :class:`~pyfe4ai.core.formats.garbled.BCFEKeyRequest`; it holds
            ``w``: send it to the authority only.

        Raises:
            FEValidationError: For ``w`` outside the key layout.
            FESchemeError: While ``pyfe4ai.backend._random_override`` is set
                (:data:`TEST_RANDOMNESS_REFUSED`). The scalars come from
                :func:`os.urandom`, which the hook does not reach; the
                request is refused like the authority's side of the issuance.
        """
        _refuse_test_randomness()
        setup = self._pp.setup
        bits = _key_bits(setup.circuit.key_layout, w)
        values = setup.circuit.key_layout.decode(bits)  # the canonical values (plain ints)
        scalars = [_random_scalar() for _ in range(setup.m + 1)]
        pks = b"".join(bytes.fromhex(eg.to_hex(eg.mul_base(s))) for s in scalars)
        with self._lock:
            self._pending[pks] = (pack_bits(bits, setup.m), b"".join(bcfe_kem.encode_scalar(s) for s in scalars))
        return BCFEKeyRequest.new(
            scheme=SCHEME_ID, bcfe_version=BCFE_VERSION, setup_digest=setup.setup_digest.hex(),
            circuit_digest=setup.circuit_digest.hex(), w=values, pks=b64(pks),
        ).to_dict()

    def finish(self, grant: dict) -> dict:
        """Check the authority's grant (SS10's CheckAuth) and return the key.

        The slot record must match its digest under this setup, have exactly
        ``2m + 1`` public keys, hold this holder's keys at ``(j, w_j)`` and
        ``(m, 0)`` for one open request, and its unselected positions must be
        ``H_P(setup_digest, slot_id, j, k)`` for the grant's ``k``
        (backend ``bcfe_unselected_ok``).

        The grant's ``pp_digest`` (the pp digest of the slot list as of the
        slot's issue) goes into the key as the authority states it: only its
        form is checked. The holder cannot recompute it -- its public
        parameters are, as a rule, older than its slot (the slots issued in
        between are unknown to it), and the cap ``max_keys`` at the issue,
        an input of the digest, is not in the grant -- and no reader of the
        key compares it (:meth:`BCFEGCRistretto255.decrypt` checks the key
        against its listed slot instead). It is informational, as in h1.

        Raises:
            FEKeyError: For a malformed grant (its ``slot_list_version`` too,
                checked against this holder's public parameters: see *Checks
                of the versions* in the module documentation), one of another
                setup, or one that fails CheckAuth (the request then stays
                open).
        """
        setup = self._pp.setup
        what = "the grant"
        with _not_too_deep(what):
            versions.check_bcfe_markers(grant, ("bcfe_version", "slot_list_version"), FEKeyError, what)
            if set(grant) != set(BCFEGrant.FIELDS) or grant.get("scheme") != SCHEME_ID:
                raise FEKeyError("{} is not a grant of {}".format(what, SCHEME_ID))
            try:
                if (from_hex(grant["setup_digest"], "setup_digest", _DIGEST) != setup.setup_digest
                        or from_hex(grant["circuit_digest"], "circuit_digest", _DIGEST) != setup.circuit_digest):
                    raise ValueError("it is for another setup")
                k = from_hex(grant["k"], "k", bcfe_kem.SLOT_KEY_BYTES)
                from_hex(grant["pp_digest"], "pp_digest", _DIGEST)  # informational: its form only (see finish)
            except ValueError as exc:
                raise FEKeyError("{}: {}".format(what, exc)) from None
            slot = _parse_slot(grant["slot"], setup, FEKeyError, what)  # exactly 2m + 1 keys, the slot digest
            try:
                _check_issue_version(grant["slot_list_version"], slot.slot_id)
            except ValueError as exc:
                raise FEKeyError("{}: {}".format(what, exc)) from None
        m = setup.m
        table = _chunks(slot.pks, _POINT)
        with self._lock:                                                 # one finish per open request
            for own, (w_packed, scalars) in self._pending.items():
                bits = unpack_bits(w_packed, m)
                placed = b"".join(_select_bytes(bits[j], table[2 * j], table[2 * j + 1]) for j in range(m))
                if hmac.compare_digest(placed + table[2 * m], own):
                    break
            else:
                raise FEKeyError("CheckAuth failed: the slot of the grant does not hold this holder's public keys at "
                                 "the positions of the requested function (a grant of another request or holder, a "
                                 "slot built for another function, or a request already finished)")
            if not backend.ops().bcfe_unselected_ok(setup.setup_digest, slot.slot_id, w_packed, k, slot.pks, m=m):
                raise FEKeyError("CheckAuth failed: the unselected positions of the slot are not derived from the "
                                 "grant's k")
            del self._pending[own]
        return BCFEKey.new(
            scheme=SCHEME_ID, bcfe_version=BCFE_VERSION, setup_digest=setup.setup_digest.hex(),
            circuit_digest=setup.circuit_digest.hex(), slot_id=slot.slot_id, slot_digest=slot.digest.hex(),
            pp_digest=grant["pp_digest"], slot_list_version=grant["slot_list_version"], scalars=b64(scalars),
            w=b64(w_packed),
        ).to_dict()


# ---------------------------------------------------------------------------
# encryption and decryption


class BCFEGCRistretto255(CircuitFEAbsCrypto):
    """Encryptor and decryptor of ``bcfe/gc_ristretto255``, built from
    ``{"keys": {"pp": public parameters}}``.

    The public parameters are checked (fail closed) before every use: the
    markers, the setup digest, every slot digest and the pp digest
    (:class:`~pyfe4ai.core.exceptions.FEValidationError` otherwise).

    Two caches live on the object and are left out of its copies and pickles
    (rebuilt on first use): the circuit of the public parameters' encoding,
    and the verdicts of the key checks of :meth:`decrypt` (by slot digest and
    a SHAKE-256 hash of the key's scalars and ``w``; at most
    256 entries, no secret). Threads may share the object: a lock of the
    object covers the lookup, eviction and insertion of a verdict (the check
    itself runs outside it), and is renewed in a forked child
    (:func:`pyfe4ai.core.forksafe.track`).
    """

    def __init__(self, config: dict, **kwargs) -> None:
        super().__init__(config, **kwargs)
        self._circuit_cache: dict = {}  # the circuit of the last encoding read (one entry)
        self._key_checks: dict = {}     # key check tag -> verdict (bool), under _key_checks_lock
        self._key_checks_lock = threading.Lock()
        forksafe.track(self)  # a forked child: a new lock (another thread may hold this one at the fork)
        _parse_pp(self.keys["pp"], self._circuit_cache)

    def _after_fork_in_child(self) -> None:
        self._key_checks_lock = threading.Lock()

    def __getstate__(self) -> dict:
        state = dict(self.__dict__)
        state["_circuit_cache"], state["_key_checks"] = {}, {}
        del state["_key_checks_lock"]
        return state

    def __setstate__(self, state: dict) -> None:
        self.__dict__.update(state)
        self._circuit_cache, self._key_checks = {}, {}
        self._key_checks_lock = threading.Lock()
        forksafe.track(self)

    def _check_key(self, slot: _Slot, scalars: bytes, w_packed: bytes, m: int) -> bool:
        """Whether the key's secret keys open its listed slot: ``s_j B = P[2j + w_j]`` for ``j < m`` and ``s_g B
        = P[2m]`` (cached per key, see the class documentation)."""
        tag = slot.digest + hashlib.shake_256(b"PYFE4AI-V06-BCFE-KEYCHECK" + scalars + w_packed).digest(32)
        with self._key_checks_lock:
            verdict = self._key_checks.get(tag)
        if verdict is None:
            table = _chunks(slot.pks, _POINT)
            bits = unpack_bits(w_packed, m)
            expected = b"".join(_select_bytes(bits[j], table[2 * j], table[2 * j + 1]) for j in range(m)) + table[2 * m]
            try:
                actual = b"".join(bytes.fromhex(eg.to_hex(eg.mul_base(bcfe_kem.decode_scalar(part))))
                                  for part in _chunks(scalars, _SCALAR))
            except ValueError:  # a scalar not below the group order
                actual = b""
            verdict = len(actual) == len(expected) and hmac.compare_digest(actual, expected)
            with self._key_checks_lock:                                  # evict the oldest, keep the bound
                while len(self._key_checks) >= _KEY_CHECKS_KEPT:
                    del self._key_checks[next(iter(self._key_checks))]
                self._key_checks[tag] = verdict
        return verdict

    def _pp(self) -> _PP:
        return _parse_pp(self.keys["pp"], self._circuit_cache)

    @staticmethod
    def key_holder(pp: dict) -> BCFEGCRistretto255KeyHolder:
        """The key holder of the holder-generated issuance (h2) for *pp*:
        ``holder.request(w)`` to the authority's ``authorize``, then
        ``holder.finish(grant)`` gives the key."""
        return BCFEGCRistretto255KeyHolder(pp)

    def encrypt(self, x, slots=None) -> dict:
        """Encrypt the data values *x* to the slots of the public parameters.

        Every selected slot gets its own component (backend
        ``bcfe_encrypt_slot``: a fresh garbling of ``guard(C)``, fresh labels
        and a fresh ``R``); the ciphertext has one fresh 16-byte ``ct_nonce``.

        Args:
            x: The data values, a dict by field name of the circuit's data layout.
            slots: ``None`` for every slot of the list; a slot label
                (``str``: every slot of that label) or a slot id (``int``,
                not ``bool``); or an iterable (list, tuple, set, generator)
                of labels and ids. A label is the authority's name for a slot
                and is not unique: it selects every slot of the list with
                that label, whoever requested it, also slots listed after
                the public parameters you checked; to encrypt to exactly the
                slots you mean, pin the ``pp_digest`` and select by slot id. Anything else is refused, bytes, dicts and
                floats included (alternative not taken: reading a dict's keys
                or the bytes of a bytes object as ids, which would hide a
                caller's mistake).

        Returns:
            The ciphertext (:class:`~pyfe4ai.core.formats.garbled.BCFECiphertext`).

        Raises:
            FEValidationError: For public parameters that do not check, *x*
                outside the data layout, an empty slot list or selection, or
                a label or id that names no slot.
            FESchemeError: While ``pyfe4ai.backend._random_override`` is set
                (:data:`TEST_RANDOMNESS_REFUSED`).
        """
        _refuse_test_randomness()
        pp = self._pp()
        setup = pp.setup
        x_packed = pack_bits(setup.circuit.data_layout.encode(x), setup.n)
        selected = self._select(pp.slots, slots)
        ct_nonce = os.urandom(bcfe_kem.NONCE_BYTES)
        components = []
        for slot in selected:
            try:
                out = backend.ops().bcfe_encrypt_slot(setup.guarded, setup.data_wires, pp.pp_digest, ct_nonce,
                                                      slot.slot_id, x_packed, slot.pks)
            except ValueError:
                raise FEValidationError("slot {} of the public parameters has a public key that is not a group "
                                        "element other than the identity".format(slot.slot_id)) from None
            components.append(BCFEComponent.new(
                slot_id=slot.slot_id, R=out["R"].hex(), label_cts=b64(out["label_cts"]), tables=b64(out["tables"]),
                decode=b64(out["decode"]), data_active=b64(out["data_active"]),
            ).to_dict())
        return BCFECiphertext.new(
            scheme=SCHEME_ID, bcfe_version=BCFE_VERSION, setup_digest=setup.setup_digest.hex(),
            circuit_digest=setup.circuit_digest.hex(), slot_list_version=pp.version, pp_digest=pp.pp_digest.hex(),
            ct_nonce=ct_nonce.hex(), components=components,
        ).to_dict()

    @staticmethod
    def _select(slots: list, wanted) -> list:
        if not slots:
            raise FEValidationError("the public parameters list no slot: nobody holds a key yet")
        if wanted is None:
            return list(slots)
        if isinstance(wanted, str) or (isinstance(wanted, int) and not isinstance(wanted, bool)):
            wanted = [wanted]
        elif (isinstance(wanted, (bool, bytes, bytearray, memoryview, Mapping)) or not isinstance(wanted, Iterable)):
            raise FEValidationError("slots is None, a slot label (str), a slot id (int) or an iterable of labels and "
                                    "ids, not {}".format(type(wanted).__name__))
        try:
            wanted = list(wanted)
        except TypeError:
            raise FEValidationError("slots is None, a slot label (str), a slot id (int) or an iterable of labels and "
                                    "ids") from None
        ids, labels = set(), set()
        for item in wanted:
            if isinstance(item, bool) or not isinstance(item, (int, str)):
                raise FEValidationError("slots selects by slot label (str) or slot id (int), got {}".format(
                    type(item).__name__))
            (ids if isinstance(item, int) else labels).add(item)
        known_ids, known_labels = {s.slot_id for s in slots}, {s.label for s in slots}
        missing = sorted(ids - known_ids)
        if missing:
            raise FEValidationError("no slot with the ids {}".format(missing))
        missing = sorted(labels - known_labels)
        if missing:
            raise FEValidationError("no slot with the labels {}".format(missing))
        chosen = [s for s in slots if s.slot_id in ids or s.label in labels]
        if not chosen:
            raise FEValidationError("slots selects no slot")
        return chosen

    @staticmethod
    def _parse_key(key, pp: _PP) -> tuple:
        setup = pp.setup
        what = "the key"
        versions.check_bcfe_markers(key, ("bcfe_version", "slot_list_version"), FEKeyError, what)
        if set(key) != set(BCFEKey.FIELDS) or key.get("scheme") != SCHEME_ID:
            raise FEKeyError("{} is not a key of {}".format(what, SCHEME_ID))
        try:
            if from_hex(key["setup_digest"], "setup_digest", _DIGEST) != setup.setup_digest:
                raise ValueError("it is a key of another setup than the public parameters'")
            if from_hex(key["circuit_digest"], "circuit_digest", _DIGEST) != setup.circuit_digest:
                raise ValueError("it is a key of another circuit than the public parameters'")
            slot_id = key["slot_id"]
            if isinstance(slot_id, bool) or not isinstance(slot_id, int) or not 0 <= slot_id < 1 << 64:
                raise ValueError("slot_id must be an int in [0, 2^64)")
            slot_digest = from_hex(key["slot_digest"], "slot_digest", _DIGEST)
            from_hex(key["pp_digest"], "pp_digest", _DIGEST)
            _check_issue_version(key["slot_list_version"], slot_id)
            scalars = from_b64(key["scalars"], "scalars", _SCALAR * (setup.m + 1))
            w_packed = from_b64(key["w"], "w", (setup.m + 7) // 8)
            unpack_bits(w_packed, setup.m)  # no padding bits
        except ValueError as exc:
            raise FEKeyError("{}: {}".format(what, exc)) from None
        return slot_id, slot_digest, key["slot_list_version"], scalars, w_packed

    def decrypt(self, ct: dict, key: dict) -> dict:
        """``C(w, x)``, decoded by the output layout, from a ciphertext of
        ``x`` and the key of ``w``.

        Raises:
            FEValidationError: For public parameters that do not check, or a
                ciphertext that is malformed, of another setup or circuit, or
                with two components of one slot or one ``R``. The decryptor's
                public parameters give the circuit; the ciphertext's own
                ``pp_digest`` (that of the encryptor's public parameters,
                which may be older or newer) goes into the hashes, so a
                ciphertext of other public parameters of this setup does not
                decrypt to ``C(w, x)``.
            FEKeyError: For a malformed key (its ``slot_list_version`` too:
                see *Checks of the versions* in the module documentation),
                one of another setup or circuit, or of another slot than the
                public parameters' slot of its id; if the ciphertext has no component for the key's slot --
                older than the key, or not encrypted to it.
            FESchemeError: :data:`DECRYPTION_FAILED`, for every failure inside
                the component (one error, whatever the cause: which position
                failed is not told), and for a
                key whose secret keys do not open its slot when the public
                parameters list the slot (a damaged key or ``w``; see *Keys
                are checked against their slot* in the module documentation).
        """
        pp = self._pp()
        with _not_too_deep("the key or the ciphertext"):
            return self._decrypt(pp, ct, key)

    def _decrypt(self, pp: _PP, ct: dict, key: dict) -> dict:
        setup = pp.setup
        slot_id, slot_digest, issued_version, scalars, w_packed = self._parse_key(key, pp)
        known = next((s for s in pp.slots if s.slot_id == slot_id), None)
        if known is not None and known.digest != slot_digest:
            raise FEKeyError("the key is not for slot {} of the public parameters (another slot digest)".format(
                slot_id))
        what = "the ciphertext"
        versions.check_bcfe_markers(ct, ("bcfe_version", "slot_list_version"), FEValidationError, what)
        if set(ct) != set(BCFECiphertext.FIELDS) or ct.get("scheme") != SCHEME_ID:
            raise FEValidationError("{} is not a ciphertext of {}".format(what, SCHEME_ID))
        try:
            if from_hex(ct["setup_digest"], "setup_digest", _DIGEST) != setup.setup_digest:
                raise ValueError("it was made for another setup than the public parameters'")
            if from_hex(ct["circuit_digest"], "circuit_digest", _DIGEST) != setup.circuit_digest:
                raise ValueError("it was made for another circuit than the public parameters'")
            ct_pp_digest = from_hex(ct["pp_digest"], "pp_digest", _DIGEST)
            ct_nonce = from_hex(ct["ct_nonce"], "ct_nonce", bcfe_kem.NONCE_BYTES)
            components = ct["components"]
            if not isinstance(components, list) or not all(
                    isinstance(c, dict) and set(c) == set(BCFEComponent.FIELDS) for c in components):
                raise ValueError("components must be a list of component records")
            ids = [c["slot_id"] for c in components]
            if any(isinstance(i, bool) or not isinstance(i, int) for i in ids) or len(set(ids)) != len(ids):
                raise ValueError("the component slot ids must be distinct ints")
            rs = [c["R"] for c in components]
            if any(not isinstance(r, str) for r in rs) or len(set(r.lower() for r in rs)) != len(rs):
                raise ValueError("two components share an ephemeral key R")
            if not len(components) <= ct["slot_list_version"] < _MAX_VERSION:
                raise ValueError("slot_list_version must be in [the number of components, 2^63)")
        except ValueError as exc:
            raise FEValidationError("{}: {}".format(what, exc)) from None
        component = next((c for c in components if c["slot_id"] == slot_id), None)
        if component is None:
            if ct["slot_list_version"] < issued_version:
                raise FEKeyError("the ciphertext is older than the key (keys decrypt only ciphertexts made after "
                                 "they were issued): it was made with slot list version {}, the key's slot came "
                                 "with version {}".format(ct["slot_list_version"], issued_version))
            raise FEKeyError("the ciphertext has no component for this slot (slot {}): its encryptor did not "
                             "encrypt to it".format(slot_id))
        if known is not None and not self._check_key(known, scalars, w_packed, setup.m):
            raise FESchemeError(DECRYPTION_FAILED)                       # which position failed: not told
        try:
            out = backend.ops().bcfe_decrypt_slot(
                setup.guarded, setup.data_wires, ct_pp_digest, ct_nonce, slot_id,
                from_hex(component["R"], "R", _POINT), from_b64(component["label_cts"], "label_cts"),
                from_b64(component["tables"], "tables"), from_b64(component["decode"], "decode"),
                from_b64(component["data_active"], "data_active"), scalars, w_packed)
            bits = unpack_bits(out, setup.n_out)
        except (ValueError, TypeError):
            raise FESchemeError(DECRYPTION_FAILED) from None
        return setup.circuit.output_layout.decode(bits)
