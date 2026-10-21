"""
The serialized objects of the threshold schemes: key shares (whole, and the
weight and label parts of the threshold MCFE over Ring-LWE), the pieces of
the replicated sharings, and partial decryptions.

:class:`ThresholdKeyShare` and :class:`PartialDecryption` are the views the
shared threshold code (:mod:`pyfe4ai.schemes.threshold`) reads any scheme's
key shares and partials through; the other classes are the layouts of each
scheme. A key share of the replicated sharings holds ``sharing`` and
``pieces`` (and, seeded, ``seeds`` and ``layout``) in place of the shared
values; one of the Shamir sharing holds the values (``sk_y``, ``z``, ...).
"""

from __future__ import annotations

from pyfe4ai.core.formats.record import Field
from pyfe4ai.core.formats.record import Record
from pyfe4ai.primitives.hashing import LEGACY_LABEL_HASH
from pyfe4ai.core.versions import LEGACY_FLOODING
from pyfe4ai.core.versions import LEGACY_SHARING


# ---------------------------------------------------------------------------
# replicated pieces


class KeyPiece(Record):
    """A piece ``r_T`` of a replicated sharing, as a server holds it."""

    set = Field("the index set ``T`` (sorted server points)")
    share = Field("the piece: the shared material's structure (``sk_y``, ``z``, ...)")


class KeySeed(Record):
    """The seed of a piece of the seed-expanded replicated sharing."""

    set = Field("the index set ``T``")
    seed = Field("32-byte hex seed")


#: The fields of a key share that hold the pieces of a replicated sharing.
PIECE_FIELDS = ("pieces", "seeds", "layout")


class ThresholdKeyShare(Record):
    """Any threshold key share, as the shared server code reads it (the
    fields of every scheme's key share that the checks and the sums of
    :mod:`pyfe4ai.schemes.threshold` use)."""

    sharing = Field("``replicated`` or ``replicated-seeded`` (Shamir key shares have none)",
                    absent=LEGACY_SHARING)
    pieces = Field("replicated: the server's pieces (:class:`KeyPiece`)")
    seeds = Field("seeded: the seeds of the other pieces (:class:`KeySeed`)")
    layout = Field("seeded: the layout of a piece")
    sk_y = Field("the shared key of the weights (LWE, Ring-LWE)")
    z = Field("the shared term of the key (LWE, Ring-LWE: of the label; DDH, FH)")
    z_chunks = Field("tMCFE-Ring-LWE: the shared terms of the chunk labels")
    d = Field("DDH: the shared key")
    k = Field("FH: the shared key in the exponent")
    part = Field("tMCFE-Ring-LWE: ``weights`` or ``label`` (whole key shares have none)", absent=None)
    share_set_id = Field("tMCFE-Ring-LWE: the weight part's sharing")
    sharing_id = Field("tMCFE-Ring-LWE: this sharing (32-char hex)", absent=None)
    label = Field("the label (tMCFE-Ring-LWE)")
    noise_key = Field("hex key of the server's flooding noise")
    sid = Field("the server id (str or int)")
    lst_sid = Field("the server ids")
    weights_id = Field("hex digest of the weights")
    key_id = Field("hex id of the sharing (tMCFE FH)")
    mac_keys = Field("``{client: hex key}`` of the ciphertext tags")
    noise_version = Field("int, the flooding derivation", absent=1)
    noise_sampler = Field("the flooding sampler")
    flooding = Field("``uniform`` or ``gaussian``", absent=LEGACY_FLOODING)
    label_hash = Field("label hash mode", absent=LEGACY_LABEL_HASH)
    pp_binding = Field("hex digest of the public parameters it was issued for")

    def without_pieces(self) -> dict:
        """The key share's other fields (a new dict, in their order)."""
        return {key: value for key, value in self._raw.items() if key not in PIECE_FIELDS}


class TMIFELWEKeyShare(Record):
    """Key share of the threshold MIFE over LWE (:mod:`pyfe4ai.schemes.mife.lwe_threshold`)."""

    sharing = Field("replicated sharings only", absent=LEGACY_SHARING)
    pieces = Field("replicated: :class:`KeyPiece` of ``{sk_y, z}``")
    seeds = Field("seeded: :class:`KeySeed`")
    layout = Field("seeded: the piece layout")
    sk_y = Field("Shamir: ``{client: decimal strings}``")
    noise_key = Field("hex key of the flooding noise")
    z = Field("Shamir: decimal string")
    noise_version = Field("int", absent=1)
    noise_sampler = Field("the flooding sampler")
    sid = Field("the server id (str or int)")
    lst_sid = Field("server ids")
    weights_id = Field("hex digest of the weights")
    mac_keys = Field("``{client: hex key}``")
    pp_binding = Field("hex digest")


class TMCFELWEKeyShare(Record):
    """Key share of the threshold MCFE over LWE (:mod:`pyfe4ai.schemes.mcfe.lwe_threshold`)."""

    sharing = Field("replicated sharings only", absent=LEGACY_SHARING)
    pieces = Field("replicated: :class:`KeyPiece` of ``{sk_y}``")
    seeds = Field("seeded: :class:`KeySeed`")
    layout = Field("seeded: the piece layout")
    sk_y = Field("Shamir: decimal strings")
    sid = Field("the server id (str or int)")
    lst_sid = Field("server ids")
    weights_id = Field("hex digest of the weights")
    noise_key = Field("hex key of the flooding noise")
    noise_version = Field("int", absent=1)
    noise_sampler = Field("the flooding sampler")
    label_hash = Field("label hash mode", absent=LEGACY_LABEL_HASH)
    pp_binding = Field("hex digest")


class TMCFERingLWEKeyShare(Record):
    """Whole key share of the threshold MCFE over Ring-LWE
    (:mod:`pyfe4ai.schemes.mcfe.ring_lwe_threshold`, ``key_part="full"``):
    shares of ``sk_y``, ``z`` and ``z_chunks`` (Shamir; binary polynomials
    under binary public parameters) or the pieces of a replicated sharing."""

    sharing = Field("replicated sharings only", absent=LEGACY_SHARING)
    pieces = Field("replicated: :class:`KeyPiece` of ``{sk_y, z, z_chunks}``")
    seeds = Field("seeded: :class:`KeySeed`")
    layout = Field("seeded: the piece layout")
    sk_y = Field("Shamir: ``{client: polynomial}``")
    z = Field("Shamir: polynomial")
    z_chunks = Field("Shamir: polynomials")
    label = Field("str")
    noise_key = Field("hex key of the flooding noise (per server, weights and label)")
    sharing_id = Field("32-char hex, this sharing", absent=None)
    sid = Field("the server id (str or int)")
    lst_sid = Field("server ids")
    weights_id = Field("hex digest of the weights")
    mac_keys = Field("``{client: hex key}``")
    noise_version = Field("int", absent=1)
    noise_sampler = Field("the flooding sampler")
    flooding = Field("``uniform`` or ``gaussian``", absent=LEGACY_FLOODING)
    label_hash = Field("label hash mode", absent=LEGACY_LABEL_HASH)
    pp_binding = Field("hex digest")


class TMCFERingLWEWeightShare(Record):
    """Weight part of a split key share of the threshold MCFE over Ring-LWE
    (``part = "weights"``; once per weights and share set)."""

    sharing = Field("replicated only", absent=LEGACY_SHARING)
    pieces = Field("replicated: :class:`KeyPiece` of ``{sk_y}``")
    sk_y = Field("Shamir: ``{client: polynomial}``")
    part = Field("``weights``")
    share_set_id = Field("32-char hex, this sharing of ``sk_y``")
    sid = Field("the server id (str or int)")
    lst_sid = Field("server ids")
    weights_id = Field("hex digest of the weights")
    mac_keys = Field("``{client: hex key}``")
    noise_version = Field("int", absent=1)
    noise_sampler = Field("the flooding sampler")
    flooding = Field("``uniform`` or ``gaussian``", absent=LEGACY_FLOODING)
    label_hash = Field("label hash mode", absent=LEGACY_LABEL_HASH)
    pp_binding = Field("hex digest")


class TMCFERingLWELabelShare(Record):
    """Label part of a split key share of the threshold MCFE over Ring-LWE
    (``part = "label"``; per label, completing the weight part of its
    ``share_set_id``)."""

    sharing = Field("replicated only", absent=LEGACY_SHARING)
    pieces = Field("replicated: :class:`KeyPiece` of ``{z, z_chunks}``")
    z = Field("Shamir: polynomial (absent with ``label_term=False``)")
    z_chunks = Field("Shamir: polynomials")
    label = Field("str")
    noise_key = Field("hex key of the flooding noise")
    part = Field("``label``")
    share_set_id = Field("the weight part's sharing")
    label_hash = Field("label hash mode", absent=LEGACY_LABEL_HASH)
    sharing_id = Field("32-char hex, this sharing of the label's terms", absent=None)
    sid = Field("the server id (str or int)")
    lst_sid = Field("server ids")
    weights_id = Field("hex digest of the weights")
    mac_keys = Field("``{client: hex key}``")
    noise_version = Field("int", absent=1)
    noise_sampler = Field("the flooding sampler")
    flooding = Field("``uniform`` or ``gaussian``", absent=LEGACY_FLOODING)
    pp_binding = Field("hex digest")


class TMIFEDDHKeyShare(Record):
    """Key share of the threshold MIFE over DDH (:mod:`pyfe4ai.schemes.mife.ddh_threshold`; Shamir)."""

    d = Field("``{client: decimal string}``")
    z = Field("decimal string")
    sid = Field("the server id (str or int)")
    lst_sid = Field("server ids")
    weights_id = Field("hex digest of the weights")


class TMCFEDDHKeyShare(Record):
    """Key share of the threshold MCFE over DDH (:mod:`pyfe4ai.schemes.mcfe.ddh_threshold`; Shamir)."""

    d = Field("two decimal strings")
    sid = Field("the server id (str or int)")
    lst_sid = Field("server ids")
    weights_id = Field("hex digest of the weights")


class TMCFEFHKeyShare(Record):
    """Key share of the threshold FH-MCFE (:mod:`pyfe4ai.schemes.mcfe.fh_multi_ipe_pairing_threshold`;
    shares in the exponent)."""

    k = Field("``{client: serialised G2 elements}``")
    z = Field("serialised GT element")
    sid = Field("the server id (str or int)")
    lst_sid = Field("server ids")
    key_id = Field("32-char hex, this sharing")
    mac_keys = Field("``{client: hex key}``")


class TaggedCiphertext(Record):
    """Any ciphertext of the threshold schemes whose servers check client tags
    (tMIFE-LWE, tMCFE-Ring-LWE, tMCFE FH; the other fields are the scheme's)."""

    tags = Field("``{server: hex HMAC-SHA256}`` under the client-server keys")


# ---------------------------------------------------------------------------
# partial decryptions


class PartialDecryption(Record):
    """Any partial decryption, as the shared combiner code reads it."""

    w = Field("the server's key part")
    c_y = Field("the weighted ciphertexts")
    u_d = Field("tMCFE DDH: the label term")
    k = Field("tMCFE-Ring-LWE: int, the batch length")
    ct_digest = Field("hex digest of the decrypted ciphertexts")
    key_id = Field("tMCFE FH: the key's sharing")
    label = Field("the encoded label")
    sid = Field("the server id (str or int)")
    lst_sid_enrolled = Field("the servers whose partials are combined")
    sharing_id = Field("tMCFE-Ring-LWE: the key share's sharing", absent=None)


class TMCFERingLWEPartial(Record):
    """Partial decryption of the threshold MCFE over Ring-LWE: ``w`` and
    ``c_y`` as decimal strings (``ring_n`` values, JSON public parameters) or
    bytes (``k`` words, binary ones)."""

    w = Field("values or bytes")
    c_y = Field("values or bytes")
    k = Field("int, the batch length")
    ct_digest = Field("hex digest")
    sid = Field("the server id (str or int)")
    lst_sid_enrolled = Field("server ids")
    sharing_id = Field("the key share's ``sharing_id`` (or null)", absent=None)


class TMCFERingLWELayerPartials(Record):
    """One server's partial decryptions of a packed ndarray layer (one per chunk)."""

    shape = Field("list of ints")
    parts = Field(":class:`TMCFERingLWEPartial`")


class TMIFELWEPartial(Record):
    """Partial decryption of the threshold MIFE over LWE."""

    w = Field("decimal string")
    c_y = Field("decimal string")
    ct_digest = Field("hex digest of the canonical ciphertexts")
    sid = Field("the server id (str or int)")
    lst_sid_enrolled = Field("server ids")


class TMCFELWEPartial(Record):
    """Partial decryption of the threshold MCFE over LWE."""

    w = Field("decimal string")
    c_y = Field("int")
    label = Field("the encoded label")
    sid = Field("the server id (str or int)")
    lst_sid_enrolled = Field("server ids")


class TMIFEDDHPartial(Record):
    """Partial decryption of the threshold MIFE over DDH."""

    c_y = Field("decimal string")
    w = Field("decimal string")
    ct_digest = Field("hex digest of the request")
    sid = Field("the server id (str or int)")
    lst_sid_enrolled = Field("server ids")


class TMCFEDDHPartial(Record):
    """Partial decryption of the threshold MCFE over DDH."""

    c_y = Field("decimal string")
    u_d = Field("decimal string")
    label = Field("the encoded label")
    sid = Field("the server id (str or int)")
    lst_sid_enrolled = Field("server ids")


class TMCFEFHPartial(Record):
    """Partial decryption of the threshold FH-MCFE."""

    w = Field("serialised GT element")
    ct_digest = Field("hex digest")
    key_id = Field("the key share's ``key_id``")
    label = Field("the encoded label")
    sid = Field("the server id (str or int)")
    lst_sid_enrolled = Field("server ids")
