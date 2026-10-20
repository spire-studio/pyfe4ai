"""
The serialized objects of the Ring-LWE schemes: SIFE (:mod:`pyfe4ai.schemes.sife.ring_lwe`),
SIFE FullySec (:mod:`pyfe4ai.schemes.sife.fullysec_ring_lwe`), and the
MIFE / MCFE family over BMMS21 (:mod:`pyfe4ai.schemes.mife.ring_lwe`,
:mod:`pyfe4ai.schemes.mcfe.ring_lwe`, its decentralised and threshold
variants; the threshold key shares and partial decryptions are in
:mod:`pyfe4ai.core.formats.threshold`).

Values are decimal strings unless a field says otherwise; a polynomial is a
list of ``ring_n`` values, or (binary formats) bytes of
``ceil(bits(q) / 8)``-byte words (:mod:`pyfe4ai.core.formats.binary`).
"""

from __future__ import annotations

from pyfe4ai.core.formats.record import Field
from pyfe4ai.core.formats.record import Record
from pyfe4ai.primitives.hashing import LEGACY_LABEL_HASH

# ---------------------------------------------------------------------------
# SIFE Ring-LWE (BMMS21 Sec. 4.2)


class SIFERingLWERecord(Record):
    """Parameter record of SIFE Ring-LWE (``SIFE_RING_LWE/param.json``)."""

    param_version = Field("record id (:data:`~pyfe4ai.core.versions.SIFE_RING_LWE_RECORD`)")
    sec_param = Field("int")
    eta = Field("int")
    ring_n = Field("int")
    bound_x = Field("decimal string")
    bound_y = Field("decimal string")
    p = Field("decimal string")
    q = Field("decimal string")
    sigma1 = Field("float")
    sigma2 = Field("float")
    sigma3 = Field("float")
    A_seed = Field("hex seed of ``a``")


class SIFERingLWEParameters(Record):
    """Public parameters of SIFE Ring-LWE."""

    A = Field("the polynomial ``a`` (expanded)")
    p = Field("decimal string")
    q = Field("decimal string")
    eta = Field("int")
    ring_n = Field("int")
    bound_x = Field("decimal string")
    bound_y = Field("decimal string")
    sigma1 = Field("float")
    sigma2 = Field("float")
    sigma3 = Field("float")
    sec_param = Field("int")
    gaussian_sampler = Field("``facct`` or ``rounded``")
    security_profile = Field("``toy`` or ``standard``")


class SIFERingLWEEncryptionKey(Record):
    """Encryption key of SIFE Ring-LWE."""

    pk = Field("``eta`` polynomials")


class SIFERingLWEFunctionalKey(Record):
    """Functional key of SIFE Ring-LWE."""

    sk_y = Field("polynomial")
    y = Field("the weights (ints)")


class SIFERingLWECiphertext(Record):
    """Ciphertext of SIFE Ring-LWE (``ct0``: the ``eta`` slot polynomials
    ``pk_i r + f_i + Delta x_i``; ``ct1``: ``a r + e``)."""

    ct0 = Field("``eta`` polynomials")
    ct1 = Field("polynomial")
    k = Field("int, the batch length (columns of the plaintext)")


# ---------------------------------------------------------------------------
# SIFE FullySec Ring-LWE (BMMS21 Sec. 5)


class FullySecRingLWERecord(Record):
    """Parameter record of SIFE FullySec Ring-LWE (``SIFE_FULLYSEC_RING_LWE/param.json``)."""

    param_version = Field(
        "record id (:data:`~pyfe4ai.core.versions.SIFE_FULLYSEC_RING_LWE_RECORD`, or the packed one)")
    sec_param = Field("int")
    eta = Field("int")
    bound_x = Field("decimal string")
    bound_y = Field("decimal string")
    ring_n_config = Field("the config's ``ring_n`` (int, ``auto`` or null)")
    m_config = Field("the config's ``m``")
    profile = Field("security profile")
    ring_n = Field("int")
    m = Field("int")
    p = Field("decimal string")
    q = Field("decimal string")
    a_seed = Field("hex seed of ``a``")
    sigma = Field("float")
    sigma_lhl = Field("float")
    sigma1 = Field("float")
    sigma2 = Field("float")
    sigma3 = Field("float")
    ndarray_layout = Field("``packed`` (only for that layout)", absent="elementwise")


class FullySecRingLWEParameters(Record):
    """Public parameters of SIFE FullySec Ring-LWE."""

    a = Field("``m`` polynomials")
    p = Field("decimal string")
    q = Field("decimal string")
    ring_n = Field("int")
    m = Field("int")
    eta = Field("int")
    bound_x = Field("decimal string")
    bound_y = Field("decimal string")
    sigma2 = Field("float")
    sigma3 = Field("float")
    sec_param = Field("int")
    gaussian_sampler = Field("``facct`` or ``rounded``")
    security_profile = Field("``toy`` or ``standard``")
    ndarray_layout = Field("``packed`` (only for that layout)", absent="elementwise")


class FullySecRingLWEEncryptionKey(Record):
    """Encryption key of SIFE FullySec Ring-LWE."""

    pk = Field("``eta`` polynomials")


class FullySecRingLWEFunctionalKey(Record):
    """Functional key of SIFE FullySec Ring-LWE."""

    sk_y = Field("``m`` polynomials")
    y = Field("the weights (ints)")


class FullySecRingLWECiphertext(Record):
    """Ciphertext of SIFE FullySec Ring-LWE (``ct0``: ``a r + f0``; ``ct``:
    the ``eta`` slots; ``k``: the batch length of a packed ciphertext)."""

    ct0 = Field("``m`` polynomials")
    ct = Field("``eta`` polynomials")
    k = Field("int (packed layout only)")


class FullySecRingLWELayer(Record):
    """One packed ndarray layer of SIFE FullySec Ring-LWE (``ndarray_layout =
    "packed"``): chunk ``j`` holds elements ``j ring_n`` onward."""

    shape = Field("list of ints")
    first_chunk = Field("int, the chunks before this layer")
    chunks = Field("packed ciphertexts (:class:`FullySecRingLWECiphertext` with ``k``)")


# ---------------------------------------------------------------------------
# MIFE / MCFE Ring-LWE (ACFGU18 over BMMS21) and their variants


class RingLWERecord(Record):
    """Parameter record of the MIFE / MCFE Ring-LWE family (``param.json``;
    ``param-rns.json`` for an RNS modulus). The threshold MCFE adds the
    fields of :class:`ThresholdRingLWERecord`."""

    construction = Field("record id (:data:`~pyfe4ai.core.versions.RING_LWE_RECORD`)")
    sec_param = Field("int")
    eta = Field("int")
    n = Field("int, the number of slots")
    ring_n = Field("int")
    bound_x = Field("decimal string")
    bound_y = Field("decimal string")
    p = Field("decimal string")
    q = Field("decimal string (the threshold MCFE's flooded modulus)")
    sigma1 = Field("float")
    sigma2 = Field("float")
    sigma3 = Field("float")
    A_seed = Field("hex seed of ``a``")
    ring_modulus = Field("``rns`` (only for that modulus)", absent="prime")
    q_factors = Field("the factors of an RNS ``q`` (decimal strings)", absent=None)


class ThresholdRingLWERecord(RingLWERecord):
    """Parameter record of the threshold MCFE over Ring-LWE."""

    threshold_construction = Field("construction id (:data:`~pyfe4ai.core.versions.TMCFE_RING_LWE_CONSTRUCTION_IDS`)")
    s = Field("int, the number of servers")
    stat_param = Field("int")
    flooding_accounting = Field("``kl`` (only under that accounting)", absent="per-sample")
    max_partial_decryptions = Field("int (``kl`` only)", absent=None)
    flooding = Field("``uniform`` (uniform flooding only)", absent="gaussian")
    flood_bits = Field("int, the uniform flooding's width")
    sigma_f = Field("decimal string, the Gaussian flooding's width")


class MIFERingLWEParameters(Record):
    """Public parameters of MIFE Ring-LWE."""

    A = Field("the polynomial ``a`` (expanded)")
    p = Field("decimal string")
    q = Field("decimal string")
    eta = Field("int")
    n = Field("int, the number of slots")
    ring_n = Field("int")
    bound_x = Field("decimal string")
    bound_y = Field("decimal string")
    sigma1 = Field("float")
    sigma2 = Field("float")
    sigma3 = Field("float")
    sec_param = Field("int")
    gaussian_sampler = Field("``facct`` or ``rounded``")
    security_profile = Field("``toy`` or ``standard``")
    q_factors = Field("the factors of an RNS ``q`` (decimal strings; RNS only)", absent=None)


class MCFERingLWEParameters(MIFERingLWEParameters):
    """Public parameters of MCFE Ring-LWE."""

    label_hash = Field("label hash mode (:data:`~pyfe4ai.primitives.hashing.LABEL_HASHES`)",
                       absent=LEGACY_LABEL_HASH)


class DecentralizedRingLWEParameters(MCFERingLWEParameters):
    """Public parameters of the decentralised MCFE over Ring-LWE."""

    lst_nid = Field("the client ids")


class ThresholdRingLWEParameters(MCFERingLWEParameters):
    """Public parameters of the threshold MCFE over Ring-LWE."""

    s = Field("int, the number of servers")
    t = Field("int, the threshold")
    lst_sid = Field("the server ids")
    flooding = Field("``uniform`` or ``gaussian``", absent="gaussian")
    sharing = Field("``shamir``, ``replicated`` or ``replicated-seeded``", absent="shamir")
    threshold_construction = Field("construction id", absent=None)
    flood_bits = Field("int, the uniform flooding's width")
    sigma_f = Field("decimal string, the Gaussian flooding's width")
    ct_format = Field("``binary`` (only for that format)", absent="json")
    flooding_accounting = Field("``kl`` (only under that accounting)", absent="per-sample")
    max_partial_decryptions = Field("int (``kl`` only)")
    stat_param = Field("int (``kl`` only)")
    flood_tail_cut = Field("int (``kl`` only)")


class MIFERingLWEEncryptionKey(Record):
    """Encryption key of a MIFE Ring-LWE slot."""

    pk = Field("``eta`` polynomials")
    u = Field("``eta`` polynomials, the slot's pad")


class MCFERingLWEEncryptionKey(MIFERingLWEEncryptionKey):
    """Encryption key of an MCFE Ring-LWE client."""

    u_prime = Field("hex secret of the label pads")
    label_hash = Field("label hash mode", absent=LEGACY_LABEL_HASH)


class DecentralizedRingLWEEncryptionKey(MCFERingLWEEncryptionKey):
    """Encryption key of a client of the decentralised MCFE over Ring-LWE."""

    s = Field("``eta`` polynomials, the client's IPFE secret")
    pair_keys = Field("``{other client: hex key}``")


class ThresholdRingLWEEncryptionKey(MCFERingLWEEncryptionKey):
    """Encryption key of a client of the threshold MCFE over Ring-LWE."""

    mac_keys = Field("``{server: hex key}`` of the ciphertext tags")


class MIFERingLWEFunctionalKey(Record):
    """Functional key of MIFE Ring-LWE."""

    sk_y = Field("``{client: polynomial}``")
    z = Field("polynomial")


class MCFERingLWEFunctionalKey(Record):
    """Functional key of MCFE Ring-LWE, bound to a label (also the combined
    key of the decentralised variant)."""

    sk_y = Field("``{client: polynomial}``")
    z = Field("polynomial, the term of the label itself")
    z_chunks = Field("polynomials, the terms of the chunk labels")
    label = Field("str, the label")
    label_hash = Field("label hash mode", absent=LEGACY_LABEL_HASH)


class DecentralizedRingLWEKeyShare(Record):
    """A client's share of a functional key of the decentralised MCFE over
    Ring-LWE (:meth:`~pyfe4ai.schemes.mcfe.ring_lwe_decentralized.DecentralizedMCFERingLWE.derive_function_decryption_key_share`)."""

    sk_y = Field("polynomial, the client's slot")
    z = Field("polynomial")
    z_chunks = Field("polynomials")
    label = Field("str")
    nid = Field("str, the client")
    weights_id = Field("hex digest of the weights")
    label_hash = Field("label hash mode", absent=LEGACY_LABEL_HASH)


class RingLWECiphertext(Record):
    """Ciphertext of the MIFE / MCFE Ring-LWE family: JSON (``ct0``, ``ct1``)
    or binary (``ct``: format 1, or format 2 under binary threshold public
    parameters), the batch length, the label hash code (MCFE, none for
    ``shake256-wide``) and the threshold MCFE's tags."""

    ct0 = Field("``eta`` polynomials (JSON)")
    ct1 = Field("polynomial (JSON)")
    ct = Field("bytes (binary formats)")
    k = Field("int, the batch length")
    label_hash = Field("int code (:data:`~pyfe4ai.primitives.hashing.LABEL_HASH_CODES`)", absent=0)
    tags = Field("``{server: hex HMAC}`` (threshold MCFE)")


class RingLWELayer(Record):
    """One layer of packed ndarray ciphertexts (MCFE Ring-LWE; SIFE FullySec
    Ring-LWE's packed layer has no ``label_hash``): chunk ``j`` holds
    elements ``j ring_n`` onward, under the chunk label ``first_chunk + j``."""

    shape = Field("list of ints")
    first_chunk = Field("int, the chunks before this layer")
    chunks = Field("ciphertexts (:class:`RingLWECiphertext`)")
    label_hash = Field("int code, the layer's", absent=0)
