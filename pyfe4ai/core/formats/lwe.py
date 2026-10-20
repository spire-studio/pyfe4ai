"""
The serialized objects of the LWE schemes: the ABDP15 IPFE with a seeded
public matrix (:mod:`pyfe4ai.schemes.sife.lwe`, :mod:`pyfe4ai.schemes.mife.lwe`
and the threshold :mod:`pyfe4ai.schemes.mife.lwe_threshold`), ALS16 FullySec
(:mod:`pyfe4ai.schemes.sife.fullysec_lwe`, :mod:`pyfe4ai.schemes.mife.fullysec_lwe`)
and the ABMPSW20 MCFE (:mod:`pyfe4ai.schemes.mcfe.lwe`, its FullySec,
threshold and decentralised variants; the threshold key shares and partial
decryptions are in :mod:`pyfe4ai.core.formats.threshold`).

Values are decimal strings unless a field says otherwise.
"""

from __future__ import annotations

from pyfe4ai.core.formats.record import Field
from pyfe4ai.core.formats.record import Record
from pyfe4ai.primitives.hashing import LEGACY_LABEL_HASH

# ---------------------------------------------------------------------------
# ABDP15 (seeded A): SIFE, MIFE, threshold MIFE


class ABDP15Record(Record):
    """Parameter record of the ABDP15 schemes (SIFE, MIFE and threshold MIFE
    LWE): the configuration it was derived for, the derived parameters and the
    seed of ``A``."""

    construction = Field("record id (:data:`~pyfe4ai.core.versions.SIFE_LWE_RECORD`, "
                         ":data:`~pyfe4ai.core.versions.MIFE_LWE_RECORD` or a tMIFE-LWE construction id)")
    sec_param = Field("int")
    eta = Field("int")
    lwe_n_config = Field("the config's ``lwe_n`` (int or ``auto``)")
    bound_x = Field("decimal string")
    bound_y = Field("decimal string")
    n = Field("int, the number of slots (MIFE)")
    s = Field("int, the number of servers (tMIFE)")
    stat_param = Field("int (tMIFE)")
    flooding_accounting = Field("``kl`` (tMIFE under that accounting)", absent="per-sample")
    max_partial_decryptions = Field("int (tMIFE, ``kl``)")
    lwe_n = Field("int")
    p = Field("decimal string")
    q = Field("decimal string")
    m = Field("int, the rows of ``A``")
    sigma_q = Field("float")
    l_sigma = Field("decimal string")
    sigma_f = Field("decimal string, the flooding width (tMIFE)")
    A_seed = Field("32-byte hex seed of ``A``")


class SIFELWEParameters(Record):
    """Public parameters of SIFE LWE (ABDP15)."""

    A_seed = Field("hex seed of ``A``")
    m = Field("int")
    lwe_n = Field("int")
    gaussian_sampler = Field("the sampler of the public keys' noise")
    security_profile = Field("``toy`` or ``standard``")
    p = Field("decimal string")
    q = Field("decimal string")
    eta = Field("int")
    bound_x = Field("decimal string")
    bound_y = Field("decimal string")
    sec_param = Field("int")


class MIFELWEParameters(Record):
    """Public parameters of MIFE LWE (ABDP15 slots)."""

    A_seed = Field("hex seed of ``A``")
    m = Field("int")
    lwe_n = Field("int")
    gaussian_sampler = Field("the sampler of the public keys' noise")
    security_profile = Field("``toy`` or ``standard``")
    p = Field("decimal string")
    q = Field("decimal string")
    eta = Field("int")
    n = Field("int, the number of slots")
    bound_x = Field("decimal string")
    bound_y = Field("decimal string")
    sec_param = Field("int")


class TMIFELWEParameters(MIFELWEParameters):
    """Public parameters of the threshold MIFE over LWE."""

    s = Field("int, the number of servers")
    t = Field("int, the threshold")
    lst_sid = Field("the server ids")
    sigma_f = Field("decimal string, the flooding width")
    sharing = Field("the key sharing", absent="shamir")
    threshold_construction = Field("construction id", absent=None)
    flooding_accounting = Field("``kl`` (only under that accounting)", absent="per-sample")
    max_partial_decryptions = Field("int (``kl`` only)")
    stat_param = Field("int (``kl`` only)")
    flood_tail_cut = Field("int (``kl`` only)")


class LWEEncryptionKey(Record):
    """Encryption key of SIFE LWE: the public key ``A S + E``."""

    pk = Field("``m x eta`` matrix")


class MIFELWEEncryptionKey(LWEEncryptionKey):
    """Encryption key of a MIFE LWE slot (the threshold MIFE adds ``mac_keys``)."""

    u = Field("the slot's pad (``eta`` values)")
    mac_keys = Field("tMIFE: ``{server: hex key}`` of the ciphertext tags")


class SIFELWEFunctionalKey(Record):
    """Functional key of SIFE LWE and SIFE FullySec LWE."""

    sk_y = Field("vector")
    y = Field("the weights (ints)")


class MIFELWEFunctionalKey(Record):
    """Functional key of MIFE LWE and MIFE FullySec LWE (shared among the
    servers by the threshold MIFE)."""

    sk_y = Field("``{slot: vector}``")
    z = Field("decimal string, the pad term")


class LWECiphertext(Record):
    """Ciphertext of the ABDP15 and FullySec LWE schemes (``ct0``: the
    ``A``-part, ``ct1``: the ``eta`` slot values; the threshold MIFE adds
    ``tags``)."""

    ct0 = Field("vector")
    ct1 = Field("vector of ``eta`` values")
    tags = Field("tMIFE: ``{server: hex HMAC}``")


# ---------------------------------------------------------------------------
# ALS16 FullySec LWE (public A)


class FullySecLWERecord(Record):
    """Parameter record of SIFE / MIFE FullySec LWE."""

    construction = Field("record id (:data:`~pyfe4ai.core.versions.SIFE_FULLYSEC_LWE_RECORD` or the MIFE one)")
    sec_param = Field("int")
    eta = Field("int")
    n = Field("int, the number of slots (MIFE)")
    lwe_n = Field("int")
    m = Field("int")
    bound_x = Field("decimal string")
    bound_y = Field("decimal string")
    k = Field("decimal string")
    q = Field("decimal string")
    sigma_q = Field("float")
    l_sigma_q = Field("decimal string")
    sigma1 = Field("float")
    l_sigma1 = Field("decimal string")
    sigma2 = Field("float")
    l_sigma2 = Field("decimal string")
    A_seed = Field("32-byte hex seed of ``A``")


class FullySecLWEParameters(Record):
    """Public parameters of SIFE / MIFE FullySec LWE (``A`` expanded)."""

    A = Field("``m x lwe_n`` matrix")
    k = Field("decimal string")
    q = Field("decimal string")
    m = Field("int")
    eta = Field("int")
    n = Field("int, the number of slots (MIFE)")
    lwe_n = Field("int")
    bound_x = Field("decimal string")
    bound_y = Field("decimal string")
    sigma_q = Field("float")
    sigma1 = Field("float")
    sigma2 = Field("float")
    sec_param = Field("int")


# ---------------------------------------------------------------------------
# ABMPSW20 MCFE over LWE (Fig. 4) and its variants


class MCFELWEParameters(Record):
    """Public parameters of the MCFE over LWE (ABMPSW20 Fig. 4; also FullySec)."""

    q = Field("decimal string")
    q0 = Field("int, the rounding modulus")
    K = Field("int")
    delta = Field("decimal string")
    bound_ip = Field("int, the decoding bound")
    n0 = Field("int")
    m0 = Field("int")
    security = Field("``selective`` or ``adaptive``")
    eta = Field("int")
    n = Field("int, the number of clients")
    lwe_n = Field("int")
    bound_x = Field("int")
    bound_y = Field("int")
    stat_param = Field("int")
    sec_param = Field("int")
    gaussian_sampler = Field("the sampler of the keys")
    security_profile = Field("``toy`` or ``standard``")
    label_hash = Field("label hash mode", absent=LEGACY_LABEL_HASH)


class TMCFELWEParameters(MCFELWEParameters):
    """Public parameters of the threshold MCFE over LWE."""

    s = Field("int, the number of servers")
    t = Field("int, the threshold")
    lst_sid = Field("the server ids")
    sigma_f = Field("decimal string, the flooding width")
    sharing = Field("the key sharing", absent="shamir")
    threshold_construction = Field("construction id", absent=None)


class DecentralizedLWEParameters(MCFELWEParameters):
    """Public parameters of the decentralised MCFE over LWE (Fig. 14)."""

    n2_mod = Field("decimal string, the DCR modulus of the key derivation")
    lst_nid = Field("the client ids")
    contribution_bound = Field("decimal string")


class DecentralizedLWERecord(Record):
    """Parameter record of the decentralised MCFE over LWE (the DCR modulus only)."""

    construction = Field("record id (:data:`~pyfe4ai.core.versions.DMCFE_LWE_RECORD`)")
    n2_prime_bits = Field("int")
    n2_mod = Field("decimal string")


class MCFELWEEncryptionKey(Record):
    """Encryption key of an MCFE LWE client (the decentralised MCFE adds ``s2``)."""

    z = Field("per slot, ``n0 + m0`` decimal strings")
    label_hash = Field("label hash mode", absent=LEGACY_LABEL_HASH)
    s2 = Field("dMCFE: the client's key of the key derivation (decimal strings)")


class MCFELWEFunctionalKey(Record):
    """Functional key of the MCFE over LWE (also the combined key of the decentralised variant)."""

    sk_y = Field("``n0 + m0`` decimal strings")


class MCFELWECiphertext(Record):
    """Ciphertext of the MCFE over LWE: one element of ``Z_q0`` per slot (ints)
    and the label hash code (none for ``shake256-wide``; in an ndarray layer
    only its first element carries it)."""

    c = Field("ints")
    label_hash = Field("int code", absent=0)
