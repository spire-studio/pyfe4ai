"""
The serialized objects of the pairing schemes: function-hiding IPFE
(:mod:`pyfe4ai.schemes.sife.fh_ipe_pairing`), partially function-hiding IPFE
(:mod:`pyfe4ai.schemes.sife.part_fh_ipe_pairing`), the function-hiding MIFE
(:mod:`pyfe4ai.schemes.mife.fh_ipe_pairing`,
:mod:`pyfe4ai.schemes.mife.fh_multi_ipe_pairing`), the function-hiding MCFE
(:mod:`pyfe4ai.schemes.mcfe.fh_multi_ipe_pairing` and its threshold and
decentralised variants; the threshold key shares and partial decryptions are
in :mod:`pyfe4ai.core.formats.threshold`) and the quadratic schemes
(:mod:`pyfe4ai.schemes.quadratic`).

Group elements are serialised strings (``group.serialize(e).decode("utf-8")``);
scalars are decimal strings unless a field says otherwise.
"""

from __future__ import annotations

from pyfe4ai.core.formats.record import Field
from pyfe4ai.core.formats.record import Record

# ---------------------------------------------------------------------------
# SIFE function-hiding IPFE (KKLW-style, dual bases)


class SIFEFHIPERecord(Record):
    """Parameter record of SIFE FH-IPE (the configuration only; no group elements)."""

    sec_param = Field("int")
    eta = Field("int")
    bound_x = Field("decimal string")
    bound_y = Field("decimal string")
    pairing_group_param = Field("the pairing group")


class SIFEFHIPEParameters(Record):
    """Public parameters of SIFE FH-IPE."""

    eta = Field("int")
    bound_x = Field("decimal string")
    bound_y = Field("decimal string")
    pairing_group_param = Field("the pairing group")


class SIFEFHIPEEncryptionKey(Record):
    """Encryption key of SIFE FH-IPE (equivalent to the master secret key)."""

    g2 = Field("G2")
    b_star = Field("``(eta + 2) x (eta + 2)`` matrix of decimal strings")


class SIFEFHIPEFunctionalKey(Record):
    """Functional key of SIFE FH-IPE (no weights: function hiding)."""

    k1 = Field("G1")
    k2 = Field("``eta + 2`` G1 elements")


class SIFEFHIPECiphertext(Record):
    """Ciphertext of SIFE FH-IPE."""

    c1 = Field("G2")
    c2 = Field("``eta + 2`` G2 elements")


# ---------------------------------------------------------------------------
# SIFE partially function-hiding IPFE (Gay20)


class PartFHIPERecord(Record):
    """Parameter record of SIFE partially FH-IPE (the configuration only)."""

    sec_param = Field("int")
    eta = Field("int")
    bound = Field("decimal string")
    subspace_dim = Field("int, ``k``")
    coeff_bound = Field("decimal string")
    pairing_group_param = Field("the pairing group")


class PartFHIPEParameters(Record):
    """Public parameters of SIFE partially FH-IPE (the public key included)."""

    eta = Field("int")
    bound = Field("decimal string")
    subspace_dim = Field("int, ``k``")
    coeff_bound = Field("decimal string")
    pairing_group_param = Field("the pairing group")
    g1 = Field("G1")
    g2 = Field("G2")
    a = Field("two G1 elements")
    ua = Field("``eta + 2`` G1 elements")
    vt_m = Field("``2 x k`` matrix of decimal strings")
    m = Field("``eta x k`` matrix of decimal strings")


class PartFHIPEFunctionalKey(Record):
    """Functional key of SIFE partially FH-IPE."""

    k = Field("``eta + 4`` G2 elements")


class PartFHIPECiphertext(Record):
    """Ciphertext of SIFE partially FH-IPE (``Enc`` and ``Enc'``)."""

    ct = Field("``eta + 4`` G1 elements")


# ---------------------------------------------------------------------------
# function-hiding MIFE (ACFGU18 Fig. 12) and MCFE (AGT21b, NPS24, threshold)


class FHParameters(Record):
    """Public parameters of the function-hiding MIFE (ACFGU18) and MCFE (AGT21b)."""

    pairing_group_param = Field("the pairing group")
    g1 = Field("G1")
    g2 = Field("G2")
    eta = Field("int")
    n = Field("int")
    lst_nid = Field("the client ids")
    bound_x = Field("int")
    bound_y = Field("int")
    sec_param = Field("int")


class TMCFEFHParameters(FHParameters):
    """Public parameters of the threshold FH-MCFE."""

    s = Field("int")
    t = Field("int")
    lst_sid = Field("the server ids")


class FHMasterKey(Record):
    """A slot's FH-IPFE master key as a client holds it (``imsk``; integers as
    decimal strings). Clients of the MIFE and MCFE get it without ``a_out``,
    which only key generation needs."""

    a_in = Field("two decimal strings")
    u = Field("rows of two decimal strings")
    w = Field("rows of two decimal strings")
    z = Field("decimal string")
    a_out = Field("two decimal strings (decentralised MCFE only)")


class FHClientKey(Record):
    """Encryption key of a client of the function-hiding MIFE / MCFE."""

    imsk = Field(":class:`FHMasterKey`")
    mac_keys = Field("tMCFE: ``{server: hex key}`` of the ciphertext tags")
    t_tilde = Field("dMCFE: the client's share of zero (decimal string)")
    seed = Field("dMCFE: 64-char hex seed of the key derivation")


class MIFEFHIPEFunctionalKey(Record):
    """Functional key of the function-hiding MIFE (ACFGU18 Fig. 12)."""

    dk = Field("``{slot: FH-IPFE key (G2 elements)}``")
    z = Field("GT, ``[r sum z_i]_T``")
    key_id = Field("32-char hex, random")


class FHKeyEntry(Record):
    """A client's part of a functional key of the FH-MCFE (also a client's
    key share of the decentralised FH-MCFE, which adds ``nid`` and ``tag_f``)."""

    k = Field("G2 elements")
    z = Field("GT")
    nid = Field("dMCFE: the client")
    tag_f = Field("dMCFE: the function tag")


class MCFEFHFunctionalKey(Record):
    """Functional key of the FH-MCFE (the decentralised MCFE's combined key has
    ``tag_f`` and no ``key_id``)."""

    dk = Field("``{client: FHKeyEntry}``")
    key_id = Field("32-char hex, random")
    tag_f = Field("dMCFE: the function tag")


class FHCiphertext(Record):
    """Ciphertext of the function-hiding MIFE / MCFE schemes (the threshold
    FH-MCFE adds the encoded label and the server tags)."""

    ct = Field("G1 elements")
    label = Field("tMCFE: the encoded label")
    tags = Field("tMCFE: ``{server: hex HMAC}``")


class MIFEFHMultiIPERecord(Record):
    """Parameter record of the full-hiding MIFE (DOT18; the configuration only)."""

    sec_param = Field("int")
    sec_level = Field("int")
    eta = Field("int")
    n = Field("int")
    bound_x = Field("decimal string")
    bound_y = Field("decimal string")
    pairing_group_param = Field("the pairing group")


class MIFEFHMultiIPEParameters(Record):
    """Public parameters of the full-hiding MIFE (DOT18)."""

    sec_level = Field("int")
    eta = Field("int")
    n = Field("int")
    lst_nid = Field("the client ids")
    bound_x = Field("decimal string")
    bound_y = Field("decimal string")
    pairing_group_param = Field("the pairing group")
    g1 = Field("G1")
    g2 = Field("G2")
    pub = Field("GT")


class MIFEFHMultiIPEEncryptionKey(Record):
    """Encryption key of a slot of the full-hiding MIFE."""

    bhat = Field("``(eta + k + 1) x dim`` matrix of decimal strings")


class MIFEFHMultiIPEFunctionalKey(Record):
    """Functional key of the full-hiding MIFE."""

    dk = Field("``{slot: G2 elements}``")


# ---------------------------------------------------------------------------
# quadratic FE


class QuadIPFEPublicKey(Record):
    """The public key of the IPFE inside Quad (Gay20 Fig. 4)."""

    a = Field("two G1 elements")
    ua = Field("G1 elements")
    vt_m = Field("rows of G1 elements")
    m = Field("rows of G1 elements")


class QuadParameters(Record):
    """Public parameters (the public key) of Quad."""

    pairing_group_param = Field("the pairing group")
    n = Field("int")
    m = Field("int")
    bound = Field("int")
    sec_param = Field("int")
    g1 = Field("G1")
    g2 = Field("G2")
    ua = Field("``n`` G1 elements")
    vb = Field("``m x 2`` G2 elements")
    ipfe = Field(":class:`QuadIPFEPublicKey`")


class QuadFunctionalKey(Record):
    """Functional key of Quad."""

    f = Field("``n x m`` matrix of decimal strings")
    k = Field("``2 m + 3 n + 4`` G2 elements")


class QuadCiphertext(Record):
    """Ciphertext of Quad."""

    ct_x = Field("``n`` G1 elements")
    ct_y = Field("``m`` G2 elements")
    ct = Field("``2 m + 3 n + 4`` G1 elements")


class SGPRecord(Record):
    """Parameter record of SGP (DSGPP18; the configuration only)."""

    sec_param = Field("int")
    n = Field("int")
    bound = Field("decimal string")
    pairing_group_param = Field("the pairing group")


class SGPParameters(Record):
    """Public parameters (the public key) of SGP."""

    n = Field("int")
    bound = Field("decimal string")
    pairing_group_param = Field("the pairing group")
    g1 = Field("G1")
    g2 = Field("G2")
    g1_s = Field("``n`` G1 elements")
    g2_t = Field("``n`` G2 elements")


class SGPFunctionalKey(Record):
    """Functional key of SGP."""

    key = Field("G2")
    f = Field("``n x n`` matrix of decimal strings")


class SGPCiphertext(Record):
    """Ciphertext of SGP."""

    c0 = Field("G1")
    a = Field("``n x 2`` G1 elements")
    b = Field("``n x 2`` G2 elements")


class MIQFEParameters(Record):
    """Public parameters of the multi-input quadratic FE (AGT22)."""

    pairing_group_param = Field("the pairing group")
    g1 = Field("G1")
    g2 = Field("G2")
    n = Field("int, the number of users")
    m = Field("int, each user's input length")
    mddh_k = Field("int")
    lst_nid = Field("the user ids")
    bound_x = Field("int")
    bound_c = Field("int")
    sec_param = Field("int")


class MIQFECiphertext(Record):
    """A master ciphertext ``MstEnc`` of the multi-input quadratic FE, and a
    user's ciphertext (their combination, the same layout)."""

    ct = Field("``m`` rows of G1 elements")
    sk = Field("``m`` rows of G2 elements")
    ct2 = Field("G1 elements")
    sk2 = Field("G2 elements")
    mg_ct = Field("G1 elements")
    mg_sk = Field("G2 elements")


class MIQFEEncryptionKey(Record):
    """Encryption key ``EK_i`` of a user: master ciphertexts only."""

    mct1 = Field("``m`` :class:`MIQFECiphertext`")
    mct0 = Field("``4 m + 2 k + 17`` :class:`MIQFECiphertext`")


class MIQFEFunctionalKey(Record):
    """Functional key of the multi-input quadratic FE."""

    c = Field("``nm x nm`` matrix of decimal strings")
    mg_sk = Field("``n`` rows of G2 elements")
    mg_ct = Field("``n`` rows of G1 elements")
    sigma = Field("``n x n`` matrix of decimal strings")
