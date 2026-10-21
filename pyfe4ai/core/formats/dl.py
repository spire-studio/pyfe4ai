"""
The serialized objects of the discrete-log and DCR schemes: DDH over
``Z_p^*`` (:mod:`pyfe4ai.schemes.sife.ddh`, ``ddh_dynamic``,
:mod:`pyfe4ai.schemes.mife.ddh`, :mod:`pyfe4ai.schemes.mcfe.ddh` and their
threshold and decentralised variants), DDH over ristretto255, Damgard /
MDDH and Paillier (DCR), and the key shares of the decentralised MCFE key
derivation (ABMPSW20 Fig. 14) that the decentralised DDH, DCR and LWE
schemes share. The threshold key shares and partial decryptions are in
:mod:`pyfe4ai.core.formats.threshold`.

Values are decimal strings unless a field says otherwise; points of
ristretto255 are 64-character hex strings.
"""

from __future__ import annotations

from pyfe4ai.core.formats.record import Field
from pyfe4ai.core.formats.record import Record

# ---------------------------------------------------------------------------
# parameter records


class DLGroupRecord(Record):
    """The group of a DDH or Damgard parameter record (``p = r q + 1``)."""

    p = Field("decimal string, the modulus")
    q = Field("decimal string, the subgroup order")
    r = Field("decimal string, the cofactor (DDH records: 2)")


class DDHRecord(Record):
    """Parameter record of the DDH schemes over ``Z_p^*`` (SIFE, MIFE, MCFE
    and their threshold and decentralised variants)."""

    sec_param = Field("int, the bits of ``p``")
    g = Field("decimal string, the generator")
    eta = Field("int (or per client)")
    group = Field(":class:`DLGroupRecord`")
    n = Field("int, the number of clients (MIFE, MCFE)")
    s = Field("the number of servers (MIFE, MCFE; tMCFE)")
    t = Field("int, the threshold (tMCFE)")
    construction = Field("record id (tMCFE :data:`~pyfe4ai.core.versions.TMCFE_DDH_RECORD`, "
                         "dMCFE :data:`~pyfe4ai.core.versions.DMCFE_DDH_RECORD`)")
    n2_prime_bits = Field("int (dMCFE)")
    n2_mod = Field("decimal string, the DCR modulus of the key derivation (dMCFE)")


class DamgardRecord(Record):
    """Parameter record of SIFE and MIFE Damgard."""

    sec_param = Field("int")
    eta = Field("int")
    modulus_length = Field("int")
    bound = Field("int")
    group = Field(":class:`DLGroupRecord` (``p`` and ``q``)")
    g = Field("decimal string")
    h_seed = Field("hex seed of ``h``")
    n = Field("int, the number of slots (MIFE)")


class MCFEDamgardRecord(Record):
    """Parameter record of MCFE Damgard (MDDH)."""

    construction = Field("record id (:data:`~pyfe4ai.core.versions.MCFE_DAMGARD_RECORD`)")
    modulus_length = Field("int")
    p = Field("decimal string")
    q = Field("decimal string")
    g = Field("decimal string")


class SIFEPaillierRecord(Record):
    """Parameter record of SIFE Paillier (the factors are never stored)."""

    sec_param = Field("int")
    eta = Field("int")
    bit_length = Field("int")
    bound_x = Field("int")
    bound_y = Field("int")
    param_format = Field("record format (:data:`~pyfe4ai.core.versions.PAILLIER_RECORD`)")
    n = Field("decimal string, the modulus ``N``")
    n_square = Field("decimal string")
    g_seed = Field("hex seed of ``g``")


class MIFEPaillierRecord(Record):
    """Parameter record of MIFE Paillier."""

    sec_param = Field("int")
    eta = Field("int")
    n = Field("int, the number of slots")
    bit_length = Field("int")
    bound_x = Field("int")
    bound_y = Field("int")
    param_format = Field("record format (:data:`~pyfe4ai.core.versions.PAILLIER_RECORD`)")
    n_mod = Field("decimal string, the modulus ``N``")
    n_square = Field("decimal string")
    g_seed = Field("hex seed of ``g``")


class MCFEPaillierRecord(Record):
    """Parameter record of MCFE Paillier and of the decentralised MCFE over DCR."""

    construction = Field("record id (:data:`~pyfe4ai.core.versions.MCFE_PAILLIER_RECORD` or the dMCFE one)")
    bit_length = Field("int")
    n2_prime_bits = Field("int (dMCFE)")
    n_mod = Field("decimal string, the modulus ``N``")
    n2_mod = Field("decimal string, the DCR modulus of the key derivation (dMCFE)")


# ---------------------------------------------------------------------------
# DDH over Z_p^*


class SIFEDDHParameters(Record):
    """Public parameters of SIFE DDH."""

    g = Field("decimal string")
    p = Field("decimal string")
    eta = Field("int")
    sec_param = Field("int")


class SIFEDDHEncryptionKey(Record):
    """Encryption key of SIFE DDH: ``g^{s_i}``."""

    gs = Field("``eta`` decimal strings")


class SIFEDDHFunctionalKey(Record):
    """Functional key of SIFE DDH (``ddh_dynamic`` adds ``size``)."""

    z = Field("decimal string, ``<s, y> mod q``")
    y = Field("the weights (ints)")
    size = Field("int, the number of weights (``ddh_dynamic``)")


class SIFEDDHCiphertext(Record):
    """Ciphertext of SIFE DDH (also SIFE DDH over ristretto255, with points)."""

    ct0 = Field("``g^r``")
    ct1 = Field("``g^{x_i} gs_i^r``")


class MIFEDDHParameters(Record):
    """Public parameters of MIFE DDH (the threshold MIFE adds ``t``, ``s`` and ``lst_sid``)."""

    construction = Field("public id (:data:`~pyfe4ai.core.versions.MIFE_DDH_CONSTRUCTION`; not checked)")
    g = Field("decimal string")
    p = Field("decimal string")
    q = Field("decimal string")
    n = Field("int")
    eta = Field("int (or per client)")
    lst_nid = Field("the client ids")
    sec_param = Field("int")
    t = Field("int (tMIFE)")
    s = Field("int (tMIFE)")
    lst_sid = Field("the server ids (tMIFE)")


class MIFEDDHEncryptionKey(Record):
    """Encryption key of a MIFE DDH slot (ristretto255: points and scalars)."""

    h = Field("``g^{s_j}``")
    u = Field("the slot's pad")


class MIFEDDHFunctionalKey(Record):
    """Functional key of MIFE DDH (also over ristretto255)."""

    d = Field("``{slot: <s_i, y_i>}``")
    z = Field("the pad term")
    weights_id = Field("hex digest of the weights")


class MIFEDDHCiphertext(Record):
    """Ciphertext of MIFE DDH (also over ristretto255)."""

    t = Field("``g^r``")
    c = Field("the slot values")


class MCFEDDHParameters(Record):
    """Public parameters of MCFE DDH (the decentralised MCFE adds its fields)."""

    g = Field("decimal string")
    p = Field("decimal string")
    n = Field("int")
    eta = Field("int (or per client)")
    sec_param = Field("int")
    n2_mod = Field("decimal string (dMCFE)")
    lst_nid = Field("the client ids (dMCFE)")
    contribution_bound = Field("decimal string (dMCFE)")
    mask_bits = Field("int (dMCFE)")


class TMCFEDDHParameters(Record):
    """Public parameters of the threshold MCFE over DDH."""

    p = Field("decimal string")
    g = Field("decimal string")
    eta = Field("int")
    s = Field("int")
    t = Field("int")
    n = Field("int")
    sec_param = Field("int")


class MCFEDDHEncryptionKey(Record):
    """Encryption key of an MCFE DDH client (also MCFE Damgard, Paillier and
    ristretto255; the decentralised MCFE adds ``s2`` and ``mask_key``)."""

    s = Field("per slot, the client's secret")
    s2 = Field("dMCFE: the client's key of the key derivation")
    mask_key = Field("dMCFE DDH: hex key of the carry masks")


class PreparedKey(Record):
    """A functional key of the element-wise MCFE schemes as
    :meth:`~pyfe4ai.schemes.mcfe.offline.OfflineMCFEMixin.prepare_key`
    returns it: the key's fields, then ``key_terms`` (an in-process object,
    never serialised)."""

    key_terms = Field("the prepared key terms (in process only)")


class MCFEDDHFunctionalKey(PreparedKey):
    """Functional key of MCFE DDH and MCFE Damgard."""

    ORDER = ("d", "key_terms")
    d = Field("decimal strings")


class MCFEPaillierFunctionalKey(PreparedKey):
    """Functional key of MCFE Paillier and of the decentralised MCFE over DCR."""

    ORDER = ("k", "key_terms")
    k = Field("decimal string (over Z)")


class MCFEDDHCiphertext(Record):
    """Ciphertext of the element-wise MCFE schemes (DDH, ristretto255,
    Damgard, Paillier): one value per slot; the threshold MCFE adds the
    encoded label."""

    c = Field("one value per slot")
    label = Field("tMCFE DDH: the encoded label")


# ---------------------------------------------------------------------------
# DDH over ristretto255


class RistrettoParameters(Record):
    """Public parameters of the ristretto255 schemes (MIFE adds its id and clients)."""

    construction = Field("MIFE: public id (not checked)")
    group = Field("``ristretto255``")
    n = Field("int (MIFE, MCFE)")
    eta = Field("int (or per client)")
    lst_nid = Field("the client ids (MIFE)")


class SIFERistrettoEncryptionKey(Record):
    """Encryption key of SIFE DDH over ristretto255."""

    h = Field("``eta`` points")


# ---------------------------------------------------------------------------
# Damgard / MDDH


class SIFEDamgardParameters(Record):
    """Public parameters of SIFE and MIFE Damgard."""

    p = Field("decimal string")
    q = Field("decimal string")
    g = Field("decimal string")
    h = Field("decimal string")
    eta = Field("int")
    n = Field("int (MIFE)")
    bound = Field("int")
    sec_param = Field("int")
    modulus_length = Field("int")


class DamgardEncryptionKey(Record):
    """Encryption key of SIFE / MIFE Damgard (MIFE adds the pad ``u``)."""

    pk = Field("decimal strings")
    u = Field("MIFE: the slot's pad")


class SIFEDamgardFunctionalKey(Record):
    """Functional key of SIFE Damgard."""

    k1 = Field("decimal string")
    k2 = Field("decimal string")
    y = Field("the weights (ints)")


class MIFEDamgardFunctionalKey(Record):
    """Functional key of MIFE Damgard."""

    k1 = Field("``{slot: decimal string}``")
    k2 = Field("``{slot: decimal string}``")
    z = Field("decimal string, the pad term")


class DamgardCiphertext(Record):
    """Ciphertext of SIFE / MIFE Damgard."""

    c = Field("``g^r``")
    d = Field("``h^r``")
    e = Field("the slot values")


class MCFEDamgardParameters(Record):
    """Public parameters of MCFE Damgard (MDDH)."""

    p = Field("decimal string")
    q = Field("decimal string")
    g = Field("decimal string")
    mddh_k = Field("int")
    eta = Field("int")
    n = Field("int")
    bound = Field("int")
    sec_param = Field("int")
    modulus_length = Field("int")


# ---------------------------------------------------------------------------
# Paillier (DCR)


class SIFEPaillierParameters(Record):
    """Public parameters of SIFE Paillier."""

    n = Field("decimal string, the modulus ``N``")
    n_square = Field("decimal string")
    g = Field("decimal string")
    eta = Field("int")
    sec_param = Field("int")
    bit_length = Field("int")
    bound_x = Field("int")
    bound_y = Field("int")


class MIFEPaillierParameters(Record):
    """Public parameters of MIFE Paillier."""

    n_mod = Field("decimal string, the modulus ``N``")
    n_square = Field("decimal string")
    g = Field("decimal string")
    eta = Field("int")
    n = Field("int, the number of slots")
    sec_param = Field("int")
    bit_length = Field("int")
    bound_x = Field("int")
    bound_y = Field("int")


class MCFEPaillierParameters(Record):
    """Public parameters of MCFE Paillier (the decentralised MCFE adds its fields)."""

    n_mod = Field("decimal string, the modulus ``N``")
    n_square = Field("decimal string")
    eta = Field("int")
    n = Field("int")
    sec_param = Field("int")
    bit_length = Field("int")
    bound_x = Field("int")
    bound_y = Field("int")
    n2_mod = Field("decimal string (dMCFE)")
    lst_nid = Field("the client ids (dMCFE)")
    contribution_bound = Field("decimal string (dMCFE)")


class PaillierEncryptionKey(Record):
    """Encryption key of SIFE / MIFE Paillier (MIFE adds the pad ``u``)."""

    pk = Field("decimal strings in ``Z_{N^2}``")
    u = Field("MIFE: the slot's pad")


class SIFEPaillierFunctionalKey(Record):
    """Functional key of SIFE Paillier."""

    k = Field("decimal string, ``<s, y>`` over Z")


class MIFEPaillierFunctionalKey(Record):
    """Functional key of MIFE Paillier."""

    k = Field("``{slot: decimal string}`` over Z")
    z = Field("decimal string, the pad term mod N")


class PaillierCiphertext(Record):
    """Ciphertext of SIFE / MIFE Paillier."""

    ct0 = Field("``g^r``")
    ct1 = Field("the slot values")


# ---------------------------------------------------------------------------
# decentralised key derivation (ABMPSW20 Fig. 14)


class MCFE2KeyShare(Record):
    """A client's share of a functional key of the decentralised MCFE
    (:mod:`pyfe4ai.schemes.mcfe.decentralize`): its contribution encrypted under
    the MCFE2 key."""

    c = Field("decimal strings, the MCFE2 ciphertexts of the contribution")
    nid = Field("the client id (str or int)")
    weights_id = Field("hex digest of the weights")
