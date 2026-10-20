"""
Function-hiding multi-input inner-product FE over pairings.

| Abdalla, Catalano, Fiore, Gay, Ursu, "Multi-Input Functional Encryption for
| Inner Products: Function-Hiding Realizations and Constructions without
| Pairings", CRYPTO 2018 (ePrint 2017/972) -- Fig. 12 over the AGRW17 MIFE
| of Fig. 11, with ``k = 1`` (SXDH) and the LV16 padding of Appendix B.

* type:     private-key multi-input encryption with function-hiding keys
* setting:  asymmetric pairing group (default BLS12-381)

Slot ``i`` has the master key ``msk_i`` of a function-hiding IPFE
(:class:`pyfe4ai.schemes.mcfe.fh_ipe.FHIPE`, which contains the inner
AGRW17 secret ``z_i``):

* Enc:      ``ct_i = FHIPE.Enc(msk_i, [x_i]_1)``.
* KeyGen:   one ``r`` for all slots; ``dk_i = FHIPE.KeyGen(msk_i, [y_i]_2; r)``
  and ``[z]_T = [r sum_i z_i]_T``.
* Dec:      ``prod_i FHIPE.Dec(ct_i, dk_i) / [z]_T = [sum_i <x_i, y_i>]_T``,
  then one bounded discrete log.

A slot's partial result is masked by ``r z_i``, so the key holder learns the
sum only, and slot keys of different functional keys (different ``r``) do
not combine. Keys carry no digest of ``y`` and decryption does not need it.
A slot's client holds that slot's outer key ``W_i`` (needed to encrypt) and
could strip the outer layer of its own slot of a functional key -- outside
the model, where clients are not corrupted. As in every MIFE there are no labels: a key holder may combine
any ciphertexts of each slot. Versions <= 0.2.0 ran independent KLMMRW
instances per slot and decoded every slot separately.
"""

from __future__ import annotations

import logging
import secrets

import gmpy2 as gp

from pyfe4ai.core.base import IPFEAbsCrypto
from pyfe4ai.core.base import IPFEAbsKeyGenerator
from pyfe4ai.schemes.mcfe.fh_ipe import FHIPE
from pyfe4ai.schemes.mcfe.fh_multi_ipe_pairing import FHContext
from pyfe4ai.core.constants import CryptoCONST
from pyfe4ai.primitives.pairing.pairing_backend import G1, G2, GT, ZR, PairingGroup, deserialize_many, pow_many
from pyfe4ai.primitives.pairing.pairing_backend import DEFAULT_PAIRING_GROUP, check_asymmetric
from pyfe4ai.primitives.pairing.pairing_backend import require_pairing_backend
from pyfe4ai.primitives.pairing.pairing_utils import bounded_discrete_log_gt
from pyfe4ai.core.exceptions import FEKeyError, FESchemeError, FEValidationError
from pyfe4ai._validation import int_vector

logger = logging.getLogger(__name__)


class _Group:
    """Deserialised group, generators and per-slot FH-IPFE of the public parameters."""

    def __init__(self, pp: dict) -> None:
        require_pairing_backend()
        self.group = PairingGroup(pp["pairing_group_param"])
        check_asymmetric(self.group)
        self.g1 = self.group.deserialize(pp["g1"].encode("utf-8"))
        self.g2 = self.group.deserialize(pp["g2"].encode("utf-8"))
        self.fhipe = FHIPE(self.group, self.g1, self.g2, pp["eta"])

    def zr(self, value):
        return self.group.init(ZR, int(value) % self.fhipe.order)

    def ser(self, element) -> str:
        return self.group.serialize(element).decode("utf-8")

    def de(self, text: str):
        return self.group.deserialize(text.encode("utf-8"))

    def de_many(self, texts: list) -> list:
        return deserialize_many(self.group, texts)


class MIFEFHIPEKeyGenerator(IPFEAbsKeyGenerator):
    """Key generator for the ACFGU18 function-hiding MIFE."""

    _scheme_type = CryptoCONST.TYPE_MIFE_FH_IPE

    def __init__(self, config: dict, **kwargs) -> None:
        """Initialise the key generator.

        Args:
            config: ``n``, ``lst_nid``, ``eta``, ``bound_x``, ``bound_y`` and
                ``pairing_group_param`` (asymmetric, default BLS12-381).
        """
        super().__init__(config, **kwargs)
        require_pairing_backend()
        self.eta = int(config.get("eta", CryptoCONST.MIFE_ETA))
        self.bound_x = int(config.get("bound_x", 16))
        self.bound_y = int(config.get("bound_y", 16))
        self.pairing_group_param = config.get("pairing_group_param", DEFAULT_PAIRING_GROUP)
        check_asymmetric(self.pairing_group_param)
        if not self.n:
            raise FEKeyError("need to provide number of clients")

    def setup(self) -> None:
        """Sample the generators and one FH-IPFE master key per slot."""
        group = PairingGroup(self.pairing_group_param)
        self.pp = {
            "pairing_group_param": self.pairing_group_param,
            "g1": group.serialize(group.random(G1)).decode("utf-8"),
            "g2": group.serialize(group.random(G2)).decode("utf-8"),
            "eta": self.eta,
            "n": self.n,
            "lst_nid": list(self.lst_nid),
            "bound_x": self.bound_x,
            "bound_y": self.bound_y,
            "sec_param": self.sec_param,
        }
        self.ctx = _Group(self.pp)
        self.msk = {nid: self.ctx.fhipe.setup() for nid in self.lst_nid}
        logger.info("FH-MIFE (ACFGU18 Fig. 12) setup successfully")

    def get_public_parameters(self) -> dict:
        """Group description, generators, dimensions and bounds."""
        return dict(self.pp)

    def get_private_keys(self, nid: str, **kwargs) -> dict | None:
        """Return slot *nid*'s FH-IPFE master key ``{imsk}`` (``None`` if unknown)."""
        if nid not in self.msk:
            return None
        # the outer A' is only needed for key generation (least privilege)
        return {"imsk": FHContext.serialise_msk(self.msk[nid], encrypt_only=True)}

    def get_decryption_keys(self, sid: str, **kwargs) -> dict | None:
        """Key for weights ``y`` covering every slot (one ``r`` for all slots).

        Returns:
            Dict with ``dk`` (per slot the FH-IPFE key), ``z`` (``[r sum z_i]_T``)
            and a random ``key_id`` (nothing derived from ``y``).
        """
        credentials = kwargs.get("credentials", None)
        if not credentials:
            raise FEKeyError("need credentials for FH-MIFE decryption key generation")
        fusion_weight = credentials.get("fusion_weight")
        if not isinstance(fusion_weight, dict) or set(fusion_weight) != set(self.lst_nid):
            raise FEValidationError("fusion weights must cover every client")
        ctx = self.ctx
        r = int(ctx.group.random(ZR))
        z_total, dk = 0, {}
        for nid in self.lst_nid:
            y = int_vector(fusion_weight[nid], "fusion weight of " + nid, length=self.eta, bound=self.bound_y)
            key, z_i = ctx.fhipe.keygen(self.msk[nid], pow_many(ctx.g2, y, ctx.group), r=r)
            dk[nid] = [ctx.ser(v) for v in key]
            z_total = (z_total + z_i) % ctx.fhipe.order
        return {
            "dk": dk,
            "z": ctx.ser(ctx.fhipe.gt ** ctx.zr(z_total)),
            "key_id": secrets.token_hex(16),
        }


class MIFEFHIPE(IPFEAbsCrypto):
    """Crypto operations for the ACFGU18 function-hiding MIFE."""

    def __init__(self, config: dict, **kwargs) -> None:
        """Initialise with ``keys.pp`` and, for clients, ``keys.sk``."""
        super().__init__(config, **kwargs)
        self.pp = self.keys["pp"]
        self.ctx = _Group(self.pp)
        if self._has_private_keys():
            self.sk = self.keys["sk"]
            self.imsk = FHContext.deserialise_msk(self.sk["imsk"])

    def encrypt(self, lst_pt: list) -> dict:
        """``FHIPE.Enc(msk_i, [x_i]_1)`` for this slot's ``eta`` integers."""
        if not self._has_private_keys():
            raise FEKeyError("no encryption key provided")
        x = int_vector(lst_pt, "plaintext", length=self.pp["eta"], bound=self.pp["bound_x"])
        ctx = self.ctx
        ct = ctx.fhipe.encrypt(self.imsk, pow_many(ctx.g1, x, ctx.group))
        return {"ct": [ctx.ser(v) for v in ct]}

    def decrypt(self, dct_ct: dict, dk: dict, fusion_weight: dict | None = None):
        """``log prod_i FHIPE.Dec(ct_i, dk_i) / [z]_T``; ``None`` if out of bound.

        *fusion_weight* is accepted for API compatibility and ignored: the
        key encodes ``y`` (function hiding).

        Raises:
            FESchemeError: If the ciphertexts do not cover every slot of the key.
        """
        if not dk or "dk" not in dk:
            raise FEKeyError("no decryption key provided")
        if set(dct_ct) != set(dk["dk"]):
            raise FESchemeError("need one ciphertext for every slot of the key")
        ctx = self.ctx
        acc = ctx.group.init(GT, 1)
        for nid, ct in dct_ct.items():
            acc *= ctx.fhipe.decrypt(ctx.de_many(ct["ct"]), ctx.de_many(dk["dk"][nid]), ctx.group.init(GT, 1))
        acc /= ctx.de(dk["z"])
        bound = int(self.pp["n"] * self.pp["eta"] * gp.mpz(self.pp["bound_x"]) * gp.mpz(self.pp["bound_y"]))
        try:
            return bounded_discrete_log_gt(ctx.group, ctx.fhipe.gt, acc, bound)
        except ValueError:
            logger.error("inner product out of bound (wrong key or ciphertexts?)")
            return None

    def encrypt_lst_ndarray(self, lst_ndarray: list, **kwargs) -> list | None:
        """Not provided: MIFE is not used for element-wise aggregation (no labels)."""
        raise NotImplementedError("ndarray helper is not implemented for FH-IPE MIFE")

    def compute_lst_ndarray_ct(self, dict_ndarray_ct: dict, **kwargs) -> list | None:
        """Not provided (see :meth:`encrypt_lst_ndarray`)."""
        raise NotImplementedError("ndarray helper is not implemented for FH-IPE MIFE")

    def decrypt_lst_ndarray_ct(self, dict_ndarray_ct: dict, **kwargs) -> list | None:
        """Not provided (see :meth:`encrypt_lst_ndarray`)."""
        raise NotImplementedError("ndarray helper is not implemented for FH-IPE MIFE")
