"""
Function-hiding multi-client inner-product FE over pairings.

| Agrawal, Goyal, Tomida, "Multi-Party Functional Encryption", TCC 2021
| (AGT21b), Section 6.2 -- the FH-IP-MCFE as recapped in
| Nguyen, Pointcheval, Schaedlich, "Decentralized Multi-Client Functional
| Encryption with Strong Security", IACR CiC 1(2), 2024, Section 2
| ("Recap: The Function-Hiding MCFE of [AGT21b]"); selective security under
| SXDH in the random-oracle model.

* type:     multi-client encryption with function-hiding keys (FH-MCFE)
* setting:  asymmetric pairing group (default BLS12-381), random oracle

Each client ``i`` holds the master key ``imsk_i`` of an independent
function-hiding IPFE (:mod:`pyfe4ai.schemes.mcfe.fh_ipe`):

* Enc:      ``[omega]_1 = H1(tag)``; ``ct_i = iEnc(imsk_i, [x_hat_i]_1)`` with
  ``x_hat_i = (x_i, omega, 0, 0^eta, 0, 0)``.
* KeyGen:   fresh ``t_1, ..., t_n`` with ``sum t_i = 0``;
  ``dk_i = iKeyGen(imsk_i, [y_hat_i]_2)``, ``y_hat_i = (y_i, t_i, 0, 0^eta, 0, 0)``.
* Dec:      ``prod_i iDec(dk_i, ct_i) = [sum <x_i, y_i> + omega sum t_i]_T = [<x, y>]_T``,
  then a bounded discrete log.

The zero coordinates are the ones the security proof uses (NPS24 Section 2).
Each client's partial result is masked by ``omega t_i`` and the IPFE hides
``y_i``; each client must encrypt at most once per tag. Symmetric pairing
groups (SS512, ...) are rejected because SXDH does not hold there.
Versions <= 0.2.0 masked inputs with a public label scalar instead.
"""

from __future__ import annotations

import logging
import secrets

import gmpy2 as gp
import numpy as np

from pyfe4ai.core.base import IPFEAbsCrypto
from pyfe4ai.core.base import decoded_layer
from pyfe4ai.core.base import IPFEAbsKeyGenerator
from pyfe4ai.schemes.mcfe.fh_ipe import FHIPE
from pyfe4ai.schemes.mcfe.labels import element_label as _element_label
from pyfe4ai.schemes.mcfe.labels import encode_label
from pyfe4ai.primitives.pairing.pairing_backend import G1, G2, GT, ZR, PairingGroup, deserialize_many, pow_many
from pyfe4ai.primitives.pairing.pairing_backend import DEFAULT_PAIRING_GROUP, check_asymmetric
from pyfe4ai.primitives.pairing.pairing_backend import require_pairing_backend
from pyfe4ai.primitives.pairing.pairing_utils import bounded_discrete_log_gt
from pyfe4ai.core.exceptions import FEKeyError, FESchemeError, FEValidationError
from pyfe4ai._validation import int_vector
from pyfe4ai._validation import scale_to_int

logger = logging.getLogger(__name__)

_H1_DOMAIN = "pyfe4ai/fh-mcfe/H1(tag)|"
_H2_DOMAIN = "pyfe4ai/fh-mcfe/H2(tag-f)|"


def vector_dim(eta: int) -> int:
    """Length of the extended vectors ``(v, scalar, 0, 0^eta, 0, 0)``."""
    return 2 * eta + 4


class FHContext:
    """Deserialised public parameters shared by the FH (D)MCFE schemes."""

    def __init__(self, pp: dict) -> None:
        require_pairing_backend()
        self.pp = pp
        self.group = PairingGroup(pp["pairing_group_param"])
        check_asymmetric(self.group)
        self.g1 = self.group.deserialize(pp["g1"].encode("utf-8"))
        self.g2 = self.group.deserialize(pp["g2"].encode("utf-8"))
        self.eta = pp["eta"]
        self.fhipe = FHIPE(self.group, self.g1, self.g2, vector_dim(self.eta))
        self.order = self.fhipe.order

    def zr(self, value):
        return self.group.init(ZR, int(value) % self.order)

    def ser(self, element) -> str:
        return self.group.serialize(element).decode("utf-8")

    def de(self, text: str):
        return self.group.deserialize(text.encode("utf-8"))

    def de_many(self, texts: list) -> list:
        return deserialize_many(self.group, texts)

    def h1(self, tag) -> object:
        return self.group.hash(_H1_DOMAIN + encode_label(tag), G1)

    def h2(self, tag_f: str) -> object:
        return self.group.hash(_H2_DOMAIN + tag_f, G2)

    def x_hat(self, x: list, slot_scalar_elem) -> list:
        """``[(x, s, 0, 0^eta, 0, 0)]_1`` with the scalar slot given as a G1 element."""
        one = self.fhipe.identity1()
        return pow_many(self.g1, x, self.group) + [slot_scalar_elem] + [one] * (self.eta + 3)

    def y_hat(self, y: list, slot_scalar_elem) -> list:
        """``[(y, s, 0, 0^eta, 0, 0)]_2`` with the scalar slot given as a G2 element."""
        one = self.fhipe.identity2()
        return pow_many(self.g2, y, self.group) + [slot_scalar_elem] + [one] * (self.eta + 3)

    @staticmethod
    def serialise_msk(msk: dict, encrypt_only: bool = False) -> dict:
        """Serialise an FH-IPFE master key; *encrypt_only* drops ``a_out``,
        which only key generation needs."""
        out = {
            "a_in": [str(v) for v in msk["a_in"]],
            "u": [[str(v) for v in row] for row in msk["u"]],
            "w": [[str(v) for v in row] for row in msk["w"]],
            "z": str(msk["z"]),
        }
        if not encrypt_only:
            out["a_out"] = [str(v) for v in msk["a_out"]]
        return out

    @staticmethod
    def deserialise_msk(msk: dict) -> dict:
        out = {
            "a_in": [int(v) for v in msk["a_in"]],
            "u": [[int(v) for v in row] for row in msk["u"]],
            "w": [[int(v) for v in row] for row in msk["w"]],
            "z": int(msk["z"]),
        }
        if "a_out" in msk:
            out["a_out"] = [int(v) for v in msk["a_out"]]
        return out

    def key_entry(self, key: list, z_sum: int) -> dict:
        return {"k": [self.ser(v) for v in key], "z": self.ser(self.fhipe.gt ** self.zr(z_sum))}

    def partial(self, ct: dict, key_entry: dict):
        return self.fhipe.decrypt(self.de_many(ct["ct"]), self.de_many(key_entry["k"]), self.de(key_entry["z"]))

    def solve(self, value) -> int | None:
        bound = int(self.pp["n"] * self.eta * gp.mpz(self.pp["bound_x"]) * gp.mpz(self.pp["bound_y"]))
        try:
            return bounded_discrete_log_gt(self.group, self.fhipe.gt, value, bound)
        except ValueError:
            logger.error("inner product out of bound (wrong tag or key?)")
            return None


class MCFEFHMultiIPEKeyGenerator(IPFEAbsKeyGenerator):
    """Key generator for the AGT21b function-hiding MCFE."""

    def __init__(self, config: dict, **kwargs) -> None:
        """Initialise the key generator.

        Args:
            config: ``n``, ``lst_nid``, ``eta``, ``bound_x``, ``bound_y`` and
                ``pairing_group_param`` (asymmetric, default BLS12-381). The keys
                ``sec_level``, ``u_bound`` and ``label_modulus`` of versions
                <= 0.2.0 are ignored.
        """
        super().__init__(config, **kwargs)
        require_pairing_backend()
        self.eta = int(config.get("eta", 1))
        self.bound_x = int(config.get("bound_x", 10))
        self.bound_y = int(config.get("bound_y", 10))
        self.pairing_group_param = config.get("pairing_group_param", DEFAULT_PAIRING_GROUP)
        check_asymmetric(self.pairing_group_param)
        if not self.n:
            raise FEKeyError("need to provide number of clients")

    def setup(self) -> None:
        """Sample the generators and one FH-IPFE master key per client."""
        group = PairingGroup(self.pairing_group_param)
        g1, g2 = group.random(G1), group.random(G2)
        self.pp = {
            "pairing_group_param": self.pairing_group_param,
            "g1": group.serialize(g1).decode("utf-8"),
            "g2": group.serialize(g2).decode("utf-8"),
            "eta": self.eta,
            "n": self.n,
            "lst_nid": list(self.lst_nid),
            "bound_x": self.bound_x,
            "bound_y": self.bound_y,
            "sec_param": self.sec_param,
        }
        self.ctx = FHContext(self.pp)
        self.msk = {nid: self.ctx.fhipe.setup() for nid in self.lst_nid}
        logger.info("FH-MCFE (AGT21b) setup successfully")

    def get_public_parameters(self) -> dict:
        """Group description, generators, dimensions and bounds."""
        return dict(self.pp)

    def get_private_keys(self, nid: str, **kwargs) -> dict | None:
        """Return client *nid*'s encryption key ``{imsk}`` (``None`` if unknown).

        This is the slot's FH-IPFE master key without the part only key
        generation needs. With it a client could strip the outer layer of its
        own slot of a functional key (outside the model: clients are not
        corrupted).
        """
        if nid not in self.msk:
            return None
        return {"imsk": self.ctx.serialise_msk(self.msk[nid], encrypt_only=True)}

    def _validate_weights(self, fusion_weight: dict) -> None:
        if not isinstance(fusion_weight, dict) or set(fusion_weight) != set(self.lst_nid):
            raise FEValidationError("fusion weights must cover every client")
        for nid, y in fusion_weight.items():
            int_vector(y, "fusion weight of {}".format(nid), length=self.eta, bound=self.bound_y)

    def get_decryption_keys(self, sid: str, **kwargs) -> dict | None:
        """Derive ``{dk_i = iKeyGen(imsk_i, [y_hat_i]_2)}`` with fresh zero-sum ``t_i``.

        Args:
            sid: Session identifier (unused).
            **kwargs: ``credentials`` with ``fusion_weight`` covering every client.

        Returns:
            Dict with ``dk`` (per client the FH-IPFE key ``k`` and ``[z]_T``)
            and a random ``key_id`` (nothing derived from ``y``: keys are
            function hiding).
        """
        credentials = kwargs.get("credentials", None)
        if not credentials:
            raise FEKeyError("need credentials for FH-MCFE decryption key generation")
        fusion_weight = credentials.get("fusion_weight")
        self._validate_weights(fusion_weight)
        ctx = self.ctx
        t = [int(ctx.group.random(ZR)) for _ in self.lst_nid[:-1]]
        t.append((-sum(t)) % ctx.order)
        dk = {}
        for t_i, nid in zip(t, self.lst_nid):
            y_hat = ctx.y_hat([int(v) for v in fusion_weight[nid]], ctx.g2 ** ctx.zr(t_i))
            key, z_sum = ctx.fhipe.keygen(self.msk[nid], y_hat)
            dk[nid] = ctx.key_entry(key, z_sum)
        return {"dk": dk, "key_id": secrets.token_hex(16)}


class MCFEFHMultiIPE(IPFEAbsCrypto):
    """Crypto operations for the AGT21b function-hiding MCFE."""

    def __init__(self, config: dict, **kwargs) -> None:
        """Initialise with ``keys.pp`` and, for clients, ``keys.sk``."""
        super().__init__(config, **kwargs)
        self.pp = self.keys["pp"]
        self.ctx = FHContext(self.pp)
        if self._has_private_keys():
            self.sk = self.keys["sk"]
            self.imsk = FHContext.deserialise_msk(self.sk["imsk"])

    def _check_plaintext(self, lst_pt) -> list:
        if not self._has_private_keys():
            raise FEKeyError("no encryption key provided")
        return int_vector(lst_pt, "plaintext", length=self.pp["eta"], bound=int(self.pp["bound_x"]))

    def encrypt(self, lst_pt: list, label: str) -> dict:
        """``iEnc(imsk_i, [(x_i, H1(label), 0, ...)]_1)``; once per client and label."""
        x_hat = self.ctx.x_hat(self._check_plaintext(lst_pt), self.ctx.h1(label))
        return {"ct": [self.ctx.ser(v) for v in self.ctx.fhipe.encrypt(self.imsk, x_hat)]}

    def decrypt(self, dct_ct: dict, dk: dict, fusion_weight: dict | None = None, label=None):
        """``log prod_i iDec(dk_i, ct_i)``; ``None`` if out of bound (wrong tag or key).

        *fusion_weight* and *label* are accepted for API compatibility and
        ignored: the key encodes ``y`` (function hiding) and the ciphertexts
        carry ``H1(label)``.

        Raises:
            FESchemeError: If the ciphertexts do not cover every client of the key.
        """
        if not dk or "dk" not in dk:
            raise FEKeyError("no decryption key provided")
        if set(dct_ct) != set(dk["dk"]):
            raise FESchemeError("need one ciphertext for every client of the key")
        acc = self.ctx.group.init(GT, 1)
        for nid, ct in dct_ct.items():
            acc *= self.ctx.partial(ct, dk["dk"][nid])
        return self.ctx.solve(acc)

    def encrypt_lst_ndarray(self, lst_ndarray: list, **kwargs) -> list | None:
        """Encrypt ndarrays element-wise (``eta = 1``), one derived label per element."""
        if self.pp["eta"] != 1:
            raise NotImplementedError("ndarray helper currently expects eta=1")
        label = kwargs.get("label", None)
        if not isinstance(label, str):
            raise FEValidationError("need a string label for encrypt_lst_ndarray")
        out = []
        for l, ary_src in enumerate(lst_ndarray):
            ary = scale_to_int(ary_src, self.precision)
            ary_ct = np.empty(ary.shape, dtype=object)
            for i, w in np.ndenumerate(ary):
                ary_ct[i] = self.encrypt([int(w)], _element_label(label, l, i))
            out.append(ary_ct)
        return out

    def compute_lst_ndarray_ct(self, dict_ndarray_ct: dict, **kwargs) -> list | None:
        """Aggregate-decrypt ndarray ciphertexts (needs ``dk`` and ``label``)."""
        return self.decrypt_lst_ndarray_ct(dict_ndarray_ct, **kwargs)

    def decrypt_lst_ndarray_ct(self, dict_ndarray_ct: dict, **kwargs) -> list | None:
        """Decrypt ndarray ciphertexts element-wise."""
        dk = kwargs.get("dk", None)
        fusion_weight = kwargs.get("fusion_weight", None)
        label = kwargs.get("label", None)
        if dk is None or label is None:
            raise FEKeyError("need to provide decryption key and label")
        sample = next(iter(dict_ndarray_ct.values()))
        out = []
        for l in range(len(sample)):
            ary_dec = np.empty(sample[l].shape, dtype=object)
            for i, _ in np.ndenumerate(sample[l]):
                dct_ct = {nid: dict_ndarray_ct[nid][l][i] for nid in dict_ndarray_ct}
                ary_dec[i] = self.decrypt(dct_ct, dk, fusion_weight, _element_label(label, l, i))
            out.append(decoded_layer(ary_dec, l, 10 ** self.precision))
        return out
