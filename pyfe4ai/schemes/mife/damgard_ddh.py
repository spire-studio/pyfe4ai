"""
DDH (Damgard-style) multi-input inner-product FE.

| Abdalla, Catalano, Fiore, Gay, Ursu:
| "Multi-Input Functional Encryption for Inner Products: Function-Hiding
|  Realizations and Constructions without Pairings", CRYPTO 2018
| (ePrint 2017/972) -- Fig. 4 (transformation for bounded-norm inner products)
| instantiated with the DDH IPFE of Agrawal, Libert, Stehle (CRYPTO 2016),
| recalled as Fig. 7 (D_k-MDDH, here k = 1) with its two-step decryption in
| Section 4.1.

* type:     private-key multi-input encryption (MIFE, DDH)
* setting:  Integer based (order-q subgroup of Z_p^*, p = 2q + 1)

Construction (slot ``i`` is client ``i``; ``[a] = g^a``, ``h = g^a``):

* Setup:    ``W_i = (s_i, t_i) <- (Z_q^eta)^2``, ``pk_{i,j} = g^{s_{i,j}} h^{t_{i,j}}``,
  one-time pad ``u_i <- Z_q^eta``; client ``i`` keeps ``(pk_i, u_i)``.
* Encrypt:  ``w = x_i + u_i mod q``; ``C = g^r``, ``D = h^r``,
  ``E_j = g^{w_j} pk_{i,j}^r``.
* KeyGen:   per slot ``(<s_i, y_i>, <t_i, y_i>)`` and ``z = sum_i <u_i, y_i> mod q``.
* Decrypt:  Dec1 per slot ``[<w_i, y_i>] = prod_j E_j^{y_j} / (C^{<s,y>} D^{<t,y>})``;
  ``[<x, y>] = prod_i [<w_i, y_i>] / [z]``, then a bounded dlog (Dec2).

The pad makes each slot's partial result ``[<x_i + u_i, y_i>]`` uniform, so
the key holder learns only ``<x, y>``; versions <= 0.2.0 omitted it and
revealed every ``<x_i, y_i>``. Deviation: all slots share the group and
``h`` (the matrix ``A`` of Fig. 7) while ``W_i`` and ``u_i`` are independent
per slot, instead of running Setup independently per slot. A decryption
needs ciphertexts of every slot named in the key.
"""

from __future__ import annotations

import logging
import os

import gmpy2 as gp
import numpy as np

from pyfe4ai.schemes.common.ddh_base import DamgardDDHKeyGeneratorBase
from pyfe4ai.core.base import IPFEAbsCrypto
from pyfe4ai.core.base import decoded_layer
from pyfe4ai.core.constants import CryptoCONST
from pyfe4ai.primitives.groups.integer_groups import _random
from pyfe4ai.primitives.groups.dlog_solver import load_or_build_dlog_table, dlog_table_solve
from pyfe4ai.primitives.groups.modular_utils import pow_signed
from pyfe4ai.core.formats.dl import DamgardCiphertext
from pyfe4ai.core.formats.dl import DamgardEncryptionKey
from pyfe4ai.core.formats.dl import MIFEDamgardFunctionalKey
from pyfe4ai.core.formats.dl import SIFEDamgardParameters
from pyfe4ai.core.exceptions import FEKeyError, FESchemeError, FEValidationError, warn_at_caller
from pyfe4ai.primitives.sampling import rand_uniform_vector
from pyfe4ai._validation import scale_to_int
from pyfe4ai._validation import strict_int

logger = logging.getLogger(__name__)

class MIFEDamgardKeyGenerator(DamgardDDHKeyGeneratorBase):
    """Key generator for DDH multi-input inner-product FE (ACFGU18 Fig. 4 + ALS16)."""

    _scheme_type = CryptoCONST.TYPE_MIFE_DAMGARD
    _verify_keys = ("sec_param", "eta", "n", "modulus_length", "bound")

    def __init__(self, config: dict, **kwargs) -> None:
        """Initialise the key generator.

        Args:
            config: Scheme configuration. Recognised keys: ``n`` (number of
                slots), ``lst_nid``, ``eta`` (slot dimension),
                ``modulus_length`` (bits of ``p``), ``bound`` (on ``|x|`` and
                ``|y|``), ``sec_param``.
        """
        super().__init__(config, **kwargs)
        self.eta = config.get("eta", CryptoCONST.MIFE_ETA)
        self.modulus_length = config.get("modulus_length", self.sec_param)
        self.bound = config.get("bound", 100)
        if not self.n:
            raise FEKeyError("need to provide number of clients")
        self._load_parameters()
        self._validate_bounds()

    def _extra_param_fields(self) -> dict:
        return {"n": self.n}

    def _validate_bounds(self) -> None:
        if 2 * self.n * self.eta * gp.mpz(self.bound) ** 2 >= self.q:
            raise FEValidationError(
                "2 * n * eta * bound^2 should be smaller than the group order"
            )

    def setup(self) -> None:
        """Sample ``W_i = (s_i, t_i)`` and the one-time pad ``u_i`` for every slot."""
        s = {nid: rand_uniform_vector(self.eta, self.q) for nid in self.lst_nid}
        t = {nid: rand_uniform_vector(self.eta, self.q) for nid in self.lst_nid}
        u = {nid: rand_uniform_vector(self.eta, self.q) for nid in self.lst_nid}
        pk = {
            nid: [
                gp.powmod(self.g, s_j, self.p) * gp.powmod(self.h, t_j, self.p) % self.p
                for s_j, t_j in zip(s[nid], t[nid])
            ]
            for nid in self.lst_nid
        }
        self.msk = {"s": s, "t": t, "u": u}
        self.mpk = {"pk": pk}
        logger.info("MIFE Damgard (ACFGU18) setup successfully")

    def get_public_parameters(self) -> dict:
        """Return serialisable public parameters."""
        return SIFEDamgardParameters.new(
            p=gp.digits(self.p),
            q=gp.digits(self.q),
            g=gp.digits(self.g),
            h=gp.digits(self.h),
            eta=self.eta,
            n=self.n,
            bound=self.bound,
            sec_param=self.sec_param,
            modulus_length=self.modulus_length,
        ).to_dict()

    def get_private_keys(self, nid: str, **kwargs) -> dict | None:
        """Return slot *nid*'s encryption key ``{pk, u}`` (``None`` if unknown).

        The key includes the one-time pad and must stay with the client.
        """
        if nid not in self.mpk["pk"]:
            return None
        return DamgardEncryptionKey.new(
            pk=[gp.digits(v) for v in self.mpk["pk"][nid]],
            u=[gp.digits(v) for v in self.msk["u"][nid]],
        ).to_dict()

    def get_decryption_keys(self, sid: str, **kwargs) -> dict | None:
        """Derive ``({(<s_i, y_i>, <t_i, y_i>)}_i, z = sum <u_i, y_i> mod q)``.

        Args:
            sid: Session identifier (unused).
            **kwargs: Must include ``credentials`` with ``fusion_weight``
                (dict mapping every client ID to an integer weight list).

        Raises:
            FEKeyError: If credentials are missing.
            FEValidationError: If a client is missing or a weight is malformed.
        """
        credentials = kwargs.get("credentials", None)
        if not credentials:
            raise FEKeyError("need credentials for MIFE Damgard decryption key generation")
        fusion_weight = credentials.get("fusion_weight")
        if not isinstance(fusion_weight, dict):
            raise FEValidationError("invalid fusion weights provided, need a dict")

        k1, k2 = {}, {}
        z = gp.mpz(0)
        for nid in self.lst_nid:
            if nid not in fusion_weight:
                raise FEValidationError("fusion weight missing client {}".format(nid))
            weights = fusion_weight[nid]
            if len(weights) != self.eta:
                raise FEValidationError("invalid fusion weight length for client {}".format(nid))
            if any(abs(strict_int(v, "fusion weight")) > self.bound for v in weights):
                raise FEValidationError("fusion weight exceeds configured bound")
            ys = [gp.mpz(int(v)) for v in weights]
            k1[nid] = gp.digits(sum(a * b for a, b in zip(self.msk["s"][nid], ys)) % self.q)
            k2[nid] = gp.digits(sum(a * b for a, b in zip(self.msk["t"][nid], ys)) % self.q)
            z += sum(a * b for a, b in zip(self.msk["u"][nid], ys))
        return MIFEDamgardFunctionalKey.new(k1=k1, k2=k2, z=gp.digits(z % self.q)).to_dict()


class MIFEDamgard(IPFEAbsCrypto):
    """Crypto operations for DDH multi-input inner-product FE (ACFGU18 Fig. 4 + ALS16)."""

    def __init__(self, config: dict, **kwargs) -> None:
        """Initialise with ``keys.pp`` and, for clients, ``keys.sk``."""
        super().__init__(config, **kwargs)
        self.pp = self.keys["pp"]
        if self._has_private_keys():
            self.sk = self.keys["sk"]
        self.dlog_table = None

    def _load_dlog_table(self) -> None:
        _dlog_file = os.path.join(
            self.config_folder, CryptoCONST.TYPE_MIFE_DAMGARD,
            f"dlog_{self.pp['bound']}.json",
        )
        bound = self.pp["n"] * self.pp["eta"] * (self.pp["bound"] ** 2) + 1
        self.dlog_table, self.func_bound, self._step_size, self._giant_step = (
            load_or_build_dlog_table(_dlog_file, self.pp["g"], self.pp["p"], bound)
        )

    def encrypt(self, lst_pt: list) -> dict:
        """Encrypt this slot's vector: ALS16 encryption of ``x + u mod q``.

        Returns:
            Dict with ``c`` (``g^r``), ``d`` (``h^r``) and ``e`` (one element per
            coordinate).
        """
        if not self._has_public_parameters():
            raise FEKeyError("no public parameters provided for encryption")
        if not self._has_private_keys() or "u" not in DamgardEncryptionKey(self.sk):
            raise FEKeyError("no private keys provided for encryption")
        if len(lst_pt) != self.pp["eta"]:
            raise FEValidationError("invalid size of input plaintext")
        if any(abs(strict_int(v, "plaintext")) > self.pp["bound"] for v in lst_pt):
            raise FEValidationError("plaintext exceeds configured bound")

        p = gp.mpz(self.pp["p"])
        q = gp.mpz(self.pp["q"])
        g = gp.mpz(self.pp["g"])
        h = gp.mpz(self.pp["h"])
        r = _random(q, int(q).bit_length())
        e = []
        sk = DamgardEncryptionKey.from_dict(self.sk)
        for x_j, u_j, pk_j in zip(lst_pt, sk.u, sk.pk):
            w_j = (gp.mpz(int(x_j)) + gp.mpz(u_j)) % q
            e.append(gp.digits(gp.powmod(g, w_j, p) * gp.powmod(gp.mpz(pk_j), r, p) % p))
        return DamgardCiphertext.new(c=gp.digits(gp.powmod(g, r, p)), d=gp.digits(gp.powmod(h, r, p)), e=e).to_dict()

    def decrypt(self, dct_ct: dict, dk: dict, fusion_weight: dict):
        """Recover ``sum_i <x_i, y_i>`` from one ciphertext per slot.

        Returns:
            The inner product, or ``None`` if it is out of the dlog bound.

        Raises:
            FEKeyError: If the key is missing.
            FESchemeError: If ciphertexts do not cover exactly the key's slots.
        """
        if not self._has_public_parameters():
            raise FEKeyError("no public parameters provided for decryption")
        if not dk or "z" not in MIFEDamgardFunctionalKey(dk):
            raise FEKeyError("no decryption key provided")
        key = MIFEDamgardFunctionalKey.from_dict(dk)
        if set(dct_ct) != set(key.k1) or set(dct_ct) != set(fusion_weight):
            raise FESchemeError("need one ciphertext for every slot of the key")

        p = gp.mpz(self.pp["p"])
        g = gp.mpz(self.pp["g"])
        acc = gp.mpz(1)
        for nid, ct in dct_ct.items():
            # Dec1 of slot nid: [<x_i + u_i, y_i>]
            ct = DamgardCiphertext.from_dict(ct)
            for e_j, y_j in zip(ct.e, fusion_weight[nid]):
                acc = acc * pow_signed(gp.mpz(e_j), gp.mpz(int(y_j)), p) % p
            den = gp.powmod(gp.mpz(ct.c), gp.mpz(key.k1[nid]), p)
            den = den * gp.powmod(gp.mpz(ct.d), gp.mpz(key.k2[nid]), p) % p
            acc = gp.divm(acc, den, p)
        acc = gp.divm(acc, gp.powmod(g, gp.mpz(key.z), p), p)
        return self._solve_dlog(acc)

    def _solve_dlog(self, encoded_value: gp.mpz) -> int | None:
        if self.dlog_table is None:
            self._load_dlog_table()
        try:
            return dlog_table_solve(
                encoded_value, self.pp["g"], self.pp["p"],
                self.func_bound, self.dlog_table, self._step_size, self._giant_step,
            )
        except (ValueError, RuntimeError):
            logger.error("inner-product is out of bound supported by crypto system")
            warn_at_caller("inner-product is out of bound supported by crypto system")
            return None

    def encrypt_lst_ndarray(self, lst_ndarray: list, **kwargs) -> list | None:
        """Encrypt ndarrays element-wise (``eta = 1``)."""
        if self.pp["eta"] != 1:
            raise NotImplementedError("ndarray helper currently expects eta=1")
        lst_ndarray_ct = []
        for ary_src in lst_ndarray:
            ary = scale_to_int(ary_src, self.precision)
            ary_ct = np.empty(ary.shape, dtype=object)
            for i, w in np.ndenumerate(ary):
                ary_ct[i] = self.encrypt([int(w)])
            lst_ndarray_ct.append(ary_ct)
        return lst_ndarray_ct

    def compute_lst_ndarray_ct(self, dict_ndarray_ct: dict, **kwargs) -> list | None:
        """Aggregate-decrypt ndarray ciphertexts (needs ``dk`` and ``fusion_weight``)."""
        dk = kwargs.get("dk", None)
        fusion_weight = kwargs.get("fusion_weight", None)
        if dk is None or fusion_weight is None:
            raise FEKeyError("need to provide decryption key and fusion weight")
        return self.decrypt_lst_ndarray_ct(dict_ndarray_ct, dk=dk, fusion_weight=fusion_weight)

    def decrypt_lst_ndarray_ct(self, dict_ndarray_ct: dict, **kwargs) -> list | None:
        """Decrypt ndarray ciphertexts element-wise."""
        if self.pp["eta"] != 1:
            raise NotImplementedError("ndarray helper currently expects eta=1")
        dk = kwargs.get("dk", None)
        fusion_weight = kwargs.get("fusion_weight", None)
        if dk is None or fusion_weight is None:
            raise FEKeyError("need to provide decryption key and fusion weight")
        sample = next(iter(dict_ndarray_ct.values()))
        lst_ndarray = []
        for l in range(len(sample)):
            ary_dec = np.empty(sample[l].shape, dtype=object)
            for i, _ in np.ndenumerate(sample[l]):
                dct_ct = {nid: dict_ndarray_ct[nid][l][i] for nid in dict_ndarray_ct}
                ary_dec[i] = self.decrypt(dct_ct, dk, fusion_weight)
            lst_ndarray.append(decoded_layer(ary_dec, l, 10 ** self.precision))
        return lst_ndarray
