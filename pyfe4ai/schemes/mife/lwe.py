"""
LWE multi-input inner-product FE (selective security).

| Abdalla, Catalano, Fiore, Gay, Ursu:
| "Multi-Input Functional Encryption for Inner Products: Function-Hiding
|  Realizations and Constructions without Pairings", CRYPTO 2018
| (ePrint 2017/972) -- Fig. 4 (transformation for bounded-norm inner
| products), instantiated with the LWE IPFE of Abdalla, Bourse, De Caro,
| Pointcheval, "Simple Functional Encryption Schemes for Inner Products",
| PKC 2015 (full version: ePrint 2015/017), using the two-step decryption
| of ACFGU18 Section 4.2.

* type:     private-key multi-input encryption (MIFE, LWE)
* setting:  Integer based (Z_q, plaintext modulus p, Delta = floor(q / p))

Construction (slot ``i`` is client ``i``, ``L = q``):

* Setup:    public ``A``; per slot ``S_i`` (``lwe_n x eta`` uniform),
  ``pk_i = A S_i + E_i`` and a one-time pad ``u_i <- Z_q^eta``;
  client ``i`` keeps ``(pk_i, u_i)``.
* Encrypt:  ``w = x_i + u_i mod q``; ``ct0 = A^T r``,
  ``ct1 = pk_i^T r + Delta * w mod q`` (``r`` binary).
* KeyGen:   per slot ``sk_i = S_i y_i mod q``, and ``z = sum_i <u_i, y_i> mod q``.
* Decrypt:  Dec1 per slot ``<y_i, ct1> - <ct0, sk_i> = Delta <w_i, y_i> + noise_i``;
  ``mu = round(centre(sum_i Dec1_i - Delta z) / Delta)`` (Dec2).

The pad hides each slot's partial result; versions <= 0.2.0 had no pad and
encoded ``floor(x q / p)``, so the key holder decoded every ``<x_i, y_i>``.
Deviations: ACFGU18 instantiates Fig. 4 with the adaptively secure ALS16 LWE
IPFE (that is :mod:`pyfe4ai.schemes.mife.fullysec_lwe`); this module uses
the selectively secure ABDP15 IPFE, whose two-step decryption is the same.
All slots share ``A`` (secrets and pads are independent per slot). A
decryption needs ciphertexts of every slot named in the key.

Security of a slot's IPFE rests on the LWE proof of the ABDP15 full version,
Thm. 4.4 with Thms. 6.6 and 6.7 (see :mod:`pyfe4ai.schemes.sife.lwe`). Here
it reduces to LWE with errors of width ``sigma / (1 + 4 X sqrt(eta))``, since
the per-slot challenge vectors of the ACFGU18 reduction differ by up to
``4 X`` per coordinate; this composition is the library's own, and neither
paper covers it. The pad makes
the slot plaintexts ``w`` uniform mod ``q``; the proof only needs their
difference to be short. ``standard`` reads the 128-bit table at that width.

``A`` is published as a seed and expanded with SHAKE-128; encryption runs
in numpy over an ``A`` kept in memory or expanded per batch
(``matrix_mode``), see :mod:`pyfe4ai.schemes.common.abdp15_lwe`. Versions <= 0.2.0
published ``A`` entry by entry.
"""

from __future__ import annotations

import json
import logging
import os

import gmpy2 as gp
import numpy as np

from pyfe4ai.schemes.common.abdp15_lwe import ENCRYPT_BATCH
from pyfe4ai.schemes.common.abdp15_lwe import ABDP15CryptoMixin
from pyfe4ai.schemes.common.abdp15_lwe import ABDP15KeyGeneratorMixin
from pyfe4ai.core.base import IPFEAbsCrypto
from pyfe4ai.core.base import decoded_layer
from pyfe4ai.core.base import IPFEAbsKeyGenerator
from pyfe4ai.core.constants import CryptoCONST
from pyfe4ai.primitives.lattice.lwe_utils import abdp15_parameters
from pyfe4ai.primitives.lattice.lwe_utils import decode_scaled
from pyfe4ai.primitives.lattice.matrix_utils import digits_to_vector
from pyfe4ai.primitives.lattice.matrix_utils import matrix_to_digits
from pyfe4ai.primitives.lattice.matrix_utils import matvec_mod
from pyfe4ai.primitives.lattice.matrix_utils import vecdot_mod
from pyfe4ai.primitives.lattice.matrix_utils import vector_to_digits
from pyfe4ai.primitives.sampling import rand_uniform_matrix
from pyfe4ai.primitives.sampling import rand_uniform_vector
from pyfe4ai.core.exceptions import FEKeyError, FESchemeError, FEValidationError
from pyfe4ai.core.formats.lwe import LWECiphertext
from pyfe4ai.core.formats.lwe import MIFELWEEncryptionKey
from pyfe4ai.core.formats.lwe import MIFELWEFunctionalKey
from pyfe4ai.core.formats.lwe import MIFELWEParameters
from pyfe4ai.core.versions import MIFE_LWE_RECORD
from pyfe4ai.core.ids import issued_id
from pyfe4ai._validation import scale_to_int
from pyfe4ai._validation import strict_int

logger = logging.getLogger(__name__)


class MIFELWEKeyGenerator(IPFEAbsKeyGenerator, ABDP15KeyGeneratorMixin):
    """Key generator for LWE-based multi-input inner-product FE."""

    _scheme_type = CryptoCONST.TYPE_MIFE_LWE
    _construction = MIFE_LWE_RECORD
    # ACFGU18, proof of Thm. 3: a slot's IPFE is challenged with
    # x^(j,0) - x^(1,0) + x^(1,1) against x^(j,1), which differ by up to 4X
    _challenge_spread = 4

    def __init__(self, config: dict, **kwargs) -> None:
        """Initialise the MIFE LWE key generator.

        Args:
            config: Scheme configuration. Recognised keys: ``sec_param``,
                ``eta``, ``n`` (number of clients), ``lwe_n`` (default 32;
                ``"auto"`` for the smallest dimension meeting the 128-bit
                table, the ``standard`` profile's choice), ``bound_x``,
                ``bound_y`` and ``matrix_mode`` (``"memory"`` or
                ``"stream"``, see :mod:`pyfe4ai.schemes.common.abdp15_lwe`).
        """
        super().__init__(config, **kwargs)
        self.eta = config.get("eta", CryptoCONST.MIFE_ETA)
        self.bound_x = gp.mpz(config.get("bound_x", 20))
        self.bound_y = gp.mpz(config.get("bound_y", 20))
        if not self.n:
            raise FEKeyError("need to provide number of clients")
        self._init_abdp15(config)

    def _record_fields(self) -> dict:
        return {**super()._record_fields(), "n": self.n}

    def _derive(self) -> dict:
        # ACFGU18, footnote 14: the ALS16 parameters for vectors of dimension
        # n * eta, since decryption adds the noise of all n slots
        return abdp15_parameters(
            self.bound_x, self.bound_y, self.n * self.eta, self.lwe_n_config,
            lwe_sigma=self._table_sigma,
        )

    def setup(self) -> None:
        """Generate per-client secret keys and public keys (one pass over ``A``)."""
        sk = {
            nid: rand_uniform_matrix(self.lwe_n, self.eta, self.q) for nid in self.lst_nid
        }
        pk = dict(zip(self.lst_nid, self._public_keys([sk[nid] for nid in self.lst_nid])))
        self.msk = {
            "sk": sk,
            "u": {nid: rand_uniform_vector(self.eta, self.q) for nid in self.lst_nid},
        }
        self.mpk = {"pk": pk}
        logger.info("MIFE LWE setup successfully")

    def get_public_parameters(self) -> dict:
        """Return serialisable public parameters.

        Returns:
            Dict containing ``A_seed``, ``m``, ``lwe_n``, ``p``, ``q``,
            ``eta``, ``n``, ``bound_x``, ``bound_y``, ``sec_param``.
        """
        return MIFELWEParameters.new(
            **self._matrix_parameters(),
            p=gp.digits(self.p),
            q=gp.digits(self.q),
            eta=self.eta,
            n=self.n,
            bound_x=gp.digits(self.bound_x),
            bound_y=gp.digits(self.bound_y),
            sec_param=self.sec_param,
        ).to_dict()

    def get_private_keys(self, nid: str, **kwargs) -> dict | None:
        """Return encryption keys for client *nid*.

        Args:
            nid: Client identifier.

        Returns:
            Dict with ``pk`` matrix, or ``None`` if *nid* is invalid.
        """
        nid = issued_id(nid)  # numpy integers as ints; a value that is no id is an unknown client
        if nid not in self.mpk["pk"]:
            return None
        return MIFELWEEncryptionKey.new(
            pk=matrix_to_digits(self.mpk["pk"][nid]),
            u=vector_to_digits(self.msk["u"][nid]),
        ).to_dict()

    def get_decryption_keys(self, sid: str, **kwargs) -> dict | None:
        """Derive a functional decryption key for the given fusion weights.

        Args:
            sid: Session identifier.
            **kwargs: Must include ``credentials`` with ``fusion_weight``
                (dict mapping client IDs to weight lists).

        Returns:
            Dict with ``sk_y`` (per-client projected secret keys).

        Raises:
            FEKeyError: If credentials are missing.
            FEValidationError: If fusion weights are invalid or exceed ``bound_y``.
        """
        credentials = kwargs.get("credentials", None)
        if not credentials:
            raise FEKeyError("need credentials for MIFE LWE decryption key generation")
        fusion_weight = credentials.get("fusion_weight")
        if not isinstance(fusion_weight, dict):
            raise FEValidationError("invalid fusion weights provided, need a dict")

        sk_y = {}
        z = gp.mpz(0)
        for nid in self.lst_nid:
            if nid not in fusion_weight:
                raise FEValidationError("fusion weight missing client {}".format(nid))
            y = fusion_weight[nid]
            if len(y) != self.eta:
                raise FEValidationError("invalid fusion weight length for client {}".format(nid))
            if any(abs(strict_int(v, "fusion weight")) > self.bound_y for v in y):
                raise FEValidationError("fusion weight exceeds configured bound_y")
            y_vec = [gp.mpz(v) for v in y]
            sk_y[nid] = vector_to_digits(
                matvec_mod(self.msk["sk"][nid], y_vec, self.q)
            )
            z += sum(u * y_j for u, y_j in zip(self.msk["u"][nid], y_vec))
        return MIFELWEFunctionalKey.new(sk_y=sk_y, z=gp.digits(z % self.q)).to_dict()


class MIFELWE(IPFEAbsCrypto, ABDP15CryptoMixin):
    """Crypto operations for LWE-based multi-input inner-product FE."""

    def __init__(self, config: dict, **kwargs) -> None:
        """Initialise the MIFE LWE crypto system.

        Args:
            config: Must contain ``keys`` with ``pp`` (public parameters)
                and optionally ``sk`` (encryption keys); optional
                ``matrix_mode`` (``"memory"`` or ``"stream"``).
        """
        super().__init__(config, **kwargs)
        self.pp = self.keys["pp"]
        if self._has_private_keys():
            self.sk = self.keys["sk"]
        self._init_abdp15(config)

    def encrypt(self, lst_pt: list) -> dict:
        """Encrypt a plaintext vector for one client.

        Args:
            lst_pt: Integer plaintext vector of length ``eta``.

        Returns:
            Dict with ``ct0`` and ``ct1`` (serialised LWE ciphertext vectors).

        Raises:
            FEKeyError: If public parameters or encryption keys are missing.
            FEValidationError: If plaintext length or values are invalid.
        """
        if not self._has_public_parameters():
            raise FEKeyError("no public parameters provided for encryption")
        if not self._has_private_keys():
            raise FEKeyError("no encryption key provided")
        if len(lst_pt) != self.pp["eta"]:
            raise FEValidationError("invalid size of input plaintext")
        if any(abs(strict_int(v, "plaintext")) > gp.mpz(self.pp["bound_x"]) for v in lst_pt):
            raise FEValidationError("plaintext exceeds configured bound_x")
        return self._encrypt_batch([lst_pt])[0]

    def _encrypt_batch(self, plaintexts: list) -> list:
        """Ciphertexts ``(A^T r, pk^T r + Delta (x + u))`` of validated plaintexts."""
        if "u" not in MIFELWEEncryptionKey(self.sk):
            raise FEKeyError("encryption key lacks the one-time pad u")
        q = gp.mpz(self.pp["q"])
        delta = q // gp.mpz(self.pp["p"])
        pad = [gp.mpz(u) for u in MIFELWEEncryptionKey(self.sk).u]
        ct0s, masks = self._masks(len(plaintexts))
        out = []
        for x, ct0, mask in zip(plaintexts, ct0s, masks):
            w = [(gp.mpz(int(x_j)) + u) % q for x_j, u in zip(x, pad)]
            out.append(LWECiphertext.new(
                ct0=vector_to_digits([gp.mpz(v) for v in ct0]),
                ct1=vector_to_digits([(gp.mpz(v) + delta * w_j) % q for v, w_j in zip(mask, w)]),
            ).to_dict())
        return out

    def decrypt(self, dct_ct: dict, dk: dict, fusion_weight: dict):
        """Decrypt aggregated ciphertexts to recover the multi-input inner product.

        Args:
            dct_ct: Mapping of client IDs to their ciphertext dicts.
            dk: Decryption key dict with ``sk_y`` (per-client).
            fusion_weight: Mapping of client IDs to weight lists.

        Returns:
            The inner product as an integer.

        Raises:
            FEKeyError: If decryption key is missing.
            FEValidationError: If fusion weight length or values are invalid.
        """
        if not dk or "z" not in MIFELWEFunctionalKey(dk):
            raise FEKeyError("no decryption key provided")
        key = MIFELWEFunctionalKey(dk)
        if set(dct_ct) != set(key.sk_y) or set(dct_ct) != set(fusion_weight):
            raise FESchemeError("need one ciphertext for every slot of the key")

        q = gp.mpz(self.pp["q"])
        p = gp.mpz(self.pp["p"])
        delta = q // p
        total = -delta * gp.mpz(key.z)
        for nid in dct_ct:
            y = fusion_weight[nid]
            if len(y) != self.pp["eta"]:
                raise FEValidationError("invalid fusion weight length for client {}".format(nid))
            if any(abs(strict_int(v, "fusion weight")) > gp.mpz(self.pp["bound_y"]) for v in y):
                raise FEValidationError("fusion weight exceeds configured bound_y")

            y_vec = [gp.mpz(v) for v in y]
            ct0 = digits_to_vector(LWECiphertext(dct_ct[nid]).ct0)
            ct1 = digits_to_vector(LWECiphertext(dct_ct[nid]).ct1)
            sk_y = digits_to_vector(key.sk_y[nid])

            total += vecdot_mod(y_vec, ct1, q) - vecdot_mod(ct0, sk_y, q)
        bound = self.pp["n"] * self.pp["eta"] * gp.mpz(self.pp["bound_x"]) * gp.mpz(self.pp["bound_y"])
        return decode_scaled(total % q, delta, q, bound)

    def encrypt_lst_ndarray(self, lst_ndarray: list, **kwargs) -> list | None:
        """Encrypt a list of ndarrays element-wise (requires ``eta=1``).

        Args:
            lst_ndarray: List of numpy arrays to encrypt.

        Returns:
            List of object ndarrays containing per-element ciphertexts.
        """
        if self.pp["eta"] != 1:
            raise NotImplementedError("ndarray helper currently expects eta=1")
        if not self._has_private_keys():
            raise FEKeyError("no encryption key provided")
        lst_ndarray_ct = []
        for ary_src in lst_ndarray:
            ary = scale_to_int(ary_src, self.precision)
            if np.abs(ary).max(initial=0) > int(self.pp["bound_x"]):
                raise FEValidationError("plaintext exceeds configured bound_x")
            flat = [[int(w)] for w in ary.ravel()]
            cts = []
            for start in range(0, len(flat), ENCRYPT_BATCH):
                cts.extend(self._encrypt_batch(flat[start:start + ENCRYPT_BATCH]))
            ary_ct = np.empty(len(cts), dtype=object)
            ary_ct[:] = cts
            lst_ndarray_ct.append(ary_ct.reshape(ary.shape))
        return lst_ndarray_ct

    def compute_lst_ndarray_ct(self, dict_ndarray_ct: dict, **kwargs) -> list | None:
        """Aggregate ndarray ciphertexts (delegates to :meth:`decrypt_lst_ndarray_ct`).

        Args:
            dict_ndarray_ct: Mapping of client IDs to ciphertext ndarray lists.
            **kwargs: Must include ``dk`` and ``fusion_weight``.

        Returns:
            Decrypted ndarray list.
        """
        dk = kwargs.get("dk", None)
        fusion_weight = kwargs.get("fusion_weight", None)
        if dk is None or fusion_weight is None:
            raise FEKeyError("need to provide decryption key and fusion weight")
        return self.decrypt_lst_ndarray_ct(
            dict_ndarray_ct, dk=dk, fusion_weight=fusion_weight
        )

    def decrypt_lst_ndarray_ct(self, dict_ndarray_ct: dict, **kwargs) -> list | None:
        """Decrypt ndarray ciphertexts from multiple clients.

        Args:
            dict_ndarray_ct: Mapping of client IDs to ciphertext ndarray lists.
            **kwargs: Must include ``dk`` and ``fusion_weight``.

        Returns:
            List of float ndarrays with decrypted values.
        """
        dk = kwargs.get("dk", None)
        fusion_weight = kwargs.get("fusion_weight", None)
        if self.pp["eta"] != 1:
            raise NotImplementedError("ndarray helper currently expects eta=1")
        if dk is None or fusion_weight is None:
            raise FEKeyError("need to provide decryption key and fusion weight")

        sample = next(iter(dict_ndarray_ct.values()))
        lst_ndarray = []
        for l in range(len(sample)):
            ary = sample[l]
            ary_dec = np.empty(ary.shape, dtype=object)
            for i, _ in np.ndenumerate(ary):
                dct_ct = {nid: dict_ndarray_ct[nid][l][i] for nid in dict_ndarray_ct}
                ary_dec[i] = self.decrypt(dct_ct, dk, fusion_weight)
            lst_ndarray.append(decoded_layer(ary_dec, l, 10 ** self.precision))
        return lst_ndarray
