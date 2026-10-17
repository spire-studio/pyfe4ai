"""
MDDH-based multi-client inner-product FE (Damgard-style DDH family).

| Abdalla, Bourse, Marival, Pointcheval, Soleimanian, Waldner:
| "Multi-Client Inner-Product Functional Encryption in the Random-Oracle Model"
| SCN 2020, https://eprint.iacr.org/2020/788 -- Fig. 2 (MCFE based on the
| D_k-MDDH assumption), proof in Appendix A (one-IND secure in the ROM).

* type:     secret-key multi-client encryption (MCFE, MDDH, random oracle)
* setting:  Integer based (order-q subgroup of Z_p^*, p = 2q + 1)

Construction (one coordinate j of client i's vector is one slot; ``[a]``
denotes ``g^a``):

* Setup:    ``H: labels -> G^{k+1}``.
* KeyGen:   ``S_{i,j} <- Z_q^{k+1}`` per slot.
* Encrypt:  ``[u_l] = H(l)``;  ``c_{i,j} = [S_{i,j} . u_l + x_{i,j}]``.
* KeyDer:   ``sk_y = sum y_{i,j} S_{i,j} in Z_q^{k+1}`` (one aggregated key).
* Decrypt:  ``[<x, y>] = prod c_{i,j}^{y_{i,j}} / [u_l . sk_y]``, then a
  bounded discrete log.

``k`` is the config key ``mddh_k`` (default 1). For ``k = 1`` (DDH) this is
exactly the CDGPP18 MCFE of :mod:`pyfe4ai.schemes.mcfe.ddh`, and it is also
what ABMPSW20's approach (replace the IPFE randomness ``(g^r, h^r)`` by
``H(l)``) yields from the Damgard-style DDH IPFE of Agrawal, Libert, Stehle
(CRYPTO 2016), which is why the registry keeps the name ``damgard`` for it.
``k >= 2`` gives the k-Lin instantiation (a weaker assumption, ``k + 1``
group elements per label hash and key).

Each client slot must encrypt at most once per label (one-IND security); the
ndarray helpers derive a distinct label per element. ``H`` hashes into the
quadratic-residue subgroup with SHAKE-256.
"""

from __future__ import annotations

import json
import logging
import os

import gmpy2 as gp
import numpy as np

from pyfe4ai.schemes.ipfe import IPFEAbsCrypto
from pyfe4ai.schemes.ipfe import decoded_layer
from pyfe4ai.schemes.ipfe import IPFEAbsKeyGenerator
from pyfe4ai.schemes.ipfe import ParameterCacheMixin
from pyfe4ai.schemes.mcfe.labels import element_label as _element_label
from pyfe4ai.schemes.mcfe.labels import encode_label
from pyfe4ai.schemes.mcfe.offline import OfflineMCFEMixin
from pyfe4ai.utils.crypto_constants import CryptoCONST
from pyfe4ai.utils.crypto_utils import group_generator_fe
from pyfe4ai.utils.crypto_utils import hash_to_qr_group
from pyfe4ai.utils.dlog_solver import dlog_table_solve
from pyfe4ai.utils.dlog_solver import load_or_build_dlog_table
from pyfe4ai.utils.exceptions import FEKeyError, FESchemeError, FEValidationError
from pyfe4ai.utils.modular_utils import pow_signed
from pyfe4ai.utils.sampling_utils import rand_uniform_vector
from pyfe4ai.utils.validation import scale_to_int
from pyfe4ai.utils.crypto_utils import is_dl_group
from pyfe4ai.utils.validation import strict_int

logger = logging.getLogger(__name__)

_LABEL_HASH_DOMAIN = "pyfe4ai/mcfe/mddh/H(label)"
_CONSTRUCTION = "abmpsw20-mddh/v1"


def _hash_label(label, p: gp.mpz, k: int) -> list:
    """Return ``[u_l] = H(label) in G^{k+1}`` (random oracle into the QR subgroup)."""
    encoded = encode_label(label)
    return [
        hash_to_qr_group(encoded, p, index=m, domain=_LABEL_HASH_DOMAIN) for m in range(k + 1)
    ]


class MCFEDamgardKeyGenerator(IPFEAbsKeyGenerator, ParameterCacheMixin):
    """Key generator for MDDH-based multi-client inner-product FE (ABMPSW20 Fig. 2)."""

    _scheme_type = CryptoCONST.TYPE_MCFE_DAMGARD

    def __init__(self, config: dict, **kwargs) -> None:
        """Initialise the key generator.

        Args:
            config: Scheme configuration. Recognised keys: ``n`` (number of
                clients), ``lst_nid``, ``eta`` (slots per client),
                ``modulus_length`` (bits of ``p``, default ``sec_param``),
                ``bound`` (bound on ``|x|`` and ``|y|``) and ``mddh_k``.
        """
        super().__init__(config, **kwargs)
        self.eta = config.get("eta", CryptoCONST.MCFE_ETA)
        self.modulus_length = config.get("modulus_length", self.sec_param)
        self.bound = config.get("bound", 100)
        self.mddh_k = int(config.get("mddh_k", 1))
        if not self.n:
            raise FEKeyError("need to provide number of clients")
        if not isinstance(self.eta, int) or self.eta < 1:
            raise FEValidationError("invalid parameter `eta`:{}".format(self.eta))
        if self.mddh_k < 1:
            raise FEValidationError("mddh_k must be >= 1")
        self._load_parameters()
        self._validate_bounds()

    def _apply_parameters(self, param: dict) -> None:
        self.p = gp.mpz(param["p"])
        self.q = gp.mpz(param["q"])
        self.g = gp.mpz(param["g"])

    def _param_verification(self, param: dict) -> bool:
        return (
            param.get("construction") == _CONSTRUCTION
            and param.get("modulus_length") == self.modulus_length
        )

    def _loaded_parameters_valid(self) -> bool:
        return is_dl_group(self.p, self.q, self.g, self.modulus_length)

    def _generate_and_save(self, param_file: str) -> None:
        self.p, self.q, _, self.g = group_generator_fe(self.modulus_length)
        with open(param_file, "w", encoding="utf-8") as f:
            json.dump(
                {
                    "construction": _CONSTRUCTION,
                    "modulus_length": self.modulus_length,
                    "p": gp.digits(self.p),
                    "q": gp.digits(self.q),
                    "g": gp.digits(self.g),
                },
                f,
            )

    def _validate_bounds(self) -> None:
        if 2 * self.n * self.eta * gp.mpz(self.bound) ** 2 >= self.q:
            raise FEValidationError(
                "2 * n * eta * bound^2 should be smaller than the group order"
            )

    def setup(self) -> None:
        """Sample ``S_{i,j} <- Z_q^{k+1}`` for every client slot."""
        self.msk = {
            "s": {
                nid: [rand_uniform_vector(self.mddh_k + 1, self.q) for _ in range(self.eta)]
                for nid in self.lst_nid
            }
        }
        logger.info("MCFE MDDH (k=%d) setup successfully", self.mddh_k)

    def get_public_parameters(self) -> dict:
        """Return serialisable public parameters."""
        return {
            "p": gp.digits(self.p),
            "q": gp.digits(self.q),
            "g": gp.digits(self.g),
            "mddh_k": self.mddh_k,
            "eta": self.eta,
            "n": self.n,
            "bound": self.bound,
            "sec_param": self.sec_param,
            "modulus_length": self.modulus_length,
        }

    def get_private_keys(self, nid: str, **kwargs) -> dict | None:
        """Return the encryption key ``{s}`` of client *nid* (``None`` if unknown)."""
        if nid not in self.msk["s"]:
            return None
        return {"s": [[gp.digits(v) for v in s_j] for s_j in self.msk["s"][nid]]}

    def get_decryption_keys(self, sid: str, **kwargs) -> dict | None:
        """Derive the functional key ``sk_y = sum y_{i,j} S_{i,j} in Z_q^{k+1}``.

        Args:
            sid: Session identifier (unused).
            **kwargs: Must include ``credentials`` with ``fusion_weight``
                (dict mapping client IDs to integer weight lists).

        Returns:
            Dict with ``d``, one vector aggregated over all clients.

        Raises:
            FEKeyError: If credentials are missing.
            FEValidationError: If a client id is unknown or a weight is
                malformed or exceeds ``bound``.
        """
        credentials = kwargs.get("credentials", None)
        if not credentials:
            raise FEKeyError("need credentials for MCFE Damgard decryption key generation")
        fusion_weight = credentials.get("fusion_weight")
        if not isinstance(fusion_weight, dict) or not fusion_weight:
            raise FEValidationError("invalid fusion weights provided, need a dict")

        d = [gp.mpz(0)] * (self.mddh_k + 1)
        for nid, weights in fusion_weight.items():
            if nid not in self.msk["s"]:
                raise FEValidationError("unknown client {} in fusion weight".format(nid))
            if len(weights) > self.eta:
                raise FEValidationError("invalid fusion weight length for client {}".format(nid))
            for s_j, y_j in zip(self.msk["s"][nid], weights):
                if abs(strict_int(y_j, "fusion weight")) > self.bound:
                    raise FEValidationError("fusion weight exceeds configured bound")
                y_j = gp.mpz(int(y_j))
                d = [acc + y_j * s for acc, s in zip(d, s_j)]
        return {"d": [gp.digits(v % self.q) for v in d]}


class MCFEDamgard(OfflineMCFEMixin, IPFEAbsCrypto):
    """Crypto operations for MDDH-based multi-client inner-product FE (ABMPSW20 Fig. 2)."""

    _online_bound = "bound"  # checked by the online phase before the precomputation is used

    def __init__(self, config: dict, **kwargs) -> None:
        """Initialise with ``keys.pp`` and, for clients, ``keys.sk``."""
        super().__init__(config, **kwargs)
        self.pp = self.keys["pp"]
        if self._has_private_keys():
            self.sk = self.keys["sk"]
        self.dlog_table = None

    def _load_dlog_table(self) -> None:
        _dlog_file = os.path.join(
            self.config_folder, CryptoCONST.TYPE_MCFE_DAMGARD,
            f"dlog_{self.pp['bound']}.json",
        )
        bound = self.pp["n"] * self.pp["eta"] * (self.pp["bound"] ** 2) + 1
        self.dlog_table, self.func_bound, self._step_size, self._giant_step = (
            load_or_build_dlog_table(_dlog_file, self.pp["g"], self.pp["p"], bound)
        )

    def encrypt(self, lst_pt: list, label: str) -> dict:
        """Encrypt one client's integer vector under a label.

        Args:
            lst_pt: Integer plaintext vector of length at most ``eta``.
            label: Encryption label; a client must not encrypt twice under
                the same label.

        Returns:
            Dict with ``c`` (one group element per slot).

        Raises:
            FEKeyError: If public parameters or encryption keys are missing.
            FEValidationError: If the plaintext is malformed or out of bound.
        """
        if not self._has_public_parameters():
            raise FEKeyError("no public parameters provided for encryption")
        if not self._has_private_keys():
            raise FEKeyError("no private keys provided for encryption")
        if len(lst_pt) > len(self.sk["s"]):
            raise FEValidationError("invalid size of input plaintext")
        if any(abs(strict_int(v, "plaintext")) > self.pp["bound"] for v in lst_pt):
            raise FEValidationError("plaintext exceeds configured bound")
        return self._encrypt_masked(lst_pt, self._label_mask(label))

    def _label_mask(self, label: str) -> list:
        """``prod_m H_m(l)^{s_jm} mod p`` for every slot j (offline part, see
        :mod:`pyfe4ai.schemes.mcfe.offline`)."""
        p = gp.mpz(self.pp["p"])
        u_l = _hash_label(label, p, self.pp["mddh_k"])
        masks = []
        for s_j in self.sk["s"]:
            m = gp.mpz(1)
            for u_m, s_jm in zip(u_l, s_j):
                m = m * gp.powmod(u_m, gp.mpz(s_jm), p) % p
            masks.append(m)
        return masks

    def _encrypt_masked(self, lst_pt: list, masks: list) -> dict:
        """``g^{x_j}`` times the label masks: the ciphertext of :meth:`encrypt`."""
        if len(lst_pt) > len(masks):
            raise FEValidationError("invalid size of input plaintext")
        if any(abs(strict_int(v, "plaintext")) > self.pp["bound"] for v in lst_pt):
            raise FEValidationError("plaintext exceeds configured bound")
        p = gp.mpz(self.pp["p"])
        g = gp.mpz(self.pp["g"])
        return {"c": [gp.digits(pow_signed(g, gp.mpz(int(x_j)), p) * m % p) for x_j, m in zip(lst_pt, masks)]}

    def decrypt(self, dct_ct: dict, dk: dict, fusion_weight: dict, label: str, term=None):
        """Recover ``sum_i <x_i, y_i>`` from all clients' ciphertexts under one label.

        Args:
            dct_ct: Mapping of client IDs to ciphertext dicts.
            dk: Functional key ``{d}`` derived for *fusion_weight*.
            fusion_weight: Mapping of client IDs to weight lists.
            label: The encryption label.
            term: ``prod_m H_m(l)^{d_m} mod p`` if already computed
                (:meth:`prepare_key`).

        Returns:
            The inner product, or ``None`` if it is out of the dlog bound
            (e.g. wrong label or key).

        Raises:
            FEKeyError: If the key is missing or malformed.
            FESchemeError: If ciphertexts and weights do not match.
        """
        k = self.pp["mddh_k"]
        if not dk or not isinstance(dk.get("d"), (list, tuple)) or len(dk["d"]) != k + 1:
            raise FEKeyError("malformed MCFE MDDH key: expected d of length k + 1")
        if dct_ct.keys() != fusion_weight.keys():
            raise FESchemeError("inconsistent input among ct and fusion weight")

        p = gp.mpz(self.pp["p"])
        num = gp.mpz(1)
        for nid, ct in dct_ct.items():
            if len(ct["c"]) != len(fusion_weight[nid]):
                raise FESchemeError(
                    "ciphertext and fusion weight of {} differ in length".format(nid)
                )
            for c_j, y_j in zip(ct["c"], fusion_weight[nid]):
                c_j = gp.mpz(c_j)
                if not 0 < c_j < p:
                    raise FEValidationError("ciphertext element of {} out of range".format(nid))
                num = num * pow_signed(c_j, gp.mpz(int(y_j)), p) % p
        den = self._key_term(dk, label) if term is None else term
        return self._solve_dlog(gp.divm(num, den, p))

    def _key_term(self, dk: dict, label: str):
        """``prod_m H_m(l)^{d_m} mod p``, the label part of :meth:`decrypt`."""
        p = gp.mpz(self.pp["p"])
        den = gp.mpz(1)
        for u_m, d_m in zip(_hash_label(label, p, self.pp["mddh_k"]), dk["d"]):
            den = den * gp.powmod(u_m, gp.mpz(d_m), p) % p
        return den

    def _solve_dlog(self, encoded_value) -> int | None:
        if self.dlog_table is None:
            self._load_dlog_table()
        try:
            return dlog_table_solve(
                encoded_value, self.pp["g"], self.pp["p"],
                self.func_bound, self.dlog_table, self._step_size, self._giant_step,
            )
        except (ValueError, RuntimeError):
            logger.error("inner-product is out of bound supported by crypto system")
            return None

    def encrypt_lst_ndarray(self, lst_ndarray: list, **kwargs) -> list | None:
        """Encrypt ndarrays element-wise (``eta = 1``), one derived label per element.

        Args:
            lst_ndarray: List of numpy arrays.
            **kwargs: Must include ``label``.
        """
        if self.pp["eta"] != 1:
            raise NotImplementedError("ndarray helper currently expects eta=1")
        label = kwargs.get("label", None)
        if not isinstance(label, str):
            raise FEValidationError("need a string label for encrypt_lst_ndarray")
        if kwargs.get("precomputed") is not None:
            return self._encrypt_precomputed(lst_ndarray, label, kwargs["precomputed"])
        lst_ndarray_ct = []
        for l, ary_src in enumerate(lst_ndarray):
            ary = scale_to_int(ary_src, self.precision)
            ary_ct = np.empty(ary.shape, dtype=object)
            for i, w in np.ndenumerate(ary):
                ary_ct[i] = self.encrypt([int(w)], _element_label(label, l, i))
            lst_ndarray_ct.append(ary_ct)
        return lst_ndarray_ct

    def compute_lst_ndarray_ct(self, dict_ndarray_ct: dict, **kwargs) -> list | None:
        """Aggregate-decrypt ndarray ciphertexts (delegates to :meth:`decrypt_lst_ndarray_ct`).

        Args:
            dict_ndarray_ct: Mapping of client IDs to ciphertext ndarray lists.
            **kwargs: Must include ``dk``, ``fusion_weight`` and ``label``.
        """
        dk = kwargs.get("dk", None)
        fusion_weight = kwargs.get("fusion_weight", None)
        label = kwargs.get("label", None)
        if dk is None or fusion_weight is None or label is None:
            raise FEKeyError("need to provide decryption key, fusion weight and label")
        return self.decrypt_lst_ndarray_ct(
            dict_ndarray_ct, dk=dk, fusion_weight=fusion_weight, label=label
        )

    def decrypt_lst_ndarray_ct(self, dict_ndarray_ct: dict, **kwargs) -> list | None:
        """Decrypt ndarray ciphertexts element-wise.

        Args:
            dict_ndarray_ct: Mapping of client IDs to ciphertext ndarray lists.
            **kwargs: Must include ``dk``, ``fusion_weight`` and ``label``.
        """
        if self.pp["eta"] != 1:
            raise NotImplementedError("ndarray helper currently expects eta=1")
        dk = kwargs.get("dk", None)
        fusion_weight = kwargs.get("fusion_weight", None)
        label = kwargs.get("label", None)
        if dk is None or fusion_weight is None or label is None:
            raise FEKeyError("need to provide decryption key, fusion weight and label")

        sample = next(iter(dict_ndarray_ct.values()))
        terms = self._prepared_terms(dk, label, sample)
        lst_ndarray = []
        for l in range(len(sample)):
            ary_dec = np.empty(sample[l].shape, dtype=object)
            for i, _ in np.ndenumerate(sample[l]):
                dct_ct = {nid: dict_ndarray_ct[nid][l][i] for nid in dict_ndarray_ct}
                ary_dec[i] = self.decrypt(
                    dct_ct, dk, fusion_weight, _element_label(label, l, i),
                    term=None if terms is None else terms[l][i],
                )
            lst_ndarray.append(decoded_layer(ary_dec, l, 10 ** self.precision))
        return lst_ndarray
