"""
LWE-based multi-client inner-product FE.

| Abdalla, Bourse, Marival, Pointcheval, Soleimanian, Waldner:
| "Multi-Client Inner-Product Functional Encryption in the Random-Oracle Model"
| SCN 2020, https://eprint.iacr.org/2020/788 -- Fig. 4 (MCFE based on LWE),
| correctness and security in Appendix C.2, parameters in Appendix C.3
| (one-IND secure in the ROM under LWE). The construction is the MCFE
| counterpart of the LWE IPFE of Agrawal, Libert, Stehle (CRYPTO 2016).

* type:     secret-key multi-client encryption (MCFE, LWE, random oracle)
* setting:  Integer based (Z_q with a rounding map to Z_q0)

Construction (one coordinate j of client i's vector is one slot; ``n`` is the
total number of slots, ``Delta = floor(q / K)``):

* Setup:    ``H: labels -> Z_q^{n0 + m0}`` full-domain hash.
* KeyGen:   ``Z_{i,j} = (s_{i,j}, t_{i,j})`` with ``s <- U(Z_q^{n0})`` and
            ``t <- D_{Z^{m0}, sigma_t}`` (realised as a rounded continuous
            Gaussian; the statistical step of the proof, Lemma 34, is stated
            for a true discrete Gaussian).
* Encrypt:  ``c_{i,j} = round_q0( Z_{i,j} . H(l) + Delta * x_{i,j} )``.
* KeyDer:   ``sk_y = sum y_{i,j} Z_{i,j}`` (one aggregated vector, kept mod q).
* Decrypt:  ``mu' = sum y_{i,j} c_{i,j} - round_q0( sk_y . H(l) ) mod q0``;
            output the ``mu`` closest to ``mu' q / (q0 Delta)``.

Deviation from the paper: messages and weights are signed (``|x| <= X``,
``|y| <= Y``) instead of ``{0..P-1}`` / ``{0..V-1}``; the inner product is
encoded with ``K = 2 n X Y + 1`` and decoded centred. Parameters are derived
by :func:`pyfe4ai.utils.lwe_utils.abmpsw20_lwe_parameters` from the
constraints of Appendix C.3; this module uses the selective-security choice
of ``m0`` and :mod:`pyfe4ai.schemes.mcfe.fullysec_lwe` the adaptive one.
The LWE dimension ``lwe_n`` defaults to a toy value; no concrete hardness
estimate is made.

Each client slot must encrypt at most once per label (one-IND security):
two ciphertexts of one slot under one label differ by about
``(q0 / q) Delta (x - x')``. The ndarray helpers therefore derive a distinct
label per element. ``H`` is SHAKE-256 (the paper's implementation uses
counter-mode SHA-256).
"""

from __future__ import annotations

import logging

import gmpy2 as gp
import numpy as np

from pyfe4ai.schemes.ipfe import IPFEAbsCrypto
from pyfe4ai.schemes.ipfe import IPFEAbsKeyGenerator
from pyfe4ai.schemes.mcfe.labels import element_label as _element_label
from pyfe4ai.schemes.mcfe.labels import encode_label
from pyfe4ai.utils.crypto_constants import CryptoCONST
from pyfe4ai.utils.crypto_utils import hash_to_zq_vector
from pyfe4ai.utils.lwe_utils import abmpsw20_lwe_parameters
from pyfe4ai.utils.lwe_utils import decode_rounded_inner_product
from pyfe4ai.utils.lwe_utils import round_to_q0
from pyfe4ai.utils.exceptions import FEKeyError, FESchemeError, FEValidationError
from pyfe4ai.utils.sampling_utils import discrete_gaussian_matrix
from pyfe4ai.utils.sampling_utils import rand_uniform_vector

logger = logging.getLogger(__name__)

_LABEL_HASH_DOMAIN = "pyfe4ai/mcfe/lwe/H(label)"
_DEFAULT_STAT_PARAM = 64


def hash_label(label, q, length: int) -> list:
    """Return ``H(label) in Z_q^length`` (random oracle)."""
    return hash_to_zq_vector(encode_label(label), q, length, domain=_LABEL_HASH_DOMAIN)


def dot_mod(left: list, right: list, q) -> gp.mpz:
    """Return ``<left, right> mod q`` for integer vectors."""
    acc = gp.mpz(0)
    for a, b in zip(left, right):
        acc += gp.mpz(a) * b
    return acc % q


class MCFELWEKeyGenerator(IPFEAbsKeyGenerator):
    """Key generator for LWE-based multi-client inner-product FE (ABMPSW20 Fig. 4)."""

    _scheme_type = CryptoCONST.TYPE_MCFE_LWE
    _security = "selective"

    def __init__(self, config: dict, **kwargs) -> None:
        """Initialise the key generator and derive the parameters.

        Args:
            config: Scheme configuration. Recognised keys: ``n`` (number of
                clients), ``lst_nid``, ``eta`` (slots per client), ``lwe_n``
                (LWE dimension ``n0``), ``bound_x``, ``bound_y``,
                ``stat_param`` (statistical parameter ``kappa``) and
                ``sec_param`` (recorded only). The keys ``bound_u`` and
                ``label_modulus`` of versions <= 0.2.0 are ignored.
        """
        super().__init__(config, **kwargs)
        self.eta = config.get("eta", CryptoCONST.MCFE_ETA)
        self.lwe_n = int(config.get("lwe_n", 32))
        self.bound_x = int(config.get("bound_x", 20))
        self.bound_y = int(config.get("bound_y", 20))
        self.stat_param = int(config.get("stat_param", _DEFAULT_STAT_PARAM))
        if not self.n:
            raise FEKeyError("need to provide number of clients")
        if not isinstance(self.eta, int) or self.eta < 1:
            raise FEValidationError("invalid parameter `eta`:{}".format(self.eta))
        self.n_slots = self.n * self.eta
        self.params = abmpsw20_lwe_parameters(
            self.n_slots, self.bound_x, self.bound_y, self.lwe_n,
            self.stat_param, security=self._security,
        )
        self.q = gp.mpz(self.params["q"])
        self.dim = self.params["n0"] + self.params["m0"]

    def setup(self) -> None:
        """Sample ``Z_{i,j} = (s, t)`` for every client slot."""
        n0, m0 = self.params["n0"], self.params["m0"]
        sigma_t = self.params["sigma_t"]
        self.msk = {"z": {}}
        for nid in self.lst_nid:
            t_rows = discrete_gaussian_matrix(self.eta, m0, sigma_t)
            self.msk["z"][nid] = [
                rand_uniform_vector(n0, self.q) + t_rows[j] for j in range(self.eta)
            ]
        logger.info("MCFE LWE (ABMPSW20, %s) setup successfully", self._security)

    def get_public_parameters(self) -> dict:
        """Return serialisable public parameters."""
        return {
            "q": gp.digits(self.q),
            "q0": self.params["q0"],
            "K": self.params["K"],
            "delta": gp.digits(gp.mpz(self.params["delta"])),
            "bound_ip": self.params["bound_ip"],
            "n0": self.params["n0"],
            "m0": self.params["m0"],
            "security": self._security,
            "eta": self.eta,
            "n": self.n,
            "lwe_n": self.lwe_n,
            "bound_x": self.bound_x,
            "bound_y": self.bound_y,
            "stat_param": self.stat_param,
            "sec_param": self.sec_param,
        }

    def get_private_keys(self, nid: str, **kwargs) -> dict | None:
        """Return the encryption key ``{z}`` of client *nid* (``None`` if unknown)."""
        if nid not in self.msk["z"]:
            return None
        return {"z": [[gp.digits(v) for v in z_j] for z_j in self.msk["z"][nid]]}

    def _aggregate_key(self, fusion_weight: dict) -> list:
        """Return ``sum y_{i,j} Z_{i,j} mod q`` after validating the weights."""
        if not isinstance(fusion_weight, dict) or not fusion_weight:
            raise FEValidationError("invalid fusion weights provided, need a dict")
        sk_y = [gp.mpz(0)] * self.dim
        for nid, weights in fusion_weight.items():
            if nid not in self.msk["z"]:
                raise FEValidationError("unknown client {} in fusion weight".format(nid))
            if len(weights) > self.eta:
                raise FEValidationError("invalid fusion weight length for client {}".format(nid))
            for z_j, y_j in zip(self.msk["z"][nid], weights):
                if abs(int(y_j)) > self.bound_y:
                    raise FEValidationError("fusion weight exceeds configured bound_y")
                y_j = gp.mpz(int(y_j))
                sk_y = [acc + y_j * z for acc, z in zip(sk_y, z_j)]
        return [v % self.q for v in sk_y]

    def get_decryption_keys(self, sid: str, **kwargs) -> dict | None:
        """Derive the functional key ``sk_y = sum y_{i,j} Z_{i,j}``.

        Args:
            sid: Session identifier (unused).
            **kwargs: Must include ``credentials`` with ``fusion_weight``
                (dict mapping client IDs to integer weight lists).

        Returns:
            Dict with ``sk_y``, one vector aggregated over all clients.

        Raises:
            FEKeyError: If credentials are missing.
            FEValidationError: If a client id is unknown or a weight is
                malformed or exceeds ``bound_y``.
        """
        credentials = kwargs.get("credentials", None)
        if not credentials:
            raise FEKeyError("need credentials for MCFE LWE decryption key generation")
        sk_y = self._aggregate_key(credentials.get("fusion_weight"))
        return {"sk_y": [gp.digits(v) for v in sk_y]}


class MCFELWE(IPFEAbsCrypto):
    """Crypto operations for LWE-based multi-client inner-product FE (ABMPSW20 Fig. 4)."""

    def __init__(self, config: dict, **kwargs) -> None:
        """Initialise with ``keys.pp`` and, for clients, ``keys.sk``."""
        super().__init__(config, **kwargs)
        self.pp = self.keys["pp"]
        if self._has_private_keys():
            self.sk = self.keys["sk"]
        self.q = gp.mpz(self.pp["q"])
        self.q0 = gp.mpz(self.pp["q0"])
        self.delta = gp.mpz(self.pp["delta"])
        self.dim = self.pp["n0"] + self.pp["m0"]

    def _hash_label(self, label: str) -> list:
        return hash_label(label, self.q, self.dim)

    def encrypt(self, lst_pt: list, label: str) -> dict:
        """Encrypt one client's integer vector under a label.

        Args:
            lst_pt: Integer plaintext vector of length at most ``eta``.
            label: Encryption label; a client must not encrypt twice under
                the same label.

        Returns:
            Dict with ``c`` (one element of ``Z_q0`` per slot).

        Raises:
            FEKeyError: If public parameters or encryption keys are missing.
            FEValidationError: If the plaintext is malformed or out of bound.
        """
        if not self._has_public_parameters():
            raise FEKeyError("no public parameters provided for encryption")
        if not self._has_private_keys():
            raise FEKeyError("no encryption key provided")
        if len(lst_pt) > len(self.sk["z"]):
            raise FEValidationError("invalid size of input plaintext")
        if any(abs(int(v)) > self.pp["bound_x"] for v in lst_pt):
            raise FEValidationError("plaintext exceeds configured bound_x")

        h_l = self._hash_label(label)
        c = []
        for x_j, z_j in zip(lst_pt, self.sk["z"]):
            v = dot_mod([gp.mpz(z) for z in z_j], h_l, self.q) + self.delta * int(x_j)
            c.append(int(round_to_q0(v, self.q, self.q0)))
        return {"c": c}

    def _weighted_sum(self, dct_ct: dict, fusion_weight: dict) -> gp.mpz:
        """Return ``sum y_{i,j} c_{i,j} mod q0`` after checking the inputs match."""
        if dct_ct.keys() != fusion_weight.keys():
            raise FESchemeError("inconsistent input among ct and fusion weight")
        acc = gp.mpz(0)
        for nid, ct in dct_ct.items():
            if len(ct["c"]) != len(fusion_weight[nid]):
                raise FESchemeError(
                    "ciphertext and fusion weight of {} differ in length".format(nid)
                )
            for c_j, y_j in zip(ct["c"], fusion_weight[nid]):
                acc += gp.mpz(int(c_j)) * int(y_j)
        return acc % self.q0

    def _decode(self, weighted_sum, key_term) -> int | None:
        """Decode ``mu' = weighted_sum - round_q0(key_term)``."""
        mu_prime = (weighted_sum - round_to_q0(key_term, self.q, self.q0)) % self.q0
        return decode_rounded_inner_product(
            mu_prime, self.q, self.q0, self.delta, self.pp["bound_ip"]
        )

    def decrypt(self, dct_ct: dict, dk: dict, fusion_weight: dict, label: str):
        """Recover ``sum_i <x_i, y_i>`` from all clients' ciphertexts under one label.

        Args:
            dct_ct: Mapping of client IDs to ciphertext dicts.
            dk: Functional key ``{sk_y}`` derived for *fusion_weight*.
            fusion_weight: Mapping of client IDs to weight lists.
            label: The encryption label.

        Returns:
            The inner product, or ``None`` if the decoded value is outside
            ``[-n X Y, n X Y]``. A wrong label or key yields an arbitrary
            value in that range rather than an error.

        Raises:
            FEKeyError: If the key is missing.
            FESchemeError: If ciphertexts and weights do not match.
        """
        if not dk or "sk_y" not in dk:
            raise FEKeyError("no decryption key provided")
        weighted = self._weighted_sum(dct_ct, fusion_weight)
        key_term = dot_mod([gp.mpz(v) for v in dk["sk_y"]], self._hash_label(label), self.q)
        return self._decode(weighted, key_term)

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
        lst_ndarray_ct = []
        for l, ary_src in enumerate(lst_ndarray):
            ary = (ary_src.copy() * pow(10, self.precision)).astype(int)
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
        lst_ndarray = []
        for l in range(len(sample)):
            ary_dec = np.empty(sample[l].shape, dtype=object)
            for i, _ in np.ndenumerate(sample[l]):
                dct_ct = {nid: dict_ndarray_ct[nid][l][i] for nid in dict_ndarray_ct}
                ary_dec[i] = self.decrypt(
                    dct_ct, dk, fusion_weight, _element_label(label, l, i)
                )
            lst_ndarray.append((ary_dec / pow(10, self.precision)).astype(float))
        return lst_ndarray
