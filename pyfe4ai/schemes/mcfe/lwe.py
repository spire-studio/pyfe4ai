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
  ``t <- D_{Z^{m0}, sigma_t}``, sampled exactly by default
  (``gaussian_sampler = "facct"``, recorded in the public parameters), as
  the statistical step of the proof (Lemma 34) assumes; ``"rounded"`` (a
  rounded continuous Gaussian, 2^-25..2^-15 from it per key in the October
  2026 crypto review) stays selectable.
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
The LWE dimension ``lwe_n`` defaults to a toy value; ``lwe_n = "auto"``
(the ``standard`` security profile) selects the smallest dimension whose
``q`` meets :data:`pyfe4ai.utils.lwe_utils.DENSE_LWE_128_LOG_Q`, a lattice
estimator bound for LWE(n0, q, ceil(2 sqrt(n0))) -- the assumption of the
proof (Appendix C.2, game G2: LWE in dimension ``n0`` modulo the scheme's
``q``) -- with a uniform secret.

Each client slot must encrypt at most once per label (one-IND security):
two ciphertexts of one slot under one label differ by about
``(q0 / q) Delta (x - x')``. The ndarray helpers therefore derive a distinct
label per element. ``H`` is SHAKE-256 (the paper's implementation uses
counter-mode SHA-256), mapped to ``Z_q`` as the key generator's
``label_hash`` says (a public parameter;
:data:`~pyfe4ai.utils.crypto_utils.LABEL_HASHES`): by default rejection
sampling mod q from SHAKE128 (exactly uniform, fewer XOF bytes). Versions
<= 0.3.0 reduced ``bits(q) + 128``-bit SHAKE-256 blocks (``"shake256-wide"``,
still selectable), so their ciphertexts differ. The mode is bound to the
public parameters, the clients' encryption keys, every single ciphertext and
every ndarray layer (``label_hash``: the code of
:data:`~pyfe4ai.utils.crypto_utils.LABEL_HASH_CODES`, absent for
``"shake256-wide"``; in an ndarray layer only its first element carries it,
for the whole layer); a missing field means ``"shake256-wide"``
(the data of versions <= 0.3.0 stay readable), and encryption with a key of
another mode or decryption of a ciphertext of another mode raises instead of
returning a wrong sum. Functional keys ``sum y Z`` do not depend on the hash.
"""

from __future__ import annotations

import logging
import operator

import gmpy2 as gp
import numpy as np

from pyfe4ai.schemes.ipfe import IPFEAbsCrypto
from pyfe4ai.schemes.ipfe import decoded_layer
from pyfe4ai.schemes.ipfe import IPFEAbsKeyGenerator
from pyfe4ai.schemes.mcfe.labels import element_label as _element_label
from pyfe4ai.schemes.mcfe.labels import encode_label
from pyfe4ai.schemes.profiles import STANDARD
from pyfe4ai.utils import native_backend
from pyfe4ai.utils.crypto_constants import CryptoCONST
from pyfe4ai.utils.crypto_utils import DEFAULT_LABEL_HASH
from pyfe4ai.utils.crypto_utils import LABEL_HASH_CODES
from pyfe4ai.utils.crypto_utils import LEGACY_LABEL_HASH
from pyfe4ai.utils.crypto_utils import check_label_hash
from pyfe4ai.utils.crypto_utils import hash_to_zq_material
from pyfe4ai.utils.crypto_utils import hash_to_zq_vector
from pyfe4ai.utils.crypto_utils import zq_rejection_parameters
from pyfe4ai.utils.lwe_utils import abmpsw20_lwe_parameters
from pyfe4ai.utils.lwe_utils import check_dense_lwe
from pyfe4ai.utils.lwe_utils import decode_rounded_inner_product
from pyfe4ai.utils.lwe_utils import resolve_lwe_n
from pyfe4ai.utils.lwe_utils import round_to_q0
from pyfe4ai.utils.exceptions import FEKeyError, FESchemeError, FEValidationError
from pyfe4ai.utils.sampling_utils import discrete_gaussian_matrix, scheme_sampler
from pyfe4ai.utils.sampling_utils import rand_uniform_vector
from pyfe4ai.utils.validation import scale_to_int
from pyfe4ai.utils.validation import strict_int

logger = logging.getLogger(__name__)

_LABEL_HASH_DOMAIN = "pyfe4ai/mcfe/lwe/H(label)"
_DEFAULT_STAT_PARAM = 64


def hash_label(label, q, length: int, mode: str) -> list:
    """Return ``H(label) in Z_q^length`` (random oracle; *mode*: the
    ``label_hash`` public parameter)."""
    return hash_to_zq_vector(encode_label(label), q, length, domain=_LABEL_HASH_DOMAIN, mode=mode)


def _label_hash(value) -> str:
    try:
        return check_label_hash(value)
    except ValueError as exc:
        raise FEValidationError(str(exc)) from exc


def dot_mod(left: list, right: list, q) -> gp.mpz:
    """Return ``<left, right> mod q`` for integer vectors."""
    return gp.mpz(sum(map(operator.mul, map(gp.mpz, left), right))) % q


def _mpz_cached(owner, attr: str, obj, convert):
    """``convert(obj)``, cached on *owner* while the same *obj* is passed
    (keys are converted from decimal strings once, not per element)."""
    cache = owner.__dict__.get(attr)
    if cache is None or cache[0] is not obj:
        cache = (obj, convert(obj))
        owner.__dict__[attr] = cache
    return cache[1]


class MCFELWEKeyGenerator(IPFEAbsKeyGenerator):
    """Key generator for LWE-based multi-client inner-product FE (ABMPSW20 Fig. 4)."""

    _scheme_type = CryptoCONST.TYPE_MCFE_LWE
    _security = "selective"

    def __init__(self, config: dict, **kwargs) -> None:
        """Initialise the key generator and derive the parameters.

        Args:
            config: Scheme configuration. Recognised keys: ``n`` (number of
                clients), ``lst_nid``, ``eta`` (slots per client), ``lwe_n``
                (LWE dimension ``n0``; default 32, ``"auto"`` for the
                smallest one meeting
                :data:`~pyfe4ai.utils.lwe_utils.DENSE_LWE_128_LOG_Q`, the
                ``standard`` profile's choice), ``bound_x``, ``bound_y``,
                ``stat_param`` (statistical parameter ``kappa``),
                ``sec_param`` (recorded only), ``gaussian_sampler`` (that of
                the keys ``t``: ``"facct"``, the default, or ``"rounded"``;
                see :func:`~pyfe4ai.utils.sampling_utils.scheme_sampler`) and
                ``label_hash`` (one of
                :data:`~pyfe4ai.utils.crypto_utils.LABEL_HASHES`, default
                ``"shake128-rejection"``, ``"shake256-wide"`` for the
                ciphertexts of versions <= 0.3.0; a public parameter).
                The keys ``bound_u`` and ``label_modulus`` of versions <= 0.2.0
                are ignored.
        """
        self.label_hash = _label_hash(config.get("label_hash", DEFAULT_LABEL_HASH))
        super().__init__(config, **kwargs)
        self.gaussian_sampler = scheme_sampler(config, profile=self.security_profile)
        self.eta = config.get("eta", CryptoCONST.MCFE_ETA)
        self.bound_x = int(config.get("bound_x", 20))
        self.bound_y = int(config.get("bound_y", 20))
        self.stat_param = int(config.get("stat_param", _DEFAULT_STAT_PARAM))
        if not self.n:
            raise FEKeyError("need to provide number of clients")
        if not isinstance(self.eta, int) or self.eta < 1:
            raise FEValidationError("invalid parameter `eta`:{}".format(self.eta))
        self.n_slots = self.n * self.eta
        self.lwe_n = resolve_lwe_n(
            config.get("lwe_n"), lambda n0: self._parameters_for(n0)["q"], default=32
        )
        self.params = self._parameters_for(self.lwe_n)
        if self.security_profile == STANDARD:
            check_dense_lwe(self.lwe_n, self.params["q"])
        self.q = gp.mpz(self.params["q"])
        self.dim = self.params["n0"] + self.params["m0"]

    def _parameters_for(self, lwe_n: int) -> dict:
        """Appendix C.3 parameters for LWE dimension *lwe_n* (``q`` under key ``"q"``)."""
        return abmpsw20_lwe_parameters(
            self.n_slots, self.bound_x, self.bound_y, lwe_n, self.stat_param, security=self._security
        )

    def setup(self) -> None:
        """Sample ``Z_{i,j} = (s, t)`` for every client slot."""
        n0, m0 = self.params["n0"], self.params["m0"]
        sigma_t = self.params["sigma_t"]
        self.msk = {"z": {}}
        for nid in self.lst_nid:
            t_rows = discrete_gaussian_matrix(self.eta, m0, sigma_t, sampler=self.gaussian_sampler)
            self.msk["z"][nid] = [
                rand_uniform_vector(n0, self.q) + t_rows[j] for j in range(self.eta)
            ]
        logger.info("MCFE LWE (ABMPSW20, %s) setup successfully", self._security)

    def get_public_parameters(self) -> dict:
        """Return serialisable public parameters."""
        pp = {
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
            "gaussian_sampler": self.gaussian_sampler,
            "security_profile": self.security_profile,
        }
        pp["label_hash"] = self.label_hash
        return pp

    def get_private_keys(self, nid: str, **kwargs) -> dict | None:
        """Return the encryption key ``{z, label_hash}`` of client *nid* (``None`` if unknown)."""
        if nid not in self.msk["z"]:
            return None
        return {"z": [[gp.digits(v) for v in z_j] for z_j in self.msk["z"][nid]], "label_hash": self.label_hash}

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
                if abs(strict_int(y_j, "fusion weight")) > self.bound_y:
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
        self.label_hash = _label_hash(self.pp.get("label_hash", LEGACY_LABEL_HASH))

    def _check_encryption_label_hash(self) -> None:
        """Raise unless the encryption key was issued for the public parameters'
        label hash (keys without ``label_hash``: ``"shake256-wide"``).

        Raises:
            FEKeyError: If they differ: the ciphertexts would decrypt to wrong
                values.
        """
        key_mode = self.sk.get("label_hash", LEGACY_LABEL_HASH)
        if key_mode != self.label_hash:
            raise FEKeyError("the encryption key was issued for label hash {!r}, the public parameters say {!r}"
                             .format(key_mode, self.label_hash))

    def _ciphertext(self, c: list) -> dict:
        """A ciphertext dict, bound to the label hash (no field for ``"shake256-wide"``,
        as in versions <= 0.3.0)."""
        if self.label_hash == LEGACY_LABEL_HASH:
            return {"c": c}
        return {"c": c, "label_hash": LABEL_HASH_CODES[self.label_hash]}

    def _mark_layer(self, ary_ct: np.ndarray) -> np.ndarray:
        """Bind the label hash to an ndarray layer: its first element carries
        the code for the whole layer (none for ``"shake256-wide"``)."""
        if ary_ct.size and self.label_hash != LEGACY_LABEL_HASH:
            first = ary_ct.flat[0]
            ary_ct.flat[0] = dict(first, label_hash=LABEL_HASH_CODES[self.label_hash])
        return ary_ct

    def _layer_codes(self, dict_ndarray_ct: dict) -> dict:
        """``{nid: [code per layer]}`` after checking every client's layers: the
        code its first element carries (none: ``"shake256-wide"``) must be the
        public parameters'.

        Raises:
            FESchemeError: For a layer of another (or an unknown) label hash.
        """
        code = LABEL_HASH_CODES[self.label_hash]
        out = {}
        for nid, layers in dict_ndarray_ct.items():
            codes = []
            for ary in layers if isinstance(layers, (list, tuple)) else ():
                first = ary.flat[0] if isinstance(ary, np.ndarray) and ary.size else None
                got = first.get("label_hash", 0) if isinstance(first, dict) else 0
                if got != code:
                    raise FESchemeError("a ciphertext layer of {} was produced with another label hash than the "
                                        "public parameters' {!r}".format(nid, self.label_hash))
                codes.append(code)
            out[nid] = codes
        return out

    def _hash_label(self, label: str) -> list:
        return hash_label(label, self.q, self.dim, self.label_hash)

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
        self._check_encryption_label_hash()
        if len(lst_pt) > len(self.sk["z"]):
            raise FEValidationError("invalid size of input plaintext")
        if any(abs(strict_int(v, "plaintext")) > self.pp["bound_x"] for v in lst_pt):
            raise FEValidationError("plaintext exceeds configured bound_x")

        h_l = self._hash_label(label)
        c = []
        z = _mpz_cached(self, "_z_mpz", self.sk["z"], lambda rows: [[gp.mpz(v) for v in row] for row in rows])
        for x_j, z_j in zip(lst_pt, z):
            v = dot_mod(z_j, h_l, self.q) + self.delta * int(x_j)
            c.append(int(round_to_q0(v, self.q, self.q0)))
        return self._ciphertext(c)

    def _weighted_sum(self, dct_ct: dict, fusion_weight: dict, in_layer: bool = False) -> gp.mpz:
        """Return ``sum y_{i,j} c_{i,j} mod q0`` after checking the inputs match
        (*in_layer*: elements of ndarray layers checked by :meth:`_layer_codes`,
        which carry no code of their own but the first)."""
        if dct_ct.keys() != fusion_weight.keys():
            raise FESchemeError("inconsistent input among ct and fusion weight")
        code = LABEL_HASH_CODES[self.label_hash]
        absent = code if in_layer else 0  # an element without a code: its layer's; a single ciphertext: v0.3.0's
        acc = gp.mpz(0)
        for nid, ct in dct_ct.items():
            if ct.get("label_hash", absent) != code:  # the hash this ciphertext was made with
                raise FESchemeError("the ciphertext of {} was produced with another label hash than the public "
                                    "parameters' {!r}".format(nid, self.label_hash))
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

    def decrypt(self, dct_ct: dict, dk: dict, fusion_weight: dict, label: str, _in_layer: bool = False):
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
        # _in_layer: elements of ndarray layers, checked by _layer_codes
        weighted = self._weighted_sum(dct_ct, fusion_weight, _in_layer)
        sk_y = _mpz_cached(self, "_sk_y_mpz", dk["sk_y"], lambda row: [gp.mpz(v) for v in row])
        key_term = dot_mod(sk_y, self._hash_label(label), self.q)
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
            ary = scale_to_int(ary_src, self.precision)
            ary_ct = np.empty(ary.shape, dtype=object)
            if ary.size and self._native_dots():
                # encrypt() of every element, the label hashes and dot products in one call
                if not self._has_public_parameters():
                    raise FEKeyError("no public parameters provided for encryption")
                if not self._has_private_keys():
                    raise FEKeyError("no encryption key provided")
                self._check_encryption_label_hash()
                if len(self.sk["z"]) < 1:
                    raise FEValidationError("invalid size of input plaintext")
                if int(np.max(np.abs(ary))) > self.pp["bound_x"]:
                    raise FEValidationError("plaintext exceeds configured bound_x")
                z = _mpz_cached(self, "_z_mpz", self.sk["z"], lambda rows: [[gp.mpz(v) for v in row] for row in rows])
                dots = self._label_dots([_element_label(label, l, i) for i in np.ndindex(ary.shape)], z[0])
                for (i, w), dot in zip(np.ndenumerate(ary), dots):
                    ary_ct[i] = {"c": [int(round_to_q0(dot + self.delta * int(w), self.q, self.q0))]}
            else:
                for i, w in np.ndenumerate(ary):
                    ct = self.encrypt([int(w)], _element_label(label, l, i))
                    ct.pop("label_hash", None)  # the layer carries it once
                    ary_ct[i] = ct
            lst_ndarray_ct.append(self._mark_layer(ary_ct))
        return lst_ndarray_ct

    def _native_dots(self) -> bool:
        return native_backend.active_backend() == "native"

    def _label_dots(self, labels: list, key_row: list) -> list:
        """``dot_mod(key_row, H(l), q)`` for every label *l*.

        With the native backend, one call hashes every label and sums the
        products of *key_row* (reduced mod q) with the hash blocks before
        their reduction mod q -- congruent mod q, so the same values.
        """
        q = self.q
        if not self._native_dots() or len(key_row) != self.dim:
            return [dot_mod(key_row, self._hash_label(lab), q) for lab in labels]
        kl = (int(q).bit_length() + 63) // 64
        key = _mpz_cached(self, "_dense_key", key_row,
                          lambda row: b"".join((int(v) % int(q)).to_bytes(8 * kl, "little") for v in row))
        materials = [hash_to_zq_material(encode_label(lab), domain=_LABEL_HASH_DOMAIN, mode=self.label_hash)
                     for lab in labels]
        if self.label_hash == LEGACY_LABEL_HASH:
            raw = native_backend.ext().dense_dots(
                materials, (int(q).bit_length() + 128 + 7) // 8, self.dim, key, kl, native_backend.get_threads(),
            )
        else:  # the accepted candidates instead of the wide blocks, also unreduced
            nb, limit = zq_rejection_parameters(q)
            raw = native_backend.ext().dense_dots(
                materials, nb, self.dim, key, kl, native_backend.get_threads(),
                limit=limit.to_bytes(nb, "big"), rate=168 if self.label_hash == "shake128-rejection" else 136,
            )
        step = len(raw) // len(labels)
        return [gp.mpz(int.from_bytes(raw[i:i + step], "little")) % q for i in range(0, len(raw), step)]

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
        self._layer_codes(dict_ndarray_ct)

        sample = next(iter(dict_ndarray_ct.values()))
        lst_ndarray = []
        for l in range(len(sample)):
            ary_dec = np.empty(sample[l].shape, dtype=object)
            if np.size(sample[l]) and self._native_dots():
                # decrypt() of every element, the key terms in one call
                if not dk or "sk_y" not in dk:
                    raise FEKeyError("no decryption key provided")
                sk_y = _mpz_cached(self, "_sk_y_mpz", dk["sk_y"], lambda row: [gp.mpz(v) for v in row])
                key_terms = self._label_dots([_element_label(label, l, i) for i in np.ndindex(sample[l].shape)], sk_y)
                for (i, _), key_term in zip(np.ndenumerate(sample[l]), key_terms):
                    dct_ct = {nid: dict_ndarray_ct[nid][l][i] for nid in dict_ndarray_ct}
                    ary_dec[i] = self._decode(self._weighted_sum(dct_ct, fusion_weight, in_layer=True), key_term)
                lst_ndarray.append(decoded_layer(ary_dec, l, 10 ** self.precision))
                continue
            for i, _ in np.ndenumerate(sample[l]):
                dct_ct = {nid: dict_ndarray_ct[nid][l][i] for nid in dict_ndarray_ct}
                ary_dec[i] = self.decrypt(
                    dct_ct, dk, fusion_weight, _element_label(label, l, i), _in_layer=True
                )
            lst_ndarray.append(decoded_layer(ary_dec, l, 10 ** self.precision))
        return lst_ndarray
