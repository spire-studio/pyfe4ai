"""
Ring-LWE multi-input inner-product FE.

| Bermudo Mera, Karmakar, Marc, Soleimanian:
| "Efficient Lattice-Based Inner-Product Functional Encryption", PKC 2022
| (ePrint 2021/046) -- Section 6, "MIFE-Compiler" (the transformation of
| Abdalla, Catalano, Fiore, Gay, Ursu, CRYPTO 2018) applied to their
| Ring-LWE IPFE, which has 2-step decryption with a linear encoding
| (Proposition 1).

* type:     private-key multi-input encryption (MIFE, Ring-LWE)
* setting:  R_q = Z_q[x] / (x^ring_n + 1), Delta = floor(q / p)

Construction (slot ``i`` is client ``i``; every row of the plaintext matrix is
one coordinate, its coefficients a batch of independent values):

* Setup:    public ``a``; per slot ``s_{i,row}`` (Gaussian),
  ``pk_{i,row} = a s_{i,row} + e``, and a one-time pad
  ``u_{i,row} <- R_q`` (uniform coefficients); client ``i`` keeps
  ``(pk_i, u_i)``.
* Encrypt:  ``w = x_i + u_i`` (coefficient-wise mod q);
  ``ct0_row = pk_row r + e_row + Delta w_row``, ``ct1 = a r + e``.
* KeyGen:   per slot ``sk_i = sum_row y_{i,row} s_{i,row}`` and
  ``z = sum_i sum_row y_{i,row} u_{i,row}`` (coefficient-wise mod q).
* Decrypt:  Dec1 per slot ``sum_row y_row ct0_row - ct1 sk_i``;
  Dec2 decodes ``sum_i Dec1_i - Delta z`` coefficient-wise.

The pad hides each slot's partial result; versions <= 0.2.0 had no pad and
decoded every slot separately. Parameters are derived as for a single-input
scheme of dimension ``n eta`` (ACFGU18 Setup*), since only the sum over all
slots is decoded, and with the Gaussian widths of BMMS21 Sec. 4.2 for
messages of norm ``3X``: ACFGU18 Fig. 4 needs an IPFE for the class
``F_{m,3X,Y}`` (footnote 12; the hybrids of Lemma 3 encrypt
``x^j - x^1 + x'^1``), while ``p`` stays sized for ``X``. Versions <= 0.3.0
used ``X``, which meets that condition only for nine or more clients.
Deviation: all slots share ``a`` (secrets and pads are independent per
slot). A decryption needs ciphertexts of every slot named in
the key.
"""

from __future__ import annotations

import json
import secrets
import logging

import gmpy2 as gp
import numpy as np

from pyfe4ai.core.base import IPFEAbsCrypto
from pyfe4ai.core.base import IPFEAbsKeyGenerator
from pyfe4ai.core.base import ParameterCacheMixin
from pyfe4ai.primitives.lattice.lwe_utils import decode_scaled
from pyfe4ai.core.profiles import STANDARD
from pyfe4ai.primitives.lattice.ring_lwe_utils import bmms21_selective_parameters
from pyfe4ai.core.formats.ring_lwe import MIFERingLWEEncryptionKey
from pyfe4ai.core.formats.ring_lwe import MIFERingLWEFunctionalKey
from pyfe4ai.core.formats.ring_lwe import MIFERingLWEParameters
from pyfe4ai.core.formats.ring_lwe import RingLWECiphertext
from pyfe4ai.core.formats.ring_lwe import RingLWERecord
from pyfe4ai.core.versions import CT_FORMATS
from pyfe4ai.core.formats.binary import canonical_ring_ciphertext
from pyfe4ai.primitives.lattice.ring_lwe_utils import check_key_rows
from pyfe4ai.primitives.lattice.ring_lwe_utils import check_ring_lwe
from pyfe4ai.primitives.lattice.ring_lwe_utils import check_ring_modulus
from pyfe4ai.primitives.lattice.ring_lwe_utils import register_rns_modulus
from pyfe4ai.primitives.lattice.ring_lwe_utils import poly_add
from pyfe4ai.primitives.lattice.ring_lwe_utils import poly_mul
from pyfe4ai.primitives.lattice.ring_lwe_utils import poly_neg
from pyfe4ai.primitives.lattice.ring_lwe_utils import resolve_ring_n
from pyfe4ai.primitives.lattice.ring_lwe_utils import RingNTT
from pyfe4ai import backend
from pyfe4ai.backend.interface import ReferenceKey
from pyfe4ai.core.constants import CryptoCONST
from pyfe4ai.primitives.sampling import discrete_gaussian_matrix, scheme_sampler
from pyfe4ai.core.exceptions import FEKeyError, FESchemeError, FEValidationError
from pyfe4ai.core.versions import record_matches
from pyfe4ai.core.versions import RING_LWE_RECORD
from pyfe4ai.primitives.sampling import rand_uniform_vector
from pyfe4ai._validation import int_vector
from pyfe4ai.primitives.lattice.ring_lwe_utils import seeded_polynomial

logger = logging.getLogger(__name__)


class MIFERingLWEKeyGenerator(IPFEAbsKeyGenerator, ParameterCacheMixin):
    """Key generator for Ring-LWE-based multi-input inner-product FE."""

    _scheme_type = CryptoCONST.TYPE_MIFE_RING_LWE
    def __init__(self, config: dict, **kwargs) -> None:
        """Initialise the MIFE Ring-LWE key generator.

        Args:
            config: Scheme configuration. Recognised keys: ``n`` (number of
                clients, required), ``lst_nid``, ``sec_param`` (``kappa`` of
                the noise bounds), ``eta``, ``bound_x``, ``bound_y`` and
                ``ring_n`` (default
                :data:`~pyfe4ai.primitives.lattice.ring_lwe_utils.DEFAULT_RING_N`;
                ``"auto"`` for the smallest dimension meeting the 128-bit
                table, the ``standard`` profile's choice); ``ring_modulus``:
                ``"prime"`` (default) or ``"rns"``, ``q`` the product of two
                NTT primes (opt-in; see docs/rns_modulus_review.md).
        """
        super().__init__(config, **kwargs)
        self.eta = config.get("eta", CryptoCONST.MIFE_ETA)
        self.bound_x = gp.mpz(config.get("bound_x", 4))
        self.bound_y = gp.mpz(config.get("bound_y", 4))
        self.ring_modulus = check_ring_modulus(config.get("ring_modulus", "prime"))
        self.q_factors = None
        if not self.n:
            raise FEKeyError("need to provide number of clients")
        self.ring_n = resolve_ring_n(config.get("ring_n"), self._params_for)
        self.gaussian_sampler = scheme_sampler(config, profile=self.security_profile)
        self._load_parameters()
        if self.security_profile == STANDARD:
            check_ring_lwe(self.ring_n, self.q, self._params_for(self.ring_n)[1])

    def _bmms21(self, ring_n: int) -> dict:
        """BMMS21 Sec. 4.2 parameters for *ring_n*: dimension ``n eta``,
        decoding bound ``X``, widths for messages of norm ``3X`` (ACFGU18
        Fig. 4, see the module docstring)."""
        return bmms21_selective_parameters(
            self.sec_param, self.n * self.eta, self.bound_x, self.bound_y, ring_n,
            security_bound_x=3 * self.bound_x, ring_modulus=self.ring_modulus,
        )

    def _bmms21_tuple(self, ring_n: int) -> tuple:
        """``(p, q, sigma1, sigma2, sigma3)`` of :meth:`_bmms21`."""
        P = self._bmms21(ring_n)
        return P["p"], P["q"], P["sigma1"], P["sigma2"], P["sigma3"]

    def _params_for(self, ring_n: int) -> tuple:
        """``(q, rlwe_width)`` that :meth:`_generate_and_save` derives for *ring_n*
        (the width is that of the RLWE problems of the BMMS21 proof)."""
        return self._modulus_for(ring_n), self._bmms21(ring_n)["rlwe_width"]

    def _modulus_for(self, ring_n: int):
        """Modulus ``q`` that :meth:`_generate_and_save` derives for *ring_n*."""
        return self._bmms21_tuple(ring_n)[1]

    def _factors_for(self, ring_n: int) -> list | None:
        """The factors of the RNS modulus for *ring_n* as decimal strings, or ``None``."""
        factors = self._bmms21(ring_n).get("q_factors")
        return None if factors is None else [gp.digits(f) for f in factors]

    def _param_file_name(self) -> str:
        """``param.json`` for the prime modulus (the records of earlier
        versions), ``param-rns.json`` for ``ring_modulus="rns"``: a folder holds
        one record of each, so alternating the two configurations loads them
        instead of regenerating each other's."""
        return "param.json" if self.ring_modulus == "prime" else "param-{}.json".format(self.ring_modulus)

    def _apply_parameters(self, param: dict) -> None:
        record = RingLWERecord.from_dict(param)
        self.p = gp.mpz(record.p)
        self.q = gp.mpz(record.q)
        if record.get("q_factors") is not None:
            self.q_factors = register_rns_modulus(self.q, record.q_factors)
        self.ring_n = int(record.ring_n)
        self.sigma1 = float(record.sigma1)
        self.sigma2 = float(record.sigma2)
        self.sigma3 = float(record.sigma3)
        self.a_seed = record.A_seed
        self.A = seeded_polynomial(self.a_seed, self.q, self.ring_n)

    def _param_verification(self, param: dict) -> bool:
        """The cached record must match the config and the parameters derived
        from it (``q`` as :meth:`_modulus_for` gives it); only the seed of ``a``
        is random."""
        if not record_matches(param, {
            "construction": RING_LWE_RECORD, "sec_param": self.sec_param, "eta": self.eta, "n": self.n,
            "bound_x": gp.digits(self.bound_x), "bound_y": gp.digits(self.bound_y), "ring_n": self.ring_n,
            "ring_modulus": self.ring_modulus,
        }):
            return False
        p, _, sigma1, sigma2, sigma3 = self._bmms21_tuple(self.ring_n)
        q = self._modulus_for(self.ring_n)
        return record_matches(param, {
            "p": gp.digits(p), "q": gp.digits(q), "sigma1": sigma1, "sigma2": sigma2, "sigma3": sigma3,
            "q_factors": self._factors_for(self.ring_n),
        })

    def _generate_and_save(self, param_file: str) -> None:
        # only the sum over all slots is decoded: size p (and the noise) as
        # for a single-input scheme of dimension n * eta (ACFGU18 Setup*),
        # with the widths for 3X (ACFGU18 Fig. 4)
        self.p, self.q, self.sigma1, self.sigma2, self.sigma3 = self._bmms21_tuple(self.ring_n)
        # the factors of this q (_factors_for: those of the derived modulus, a
        # subclass's flooded one for the threshold MCFE)
        base = self._bmms21(self.ring_n).get("q_factors")
        factors = None if base is None else [gp.digits(f) for f in base]
        if factors is not None:
            self.q_factors = register_rns_modulus(self.q, factors)
        self.a_seed = secrets.token_hex(32)
        self.A = seeded_polynomial(self.a_seed, self.q, self.ring_n)
        record = RingLWERecord.new(
            construction=RING_LWE_RECORD,
            sec_param=self.sec_param,
            eta=self.eta,
            n=self.n,
            ring_n=self.ring_n,
            bound_x=gp.digits(self.bound_x),
            bound_y=gp.digits(self.bound_y),
            p=gp.digits(self.p),
            q=gp.digits(self.q),
            sigma1=self.sigma1,
            sigma2=self.sigma2,
            sigma3=self.sigma3,
            A_seed=self.a_seed,
        )
        if factors is not None:  # the default's records stay as they were
            record.update(ring_modulus=self.ring_modulus, q_factors=factors)
        with open(param_file, "w", encoding="utf-8") as f:
            json.dump(record.to_dict(), f)

    def setup(self) -> None:
        sampler = self.gaussian_sampler
        sk = {
            nid: discrete_gaussian_matrix(self.eta, self.ring_n, self.sigma1, sampler=sampler)
            for nid in self.lst_nid
        }
        pk = {}
        for nid in self.lst_nid:
            noise = discrete_gaussian_matrix(self.eta, self.ring_n, self.sigma1, sampler=sampler)
            pk_nid = []
            for i in range(self.eta):
                pk_i = poly_mul(self.A, sk[nid][i], self.q)
                pk_nid.append(poly_add(pk_i, noise[i], self.q))
            pk[nid] = pk_nid
        pad = {
            nid: [rand_uniform_vector(self.ring_n, self.q) for _ in range(self.eta)]
            for nid in self.lst_nid
        }
        self.msk = {"sk": sk, "u": pad}
        self.mpk = {"pk": pk}
        logger.info("MIFE Ring-LWE (ACFGU18 / BMMS21) setup successfully")

    def get_public_parameters(self) -> dict:
        pp = MIFERingLWEParameters.new(
            A=[gp.digits(v) for v in self.A],
            p=gp.digits(self.p),
            q=gp.digits(self.q),
            eta=self.eta,
            n=self.n,
            ring_n=self.ring_n,
            bound_x=gp.digits(self.bound_x),
            bound_y=gp.digits(self.bound_y),
            sigma1=self.sigma1,
            sigma2=self.sigma2,
            sigma3=self.sigma3,
            sec_param=self.sec_param,
            gaussian_sampler=self.gaussian_sampler,
            security_profile=self.security_profile,
        )
        if self.q_factors is not None:  # RNS: the NTT and the native code need them
            pp.q_factors = [gp.digits(f) for f in self.q_factors]
        return pp.to_dict()

    def get_private_keys(self, nid: str, **kwargs) -> dict | None:
        """Return client *nid*'s encryption key (``pk`` and its pad ``u``).

        Args:
            nid: Client identifier.

        Returns:
            The key dict, or ``None`` for an unknown client.
        """
        if nid not in self.mpk["pk"]:
            return None
        return MIFERingLWEEncryptionKey.new(
            pk=[[gp.digits(v) for v in row] for row in self.mpk["pk"][nid]],
            u=[[gp.digits(v) for v in row] for row in self.msk["u"][nid]],
        ).to_dict()

    def get_decryption_keys(self, sid: str, **kwargs) -> dict | None:
        """Derive a functional decryption key for the given fusion weights.

        Args:
            sid: Session identifier.
            **kwargs: Must include ``credentials`` with ``fusion_weight`` (dict
                ``{nid: list of int}``).
        """
        credentials = kwargs.get("credentials", None)
        if not credentials:
            raise FEKeyError("need credentials for MIFE Ring-LWE decryption key generation")
        ys = self._key_weights(credentials.get("fusion_weight"))
        ring, sk_rows, u_rows = self._prepared_master_key()  # sum_r y_r sk_r and z mod q, as decimal strings
        sk_y = {nid: ring.lincomb(sk_rows[nid], ys[nid]) for nid in self.lst_nid}
        z = ring.lincomb([row for nid in self.lst_nid for row in u_rows[nid]], [v for nid in self.lst_nid for v in ys[nid]])
        return MIFERingLWEFunctionalKey.new(sk_y=sk_y, z=z).to_dict()

    def _key_weights(self, fusion_weight) -> dict:
        """The validated weights ``{nid: y_i}`` of a functional key."""
        if not isinstance(fusion_weight, dict):
            raise FEValidationError("invalid fusion weights provided, need a dict")
        ys = {}
        for nid in self.lst_nid:
            if nid not in fusion_weight:
                raise FEValidationError("fusion weight missing client {}".format(nid))
            ys[nid] = int_vector(
                fusion_weight[nid], "fusion weight of {}".format(nid),
                length=self.eta, bound=int(self.bound_y),
            )
        return ys

    def _key_z(self, fusion_weight) -> list:
        """The ``z = sum_i sum_row y_{i,row} u_{i,row}`` of :meth:`get_decryption_keys`
        alone, as decimal strings (the same values)."""
        ys = self._key_weights(fusion_weight)
        ring, _, u_rows = self._prepared_master_key()
        return ring.lincomb([row for nid in self.lst_nid for row in u_rows[nid]], [v for nid in self.lst_nid for v in ys[nid]])

    def __getstate__(self) -> dict:
        """Pickling and copying without the prepared key cache (rebuilt on use)."""
        state = self.__dict__.copy()
        state.pop("_native_msk", None)
        return state

    def _prepared_master_key(self):
        """``(ring, sk rows, u rows)``: the backend's ring of ``(q, ring_n)`` and
        the master key's rows per client in the form its
        :meth:`~pyfe4ai.backend.interface.Ring.lincomb` reads fastest
        (:meth:`~pyfe4ai.backend.interface.Ring.prepare_rows`; natively packed),
        prepared once per master key and ring. Rows the extension does not read
        are combined by the ring's reference
        (:meth:`~pyfe4ai.backend.interface.Ring.reference`)."""
        ring = backend.ring(self.q, self.ring_n)
        cache = self.__dict__.get("_native_msk")
        if cache is None or cache[0] is not self.msk or cache[1] is not ring:
            try:
                prepared = (
                    {nid: ring.prepare_rows(self.msk["sk"][nid]) for nid in self.lst_nid},
                    {nid: ring.prepare_rows(self.msk["u"][nid]) for nid in self.lst_nid},
                )
            except (KeyError, TypeError, ValueError):
                prepared = None
            cache = self._native_msk = (self.msk, ring, prepared)
        if cache[2] is None:
            return ring.reference(), self.msk["sk"], self.msk["u"]
        return (ring,) + cache[2]


class MIFERingLWE(IPFEAbsCrypto):
    """Crypto operations for Ring-LWE-based multi-input inner-product FE."""

    #: Whether ciphertexts may use the binary format (not where tags cover
    #: the JSON form, as in the threshold variant).
    _binary_ciphertexts = True

    def __init__(self, config: dict, **kwargs) -> None:
        """Initialise the MIFE Ring-LWE crypto object.

        Args:
            config: Must contain ``keys`` with ``pp`` (public parameters) and
                optionally ``sk`` (encryption keys). ``ct_format``: ``"json"``
                (default: ``ct0`` and ``ct1`` as decimal strings) or
                ``"binary"`` (``ct``: bytes, see
                :func:`~pyfe4ai.core.formats.binary.encode_binary_ciphertext`;
                about a third of the size, and faster to produce and read).
                Decryption reads both formats. ``gaussian_sampler``: that of
                the encryption noise (default: that of ``pp``, else
                ``"facct"``; see :func:`~pyfe4ai.primitives.sampling.scheme_sampler`).
        """
        super().__init__(config, **kwargs)
        self.pp = self.keys["pp"]
        self.pp["A"]  # a public parameter set without a raises here, as before
        self.gaussian_sampler = scheme_sampler(config, self.pp)
        if self.pp.get("q_factors") is not None:
            register_rns_modulus(self.pp["q"], self.pp["q_factors"])
        self.ct_format = config.get("ct_format", "json")
        if self.ct_format not in CT_FORMATS or (self.ct_format == "binary" and not self._binary_ciphertexts):
            raise FEValidationError("unsupported ciphertext format {!r}".format(self.ct_format))
        if self._has_private_keys():
            self.sk = self.keys["sk"]

    @property
    def A(self) -> list:
        """The public polynomial ``a``, converted from ``pp["A"]`` on first use
        (decryption never needs it)."""
        a = self.__dict__.get("_A")
        if a is None:
            a = self._A = [gp.mpz(v) for v in self.pp["A"]]
        return a

    @A.setter
    def A(self, value) -> None:
        self._A = value

    def _check_encryption_keys(self) -> None:
        if not self._has_public_parameters():
            raise FEKeyError("no public parameters provided for encryption")
        if not self._has_private_keys():
            raise FEKeyError("no encryption key provided")
        self._check_private_key()

    def _check_plaintext(self, matrix_pt) -> list:
        """Return *matrix_pt* as ``eta`` integer rows of one length ``1..ring_n``."""
        self._check_encryption_keys()
        if not isinstance(matrix_pt, list) or not matrix_pt or not isinstance(matrix_pt[0], list):
            raise FEValidationError("plaintext matrix must be a list of rows")
        if len(matrix_pt) != self.pp["eta"]:
            raise FEValidationError("invalid plaintext row count")
        rows = [int_vector(row, "plaintext row", bound=int(self.pp["bound_x"])) for row in matrix_pt]
        k = len(rows[0])
        if not 1 <= k <= self.pp["ring_n"] or any(len(row) != k for row in rows):
            raise FEValidationError(
                "plaintext rows must all have the same length, between 1 and ring_n"
            )
        if "u" not in MIFERingLWEEncryptionKey(self.sk):
            raise FEKeyError("encryption key lacks the one-time pad u")
        self._check_key_rows()
        return rows

    def _check_key_rows(self) -> None:
        """Raise unless the encryption key's ``pk`` and pad ``u`` have ``eta``
        rows (at least) of ``ring_n`` values: with fewer rows, the encryption
        silently made a ciphertext without the missing rows, with a row of
        another length a ciphertext of another length, or failed deep inside
        the backend (:func:`~pyfe4ai.primitives.lattice.ring_lwe_utils.check_key_rows`).

        Raises:
            FEKeyError: For such rows (the message holds no key data).
        """
        sk = MIFERingLWEEncryptionKey(self.sk)
        check_key_rows({"pk": sk.get("pk"), "u": sk.get("u")}, self.pp["eta"], self.pp["ring_n"], "pk or u")

    def __getstate__(self) -> dict:
        """Pickling and copying without the key cache (it may hold native
        objects; rebuilt on use)."""
        state = self.__dict__.copy()
        state.pop("_enc_key", None)
        return state

    def _encryption_key(self) -> dict:
        """This slot's key in the form encryption uses, computed once per key:
        ``pk`` and the pad ``u`` as integers, and with an NTT-friendly ``q``
        (``ring`` not ``None``) the transforms ``NTT(a)`` and ``NTT(pk_row)``;
        ``"reference"``: all of it as the backend reference's key
        (:class:`~pyfe4ai.backend.interface.ReferenceKey`)."""
        sk = MIFERingLWEEncryptionKey(self.sk)
        pk, u = sk.pk, sk.get("u")
        cache = self.__dict__.get("_enc_key")
        if cache is None or cache["src"] != (id(pk), id(u), id(self.A)):
            q = gp.mpz(self.pp["q"])
            ring = RingNTT.get(q, self.pp["ring_n"])
            pk_m = [[gp.mpz(v) for v in row] for row in pk]
            cache = {
                "src": (id(pk), id(u), id(self.A)),
                "refs": (pk, u, self.A),  # keeps the ids above valid
                "pk": pk_m,
                "u": None if u is None else [[gp.mpz(v) for v in row] for row in u],
                "ring": ring,
                "a_hat": None if ring is None else ring.forward(self.A),
                "pk_hat": None if ring is None else [ring.forward(row) for row in pk_m],
            }
            cache["reference"] = ReferenceKey(self.A, pk_m, cache["u"], q // gp.mpz(self.pp["p"]), ring,
                                              cache["a_hat"], cache["pk_hat"])
            self._enc_key = cache
        return cache

    def _reference_key(self) -> ReferenceKey:
        """The key of the backend's reference encryption (:meth:`_encryption_key`)."""
        return self._encryption_key()["reference"]

    def _ring(self):
        """The backend's ring of the public parameters (see :mod:`pyfe4ai.backend`)."""
        return backend.ring(self.pp["q"], self.pp["ring_n"])

    def _pad_hash_mode(self):
        """The label-hash mode of the pads the native encryption key hashes
        (MCFE; ``None``: no pads)."""
        return None

    def _native_encryption_key(self):
        """This slot's native encryption key (``NTT(a)``, ``NTT(pk)``, ``u``;
        :meth:`~pyfe4ai.backend.interface.Ring.encryption_key`), or ``None``
        when the backend's reference encrypts (a Python ring, widths or a
        Gaussian sampler the native code does not reproduce, a malformed key)."""
        ring = self._ring()
        if not ring.samples_natively(self.pp["sigma2"], self.pp["sigma3"], sampler=self.gaussian_sampler):
            return None
        key = self._encryption_key()
        cached = key.get("native")
        mode = self._pad_hash_mode()  # the label-hash mode of the pads (MCFE)
        if cached is None or cached[0] is not ring or cached[2] != mode:
            q = gp.mpz(self.pp["q"])
            sk = MIFERingLWEEncryptionKey(self.sk)
            try:
                a, pk, u, delta = self.A, sk.pk, sk.get("u"), q // gp.mpz(self.pp["p"])
            except (TypeError, ValueError, OverflowError):
                nkey = None  # malformed: the reference reports it
            else:
                nkey = ring.encryption_key(a, pk, u, delta, hash_mode=mode)
            cached = key["native"] = (ring, nkey, mode)
        return cached[1]

    def _padded_rows(self, matrix_pt, extra_pad=None) -> list:
        """Rows ``x + u (+ extra_pad)`` mod q, zero-extended to ``ring_n``."""
        q = gp.mpz(self.pp["q"])
        ring_n = self.pp["ring_n"]
        rows = []
        for i, (row, u_row) in enumerate(zip(matrix_pt, self._encryption_key()["u"])):
            padded = [gp.mpz(int(v)) for v in row] + [gp.mpz(0)] * (ring_n - len(row))
            if extra_pad is None:
                rows.append([(x + u) % q for x, u in zip(padded, u_row)])
            else:
                rows.append([(x + u + e) % q for x, u, e in zip(padded, u_row, extra_pad[i])])
        return rows

    def _sigmas(self) -> tuple:
        return self.pp["sigma2"], self.pp["sigma3"]

    def _encrypt_rows(self, w_rows: list, k: int) -> dict:
        """BMMS21 Ring-LWE encryption of rows already reduced mod q
        (:meth:`~pyfe4ai.backend.interface.Ring.encrypt_rows`)."""
        ring = self._ring()
        nkey = self._native_encryption_key()
        binary = self.ct_format == "binary"
        out = ring.encrypt_rows(nkey, self._reference_key, w_rows, self.pp["eta"], self._sigmas(),
                                self.gaussian_sampler, binary)
        if binary:
            return RingLWECiphertext.new(ct=out, k=k).to_dict()
        return RingLWECiphertext.new(ct0=out[0], ct1=out[1], k=k).to_dict()

    def encrypt(self, matrix_pt: list[list[int]]) -> dict:
        """Encrypt this slot's matrix: Ring-LWE encryption of ``x + u mod q``.

        Returns:
            Dict with ``ct0`` (one polynomial per row), ``ct1`` and ``k``
            (number of used coefficients).
        """
        rows = self._check_plaintext(matrix_pt)
        return self._encrypt_rows(self._padded_rows(rows), len(rows[0]))

    def decrypt(self, dct_ct: dict, dk: dict, fusion_weight: dict):
        """Recover the batch of inner products ``sum_i <x_i[:, k], y_i>``.

        Returns:
            List with one inner product per batch position (``None`` entries
            if a decoded value exceeds ``n eta X Y``).

        Raises:
            FEKeyError: If the key is missing.
            FESchemeError: If ciphertexts do not cover exactly the key's slots
                or have different batch lengths.
        """
        if not dk or "z" not in MIFERingLWEFunctionalKey(dk):
            raise FEKeyError("no decryption key provided")
        key = MIFERingLWEFunctionalKey(dk)
        if set(dct_ct) != set(key.sk_y) or set(dct_ct) != set(fusion_weight):
            raise FESchemeError("need one ciphertext for every slot of the key")
        batch = {int(RingLWECiphertext(ct).k) for ct in dct_ct.values()}
        if len(batch) != 1:
            raise FESchemeError("ciphertexts of different batch lengths")

        return self._decrypt_with_pad_term(dct_ct, key.sk_y, key.z, fusion_weight, batch.pop())

    def _decrypt_with_pad_term(self, dct_ct, sk_y_map, z, fusion_weight, k, sk_hats=None) -> list:
        """Dec2(sum_i Dec1_i - Delta z) for the first *k* coefficients.

        The products ``ct1_i sk_i`` are summed in the NTT domain and
        transformed back once (NTT linearity: the same sum mod q as summing
        ``poly_mul(ct1_i, sk_i)``), from ``slots + 1`` NTTs instead of
        ``3 slots``. *sk_hats*, a dict the caller keeps for all chunks
        decrypted with one key, caches ``NTT(sk_i)`` per slot.
        """
        ring = self._ring()
        if ring.native:  # one native call, with the checks of _decrypt_native
            out = self._decrypt_native(ring, [dct_ct], [z], [k], sk_y_map, fusion_weight, sk_hats)
            if out is not None:
                values, none = out
                values = values.tolist()
                if none is not None:
                    values = [None if bad else v for v, bad in zip(values, none.tolist())]
                return values
        q = gp.mpz(self.pp["q"])
        delta = q // gp.mpz(self.pp["p"])
        ring = RingNTT.get(q, self.pp["ring_n"])
        total = [(-delta * gp.mpz(v)) % q for v in z]
        acc = None
        for nid, ct in dct_ct.items():
            y = int_vector(
                fusion_weight[nid], "fusion weight of {}".format(nid),
                length=self.pp["eta"], bound=int(self.pp["bound_y"]),
            )
            try:
                _, ct0, ct1 = canonical_ring_ciphertext(ct, int(q), self.pp["eta"], self.pp["ring_n"])
            except FESchemeError as exc:
                raise FESchemeError("malformed ciphertext of {}".format(nid)) from exc
            y_vec = [gp.mpz(int(v)) for v in y]
            # Dec1 of slot nid: Delta <w_i, y_i> + noise (coefficient-wise)
            for y_row, ct0_row in zip(y_vec, ct0):
                total = [(t + c * y_row) % q for t, c in zip(total, ct0_row)]
            if ring is None:
                rhs = poly_mul([gp.mpz(v) for v in ct1], [gp.mpz(v) for v in sk_y_map[nid]], q)
                total = poly_add(total, poly_neg(rhs, q), q)
                continue
            sk_hat = None if sk_hats is None else sk_hats.get(nid)
            if sk_hat is None:
                sk_hat = ring.forward([gp.mpz(v) for v in sk_y_map[nid]])
                if sk_hats is not None:
                    sk_hats[nid] = sk_hat
            acc = ring.mul_add(acc, ring.forward(ct1), sk_hat)
        if acc is not None:
            total = [(t - v) % q for t, v in zip(total, ring.inverse(acc))]
        bound = (
            self.pp["n"] * self.pp["eta"]
            * gp.mpz(self.pp["bound_x"]) * gp.mpz(self.pp["bound_y"])
        )
        return [decode_scaled(v, delta, q, bound) for v in total[:k]]

    def _decrypt_native(self, ring, chunk_cts, zs, ks, sk_y_map, fusion_weight, sk_hats=None):
        """:meth:`_decrypt_with_pad_term` of several chunks in one native call
        (:meth:`~pyfe4ai.backend.interface.Ring.decrypt_tensor` of the native *ring*).

        ``chunk_cts[j]`` maps every slot to its ciphertext of chunk ``j``,
        ``zs[j]`` and ``ks[j]`` are that chunk's key term and batch length.
        The same checks as the Python path come first (and raise the same
        errors). Returns ``(values, none)``: the decoded values as ``int64``
        and a boolean mask of those that are ``None`` (or ``None`` if all
        decode); ``None`` if the native code does not handle the input (the
        caller then runs the Python path).
        """
        eta, ring_n = self.pp["eta"], self.pp["ring_n"]
        nids = list(chunk_cts[0])
        ys, chunks = {}, []
        for dct_ct in chunk_cts:
            slots = []
            for nid, ct in dct_ct.items():
                if nid not in ys:
                    ys[nid] = int_vector(
                        fusion_weight[nid], "fusion weight of {}".format(nid),
                        length=eta, bound=int(self.pp["bound_y"]),
                    )
                try:  # the shape checks of canonical_ring_ciphertext; values are parsed natively
                    c = RingLWECiphertext(ct)
                    k = c.k
                    if "ct" in c and "ct0" not in c:  # binary format
                        if not isinstance(c.ct, (bytes, bytearray, memoryview)):
                            raise TypeError("binary ciphertext must be bytes")
                        entry = bytes(c.ct)
                    else:
                        entry = (c.ct0, c.ct1)
                except (KeyError, TypeError) as exc:
                    raise FESchemeError("malformed ciphertext of {}".format(nid)) from exc
                if not isinstance(k, int) or isinstance(k, bool) or not 1 <= k <= ring_n:
                    raise FESchemeError("malformed ciphertext of {}".format(nid))
                slots.append(entry)
            if list(dct_ct) != nids:
                return None
            chunks.append(slots)
        hats = []
        for nid in nids:
            hat = None if sk_hats is None else sk_hats.get(("native", nid))
            if hat is None:
                hat = ring.decryption_key_hat(sk_y_map[nid])
                if hat is None:  # a key the extension does not read: the Python path
                    return None
                if sk_hats is not None:
                    sk_hats[("native", nid)] = hat
            hats.append(hat)
        q = gp.mpz(self.pp["q"])
        bound = self.pp["n"] * self.pp["eta"] * gp.mpz(self.pp["bound_x"]) * gp.mpz(self.pp["bound_y"])
        try:
            out = ring.decrypt_tensor(chunks, zs, ks, hats, [ys[nid] for nid in nids],
                                      int(q // gp.mpz(self.pp["p"])), int(bound))
        except ValueError as exc:
            if exc.args[:1] == ("malformed ciphertext",):
                raise FESchemeError("malformed ciphertext of {}".format(nids[exc.args[1]])) from exc
            raise
        if out is None:  # values beyond an int64: the Python path
            return None
        data, mask = out
        values = np.frombuffer(data, dtype=np.int64)
        return values, (None if mask is None else np.frombuffer(mask, dtype=np.uint8).astype(bool))

    def encrypt_lst_ndarray(self, lst_ndarray: list, **kwargs) -> list | None:
        """Not implemented for this scheme; raises :class:`NotImplementedError`."""
        raise NotImplementedError()

    def compute_lst_ndarray_ct(self, dict_ndarray_ct: dict, **kwargs) -> list | None:
        """Compute inner products on encrypted ndarray ciphertexts.

            Args:
                dict_ndarray_ct: Encrypted ndarray ciphertexts (list or dict).
        """
        raise NotImplementedError()

    def decrypt_lst_ndarray_ct(self, dict_ndarray_ct: dict, **kwargs) -> list | None:
        """Decrypt encrypted ndarray ciphertexts element-wise.

            Args:
                dict_ndarray_ct: Encrypted ndarray ciphertexts (list or dict).
        """
        raise NotImplementedError()
