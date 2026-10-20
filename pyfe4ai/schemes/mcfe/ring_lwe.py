"""
Ring-LWE identity-based multi-client inner-product FE.

| Bermudo Mera, Karmakar, Marc, Soleimanian:
| "Efficient Lattice-Based Inner-Product Functional Encryption", PKC 2022
| (ePrint 2021/046) -- Section 6, "MCFE-Compiler" (from multi-input to
| identity-based multi-client IPFE, random oracle; Theorem 5, static
| corruptions) applied to the MIFE of :mod:`pyfe4ai.schemes.mife.ring_lwe`.

* type:     identity-based multi-client encryption (MCFE, Ring-LWE, random oracle)
* setting:  R_q = Z_q[x] / (x^ring_n + 1)

Construction (MIFE notation; ``H(u'_i, l) in R_q^eta`` is a hash keyed by the
client secret ``u'_i``):

* Setup:    MIFE setup plus a secret ``u'_i`` per client.
* Encrypt:  ``ct_{i,l} = MIFE.Enc(msk_i, x_i + H(u'_i, l))``, i.e. the slot
  encrypts ``x_i + u_i + H(u'_i, l)``.
* KeyGen:   for weights ``y`` and label ``l``: the MIFE key
  ``({sk_{i,y}}, z)`` with ``z'' = z + sum_i y_i H(u'_i, l)``.
* Decrypt:  ``MIFE.Dec`` with ``z''``.

Functional keys are bound to a label (identity-based MCFE): a key for
``(y, l)`` only decrypts ciphertexts produced under ``l``. Each client must
encrypt at most once per label. The ndarray helpers pack values into the
``ring_n`` coefficients of one ciphertext per chunk; chunk ``c`` of label
``l`` uses ``H(u'_i, (l, c))``, and a key for ``n_chunks`` chunks carries one
``z''`` per chunk. ``H`` is SHAKE-256 over the client secret and the encoded
label (:func:`~pyfe4ai.primitives.hashing.hash_to_zq_vector`), mapped to
``Z_q`` as the key generator's ``label_hash`` says (a public parameter;
:data:`~pyfe4ai.primitives.hashing.LABEL_HASHES`): by default rejection
sampling mod q from SHAKE128 (``"shake128-rejection"``: exactly uniform,
fewer XOF bytes). Versions <= 0.3.0 reduced ``bits(q) + 128``-bit SHAKE-256
blocks (``"shake256-wide"``, still selectable), so their ciphertexts and
key terms differ. The mode is bound to everything that depends on it: the
public parameters, the clients' encryption keys, every functional key (and
threshold key share), every single ciphertext and every ndarray layer
(``label_hash``: the code of :data:`~pyfe4ai.primitives.hashing.LABEL_HASH_CODES`,
once per layer dict, not per chunk; covered by the threshold variant's
tags); a missing field means ``"shake256-wide"`` (the
data of versions <= 0.3.0, which stay readable), and encryption with a key
of another mode, or decryption of ciphertexts or with keys of another mode,
raises instead of returning wrong sums. Versions <= 0.2.0 masked inputs with
a small public label scalar.
"""

from __future__ import annotations

import logging
import math
import secrets

import gmpy2 as gp
import numpy as np

from pyfe4ai.schemes.mcfe.labels import chunk_label, encode_label
from pyfe4ai.core.base import decoded_layer
from pyfe4ai.schemes.mife.ring_lwe import MIFERingLWE
from pyfe4ai.schemes.mife.ring_lwe import MIFERingLWEKeyGenerator
from pyfe4ai.core.one_time import OneTimePrecomputation
from pyfe4ai import backend as native_backend
from pyfe4ai.core.constants import CryptoCONST
from pyfe4ai.primitives.hashing import DEFAULT_LABEL_HASH
from pyfe4ai.primitives.hashing import LABEL_HASH_CODES
from pyfe4ai.primitives.hashing import LEGACY_LABEL_HASH
from pyfe4ai.primitives.hashing import hash_to_zq_material
from pyfe4ai.primitives.hashing import hash_to_zq_vector
from pyfe4ai.core.exceptions import FEKeyError, FESchemeError, FEValidationError
from pyfe4ai.core.formats.ring_lwe import MCFERingLWEEncryptionKey
from pyfe4ai.core.formats.ring_lwe import MCFERingLWEFunctionalKey
from pyfe4ai.core.formats.ring_lwe import MCFERingLWEParameters
from pyfe4ai.core.formats.ring_lwe import MIFERingLWEFunctionalKey
from pyfe4ai.core.formats.ring_lwe import RingLWECiphertext
from pyfe4ai.core.formats.ring_lwe import RingLWELayer
from pyfe4ai.core.versions import check_ciphertext_label_hashes
from pyfe4ai.core.versions import check_encryption_key_label_hash
from pyfe4ai.core.versions import check_key_label_hash
from pyfe4ai.core.versions import label_hash_code
from pyfe4ai.core.versions import label_hash_mode
from pyfe4ai.core.versions import pp_label_hash
from pyfe4ai.core.formats.binary import encode_values
from pyfe4ai._validation import scale_to_int

logger = logging.getLogger(__name__)

_PAD_DOMAIN = "pyfe4ai/mcfe/ring_lwe/H(u',label)"


def label_pad(u_prime: str, label, q, eta: int, ring_n: int, mode: str) -> list:
    """Return ``H(u'_i, label)`` as ``eta`` rows of ``ring_n`` coefficients mod q
    (*mode*: the ``label_hash`` public parameter; expanded natively when the
    extension has a context for ``(q, ring_n)``)."""
    ctx = native_backend.ring_context(q, ring_n)
    if ctx is not None:
        words = native_backend.ext().hash_to_zq(
            ctx.cap, label_pad_material(u_prime, label, mode), eta * ring_n, **native_hash_kwargs(mode)
        )
        flat = [gp.mpz(v) for v in ctx.unpack(words)]
    else:
        flat = hash_to_zq_vector(
            "{}|{}".format(u_prime, encode_label(label)), q, eta * ring_n, domain=_PAD_DOMAIN, mode=mode
        )
    return [flat[r * ring_n:(r + 1) * ring_n] for r in range(eta)]


def label_pad_material(u_prime: str, label, mode: str) -> bytes:
    """The hash input of :func:`label_pad` (the native backend expands it itself)."""
    return hash_to_zq_material("{}|{}".format(u_prime, encode_label(label)), domain=_PAD_DOMAIN, mode=mode)


def native_hash_kwargs(mode: str) -> dict:
    """The ``hash_mode`` keyword of the native pad kernels (none for the kernels' own ``"shake256-wide"``)."""
    return {"hash_mode": LABEL_HASH_CODES[mode]} if mode != LEGACY_LABEL_HASH else {}


def chunk_key_index(layer: dict, j: int, n_keys: int) -> int:
    """Index of the chunk key for chunk *j* of an ndarray ciphertext layer.

    Raises:
        FESchemeError: Unless the layer's ``first_chunk`` is a non-negative
            integer and all its chunks have a key among the *n_keys*.
    """
    layer = RingLWELayer(layer)
    first = layer.get("first_chunk")
    n_chunks = len(layer.get("chunks", []))
    if (
        not isinstance(first, int) or isinstance(first, bool)
        or first < 0 or first + n_chunks > n_keys or not 0 <= j < n_chunks
    ):
        raise FESchemeError(
            "ciphertext chunks starting at {!r} are not covered by the {} chunk keys".format(first, n_keys)
        )
    return first + j

class MCFERingLWEKeyGenerator(MIFERingLWEKeyGenerator):
    """Key generator for BMMS21 identity-based MCFE over Ring-LWE."""

    _scheme_type = CryptoCONST.TYPE_MCFE_RING_LWE

    def __init__(self, config: dict, **kwargs) -> None:
        """As :class:`~pyfe4ai.schemes.mife.ring_lwe.MIFERingLWEKeyGenerator`,
        plus ``label_hash`` (one of
        :data:`~pyfe4ai.primitives.hashing.LABEL_HASHES`, default
        ``"shake128-rejection"``; ``"shake256-wide"`` for the ciphertexts of
        versions <= 0.3.0), published in the public parameters."""
        self.label_hash = label_hash_mode(config.get("label_hash", DEFAULT_LABEL_HASH))
        super().__init__(config, **kwargs)

    def get_public_parameters(self) -> dict:
        """MIFE public parameters plus ``label_hash``."""
        pp = MCFERingLWEParameters(super().get_public_parameters())
        pp.label_hash = self.label_hash
        return pp.to_dict()

    def setup(self) -> None:
        """MIFE setup plus a secret ``u'_i`` for every client."""
        super().setup()
        self.msk["u_prime"] = {nid: secrets.token_hex(32) for nid in self.lst_nid}

    def get_private_keys(self, nid: str, **kwargs) -> dict | None:
        """Return client *nid*'s keys: MIFE ``pk``, ``u`` and the label-pad secret ``u_prime``."""
        sk = super().get_private_keys(nid)
        if sk is None:
            return None
        key = MCFERingLWEEncryptionKey(sk)
        key.u_prime = self.msk["u_prime"][nid]
        key.label_hash = self.label_hash
        return key.to_dict()

    def get_decryption_keys(self, sid: str, **kwargs) -> dict | None:
        """Derive the key for weights ``y`` and label ``l`` (identity-based).

        Args:
            sid: Session identifier (unused).
            **kwargs: ``credentials`` with ``fusion_weight``, ``label`` and
                optionally ``n_chunks`` (number of ndarray chunks to cover).

        Returns:
            Dict with ``sk_y`` (per-slot MIFE keys), ``z`` (pad term of the
            label itself), ``z_chunks`` (pad terms of chunks ``0..n_chunks-1``)
            and ``label``.
        """
        credentials = kwargs.get("credentials", None)
        if not credentials:
            raise FEKeyError("need credentials for MCFE Ring-LWE decryption key generation")
        label = credentials.get("label")
        if not isinstance(label, str):
            raise FEValidationError("identity-based MCFE keys need a string label")
        n_chunks = int(credentials.get("n_chunks", 0))
        fusion_weight = credentials.get("fusion_weight")
        key = super().get_decryption_keys(sid, credentials={"fusion_weight": fusion_weight})
        labels = [label] + [chunk_label(label, c) for c in range(n_chunks)]
        key = MIFERingLWEFunctionalKey(key)
        terms = self._label_terms(key.z, fusion_weight, labels)
        return self._with_label_hash(
            MCFERingLWEFunctionalKey.new(sk_y=key.sk_y, z=terms[0], z_chunks=terms[1:], label=label).to_dict())

    def _label_terms(self, base, fusion_weight: dict, labels: list, binary: bool = False) -> list:
        """``z'' = base + sum_i y_i H(u'_i, l)`` of every label of *labels* as
        decimal strings (*base*: the MIFE key's ``z``; weights already
        validated), natively in one call where possible (same values); with
        *binary* as bytes of ``ceil(bits(q) / 8)``-byte words."""
        ctx = native_backend.ring_context(self.q, self.ring_n)
        if ctx is not None and not (binary and ctx.legacy_rns):
            return native_backend.ext().pad_terms(
                ctx.cap, base,
                [[label_pad_material(self.msk["u_prime"][nid], lab, self.label_hash) for nid in self.lst_nid]
                 for lab in labels],
                [[int(v) for v in fusion_weight[nid]] for nid in self.lst_nid],
                self.eta, native_backend.get_threads(), **native_hash_kwargs(self.label_hash),
                **({"binary": True} if binary else {}),
            )
        base = [gp.mpz(v) for v in base]

        def pad_term(lab) -> list:
            z = list(base)
            for nid in self.lst_nid:
                pad = label_pad(self.msk["u_prime"][nid], lab, self.q, self.eta, self.ring_n, self.label_hash)
                for y_row, pad_row in zip(fusion_weight[nid], pad):
                    z = [acc + int(y_row) * h for acc, h in zip(z, pad_row)]
            if binary:
                return encode_values(z, self.q)
            return [gp.digits(v % self.q) for v in z]

        return [pad_term(lab) for lab in labels]

    def _with_label_hash(self, key: dict) -> dict:
        """The key, marked with its ``label_hash``."""
        MCFERingLWEFunctionalKey(key).label_hash = self.label_hash
        return key


class RingLWEPrecomputation(OneTimePrecomputation):
    """Offline phase of :meth:`MCFERingLWE.encrypt_lst_ndarray` for one label
    and fixed layer shapes (see :meth:`MCFERingLWE.precompute`). Secret and
    one-time (:mod:`pyfe4ai.core.one_time`)."""

    _secret = ("native", "rows")

    def __init__(self, owner, label: str, shapes: list, first_chunks: list, native=None, rows=None) -> None:
        super().__init__()
        self.owner = owner
        self.key_src = owner._key_src()
        self.label = label
        self.shapes = shapes
        self.first_chunks = first_chunks
        self.native = native  # (native key, capsule) or None
        self.rows = rows      # Python: [(pk r + e_0 + Delta (u + H), a r + e)] per chunk

    def __repr__(self) -> str:
        return "RingLWEPrecomputation(label={!r}, shapes={}, used={})".format(self.label, self.shapes, self.used)


class MCFERingLWE(MIFERingLWE):
    """Crypto operations for BMMS21 identity-based MCFE over Ring-LWE."""

    @property
    def label_hash(self) -> str:
        """The ``label_hash`` public parameter (see :func:`~pyfe4ai.core.versions.pp_label_hash`)."""
        return pp_label_hash(self.pp)

    def _native_key_kwargs(self) -> dict:
        return native_hash_kwargs(self.label_hash)

    def _check_key_label_hash(self, dk: dict) -> None:
        check_key_label_hash(dk, self.label_hash)

    def _encryption_label_hash(self) -> str:
        """The label hash to encrypt with: the public parameters', which must
        be the one the encryption key was issued for (keys without
        ``label_hash``: ``"shake256-wide"``).

        Raises:
            FEKeyError: If they differ (e.g. public parameters of another
                version than the key): the ciphertexts would decrypt to wrong
                values.
        """
        mode = self.label_hash
        return check_encryption_key_label_hash(self.sk or {}, mode) if self._has_private_keys() else mode

    def _mark(self, ct: dict) -> dict:
        """Bind the label hash to a ciphertext (no field for ``"shake256-wide"``,
        as in versions <= 0.3.0)."""
        code = label_hash_code(self.label_hash)
        if code is not None:
            RingLWECiphertext(ct).label_hash = code
        return ct

    def _mark_layers(self, layers: list) -> list:
        """Bind the label hash to every ndarray layer, once per layer (its
        chunks carry none)."""
        for layer in layers:
            self._mark(layer)
        return layers

    def _check_layers_label_hash(self, dict_ndarray_ct: dict) -> None:
        """Raise unless every client's ndarray layers (and any chunk that carries
        a code of its own) are of the public parameters' label hash."""
        mode = self.label_hash
        for nid, layers in dict_ndarray_ct.items():
            for layer in layers if isinstance(layers, (list, tuple)) else ():
                if not isinstance(layer, dict):
                    continue  # malformed: the decryption reports it
                check_ciphertext_label_hashes((layer,), mode, nid)
                chunks = RingLWELayer(layer).get("chunks")
                check_ciphertext_label_hashes(
                    [ct for ct in chunks if isinstance(ct, dict) and "label_hash" in RingLWECiphertext(ct)]
                    if isinstance(chunks, (list, tuple)) else (), mode, nid)

    def _decrypt_layer_native(self, ctx, dict_ndarray_ct, l, layer, dk, fusion_weight, sk_hats):
        """Layer *l* of :meth:`decrypt_lst_ndarray_ct` in one native call, with
        the checks of :meth:`_decrypt_with_key_term` per chunk; ``None`` if the
        native code does not handle the input (then the Python loop runs)."""
        chunk_cts, zs, ks = [], [], []
        key = MCFERingLWEFunctionalKey(dk)
        for j in range(len(RingLWELayer(layer).chunks)):
            dct_ct = {nid: RingLWELayer(dict_ndarray_ct[nid][l]).chunks[j] for nid in dict_ndarray_ct}
            zs.append(key.z_chunks[chunk_key_index(layer, j, len(key.z_chunks))])
            if set(dct_ct) != set(key.sk_y) or set(dct_ct) != set(fusion_weight):
                raise FESchemeError("need one ciphertext for every client of the key")
            batch = {int(RingLWECiphertext(ct).k) for ct in dct_ct.values()}
            if len(batch) != 1:
                raise FESchemeError("ciphertexts of different batch lengths")
            chunk_cts.append(dct_ct)
            ks.append(batch.pop())
        if not chunk_cts:
            return None
        out = self._decrypt_native(ctx, chunk_cts, zs, ks, key.sk_y, fusion_weight, sk_hats)
        if out is None:
            return None
        values, none = out
        scale = 10 ** self.precision
        bound = self.pp["n"] * self.pp["eta"] * int(self.pp["bound_x"]) * int(self.pp["bound_y"])
        if none is not None or bound >= 2 ** 53 or scale >= 2 ** 53:
            # decoded_layer's checks and division on Python ints
            ary = np.empty(len(values), dtype=object)
            ary[:] = values.tolist()
            if none is not None:
                ary[none] = None
            return decoded_layer(ary.reshape(RingLWELayer(layer).shape), l, scale)
        # int64 / int as float64: exact here, the same floats as decoded_layer's int / int
        return (values.reshape(RingLWELayer(layer).shape) / scale).astype(float)

    def _pad(self, label) -> list:
        return label_pad(
            MCFERingLWEEncryptionKey(self.sk).u_prime, label, gp.mpz(self.pp["q"]), self.pp["eta"], self.pp["ring_n"],
            self._encryption_label_hash(),
        )

    def encrypt(self, matrix_pt: list[list[int]], label: str) -> dict:
        """Encrypt this client's matrix under *label* (once per client and label)."""
        rows = self._check_plaintext(matrix_pt)
        if "u_prime" not in MCFERingLWEEncryptionKey(self.sk):
            raise FEKeyError("encryption key lacks u_prime")
        if not isinstance(label, str):
            raise FEValidationError("label must be a string")
        return self._mark(self._encrypt_rows(self._padded_rows(rows, extra_pad=self._pad(label)), len(rows[0])))

    def decrypt(self, dct_ct: dict, dk: dict, fusion_weight: dict, label: str):
        """Recover the batch of inner products for a key bound to *label*.

        Raises:
            FESchemeError: If the key was derived for another label or the
                ciphertexts do not cover exactly the key's clients.
        """
        if not dk or "z" not in MCFERingLWEFunctionalKey(dk):
            raise FEKeyError("no decryption key provided")
        if MCFERingLWEFunctionalKey(dk).get("label") != label:
            raise FESchemeError("the key was derived for another label")
        self._check_key_label_hash(dk)
        for nid, ct in dct_ct.items():
            check_ciphertext_label_hashes((ct,), self.label_hash, nid)
        return self._decrypt_with_key_term(dct_ct, dk, MCFERingLWEFunctionalKey(dk).z, fusion_weight)

    def _decrypt_with_key_term(self, dct_ct, dk, z, fusion_weight, sk_hats=None) -> list:
        sk_y = MCFERingLWEFunctionalKey(dk).sk_y
        if set(dct_ct) != set(sk_y) or set(dct_ct) != set(fusion_weight):
            raise FESchemeError("need one ciphertext for every client of the key")
        batch = {int(RingLWECiphertext(ct).k) for ct in dct_ct.values()}
        if len(batch) != 1:
            raise FESchemeError("ciphertexts of different batch lengths")
        return self._decrypt_with_pad_term(dct_ct, sk_y, z, fusion_weight, batch.pop(), sk_hats)

    @staticmethod
    def key_chunks(lst_ndarray_ct: list) -> int:
        """Number of chunk keys needed for the output of :meth:`encrypt_lst_ndarray`."""
        return sum(len(RingLWELayer(layer).chunks) for layer in lst_ndarray_ct)

    def _check_layer(self, ary_src) -> np.ndarray:
        """The checks of :meth:`_check_plaintext` for a whole ndarray layer
        (``scale_to_int`` already returns integers); its values as ``int64``."""
        ints = scale_to_int(ary_src, self.precision).ravel()
        self._check_encryption_keys()
        if "u" not in MCFERingLWEEncryptionKey(self.sk):
            raise FEKeyError("encryption key lacks the one-time pad u")
        if ints.size and int(np.max(np.abs(ints))) > int(self.pp["bound_x"]):
            raise FEValidationError("plaintext row exceeds the bound {}".format(int(self.pp["bound_x"])))
        return ints

    def encrypt_lst_ndarray(self, lst_ndarray: list, **kwargs) -> list | None:
        """Pack ndarrays into ``ring_n``-coefficient chunks (``eta = 1``).

        Chunk ``c`` (counted across all layers) is encrypted under the chunk
        label ``(label, c)``, so each chunk has its own label pad.

        Args:
            lst_ndarray: The layers.
            **kwargs: ``label`` (required) and optionally ``precomputed``,
                the result of :meth:`precompute` for this label and these
                layer shapes (the online phase; one-time).

        Returns:
            One dict per layer with ``shape``, ``first_chunk`` and ``chunks``.
        """
        if self.pp["eta"] != 1:
            raise NotImplementedError("ndarray helper currently expects eta=1")
        label = kwargs.get("label", None)
        if not isinstance(label, str):
            raise FEValidationError("need a string label for encrypt_lst_ndarray")
        if kwargs.get("precomputed") is not None:
            return self._mark_layers(self._encrypt_precomputed(lst_ndarray, label, kwargs["precomputed"]))
        mode = self._encryption_label_hash()
        ring_n = self.pp["ring_n"]
        out, chunk = [], 0
        for ary_src in lst_ndarray:
            ints = self._check_layer(ary_src)
            n_layer = max(1, math.ceil(ints.size / ring_n))
            nkey = self._native_encryption_key() if ints.size else None
            if nkey is not None:  # one native call for the whole layer, same ciphertexts
                u_prime = MCFERingLWEEncryptionKey(self.sk).u_prime
                res = native_backend.ext().encrypt_tensor(
                    nkey, np.ascontiguousarray(ints, dtype=np.int64),
                    [label_pad_material(u_prime, chunk_label(label, chunk + j), mode) for j in range(n_layer)],
                    float(self.pp["sigma2"]), float(self.pp["sigma3"]),
                    native_backend.random_input(3 * native_backend.gauss_bytes(ring_n) * n_layer),
                    native_backend.get_threads(), self.ct_format == "binary", **native_backend.seed_kwargs(),
                    **native_backend.noise_kwargs(self.pp["sigma2"], self.pp["sigma3"], sampler=self.gaussian_sampler),
                )
                out.append(RingLWELayer.new(shape=list(np.shape(ary_src)), first_chunk=chunk,
                                            chunks=self._native_cts(res)).to_dict())
                chunk += n_layer
                continue
            flat = ints.tolist()
            cts = []
            for j in range(n_layer):
                part = flat[j * ring_n:(j + 1) * ring_n]
                if not part:
                    self._check_plaintext([part])  # an empty layer: refused as before
                rows = self._padded_rows([part], extra_pad=self._pad(chunk_label(label, chunk + j)))
                cts.append(self._encrypt_rows(rows, len(part)))
            out.append(RingLWELayer.new(shape=list(np.shape(ary_src)), first_chunk=chunk, chunks=cts).to_dict())
            chunk += n_layer
        return self._mark_layers(out)

    def _native_cts(self, res: list) -> list:
        if self.ct_format == "binary":
            return [RingLWECiphertext.new(ct=data, k=k).to_dict() for data, k in res]
        return [RingLWECiphertext.new(ct0=ct0, ct1=ct1, k=k).to_dict() for ct0, ct1, k in res]

    def precompute(self, label: str, shapes) -> "RingLWEPrecomputation":
        """Offline phase of :meth:`encrypt_lst_ndarray` for layers of these shapes.

        Everything of the ciphertexts but the plaintext term -- label pads,
        Gaussian samples, ``pk r + e_0 + Delta (u + H)`` and ``a r + e`` per
        chunk -- is computed now; the online phase,
        ``encrypt_lst_ndarray(layers, label=label, precomputed=...)``, only
        adds ``Delta x``. The ciphertexts are those of a direct
        :meth:`encrypt_lst_ndarray` with the same samples.

        The result determines the encryption randomness: keep it as secret as
        the key, and use it once (an MCFE client encrypts at most once per
        label). The use is atomic, and ``copy.copy``, ``copy.deepcopy``,
        pickling and use in another process (after ``os.fork()``) are refused
        (:mod:`pyfe4ai.core.one_time`). Natively each chunk is overwritten
        in memory when used; on the Python path the references to the chunks
        are dropped (Python integers cannot be wiped).

        Args:
            label: The label the layers will be encrypted under.
            shapes: One shape (tuple of ints, or an int for a 1-D layer) per layer.
        """
        if self.pp["eta"] != 1:
            raise NotImplementedError("ndarray helper currently expects eta=1")
        if not isinstance(label, str):
            raise FEValidationError("label must be a string")
        self._check_encryption_keys()
        sk = MCFERingLWEEncryptionKey(self.sk)
        if "u" not in sk:
            raise FEKeyError("encryption key lacks the one-time pad u")
        u_prime = sk.u_prime
        ring_n = self.pp["ring_n"]
        shapes = [tuple(int(d) for d in shape) if isinstance(shape, (tuple, list)) else (int(shape),)
                  for shape in shapes]
        firsts, total = [], 0
        for shape in shapes:
            size = int(np.prod(shape, dtype=np.int64))
            if size < 1:
                raise FEValidationError("plaintext rows must all have the same length, between 1 and ring_n")
            firsts.append(total)
            total += math.ceil(size / ring_n)
        nkey = self._native_encryption_key()
        if nkey is not None:
            cap = native_backend.ext().precompute_tensor(
                nkey, [label_pad_material(u_prime, chunk_label(label, c), self._encryption_label_hash())
                       for c in range(total)],
                float(self.pp["sigma2"]), float(self.pp["sigma3"]),
                native_backend.random_input(3 * native_backend.gauss_bytes(ring_n) * total),
                native_backend.get_threads(), **native_backend.seed_kwargs(),
                **native_backend.noise_kwargs(self.pp["sigma2"], self.pp["sigma3"], sampler=self.gaussian_sampler),
            )
            return RingLWEPrecomputation(self, label, shapes, firsts, native=(nkey, cap))
        rows = []
        for c in range(total):  # x = 0: the stored ct0 is pk r + e_0 + Delta (u + H)
            ct0, ct1 = self._encrypt_rows_python(self._padded_rows([[]], extra_pad=self._pad(chunk_label(label, c))))
            rows.append((ct0[0], ct1))
        return RingLWEPrecomputation(self, label, shapes, firsts, rows=rows)

    def _encrypt_precomputed(self, lst_ndarray: list, label: str, pre) -> list:
        """Online phase: the precomputed chunks plus ``Delta x``."""
        if not isinstance(pre, RingLWEPrecomputation) or pre.owner is not self or pre.key_src != self._key_src():
            raise FEValidationError("the precomputation belongs to another client or key")
        if pre.label != label:
            raise FEValidationError("the precomputation is for another label")
        if pre.used:
            raise FEValidationError("a precomputation can be used once")
        if [tuple(np.shape(a)) for a in lst_ndarray] != pre.shapes:
            raise FEValidationError("layer shapes differ from the precomputed ones {}".format(pre.shapes))
        layers = [self._check_layer(a) for a in lst_ndarray]
        # atomic: of two callers one gets the chunks. The Python rows are moved
        # out; the native capsule stays referenced (it marks and wipes its
        # chunks itself, see precomputed_info).
        (rows,) = pre._claim("rows")
        native = pre.native
        ring_n = self.pp["ring_n"]
        q = gp.mpz(self.pp["q"])
        delta = q // gp.mpz(self.pp["p"])
        out = []
        for ints, first, shape in zip(layers, pre.first_chunks, pre.shapes):
            if native is not None:
                nkey, cap = native
                res = native_backend.ext().encrypt_precomputed(
                    nkey, cap, first, np.ascontiguousarray(ints, dtype=np.int64),
                    native_backend.get_threads(), self.ct_format == "binary",
                )
                cts = self._native_cts(res)
            else:
                flat, cts = ints.tolist(), []
                for j in range(math.ceil(len(flat) / ring_n)):
                    part = flat[j * ring_n:(j + 1) * ring_n]
                    p_row, ct1 = rows[first + j]
                    rows[first + j] = None
                    ct0 = [(p + delta * x) % q for p, x in zip(p_row, part)] + p_row[len(part):]
                    cts.append(self._format_ciphertext([ct0], ct1, len(part)))
            out.append(RingLWELayer.new(shape=list(shape), first_chunk=first, chunks=cts).to_dict())
        return out

    def _key_src(self) -> tuple:
        return self._encryption_key()["src"]

    def compute_lst_ndarray_ct(self, dict_ndarray_ct: dict, **kwargs) -> list | None:
        """Aggregate-decrypt packed ndarray ciphertexts (needs ``dk``, ``fusion_weight``, ``label``)."""
        dk = kwargs.get("dk", None)
        fusion_weight = kwargs.get("fusion_weight", None)
        label = kwargs.get("label", None)
        if dk is None or fusion_weight is None or label is None:
            raise FEKeyError("need to provide decryption key, fusion weight and label")
        return self.decrypt_lst_ndarray_ct(
            dict_ndarray_ct, dk=dk, fusion_weight=fusion_weight, label=label
        )

    def decrypt_lst_ndarray_ct(self, dict_ndarray_ct: dict, **kwargs) -> list | None:
        """Decrypt packed ndarray ciphertexts with a key covering their chunks."""
        if self.pp["eta"] != 1:
            raise NotImplementedError("ndarray helper currently expects eta=1")
        dk = kwargs.get("dk", None)
        fusion_weight = kwargs.get("fusion_weight", None)
        label = kwargs.get("label", None)
        if dk is None or fusion_weight is None or label is None:
            raise FEKeyError("need to provide decryption key, fusion weight and label")
        key = MCFERingLWEFunctionalKey(dk)
        if key.get("label") != label:
            raise FESchemeError("the key was derived for another label")
        self._check_key_label_hash(dk)
        self._check_layers_label_hash(dict_ndarray_ct)
        sample = next(iter(dict_ndarray_ct.values()))
        if self.key_chunks(sample) > len(key.get("z_chunks", [])):
            raise FEKeyError("the key covers fewer chunks than the ciphertexts")
        sk_hats = {}  # NTT(sk_i), transformed once for all chunks
        ctx = self._native()
        result = []
        for l, layer in enumerate(sample):
            if ctx is not None:
                ary = self._decrypt_layer_native(ctx, dict_ndarray_ct, l, layer, dk, fusion_weight, sk_hats)
                if ary is not None:
                    result.append(ary)
                    continue
            values = []
            for j in range(len(RingLWELayer(layer).chunks)):
                dct_ct = {nid: RingLWELayer(dict_ndarray_ct[nid][l]).chunks[j] for nid in dict_ndarray_ct}
                z = key.z_chunks[chunk_key_index(layer, j, len(key.z_chunks))]
                values.extend(self._decrypt_with_key_term(dct_ct, dk, z, fusion_weight, sk_hats))
            ary = np.empty(len(values), dtype=object)
            ary[:] = values
            result.append(decoded_layer(ary.reshape(RingLWELayer(layer).shape), l, 10 ** self.precision))
        return result
