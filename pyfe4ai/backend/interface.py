"""The operations the scheme and primitive layers take from a backend.

Two implementations stand behind this interface:
:class:`~pyfe4ai.backend.python_impl.PythonBackend` (the reference) and
:class:`~pyfe4ai.backend.native_impl.NativeBackend` (the Rust extension),
which subclasses it and keeps the reference for every input its kernels do
not take (``super()``): that is where "native or Python" is decided for the
given parameters. :func:`pyfe4ai.backend.ops` returns the implementation
selected for the calling thread, :func:`pyfe4ai.backend.ring` its
:class:`~pyfe4ai.backend.interface.Ring` of ``(q, ring_n)``. The scheme and primitive layers call
nothing else of the backend; ``tests/backend/test_backend_interface.py`` checks that
no module outside :mod:`pyfe4ai.backend` reaches the extension, and runs
every operation through both implementations.

Each operation documents what both implementations guarantee:

* **outputs**: the same values, of the documented types, for the same inputs
  and, where randomness is drawn, the same random bytes;
* **randomness**: the source each implementation draws from, how much and in
  which order. The reference draws from the Python sources of its callers
  (:mod:`pyfe4ai.primitives.sampling`: ``_CSPRNG``, ``_OS_RANDOM``, an
  :class:`~pyfe4ai.primitives.sampling.XofRandom`); the extension from a
  per-call DRBG seeded with 32 bytes of the operating system, or from the
  test hooks ``_random_override`` / ``_seed_override`` of
  :mod:`pyfe4ai.backend` (:func:`~pyfe4ai.backend.native_impl.random_input`,
  :func:`~pyfe4ai.backend.native_impl.seed_kwargs`), with the same number of
  bytes per sample, in the same order;
* **errors**: what each implementation raises for malformed inputs;
* **secrets**: the extension wipes its copies of secret inputs (keys, pads,
  pad materials, shares, noise, residues) and the temporaries derived from
  them before it frees them, on every path, including errors; Python objects
  cannot be wiped, and the reference keeps no reference to its inputs beyond
  the call, except the handles documented as such (the caller owns them).

Two kinds of operations:

* **operations proper** -- both implementations compute the result; the
  scheme calls them unconditionally;
* **native kernels of a different step sequence** -- where the Python steps
  of a scheme check, convert or batch differently from the native call
  (element by element instead of a whole layer, values read one by one with
  other errors, an exact fallback on unusual inputs), the scheme keeps both
  step sequences and asks the backend which applies (:attr:`~pyfe4ai.backend.interface.Ring.native`,
  :attr:`~pyfe4ai.backend.interface.Ring.threshold_native`, :attr:`~pyfe4ai.backend.interface.Backend.batched_layers`,
  :meth:`~pyfe4ai.backend.interface.Backend.zq_kernels`); the reference of such a kernel is then the
  scheme's own Python steps, named in the kernel's documentation, which the
  tests compare with it. Their Python ring raises ``NotImplementedError`` (or
  returns ``None`` where documented).
"""

from __future__ import annotations

import abc

import gmpy2 as gp


class ReferenceKey:
    """The reference's Ring-LWE encryption key (:meth:`Ring.encrypt_rows`):
    ``a`` and the rows ``pk`` as mpz, the pad rows ``u`` (or ``None``),
    ``Delta``, the :class:`~pyfe4ai.primitives.lattice.ring_lwe_utils.RingNTT`
    of ``(q, ring_n)`` (``None`` without an NTT root) and ``NTT(a)``,
    ``NTT(pk_i)`` with it. Built by the scheme, in the order it reads its key."""

    __slots__ = ("a", "pk", "u", "delta", "ntt", "a_hat", "pk_hat")

    def __init__(self, a, pk, u, delta, ntt, a_hat, pk_hat) -> None:
        self.a, self.pk, self.u, self.delta = a, pk, u, delta
        self.ntt, self.a_hat, self.pk_hat = ntt, a_hat, pk_hat


class Ring(abc.ABC):
    """``Z_q[x] / (x^ring_n + 1)`` of one implementation (:func:`pyfe4ai.backend.ring`).

    Values cross the interface as Python ints, ``gmpy2.mpz`` or decimal
    strings (inputs) and as decimal strings or mpz in ``[0, q)`` (outputs), or
    as the binary words of :func:`~pyfe4ai.core.formats.binary.encode_values`
    (``ceil(bits(q) / 8)`` little-endian bytes per value) where *binary* is
    set. Handles (native encryption keys, precomputations, NTT transforms)
    belong to the ring that made them.
    """

    #: Whether the extension computes this ring's operations.
    native = False
    #: Whether the threshold MCFE kernels run natively on this ring (a native
    #: ring other than the two-prime context of the MIFE / MCFE prototype).
    threshold_native = False

    def __init__(self, q, ring_n: int) -> None:
        self.q = gp.mpz(q)
        self.ring_n = ring_n

    @abc.abstractmethod
    def reference(self) -> "Ring":
        """The Python ring of the same ``(q, ring_n)`` (this ring if it is one)."""

    # -- values mod q ------------------------------------------------------------

    @abc.abstractmethod
    def hash_to_zq(self, material: bytes, count: int, mode: str = "shake256-wide") -> list:
        """The *count* values in ``[0, q)`` of
        :func:`~pyfe4ai.primitives.hashing.hash_to_zq_vector` for the XOF input
        *material* (:func:`~pyfe4ai.primitives.hashing.hash_to_zq_material`)
        and the label-hash *mode*, as mpz. Deterministic. The extension wipes
        its words."""

    def prepare_rows(self, rows: list):
        """*rows* (of ``ring_n`` values) in the form :meth:`~pyfe4ai.backend.interface.Ring.lincomb` reads
        fastest, for rows combined many times: the rows themselves (Python),
        packed words (native).

        Raises:
            KeyError, TypeError, ValueError: Natively, for rows the extension
                does not read (the caller then combines the rows themselves
                with :meth:`~pyfe4ai.backend.interface.Ring.reference`)."""
        return rows

    @abc.abstractmethod
    def lincomb(self, rows, coeffs: list) -> list:
        """``sum_r coeffs[r] rows[r] mod q`` (rows of ``ring_n`` values, or
        :meth:`~pyfe4ai.backend.interface.Ring.prepare_rows`'s) as ``ring_n`` decimal strings. *rows* may be
        an iterator: each row is read when it is reached.

        The reference accumulates ``coeffs[r] * gmpy2.mpz(v)`` row by row
        (``zip``, so a longer row is cut to ``ring_n``) and raises gmpy2's
        ``ValueError`` / ``TypeError`` for a malformed value; the extension
        packs each row first and raises its own."""

    @abc.abstractmethod
    def label_terms(self, base: list, materials: list, weights: list, eta: int, mode: str,
                    binary: bool = False) -> list:
        """Per label ``l``: ``base + sum_i sum_r weights[i][r] H_r(materials[l][i]) mod q``,
        ``H`` the ``eta`` rows of ``ring_n`` values of :meth:`~pyfe4ai.backend.interface.Ring.hash_to_zq`
        (*mode*), as decimal strings or (*binary*) words: the MCFE key terms
        ``z'' = z + sum_i y_i H(u'_i, l)`` and the dMCFE pad and GDSum terms.

        The base (``sum_r y_r u_r``) and the materials (a client's ``u'``,
        the pairwise keys of dMCFE) are secret: the extension wipes its copies."""

    # -- Ring-LWE encryption (MIFE, MCFE, SIFE) ----------------------------------

    @abc.abstractmethod
    def samples_natively(self, *sigmas, sampler: str) -> bool:
        """Whether the encryption kernels sample Gaussians of these widths with
        *sampler* natively (widths up to ``2^40``; the rounded or, where built,
        the FACCT sampler): the caller then builds an :meth:`~pyfe4ai.backend.interface.Ring.encryption_key`."""

    def encryption_key(self, a: list, pk: list, u, delta, hash_mode=None):
        """The native encryption key of ``(a, pk, u, Delta)`` -- ``NTT(a)``,
        ``NTT(pk_i)``, the pad ``u`` (or ``None``) and the label-hash mode of
        the pads (MCFE) --, or ``None``: then the encryption operations run the
        reference with the caller's :class:`~pyfe4ai.backend.interface.ReferenceKey`. ``None`` for a
        Python ring and for a key the extension does not read (the reference
        then reports it). The key holds secrets (``u``) and wipes them when freed."""
        return None

    @abc.abstractmethod
    def encrypt_rows(self, key, reference, w_rows: list, eta: int, sigmas: tuple, sampler: str, binary: bool):
        """BMMS21 Ring-LWE encryption of *w_rows* (``eta`` rows of ``ring_n``
        integers): ``ct0_i = pk_i r + e_i + Delta w_i``, ``ct1 = a r + e`` mod
        q, ``r, e`` of width ``sigmas[0]``, the ``e_i`` of width ``sigmas[1]``.
        Returns ``(ct0 rows, ct1)`` as decimal strings, or the bytes of
        :func:`~pyfe4ai.core.formats.binary.encode_binary_ciphertext` (*binary*).

        With :meth:`~pyfe4ai.backend.interface.Ring.encryption_key`'s *key* (and ``eta`` rows of ``ring_n``
        values): one native call, which samples ``r``, the noise rows and
        ``e`` in that order from ``gauss_bytes(ring_n) (2 + eta)`` bytes of
        its DRBG (or the test bytes), with the rounded sampler on 7 bytes per
        uniform or the FACCT tables of
        :func:`~pyfe4ai.backend.native_impl.noise_kwargs`. Else the reference:
        ``reference()`` (a callable returning the :class:`~pyfe4ai.backend.interface.ReferenceKey`), and
        :func:`~pyfe4ai.primitives.sampling.discrete_gaussian_matrix` with
        *sampler* for ``r``, the ``eta`` noise rows and ``e``, in that order;
        ``r`` is transformed once and multiplied with ``NTT(pk_i)`` and
        ``NTT(a)`` (the products of ``poly_mul`` with the same samples)."""

    @abc.abstractmethod
    def encrypt_tensor(self, key, reference, ints, materials: list, mode: str, sigmas: tuple, sampler: str,
                       binary: bool) -> list:
        """MCFE Ring-LWE encryption of a layer (``eta = 1``): chunk ``j`` of
        ``ring_n`` values of the ``int64`` array *ints* encrypted as
        :meth:`~pyfe4ai.backend.interface.Ring.encrypt_rows` the row ``x_j + u + H(materials[j]) mod q``
        (``x_j`` zero-extended; ``H``: :meth:`~pyfe4ai.backend.interface.Ring.hash_to_zq` with *mode*).
        Returns ``[(ct0 rows, ct1, k)]`` (decimal strings) or ``[(bytes, k)]``
        (*binary*), one per material. The reference encrypts chunk by chunk
        (draws as :meth:`~pyfe4ai.backend.interface.Ring.encrypt_rows`, chunk after chunk); natively one call
        (chunk ``c`` reads DRBG stream ``c``; test bytes: ``3
        gauss_bytes(ring_n)`` per chunk, in chunk order). The pad and the
        rows are secret: the extension wipes them."""

    @abc.abstractmethod
    def precompute_tensor(self, key, reference, materials: list, mode: str, sigmas: tuple, sampler: str) -> tuple:
        """The offline phase of :meth:`~pyfe4ai.backend.interface.Ring.encrypt_tensor` for ``len(materials)``
        chunks (``x = 0``): ``(native, rows)`` -- the native handle
        (:class:`~pyfe4ai.backend.native_impl.NativePrecomputation`, ``(key,
        capsule)``; the capsule marks every chunk used and wipes it when it is
        used) and ``None``, or ``None`` and the reference's list of ``(pk r +
        e_0 + Delta (u + H), a r + e)`` per chunk. Draws as
        :meth:`~pyfe4ai.backend.interface.Ring.encrypt_tensor`. Secret and one-time
        (:class:`~pyfe4ai.core.one_time.OneTimePrecomputation`)."""

    @abc.abstractmethod
    def encrypt_precomputed(self, native, rows, first: int, ints, delta, binary: bool) -> list:
        """The online phase: chunks ``first, first + 1, ...`` of the handles
        of :meth:`~pyfe4ai.backend.interface.Ring.precompute_tensor` plus ``Delta x`` for the ``int64`` values
        *ints*, as :meth:`~pyfe4ai.backend.interface.Ring.encrypt_tensor` returns them. The native handle runs
        its kernel whatever the selection (it marks and wipes each chunk; a
        chunk used twice raises ``ValueError``); the reference drops each
        chunk from *rows* as it uses it."""

    # -- Ring-LWE decryption (native kernels of a different step sequence) --------

    def decryption_key_hat(self, poly):
        """The transform of a functional-key polynomial that
        :meth:`~pyfe4ai.backend.interface.Ring.decrypt_tensor` takes (``NTT`` in Montgomery form), or ``None``
        when the extension does not read *poly* (and for a Python ring)."""
        return None

    def decrypt_tensor(self, chunks: list, zs: list, ks: list, hats: list, ys: list, delta, bound: int):
        """``Dec2(sum_i (sum_r y_{i,r} ct0_{i,r} - ct1_i sk_i) - Delta z)`` of
        every chunk of a layer (MIFE / MCFE Ring-LWE) in one native call:
        ``(values, mask)``, the bytes of the ``int64`` values and of a
        ``uint8`` mask of those that do not decode (``None`` if all do); or
        ``None`` when the extension does not decrypt them (a Python ring,
        layouts or values it does not read, values beyond an ``int64``). The
        reference is the scheme's own loop
        (``MIFERingLWE._decrypt_with_pad_term``).

        Raises:
            ValueError: ``("malformed ciphertext", slot)`` for a ciphertext
                the extension cannot read.
        """
        return None

    # -- threshold MCFE over Ring-LWE (native kernels of a different step sequence)

    def ntt_hat(self, poly):
        """``NTT(poly)`` of a key-share polynomial (values, or binary words)
        in the form the threshold kernels take.

        Raises:
            TypeError, ValueError: For values the extension does not read."""
        raise NotImplementedError("a native kernel of the threshold MCFE")

    def shamir_share(self, values, threshold: int, points: list, binary: bool) -> list:
        """Shamir shares of every value (ints or decimal strings, or the words
        of a bytes object): per point, its share values in order, as decimal
        strings or (*binary*) one bytes object of words. Randomness: the
        ``threshold - 1`` coefficients of each value, from stream 0 of the
        per-call DRBG (or the test bytes), drawn in the order and with the
        rejection sampling of ``random.SystemRandom.randrange(0, q)``. The
        reference: :func:`~pyfe4ai.schemes.threshold.sharing.share_polys`'s
        :func:`~pyfe4ai.schemes.threshold.sharing.share_secret` per value.

        Raises:
            FEKeyError: For malformed key material.
            RuntimeError: If the test bytes run out.
        """
        raise NotImplementedError("a native kernel of the threshold MCFE")

    def floods_natively(self, width: int) -> bool:
        """Whether the partial-decryption kernels draw the uniform flooding of
        *width* bits themselves (``flood=(width, material)``)."""
        return False

    def threshold_inputs(self, keys: list, ks: list, rows: list, eta: int):
        """JSON ciphertexts of a partial decryption read natively: ``(packed,
        ct_text, key_text)`` (the texts of
        :func:`~pyfe4ai.schemes.threshold.codec.ciphertexts_digest`), or
        ``None`` when the reference reads them
        (:func:`~pyfe4ai.schemes.threshold.codec.read_json_ciphertexts`)."""
        raise NotImplementedError("a native kernel of the threshold MCFE")

    def threshold_partial(self, packed: list, eta: int, ys: list, sk_hats: list, z_share, delta, multiplier,
                          noise=None, flood=None) -> tuple:
        """``(w, c_y)`` of a partial decryption of the ciphertexts of
        :meth:`~pyfe4ai.backend.interface.Ring.threshold_inputs` as decimal strings: ``multiplier (Delta z +
        sum_i ct1_i sk_i) + noise`` and ``sum_i sum_r y_{i,r} ct0_{i,r}``.
        *noise* is the flooding polynomial, or *flood* ``(bits, material)``
        draws it natively (the values of
        :func:`~pyfe4ai.schemes.threshold.roles.uniform_flooding_poly`). The
        reference: the Python steps of
        :meth:`~pyfe4ai.schemes.mcfe.ring_lwe_threshold.ThresholdMCFERingLWE.share_decrypt`.
        The key share, its transforms and the noise are secret: wiped.

        Raises:
            TypeError, ValueError: For a key share the extension does not read.
        """
        raise NotImplementedError("a native kernel of the threshold MCFE")

    def threshold_partial_binary(self, cts: list, eta: int, ys: list, sk_hats: list, z_share, delta, multiplier,
                                 noise=None, flood=None) -> tuple:
        """:meth:`~pyfe4ai.backend.interface.Ring.threshold_partial` of format-2 ciphertexts (*cts*: their
        bytes), as words: ``(w, c_y, k)``.

        Raises:
            ValueError: ``("malformed ciphertext", client)``, or for a key
                share the extension does not read.
        """
        raise NotImplementedError("a native kernel of the threshold MCFE")

    def threshold_combine(self, c_y, ws: list, delta, bound, k: int):
        """``decode(c_y - sum w)`` of JSON partial decryptions: the first *k*
        values (``None`` for those beyond the bound), or ``None`` for values or
        layouts the reference reads (``json_partial_difference`` and
        ``decode_scaled``)."""
        raise NotImplementedError("a native kernel of the threshold MCFE")

    def threshold_combine_binary(self, c_y: bytes, ws: list, delta, bound, k: int):
        """:meth:`~pyfe4ai.backend.interface.Ring.threshold_combine` of binary partial decryptions (``k``
        words each), or ``None`` for a value beyond an ``int64``.

        Raises:
            ValueError: For a malformed partial decryption.
        """
        raise NotImplementedError("a native kernel of the threshold MCFE")


class Backend(abc.ABC):
    """The operations of one implementation (:func:`pyfe4ai.backend.ops`)."""

    #: ``"python"`` or ``"native"``.
    name = "python"

    #: ``True`` if whole tensor layers go through one call (the dense-LWE
    #: label terms of :meth:`~pyfe4ai.backend.interface.Backend.label_dots`,
    #: the ristretto255 MCFE layers of
    #: :meth:`~pyfe4ai.backend.interface.Backend.r255_mcfe_encrypt` and
    #: :meth:`~pyfe4ai.backend.interface.Backend.r255_mcfe_decrypt`). The
    #: scheme then runs its batched steps, with the checks of a whole layer,
    #: and its element-by-element steps otherwise.
    batched_layers = False

    @abc.abstractmethod
    def ring(self, q, ring_n: int) -> Ring:
        """The ring ``Z_q[x] / (x^ring_n + 1)`` (the same object for the same arguments)."""

    # -- lattices and sampling -------------------------------------------------------

    @abc.abstractmethod
    def ntt(self, a: list, q, root, inverse: bool = False) -> list:
        """The NTT (or its inverse) of :mod:`~pyfe4ai.primitives.lattice.ring_lwe_utils`
        of *a* with *root*, as mpz in ``[0, q)``: natively where the extension
        has a context for ``(q, len(a))`` with this root."""

    @abc.abstractmethod
    def poly_mul(self, a: list, b: list, q) -> list:
        """:func:`~pyfe4ai.primitives.lattice.ring_lwe_utils.poly_mul` (with
        its warning when ``q`` has no NTT root)."""

    @abc.abstractmethod
    def gaussian_bigint(self, sigma, count: int, rng, prec: int) -> list:
        """The rounded samples of
        :func:`~pyfe4ai.primitives.sampling.discrete_gaussian_bigint` (``count
        > 0``): two ``getrandbits(prec)`` per sample, from *rng* (``None``: the
        module's ``_CSPRNG``; natively its per-call DRBG, stream ``c`` for
        samples ``256 c ..``, or an :class:`~pyfe4ai.primitives.sampling.XofRandom`'s
        stream, whose counter advances as in Python)."""

    @abc.abstractmethod
    def gaussian_facct(self, sigma, count: int, rng) -> list:
        """The samples of :func:`~pyfe4ai.primitives.sampling.discrete_gaussian_facct`
        (``count > 0``), drawing ``getrandbits`` in its order from *rng*
        (``None``: the module's ``_OS_RANDOM``; natively the per-call DRBG)."""

    # -- threshold key sharings over Z_q ---------------------------------------------

    def zq_kernels(self, q) -> bool:
        """Whether the extension does the ``Z_q`` arithmetic of the replicated
        sharings for *q* (the dealing and sums of
        :mod:`~pyfe4ai.schemes.threshold.sharing`)."""
        return False

    def replicated_deal(self, q, values, n_random: int, binary: bool):
        """The pieces of a replicated sharing of the flat *values* (ints or
        decimal strings, or the words of a bytes object), natively: the
        correction ``values - sum`` first, then ``n_random`` uniform pieces,
        as lists of decimal strings or (*binary*) bytes; ``None`` when the
        reference reads the values instead (the Python steps of
        :func:`~pyfe4ai.schemes.threshold.sharing.share_replicated`, and
        always on the Python backend).

        Randomness: every random value as ``random.SystemRandom.randrange(0,
        q)`` draws it, in the order of the Python steps, from stream 0 of the
        per-call DRBG (or the test bytes). The values, the pieces, their digits
        and the caller's bytes are wiped.

        Raises:
            FEKeyError: For binary values that are not whole words below q.
            RuntimeError: If the test bytes run out.
        """
        return None

    @abc.abstractmethod
    def zq_sum(self, q, count: int, rows, materials: list, subtract: bool = False, binary: bool = False,
               reference: bool = True):
        """``sum(rows) +- sum_m H(m)`` mod q, ``H``: ``count`` values of
        :meth:`~pyfe4ai.backend.interface.Ring.hash_to_zq` (``"shake256-wide"``), as decimal strings; with
        *binary* the sum of rows of words, as words. The reference converts
        with ``gmpy2.mpz`` row by row and raises ``FESchemeError`` for a row
        of another length than *count*. Without *reference*: ``None`` instead
        of the reference for what the extension does not take (the caller
        then has reference steps of its own). The materials hold seeds: the
        extension wipes its copies."""

    # -- ristretto255 ------------------------------------------------------------------

    @abc.abstractmethod
    def r255_add(self, a, b):
        """``a + b``: group elements as 32-byte encodings (native) or points
        (Python); both implementations read both forms
        (:mod:`~pyfe4ai.primitives.groups.ec_group`)."""

    @abc.abstractmethod
    def r255_sub(self, a, b):
        """``a - b``."""

    @abc.abstractmethod
    def r255_neg(self, a):
        """``-a``."""

    @abc.abstractmethod
    def r255_mul(self, point, k: int):
        """``k point`` (*k* an integer, reduced mod the group order natively)."""

    @abc.abstractmethod
    def r255_mul_base(self, k: int):
        """``k G``."""

    @abc.abstractmethod
    def r255_decode(self, data: bytes):
        """The element of a canonical encoding.

        Raises:
            ValueError: If *data* is not one."""

    @abc.abstractmethod
    def r255_hash(self, msg: bytes, tag: bytes):
        """RFC 9380 ``hash_to_ristretto255(msg, tag)``."""

    @abc.abstractmethod
    def r255_dlog(self, point, bound: int):
        """The ``x`` with ``x G = point`` and ``|x| <= bound``, or ``None``."""

    # -- batched layers (see batched_layers) -----------------------------------------

    @abc.abstractmethod
    def dense_key(self, key_row: list, q):
        """A dense-LWE key row in the form :meth:`~pyfe4ai.backend.interface.Backend.label_dots` reads (natively
        packed words with the row; secret, cached by the caller per key and
        pickled with it, so built of plain values only)."""

    @abc.abstractmethod
    def label_dots(self, materials: list, key, q, dim: int, mode: str) -> list:
        """``<key_row, H(m)> mod q`` for every material ``m`` (*key*: the
        :meth:`~pyfe4ai.backend.interface.Backend.dense_key` of ``key_row``; ``H``: *dim* values of
        :func:`~pyfe4ai.primitives.hashing.hash_to_zq_vector` with *mode*), as
        mpz: the dense-LWE MCFE label terms. Natively the products are summed
        before the hash blocks are reduced (congruent mod q); a key row of
        another length than *dim* goes to the reference."""

    @abc.abstractmethod
    def r255_mcfe_encrypt(self, msgs: list, dst: bytes, values: list, s0: int, s1: int) -> list:
        """MCFE ristretto255 ciphertexts ``x G + s0 H0(l) + s1 H1(l)`` of the
        values ``x`` under the labels of *msgs* (``H_k(l)``:
        :meth:`~pyfe4ai.backend.interface.Backend.r255_hash` of ``bytes([k]) + msg``), as hex strings."""

    @abc.abstractmethod
    def r255_mcfe_decrypt(self, cts: list, ys: list, d0: int, d1: int, msgs: list, dst: bytes, bound: int):
        """The decryptions ``dlog(sum_i y_i ct_i - d0 H0(l) - d1 H1(l))`` of a
        layer (``cts[i][e]``: client ``i``'s hex ciphertext of element ``e``):
        ``(values, mask)`` as in :meth:`~pyfe4ai.backend.interface.Ring.decrypt_tensor`.

        Raises:
            ValueError: ``("malformed ciphertext", client)``.
        """

    # -- BLS12-381 -----------------------------------------------------------------------

    @abc.abstractmethod
    def bls12_381(self):
        """The BLS12-381 kernels (:class:`~pyfe4ai.backend.native_impl.Bls12381Kernels`),
        or ``None`` when the extension is not built: the pairing group has no
        Python implementation, so both backends return the extension's."""

    # -- garbled circuits (bcfe) ---------------------------------------------------------

    @abc.abstractmethod
    def gc_garble(self, circuit: bytes, mode: str, salt: bytes, x: bytes) -> dict:
        """Half-gates garbling (ZRE15 Fig. 2) of a ``CIRC v1`` circuit for the
        data bits *x*: :func:`~pyfe4ai.primitives.garbling.garble` with a
        fresh seed.

        Args:
            circuit: The ``CIRC v1`` encoding (usually of ``guard(C)``).
            mode: ``"hardwired"`` (data wires known to the garbler) or ``"labels"``.
            salt: The 56-byte instance salt ``pp_digest || ct_nonce ||
                u64(slot_id)`` of the gate hash, opaque here.
            x: The ``n`` data bits, packed least significant bit first.

        Returns:
            ``{"delta": 16 B, "key0": 16 B x m (the zero labels of the key
            wires, in wire order), "data_active": 16 B x n (mode ``"labels"``)
            or b"", "tables": 16 B x table entries, "decode": the n_out
            decoding bits packed}``, all bytes.

        Randomness: one 32-byte seed per call, the whole randomness of the
        garbling (expanded by :func:`~pyfe4ai.primitives.garbling.expand_seed`):
        from the operating system (:func:`os.urandom` in the reference, the
        extension's OS source natively), or ``override(32)`` when a test set
        ``pyfe4ai.backend._random_override`` -- read once, before the inputs
        are checked, by both implementations (:func:`~pyfe4ai.backend.native_impl.random_input`
        natively). There is no seed parameter: a fixed seed is for the tests
        only (``_seed_override`` does not apply). The same seed gives the
        same bytes in both implementations.

        Secrets: *x*, the seed and the returned ``delta``, ``key0`` and
        ``data_active``; the extension wipes its copies of the seed, ``Delta``
        and the labels. ``delta`` is returned for the known-answer tests;
        the encryption path of the schemes keeps it inside the extension.

        Raises:
            TypeError: For inputs that are not bytes.
            ValueError: For a malformed circuit
                (:func:`~pyfe4ai.primitives.garbling.parse_circuit`), an
                unknown mode, a salt of another length than 56 bytes, *x*
                not ``ceil(n / 8)`` bytes or with nonzero padding bits, an
                output wire that is not a labelled wire in *mode*, or test
                bytes of another length than 32.
        """

    @abc.abstractmethod
    def gc_evaluate(self, circuit: bytes, mode: str, salt: bytes, tables: bytes, decode: bytes, key_active: bytes,
                    data_active: bytes) -> bytes:
        """Evaluate a garbling of :meth:`~pyfe4ai.backend.interface.Backend.gc_garble`:
        :func:`~pyfe4ai.primitives.garbling.evaluate`. *key_active* holds
        one 16-byte active label per key wire (``m``), *data_active* one per
        data wire (``n``) in mode ``"labels"`` and is ``b""`` in mode
        ``"hardwired"``. Returns the ``n_out`` output bits, packed least
        significant bit first. Deterministic.

        Secrets: the active key labels (they determine ``w``); the extension
        wipes its copies.

        Raises:
            TypeError: For inputs that are not bytes.
            ValueError: For a malformed circuit, an unknown mode, a salt of
                another length than 56 bytes, an output wire that is not a
                labelled wire, another number of table entries than the
                circuit has, another number of key or data labels, or
                decoding bits that are not ``n_out`` packed bits.
        """

    # -- bounded-key FE: slot public keys and the label KEM (bcfe) -------------------------

    @abc.abstractmethod
    def bcfe_new_slot(self, setup_digest: bytes, slot_id: int, w: bytes, *, m: int) -> dict:
        """Issue a slot for the key bits *w* (h1 with h', plan 0.6 P1 section
        3.9): :func:`~pyfe4ai.primitives.bcfe_kem.new_slot`. The center draws
        the secret keys of the selected positions ``(j, w_j)`` and of the g
        bit; the unselected positions are ``H_P(setup_digest, slot_id, j,
        k)`` (:func:`~pyfe4ai.primitives.bcfe_kem.unselected_pk`), whose
        discrete logarithms nobody knows.

        Args:
            setup_digest: The 32-byte setup digest
                (:func:`~pyfe4ai.primitives.bcfe_kem.setup_digest`).
            slot_id: The slot number, in ``[0, 2^64)``.
            w: The ``m`` key bits, packed least significant bit first.
            m: The number of key bits of ``C`` (``guard(C)`` has ``m + 1``
                key wires), in ``[1, 2^31 - 1)``: the packed *w* does not
                determine it.

        Returns:
            ``{"scalars": 32 B x (m + 1), "pks": 32 B x (2m + 1), "k": 16
            B}``, all bytes: the secret keys of ``(j, w_j)`` for ``j = 0 ..
            m - 1`` and then of ``(g, 0)``, canonical little-endian below the
            group order; the table ``P[j, 0] || P[j, 1]`` for ``j = 0 .. m -
            1`` and then ``P[m, 0]`` (32-byte encodings); the slot secret
            ``k``.

        Randomness: one read of ``64 (m + 1) + 16`` bytes per call, after
        *m* is checked and before the other inputs are checked: from the
        operating system (:func:`os.urandom` in the reference, the
        extension's OS source natively), or ``override(64 (m + 1) + 16)``
        when a test set ``pyfe4ai.backend._random_override`` (natively
        :func:`~pyfe4ai.backend.native_impl.random_input`). In that order:
        64 bytes per secret key, ``j = 0 .. m - 1`` then the g bit, each read
        little-endian and reduced mod the group order (a key 0 raises
        ``ValueError``, there is no redraw: probability 2^-252); then the 16
        bytes of ``k``. The same bytes give the same output in both
        implementations.

        Secrets: *w* (the table does not reveal it), the random bytes, the
        returned ``scalars`` and ``k`` (never published); the extension
        wipes its copies.

        Raises:
            TypeError: For inputs that are not bytes.
            ValueError: For *m* out of range, a setup digest of another
                length than 32 bytes, *slot_id* out of range, *w* not
                ``ceil(m / 8)`` bytes or with nonzero padding bits, test
                bytes of another length, or a secret key 0.
        """

    @abc.abstractmethod
    def bcfe_place_slot(self, setup_digest: bytes, slot_id: int, w: bytes, holder_pks: bytes, *, m: int) -> dict:
        """Issue a slot from the key holder's public keys (h2 with h'):
        :func:`~pyfe4ai.primitives.bcfe_kem.place_slot`. The holder's keys go
        to ``(j, w_j)`` and ``(g, 0)``; each must be the canonical encoding
        of a group element other than the identity (I8, no proof of
        knowledge asked); the unselected positions are derived from a fresh
        ``k``.

        Args:
            setup_digest, slot_id, w, m: As :meth:`~pyfe4ai.backend.interface.Backend.bcfe_new_slot`.
            holder_pks: ``32 B x (m + 1)``: the holder's keys for ``(j,
                w_j)``, ``j = 0 .. m - 1``, then for ``(g, 0)``.

        Returns:
            ``{"pks": 32 B x (2m + 1), "k": 16 B}`` as
            :meth:`~pyfe4ai.backend.interface.Backend.bcfe_new_slot`; the
            center hands ``k`` to the holder for
            :meth:`~pyfe4ai.backend.interface.Backend.bcfe_unselected_ok`.

        Randomness: the 16 bytes of ``k``, one read per call before the
        inputs are checked (the operating system, or ``override(16)``).

        Secrets: *w*, ``k``; the extension wipes its copies.

        Raises:
            TypeError: For inputs that are not bytes.
            ValueError: For malformed fields as
                :meth:`~pyfe4ai.backend.interface.Backend.bcfe_new_slot`,
                *holder_pks* of another length than ``32 (m + 1)`` bytes, a
                holder key that is not a canonical encoding or is the
                identity, or test bytes of another length.
        """

    @abc.abstractmethod
    def bcfe_unselected_ok(self, setup_digest: bytes, slot_id: int, w: bytes, k: bytes, pks: bytes, *,
                           m: int) -> bool:
        """The h' part of the key holder's CheckAuth (I13):
        :func:`~pyfe4ai.primitives.bcfe_kem.unselected_ok`. ``True`` if the
        table *pks* has exactly ``2m + 1`` entries (``32 (2m + 1)`` bytes)
        and every unselected position ``(j, 1 - w_j)`` holds
        ``H_P(setup_digest, slot_id, j, k)``; a table of another length is
        ``False``. The caller checks the rest: its own keys at ``(j, w_j)``
        and ``(g, 0)``. Deterministic.

        Secrets: *w* and *k*; the extension wipes its copies.

        Raises:
            TypeError: For inputs that are not bytes.
            ValueError: For *m* out of range, a setup digest of another
                length than 32 bytes, *slot_id* out of range, *k* of another
                length than 16 bytes, *w* not ``ceil(m / 8)`` bytes or with
                nonzero padding bits.
        """

    @abc.abstractmethod
    def bcfe_encrypt_slot(self, circuit: bytes, mode: str, pp_digest: bytes, ct_nonce: bytes, slot_id: int,
                          x: bytes, pks: bytes) -> dict:
        """One slot's component of a ciphertext (garbling and label
        encryption fused): :func:`~pyfe4ai.primitives.bcfe_kem.encrypt_slot`
        with a fresh seed. ``guard(C)`` is garbled with the instance salt
        ``pp_digest || ct_nonce || u64(slot_id)``; the labels of its key
        wires are encrypted under the table with one ephemeral key ``R``:
        ``c[j, b] = L_j^b xor H_K(pp_digest, ct_nonce, slot_id, j, b, R, r
        P[j, b])`` for ``j = 0 .. m - 1``, ``b = 0, 1``, then ``c[m, 0]``
        alone (``L_g^1`` is never encrypted or sent, I3).

        Args:
            circuit: The ``CIRC v1`` encoding of ``guard(C)`` (``m + 1`` key
                wires).
            mode: ``"hardwired"`` or ``"labels"``.
            pp_digest: 32 bytes; ct_nonce: 16 bytes; slot_id: in ``[0,
                2^64)``.
            x: The ``n`` data bits, packed least significant bit first.
            pks: The slot's table, ``32 B x (2m + 1)``.

        Returns:
            ``{"R": 32 B, "label_cts": 16 B x (2m + 1), "tables": 16 B x
            table entries, "decode": the n_out decoding bits packed,
            "data_active": 16 B x n (mode ``"labels"``) or b""}``, all bytes.

        Randomness: one 32-byte seed per call, before the inputs are
        checked (the operating system, or ``override(32)``), the whole
        randomness of the component: the garbling
        (:func:`~pyfe4ai.primitives.garbling.expand_seed`) and ``r``
        (:func:`~pyfe4ai.primitives.bcfe_kem.ephemeral_scalar`). There is no
        seed parameter: a fixed seed is for the tests only (I16). The same
        seed gives the same bytes in both implementations.

        Secrets: *x*, the seed, ``Delta``, ``r`` and the zero labels: none of
        them is returned; the extension wipes them. The returned
        ``data_active`` (mode ``"labels"``) is part of the ciphertext.

        Raises:
            TypeError: For inputs that are not bytes.
            ValueError: For a malformed circuit or one that is not
                ``guard(C)`` (:func:`~pyfe4ai.primitives.bcfe_kem.guarded_key_bits`),
                an unknown mode, fields of another length, *slot_id* out of
                range, *x* not ``ceil(n / 8)`` bytes or with nonzero padding
                bits, *pks* of another length than ``32 (2m + 1)`` bytes or
                with an entry that is not a canonical encoding or is the
                identity, test bytes of another length, or ``r = 0``.
        """

    @abc.abstractmethod
    def bcfe_decrypt_slot(self, circuit: bytes, mode: str, pp_digest: bytes, ct_nonce: bytes, slot_id: int,
                          R: bytes, label_cts: bytes, tables: bytes, decode: bytes, data_active: bytes,
                          scalars: bytes, w: bytes) -> bytes:
        """Decrypt one slot's component with the slot's key:
        :func:`~pyfe4ai.primitives.bcfe_kem.decrypt_slot`. Opens the ``m +
        1`` active labels of ``w || 0`` (``c[j, w_j] xor H_K(..., s_j R)``,
        then ``c[m, 0]``) and evaluates the garbling. Returns the ``n_out``
        output bits, packed least significant bit first: ``C(w, x)`` for a
        well-formed component, unrelated bits for one of another slot,
        ciphertext or key. Deterministic.

        Args:
            circuit, mode, pp_digest, ct_nonce, slot_id: As
                :meth:`~pyfe4ai.backend.interface.Backend.bcfe_encrypt_slot`.
            R, label_cts, tables, decode, data_active: The component.
            scalars: ``32 B x (m + 1)``: the slot's secret keys.
            w: The ``m`` key bits, packed.

        Secrets: *scalars*, *w* and the opened labels; the extension wipes
        its copies.

        Raises:
            TypeError: For inputs that are not bytes.
            ValueError: For a malformed circuit or one that is not
                ``guard(C)``, an unknown mode, any field of another length,
                ``R`` not a canonical encoding or the identity, a scalar not
                below the group order, another number of table entries or
                data labels, decoding bits that are not ``n_out`` packed
                bits, *w* with nonzero padding bits.
        """

    # -- the tests' hooks (appended, 0.6 review R06-03) ------------------------------------

    def random_override_active(self) -> bool:
        """Whether a test has set ``pyfe4ai.backend._random_override``: the
        operations that read it (``gc_garble``, the ``bcfe_*`` operations and
        every native draw of :func:`~pyfe4ai.backend.native_impl.random_input`,
        in both implementations) then take their random bytes from the test,
        not from the operating system. A scheme that must never run on such
        bytes asks before it draws (``bcfe/gc_ristretto255`` refuses). The
        other hook, ``_seed_override`` (the DRBG seed of other native kernels,
        which the ``bcfe_*`` operations do not read), is not reported. The
        scheme and primitive layers read it through
        :func:`pyfe4ai.backend.ops`, as every operation."""
        from pyfe4ai import backend

        return backend._random_override is not None
