"""Parts shared by the ABDP15 LWE IPFEs: ``sife/lwe``, ``mife/lwe``, ``mife/lwe_threshold``.

The public matrix ``A in Z_q^{m x lwe_n}`` is published as a 32-byte seed and
expanded by :class:`pyfe4ai.primitives.lattice.lwe_matrix.SeededMatrix` (FrodoKEM-style;
the expansion is modelled as a random oracle). Encryption draws a binary
``r`` and needs ``A^T r`` and ``pk^T r``; both are computed exactly with
numpy, for a whole batch of plaintexts at once.

``matrix_mode`` (key generator and crypto configs; default ``"memory"``)
chooses between keeping the expanded ``A`` in memory (fast, ``8 L`` bytes per
entry with ``L = ceil(bits(q) / 31)``) and expanding it again for every batch
(``"stream"``: little memory, slow). The public keys ``pk = A S + E``
(``m x eta`` per slot) are always held in memory.

Under the ``standard`` security profile ``lwe_n = "auto"`` selects the
smallest dimension whose ``(q, sigma')`` meets
:data:`pyfe4ai.primitives.lattice.lwe_utils.WIDE_LWE_128_LOG_RATIO`, a lattice-estimator
table for LWE with a uniform secret; an explicit ``lwe_n`` is checked against
it. ``sigma' = sigma / (1 + B sqrt(eta))`` is the error width of the LWE
problem the security proof reduces to
(:func:`pyfe4ai.primitives.lattice.lwe_utils.abdp15_reduction_sigma`), where ``B`` bounds
how far the two challenge vectors of the IPFE game differ per coordinate:
``2 X`` for ``sife/lwe``, and ``4 X`` in the slots of ``mife/lwe``, whose
reduction (ACFGU18, proof of Thm. 3) asks for ``x^(j,0) - x^(1,0) + x^(1,1)``
against ``x^(j,1)``. The width ``sigma`` of ``pk`` itself is ``1 + B sqrt(eta)``
times larger, so the dimension is larger than an attack on ``pk`` alone would
need.

The reduction is the ABDP15 full version (ePrint 2015/017, Oct. 2015),
Thm. 4.4 with Thms. 6.6 and 6.7 (Construction 6.5, Sec. 6.2), written out in
F. Bourse's PhD thesis (ENS 2017), Thm. 4.3.5 with Thms. 8.2.2, 8.2.3,
8.3.2. Its width is the condition derived in the proof of Thm. 6.6
(= Bourse, Thm. 8.2.2); the parameter list of the full version's Sec. 6.2
states the stronger, sufficient ``(l B + 1) sigma'``.

Hypotheses of the papers that the library meets differently:

* The literal ``p > l B^2`` of Thm. 4.4 / Sec. 6.2 is not needed for this
  Regev instance. The change of basis acts on secret keys mod the LWE prime
  ``q``, where ``||x1 - x0||^2 != 0`` with a wide margin. Decryption decodes
  the integer inner product (``p > 2 eta X Y + 1``), so admissible keys are
  orthogonal to ``x1 - x0`` over ``Z``. The last hybrid only needs ``a + b``
  uniform mod ``p``.
* The ``4 X`` per MIFE slot is the library's ACFGU18 composition; neither
  text covers MIFE.
* The thesis's LWE hardness bound (an absolute width above ``2 sqrt(n)``),
  which the full version does not state, is met under ``standard``:
  ``sigma' >= 2^16``, the smallest width of the table that ``standard``
  enforces.

The public keys' noise ``E`` -- the only Gaussian of these schemes (encryption
draws a binary ``r``) -- enters the proof as LWE with noise ``D_{Z,sigma}``.
Under ``standard`` it is sampled exactly (``gaussian_sampler = "facct"``; an
explicit ``"rounded"`` is refused, a ``gaussian_sampler()`` block ignored):
the double-precision rounded sampler is far from ``D_{Z,sigma}`` over the
``m eta`` samples of a key (crypto review, October 2026). Under ``toy`` it
follows the config's ``gaussian_sampler``, else the library-wide choice. The
public parameters record the sampler used.
"""

from __future__ import annotations

import json
import secrets

import gmpy2 as gp

from pyfe4ai.core.base import ParameterCacheMixin
from pyfe4ai.core.formats.lwe import ABDP15Record
from pyfe4ai.core.formats.lwe import LWEEncryptionKey
from pyfe4ai.core.versions import record_matches
from pyfe4ai.core.versions import SIFE_LWE_RECORD
from pyfe4ai.core.profiles import STANDARD
from pyfe4ai.primitives.lattice.lwe_matrix import LimbMatrix
from pyfe4ai.primitives.lattice.lwe_matrix import SeededMatrix
from pyfe4ai.primitives.lattice.lwe_matrix import check_matrix_mode
from pyfe4ai.primitives.lattice.lwe_utils import abdp15_parameters
from pyfe4ai.primitives.lattice.lwe_utils import abdp15_reduction_sigma
from pyfe4ai.primitives.lattice.lwe_utils import check_wide_lwe
from pyfe4ai.primitives.lattice.matrix_utils import digits_to_matrix
from pyfe4ai.primitives.sampling import PROOF_SAMPLER
from pyfe4ai.primitives.sampling import discrete_gaussian_matrix
from pyfe4ai.primitives.sampling import get_gaussian_sampler
from pyfe4ai.primitives.sampling import rand_bit_matrix
from pyfe4ai.primitives.sampling import scheme_sampler

DEFAULT_MATRIX_MODE = "memory"
#: Plaintexts encrypted per pass over ``A`` by the ndarray helpers.
ENCRYPT_BATCH = 64


class ABDP15KeyGeneratorMixin(ParameterCacheMixin):
    """Parameter cache, seeded ``A`` and public keys of an ABDP15 LWE key generator.

    The host class sets ``eta``, ``bound_x`` and ``bound_y`` and calls
    ``_init_abdp15``.
    """

    _construction = SIFE_LWE_RECORD
    #: ``B / X``, where ``B`` bounds ``|x1_i - x0_i|`` for the challenge
    #: vectors ``x0, x1`` of the IPFE security game (signed ``|x_i| <= X``).
    _challenge_spread = 2

    def _init_abdp15(self, config: dict) -> None:
        self.lwe_n_config = config.get("lwe_n", 32)
        self.matrix_mode = check_matrix_mode(config.get("matrix_mode", DEFAULT_MATRIX_MODE))
        # the noise E of pk: exact under standard, else the config's or the library-wide sampler
        standard = self.security_profile == STANDARD
        self.gaussian_sampler = scheme_sampler(
            config, default=PROOF_SAMPLER if standard else get_gaussian_sampler(), profile=self.security_profile
        )
        self._load_parameters()
        if self.security_profile == STANDARD:
            check_wide_lwe(self.lwe_n, self.q, self._reduction_sigma(self.sigma_q), what="sigma'")

    def _reduction_sigma(self, sigma_q) -> float:
        """``sigma'``: the error width of the LWE problem the security proof
        reduces to, at which the 128-bit table is read."""
        return abdp15_reduction_sigma(sigma_q, self._challenge_spread * self.bound_x, self.eta)

    def _table_sigma(self, params: dict) -> float:
        """:meth:`_reduction_sigma` of derived *params*, for ``lwe_n = "auto"``."""
        return self._reduction_sigma(params["sigma_q"])

    def _record_fields(self) -> dict:
        """Configuration a cached parameter file must match."""
        return {
            "sec_param": self.sec_param,
            "eta": self.eta,
            "lwe_n_config": self.lwe_n_config,
            "bound_x": gp.digits(self.bound_x),
            "bound_y": gp.digits(self.bound_y),
        }

    def _param_verification(self, param: dict) -> bool:
        """The cached record must match the config and the parameters derived
        from it (the derivation is deterministic); only the seed of ``A`` is random."""
        if not record_matches(param, {"construction": self._construction, **self._record_fields()}):
            return False
        seed = ABDP15Record(param).get("A_seed")
        if not isinstance(seed, str) or len(bytes.fromhex(seed)) != 32:
            return False
        return record_matches(param, self._serialised(self._derive()))

    @staticmethod
    def _serialised(derived: dict) -> dict:
        record = {
            key: value if isinstance(value, float) else gp.digits(gp.mpz(value))
            for key, value in derived.items()
        }
        record["lwe_n"], record["m"] = int(record["lwe_n"]), int(record["m"])
        return record

    def _derive(self) -> dict:
        """Fresh parameters: ``lwe_n``, ``p``, ``q``, ``m``, ``sigma_q``, ``l_sigma``."""
        return abdp15_parameters(
            self.bound_x, self.bound_y, self.eta, self.lwe_n_config, lwe_sigma=self._table_sigma
        )

    def _apply_parameters(self, param: dict) -> None:
        record = ABDP15Record.from_dict(param)
        self.lwe_n = int(record.lwe_n)
        self.p = gp.mpz(record.p)
        self.q = gp.mpz(record.q)
        self.m = int(record.m)
        self.sigma_q = float(record.sigma_q)
        self.l_sigma = gp.mpz(record.l_sigma)
        self.a_seed = record.A_seed
        self.A = SeededMatrix.shared(self.a_seed, self.m, self.lwe_n, self.q, self.matrix_mode)

    def _generate_and_save(self, param_file: str) -> None:
        record = ABDP15Record.new(construction=self._construction, **self._record_fields())
        record.update(**self._serialised(self._derive()))
        record.A_seed = secrets.token_hex(32)
        self._apply_parameters(record.to_dict())
        with open(param_file, "w", encoding="utf-8") as f:
            json.dump(record.to_dict(), f)

    def _public_keys(self, secret_keys: list) -> list:
        """``[A S_i + E_i mod q]_i`` for secret matrices ``S_i`` (``lwe_n x eta`` each),
        computed in one pass over ``A``."""
        q = self.q
        joint = [sum((list(s[row]) for s in secret_keys), []) for row in range(self.lwe_n)]
        product = self.A.times(joint)
        out = []
        for i in range(len(secret_keys)):
            noise = discrete_gaussian_matrix(self.m, self.eta, self.sigma_q, sampler=self.gaussian_sampler)
            cols = slice(i * self.eta, (i + 1) * self.eta)
            out.append([
                [gp.mpz((a + e) % q) for a, e in zip(row[cols], noise_row)]
                for row, noise_row in zip(product, noise)
            ])
        return out

    def _matrix_parameters(self) -> dict:
        """Public parameters describing ``A`` and the public keys: ``A``'s seed,
        ``m``, ``lwe_n``, the sampler of the noise ``E`` and the profile."""
        return {"A_seed": self.a_seed, "m": self.m, "lwe_n": self.lwe_n,
                "gaussian_sampler": self.gaussian_sampler, "security_profile": self.security_profile}


class ABDP15CryptoMixin:
    """Lazily expanded ``A``, limb form of ``pk`` and batched masks ``(A^T r, pk^T r)``."""

    def _init_abdp15(self, config: dict) -> None:
        self.matrix_mode = check_matrix_mode(config.get("matrix_mode", DEFAULT_MATRIX_MODE))
        self._a_matrix = None
        self._pk_matrix = None

    @property
    def A(self) -> SeededMatrix:
        """The public matrix, shared by all crypto objects of this process."""
        if self._a_matrix is None:
            pp = self.pp
            self._a_matrix = SeededMatrix.shared(
                pp["A_seed"], pp["m"], pp["lwe_n"], gp.mpz(pp["q"]), self.matrix_mode
            )
        return self._a_matrix

    def _masks(self, count: int) -> tuple:
        """``count`` fresh pairs ``(A^T r, pk^T r)`` (Python ints mod ``q``)."""
        if self._pk_matrix is None:
            self._pk_matrix = LimbMatrix(digits_to_matrix(LWEEncryptionKey(self.sk).pk), gp.mpz(self.pp["q"]))
        r = rand_bit_matrix(count, self.pp["m"])
        return self.A.binary_combinations(r), self._pk_matrix.binary_combinations(r)
