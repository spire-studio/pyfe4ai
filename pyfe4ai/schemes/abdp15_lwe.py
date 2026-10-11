"""Parts shared by the ABDP15 LWE IPFEs: ``sife/lwe``, ``mife/lwe``, ``mife/lwe_threshold``.

The public matrix ``A in Z_q^{m x lwe_n}`` is published as a 32-byte seed and
expanded by :class:`pyfe4ai.utils.lwe_matrix.SeededMatrix` (FrodoKEM-style;
the expansion is modelled as a random oracle). Encryption draws a binary
``r`` and needs ``A^T r`` and ``pk^T r``; both are computed exactly with
numpy, for a whole batch of plaintexts at once.

``matrix_mode`` (key generator and crypto configs; default ``"memory"``)
chooses between keeping the expanded ``A`` in memory (fast, ``8 L`` bytes per
entry with ``L = ceil(bits(q) / 31)``) and expanding it again for every batch
(``"stream"``: little memory, slow). The public keys ``pk = A S + E``
(``m x eta`` per slot) are always held in memory.

Under the ``standard`` security profile ``lwe_n = "auto"`` selects the
smallest dimension whose ``(q, sigma)`` meets
:data:`pyfe4ai.utils.lwe_utils.WIDE_LWE_128_LOG_RATIO`, a lattice-estimator
table for the LWE problem of ``pk``; an explicit ``lwe_n`` is checked against
it.
"""

from __future__ import annotations

import json
import secrets

import gmpy2 as gp

from pyfe4ai.schemes.ipfe import ParameterCacheMixin
from pyfe4ai.schemes.profiles import STANDARD
from pyfe4ai.utils.lwe_matrix import LimbMatrix
from pyfe4ai.utils.lwe_matrix import SeededMatrix
from pyfe4ai.utils.lwe_matrix import check_matrix_mode
from pyfe4ai.utils.lwe_utils import abdp15_parameters
from pyfe4ai.utils.lwe_utils import check_wide_lwe
from pyfe4ai.utils.matrix_utils import digits_to_matrix
from pyfe4ai.utils.sampling_utils import discrete_gaussian_matrix
from pyfe4ai.utils.sampling_utils import rand_bit_matrix

DEFAULT_MATRIX_MODE = "memory"
#: Plaintexts encrypted per pass over ``A`` by the ndarray helpers.
ENCRYPT_BATCH = 64
CONSTRUCTION = "abdp15-lwe-seeded-a/v1"


class ABDP15KeyGeneratorMixin(ParameterCacheMixin):
    """Parameter cache, seeded ``A`` and public keys of an ABDP15 LWE key generator.

    The host class sets ``eta``, ``bound_x`` and ``bound_y`` and calls
    :meth:`_init_abdp15`; ``_extra_dimension`` (``None``, or the number of
    clients of a MIFE) is passed to the parameter recipe.
    """

    _construction = CONSTRUCTION
    _extra_dimension = None

    def _init_abdp15(self, config: dict) -> None:
        self.lwe_n_config = config.get("lwe_n", 32)
        self.matrix_mode = check_matrix_mode(config.get("matrix_mode", DEFAULT_MATRIX_MODE))
        self._load_parameters()
        if self.security_profile == STANDARD:
            check_wide_lwe(self.lwe_n, self.q, self.sigma_q)

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
        return param.get("construction") == self._construction and all(
            param.get(key) == value for key, value in self._record_fields().items()
        )

    def _derive(self) -> dict:
        """Fresh parameters: ``lwe_n``, ``p``, ``q``, ``m``, ``sigma_q``, ``l_sigma``."""
        return abdp15_parameters(
            self.bound_x, self.bound_y, self.eta, self.lwe_n_config,
            extra_dimension=self._extra_dimension,
        )

    def _apply_parameters(self, param: dict) -> None:
        self.lwe_n = int(param["lwe_n"])
        self.p = gp.mpz(param["p"])
        self.q = gp.mpz(param["q"])
        self.m = int(param["m"])
        self.sigma_q = float(param["sigma_q"])
        self.l_sigma = gp.mpz(param["l_sigma"])
        self.a_seed = param["A_seed"]
        self.A = SeededMatrix.shared(self.a_seed, self.m, self.lwe_n, self.q, self.matrix_mode)

    def _generate_and_save(self, param_file: str) -> None:
        record = {"construction": self._construction, **self._record_fields()}
        for key, value in self._derive().items():
            record[key] = value if isinstance(value, float) else gp.digits(gp.mpz(value))
        record["lwe_n"], record["m"] = int(record["lwe_n"]), int(record["m"])
        record["A_seed"] = secrets.token_hex(32)
        self._apply_parameters(record)
        with open(param_file, "w", encoding="utf-8") as f:
            json.dump(record, f)

    def _public_keys(self, secret_keys: list) -> list:
        """``[A S_i + E_i mod q]_i`` for secret matrices ``S_i`` (``lwe_n x eta`` each),
        computed in one pass over ``A``."""
        q = self.q
        joint = [sum((list(s[row]) for s in secret_keys), []) for row in range(self.lwe_n)]
        product = self.A.times(joint)
        out = []
        for i in range(len(secret_keys)):
            noise = discrete_gaussian_matrix(self.m, self.eta, self.sigma_q)
            cols = slice(i * self.eta, (i + 1) * self.eta)
            out.append([
                [gp.mpz((a + e) % q) for a, e in zip(row[cols], noise_row)]
                for row, noise_row in zip(product, noise)
            ])
        return out

    def _matrix_parameters(self) -> dict:
        """Public parameters describing ``A``."""
        return {"A_seed": self.a_seed, "m": self.m, "lwe_n": self.lwe_n}


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
            self._pk_matrix = LimbMatrix(digits_to_matrix(self.sk["pk"]), gp.mpz(self.pp["q"]))
        r = rand_bit_matrix(count, self.pp["m"])
        return self.A.binary_combinations(r), self._pk_matrix.binary_combinations(r)
