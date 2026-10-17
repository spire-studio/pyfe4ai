"""Accounting of the statistical flooding of the threshold LWE layers.

``mife/lwe_threshold`` and ``mcfe/ring_lwe_threshold`` flood every partial
decryption with ``e_k <- D_{Z, sigma_f}``, so that it can be simulated from
the result: real and simulated partials differ by a shift ``c`` of the
noise, the honest part of the decryption noise (``|c| <= B``). The
``flooding_accounting`` entry of their key generator config chooses how
``sigma_f`` is sized against that shift:

``"per-sample"`` (default)
    ``sigma_f = 2^stat_param B``: statistical distance below
    ``2^-stat_param`` per flooded coefficient, cut at
    :data:`~pyfe4ai.utils.lwe_utils.GAUSSIAN_TAIL_CUT` ``sigma_f``, no limit
    on the number of partial decryptions. This is the guarantee the
    ``standard`` profile documents.

``"kl"`` (opt-in; needs ``max_partial_decryptions``)
    KL divergence below ``2^-stat_param`` for the adversary's whole view of
    a run in which every server produces at most ``max_partial_decryptions``
    partial decryptions. With ``N = s * max_partial_decryptions * k`` flooded
    coefficients (``k`` per partial decryption) and
    ``KL(D + c || D) = c^2 / (2 sigma^2)`` per coefficient, adding up over
    the adaptive run (chain rule),
    ``sigma_f = ceil(sqrt(B^2 N 2^(stat_param - 1)))``. In the bit-security
    framework of Micciancio and Walter (EUROCRYPT 2018) this keeps about
    ``stat_param - 7`` bits of statistical security for decision games
    (Hellinger replacement, Yasunaga, ITC 2021, Thm. 5, with
    ``HD^2 <= KL / 2``), which is about ``2^-(stat_param + 1) / 2`` of total
    statistical distance read conventionally: a different and weaker
    guarantee than ``"per-sample"``, and valid only up to the limit, which
    the servers enforce (:class:`PartialDecryptionBudget`). Because KL is
    infinite under a hard cut, the cut moves to :data:`KL_TAIL_CUT` sigma
    (``N Pr[|D| > 20 sigma] < N 2^-293``), and the rounded sampler runs at
    :func:`kl_sampler_precision` bits so that its own deviation, summed
    over ``N`` samples, stays below ``2^-(stat_param + 3)``.

Both accountings assume that an honest partial differs from its simulation
by exactly the decryption noise. That holds when the ``t - 1`` corrupted
servers are in the enrolled set and it has ``t`` members (always when
``t = s``); for other enrolled sets the Lagrange factors make the shift a
rational multiple of the noise, which no flooding width hides.
"""

from __future__ import annotations

import math

import gmpy2 as gp

from pyfe4ai.utils.exceptions import FESchemeError, FEValidationError
from pyfe4ai.utils.lwe_utils import GAUSSIAN_TAIL_CUT
from pyfe4ai.utils.sampling_utils import FACCT_PRECISION
from pyfe4ai.utils.sampling_utils import discrete_gaussian_bigint
from pyfe4ai.utils.sampling_utils import get_gaussian_sampler

PER_SAMPLE = "per-sample"
KL = "kl"
FLOODING_ACCOUNTINGS = (PER_SAMPLE, KL)

#: Tail cut of the flooding noise under ``"kl"``, in units of ``sigma_f``.
KL_TAIL_CUT = 20

# extra bits of the rounded sampler: its deviation is about 2^4.3 sigma 2^-P
# per sample, so P = bits(sigma) + stat_param + ceil(log2 N) + 8 keeps the
# sum over N samples below 2^-(stat_param + 3)
_SAMPLER_MARGIN = 8
# FACCT samples are within about 2^-(FACCT_PRECISION - 7) of D_{Z,sigma}
_FACCT_LOG_DEVIATION = FACCT_PRECISION - 7


def flooding_config(config: dict) -> tuple[str, int | None]:
    """``(flooding_accounting, max_partial_decryptions)`` of a key generator config.

    Raises:
        FEValidationError: For an unknown accounting, ``"kl"`` without a
            positive integer ``max_partial_decryptions``, or a limit given
            with the per-sample accounting (it would not change any
            parameter there).
    """
    accounting = config.get("flooding_accounting", PER_SAMPLE)
    if accounting not in FLOODING_ACCOUNTINGS:
        raise FEValidationError(
            "unknown flooding_accounting {!r}; use one of {}".format(accounting, FLOODING_ACCOUNTINGS)
        )
    limit = config.get("max_partial_decryptions")
    if accounting == PER_SAMPLE:
        if limit is not None:
            raise FEValidationError(
                "max_partial_decryptions applies to flooding_accounting='kl' only"
            )
        return accounting, None
    if isinstance(limit, bool) or not isinstance(limit, int) or limit < 1:
        raise FEValidationError(
            "flooding_accounting='kl' needs max_partial_decryptions, the number of partial "
            "decryptions each server may produce (a positive int), got {!r}".format(limit)
        )
    return accounting, limit


def kl_flooded_samples(servers: int, max_partial_decryptions: int, coefficients: int) -> int:
    """``N``: flooded coefficients of a run, ``servers * max_partial_decryptions * coefficients``."""
    return servers * max_partial_decryptions * coefficients


def kl_flooding_width(noise_bound: int, samples: int, stat_param: int) -> int:
    """Smallest integer ``sigma`` with ``samples * noise_bound^2 / (2 sigma^2) <= 2^-stat_param``."""
    if min(noise_bound, samples, stat_param) < 1:
        raise FEValidationError("noise_bound, samples and stat_param must be positive")
    target = (noise_bound * noise_bound * samples) << (stat_param - 1)
    root = math.isqrt(target)
    return root if root * root == target else root + 1


def kl_divergence_log2(noise_bound: int, samples: int, sigma: int) -> float:
    """``log2`` of the KL bound ``samples * noise_bound^2 / (2 sigma^2)``."""
    return 2 * math.log2(noise_bound) + math.log2(samples) - 1 - 2 * math.log2(sigma)


def kl_sampler_precision(sigma: int, samples: int, stat_param: int) -> int:
    """Bits of the rounded Gaussian sampler for ``"kl"``: ``bits(sigma) +
    stat_param + ceil(log2 samples) + 8``, and at least the sampler's
    default ``bits(sigma) + 128``."""
    extra = max(128, stat_param + (samples - 1).bit_length() + _SAMPLER_MARGIN)
    # bits(sigma) as discrete_gaussian_bigint counts them
    return int(gp.ceil(gp.mpfr(sigma))).bit_length() + extra


class KLFlooding:
    """Flooding parameters of ``"kl"`` public parameters, as the servers use them.

    Args:
        pp: Public parameters with ``flooding_accounting = "kl"``.
        coefficients: Flooded coefficients per partial decryption.
    """

    def __init__(self, pp: dict, coefficients: int) -> None:
        self.max_partial_decryptions = int(pp["max_partial_decryptions"])
        self.stat_param = int(pp["stat_param"])
        self.tail_cut = int(pp["flood_tail_cut"])
        self.sigma = int(pp["sigma_f"])
        self.samples = kl_flooded_samples(int(pp["s"]), self.max_partial_decryptions, coefficients)
        self.precision = kl_sampler_precision(self.sigma, self.samples, self.stat_param)

    def sample(self, count: int, rng) -> list:
        """*count* flooding values: ``D_{Z, sigma_f}`` cut at ``tail_cut sigma_f``, in order.

        Raises:
            FEValidationError: If the FACCT sampler is in effect and the run is
                too long for its precision (``N 2^-185 > 2^-(stat_param + 2)``).
        """
        if get_gaussian_sampler() == "facct" and (
            (self.samples - 1).bit_length() > _FACCT_LOG_DEVIATION - self.stat_param - 2
        ):
            raise FEValidationError(
                "the FACCT sampler's precision covers at most 2^{} flooded samples at "
                "stat_param = {}; use the rounded sampler".format(
                    _FACCT_LOG_DEVIATION - self.stat_param - 2, self.stat_param
                )
            )
        return flooding_samples(self.sigma, count, rng, self.tail_cut, self.precision)


def kl_flooding(pp: dict, coefficients: int) -> KLFlooding | None:
    """:class:`KLFlooding` of *pp*, or ``None`` for the per-sample accounting."""
    if pp.get("flooding_accounting", PER_SAMPLE) == PER_SAMPLE:
        return None
    if pp["flooding_accounting"] != KL:
        raise FESchemeError("unknown flooding_accounting {!r}".format(pp["flooding_accounting"]))
    return KLFlooding(pp, coefficients)


def flooding_samples(sigma: int, count: int, rng, tail_cut: int = GAUSSIAN_TAIL_CUT, precision=None) -> list:
    """*count* samples of ``D_{Z, sigma}`` cut at ``tail_cut sigma``, drawn in order
    (*precision* bits for the rounded sampler, default ``bits(sigma) + 128``)."""
    bound = tail_cut * sigma
    out = [e for e in discrete_gaussian_bigint(sigma, count, rng=rng, precision=precision) if abs(e) <= bound]
    while len(out) < count:  # beyond the tail cut (rare): draw on, one at a time
        e = discrete_gaussian_bigint(sigma, 1, rng=rng, precision=precision)[0]
        if abs(e) <= bound:
            out.append(e)
    return out


class PartialDecryptionBudget:
    """The number of partial decryptions a server may still produce.

    A server object keeps one; servers that run as several objects or
    processes, or restart, must share one (pass it as ``partial_budget`` in
    the crypto config) or persist ``used``: the ``"kl"`` guarantee holds only
    up to ``limit`` partial decryptions per server for the whole setup.
    Every request counts, repeated ones too.

    Args:
        limit: ``max_partial_decryptions`` of the public parameters.
        used: Partial decryptions already produced.
    """

    def __init__(self, limit: int, used: int = 0) -> None:
        self.limit = int(limit)
        self.used = int(used)

    @property
    def remaining(self) -> int:
        return self.limit - self.used

    def consume(self, count: int = 1) -> None:
        """Account for *count* partial decryptions about to be produced.

        Raises:
            FESchemeError: If they would exceed the limit (nothing is consumed then).
        """
        if count < 0:
            raise ValueError("count must be non-negative")
        if self.used + count > self.limit:
            raise FESchemeError(
                "partial decryption limit reached: {} of max_partial_decryptions = {} used, "
                "{} requested; the KL flooding guarantee does not cover more".format(
                    self.used, self.limit, count
                )
            )
        self.used += count
