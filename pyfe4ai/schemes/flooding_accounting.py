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

``"kl"`` (opt-in; needs ``max_partial_decryptions`` and ``t = s``)
    A bound for the adversary's whole view of a run in which every server
    produces at most ``Q = max_partial_decryptions`` partial decryptions:
    ``N = s Q k`` flooded coefficients (``k`` per partial decryption). With
    ``lambda = stat_param`` the Hellinger distance between the real and the
    simulated view stays below ``2^(-lambda/2)``, split as

    * ``KL(D + c || D) = c^2 / (2 sigma^2)`` per coefficient, adding up over
      the adaptive run (chain rule): ``sigma_f = ceil(sqrt(B^2 N 2^lambda))``
      gives ``KL <= 2^-(lambda+1)``, ``HD <= sqrt(KL / 2) <= 2^(-lambda/2) / 2``;
    * the tail cut, on both sides: ``HD^2 <= N Pr[|D| > T sigma]``, at most
      ``2^-(lambda+6)`` with the cut ``T`` of :func:`kl_tail_cut` (16 for
      ``N = 2^53``, ``lambda = 128``);
    * the sampler, on both sides: ``HD^2 <= N delta`` with the rounded
      sampler's deviation ``delta <= 2^4.3 sigma 2^-P`` per sample (an
      estimate of its Box-Muller grid, not a proof) at
      ``P = bits(sigma) + lambda + log2 N + 11`` bits
      (:func:`kl_sampler_precision`), or FACCT's ``2^-185``; at most
      ``2^-(lambda+6)``.

    By the triangle inequality ``HD <= (1/2 + 2/8 + 2/8) 2^(-lambda/2)``. In
    the bit-security framework of Micciancio and Walter (EUROCRYPT 2018)
    that keeps ``lambda - 6.85`` bits for decision games (Hellinger
    replacement, Yasunaga, ITC 2021, Thm. 5), whatever the adversary's
    running time; read conventionally it is a total statistical distance of
    about ``2^-(lambda+2)/2`` (``2^-65`` at ``lambda = 128``). A different
    and weaker guarantee than ``"per-sample"``, valid only for the first
    ``Q`` partial decryptions of every server.

    The servers count against ``Q`` through a :class:`PartialDecryptionBudget`
    that the deployment passes to each server's crypto object
    (``partial_budget``); without one they refuse. Where the count lives
    (one object, shared by every object, process and restart of the server,
    persisted) is the deployment's decision: the library only enforces it
    per budget object.

Both accountings assume that an honest partial differs from its simulation
by exactly the decryption noise. With Shamir shares and Lagrange
coefficients mod ``q`` that holds only for ``t = s`` (every enrolled set is
then all servers, and the multipliers are 0 or 1); for ``t < s`` some
enrolled sets make the shift a rational multiple ``(r / d) nu``, and ``d``
times a partial reveals ``nu mod d`` whatever the width. ``"kl"`` therefore
refuses ``t < s``. (The default accounting does not check this.)

TODO (merge with ``fix/threshold-lagrange``): that branch adds replicated
sharing (``sharing="replicated"``, its default for ``t < s``), whose
multipliers are 0 or 1; after the merge allow ``"kl"`` for ``t < s`` with
``sharing="replicated"`` only, and keep refusing it with Shamir shares.
"""

from __future__ import annotations

import math
import threading

import gmpy2 as gp

from pyfe4ai.utils.exceptions import FESchemeError, FEValidationError
from pyfe4ai.utils.lwe_utils import GAUSSIAN_TAIL_CUT
from pyfe4ai.utils.sampling_utils import FACCT_PRECISION
from pyfe4ai.utils.sampling_utils import discrete_gaussian_bigint
from pyfe4ai.utils.sampling_utils import get_gaussian_sampler

PER_SAMPLE = "per-sample"
KL = "kl"
FLOODING_ACCOUNTINGS = (PER_SAMPLE, KL)

# each of the tail cut and the sampler may cost HD^2 <= 2^-(lambda + _SIDE_MARGIN)
# per side of the hybrid, i.e. HD <= 2^(-lambda/2) / 8
_SIDE_MARGIN = 6
# the rounded sampler's deviation is about 2^4.3 sigma 2^-P per sample; with
# P = bits(sigma) + lambda + ceil(log2 N) + 11 the sum over N samples is at
# most 2^-(lambda + 6.7)
_SAMPLER_MARGIN = 11
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


def check_kl_threshold(t: int, s: int) -> None:
    """Refuse ``"kl"`` for ``t < s`` (see the module docstring).

    Raises:
        FEValidationError: Unless ``t == s``.
    """
    if t != s:
        raise FEValidationError(
            "flooding_accounting='kl' needs t = s (got t = {}, s = {}): for t < s the Lagrange "
            "coefficients of some enrolled sets make an honest partial decryption reveal the "
            "decryption noise modulo a small integer, whatever the flooding width".format(t, s)
        )


def kl_flooded_samples(servers: int, max_partial_decryptions: int, coefficients: int) -> int:
    """``N``: flooded coefficients of a run, ``servers * max_partial_decryptions * coefficients``."""
    return servers * max_partial_decryptions * coefficients


def kl_flooding_width(noise_bound: int, samples: int, stat_param: int) -> int:
    """Smallest integer ``sigma`` with ``samples * noise_bound^2 / (2 sigma^2) <= 2^-(stat_param + 1)``."""
    if min(noise_bound, samples, stat_param) < 1:
        raise FEValidationError("noise_bound, samples and stat_param must be positive")
    target = (noise_bound * noise_bound * samples) << stat_param
    root = math.isqrt(target)
    return root if root * root == target else root + 1


def kl_divergence_log2(noise_bound: int, samples: int, sigma: int) -> float:
    """``log2`` of the KL bound ``samples * noise_bound^2 / (2 sigma^2)``."""
    return 2 * math.log2(noise_bound) + math.log2(samples) - 1 - 2 * math.log2(sigma)


def gaussian_tail_log2(cut: float) -> float:
    """An upper bound on ``log2 Pr[|Z| > cut]`` for a standard normal ``Z``
    (Mills' ratio: ``sqrt(2 / pi) exp(-cut^2 / 2) / cut``), ``cut >= 1``."""
    return 0.5 * math.log2(2 / math.pi) - cut * cut / (2 * math.log(2)) - math.log2(cut)


def kl_tail_cut(samples: int, stat_param: int, sigma: int | None = None) -> int:
    """Smallest integer cut ``T >= 12`` with ``N Pr[|D| > T sigma] <= 2^-(stat_param + 6)``.

    The bound is evaluated at ``T - 1 / sigma`` (rounding) and doubled (the
    discrete Gaussian's tail; the rounded one needs no factor).
    """
    shift = 1.0 / float(sigma) if sigma else 0.0
    cut = GAUSSIAN_TAIL_CUT
    while math.log2(samples) + 1 + gaussian_tail_log2(cut - shift) > -(stat_param + _SIDE_MARGIN):
        cut += 1
    return cut


def kl_sampler_precision(sigma: int, samples: int, stat_param: int) -> int:
    """Bits of the rounded Gaussian sampler for ``"kl"``: ``bits(sigma) +
    stat_param + ceil(log2 samples) + 11``, and at least the sampler's
    default ``bits(sigma) + 128``."""
    extra = max(128, stat_param + (samples - 1).bit_length() + _SAMPLER_MARGIN)
    # bits(sigma) as discrete_gaussian_bigint counts them
    return int(gp.ceil(gp.mpfr(sigma))).bit_length() + extra


def facct_max_log_samples(stat_param: int) -> int:
    """Largest ``log2 N`` for which FACCT's deviation, summed over ``N``
    samples, stays below ``2^-(stat_param + 6)``."""
    return _FACCT_LOG_DEVIATION - stat_param - _SIDE_MARGIN


class KLFlooding:
    """Flooding parameters of ``"kl"`` public parameters, as the servers use them.

    Args:
        pp: Public parameters with ``flooding_accounting = "kl"``.
        coefficients: Flooded coefficients per partial decryption.

    Raises:
        FESchemeError: If *pp* has ``t != s`` or a tail cut other than
            :func:`kl_tail_cut` derives.
    """

    def __init__(self, pp: dict, coefficients: int) -> None:
        if int(pp["t"]) != int(pp["s"]):
            raise FESchemeError("flooding_accounting='kl' public parameters with t != s")
        self.max_partial_decryptions = int(pp["max_partial_decryptions"])
        self.stat_param = int(pp["stat_param"])
        self.sigma = int(pp["sigma_f"])
        self.samples = kl_flooded_samples(int(pp["s"]), self.max_partial_decryptions, coefficients)
        self.tail_cut = int(pp["flood_tail_cut"])
        if self.tail_cut != kl_tail_cut(self.samples, self.stat_param, self.sigma):
            raise FESchemeError("flood_tail_cut does not match max_partial_decryptions and stat_param")
        self.precision = kl_sampler_precision(self.sigma, self.samples, self.stat_param)

    def check_sampler(self) -> None:
        """Raise unless the Gaussian sampler in effect covers ``N`` samples.

        Raises:
            FEValidationError: If FACCT is selected and ``log2 N`` exceeds
                :func:`facct_max_log_samples`.
        """
        limit = facct_max_log_samples(self.stat_param)
        if get_gaussian_sampler() == "facct" and (self.samples - 1).bit_length() > limit:
            raise FEValidationError(
                "the FACCT sampler's precision covers at most 2^{} flooded samples at "
                "stat_param = {}; use the rounded sampler".format(limit, self.stat_param)
            )

    def sample(self, count: int, rng) -> list:
        """*count* flooding values: ``D_{Z, sigma_f}`` cut at ``tail_cut sigma_f``, in order."""
        self.check_sampler()
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


def _count(value, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise FEValidationError("{} must be an int, got {!r}".format(name, value))
    return value


class PartialDecryptionBudget:
    """How many partial decryptions a server may still produce under ``"kl"``.

    The deployment creates one per server and passes it to every crypto
    object that answers for that server (``partial_budget`` in the crypto
    config); its ``limit`` may not exceed the public parameters'
    ``max_partial_decryptions``. The count only grows, and :meth:`consume`
    checks and increments it as one step under a lock, so the threads of a
    process can share a budget. Every partial decryption a server produces
    counts, repeated ones too; refused requests do not. The guarantee holds
    only if a server's count covers all its partial decryptions under the
    setup: a server that runs as several processes or restarts must share or
    persist it, for instance by subclassing with an atomic, durable
    :meth:`consume` and starting from the stored ``used``.

    Args:
        limit: At most ``max_partial_decryptions`` of the public parameters.
        used: Partial decryptions already produced (``0 <= used <= limit``).

    Raises:
        FEValidationError: For non-integer or out-of-range values.
    """

    def __init__(self, limit: int, used: int = 0) -> None:
        self._limit = _count(limit, "limit")
        self._used = _count(used, "used")
        if self._limit < 1 or not 0 <= self._used <= self._limit:
            raise FEValidationError(
                "a partial decryption budget needs limit >= 1 and 0 <= used <= limit, "
                "got limit = {}, used = {}".format(limit, used)
            )
        # re-entrant: consume() holds it while it calls check()
        self._lock = threading.RLock()

    @property
    def limit(self) -> int:
        return self._limit

    @property
    def used(self) -> int:
        return self._used

    @property
    def remaining(self) -> int:
        with self._lock:
            return self._limit - self._used

    def check(self, count: int = 1) -> None:
        """Raise unless *count* more partial decryptions fit (consumes nothing).

        Raises:
            FESchemeError: If they would exceed the limit.
        """
        if _count(count, "count") < 0:
            raise FEValidationError("count must be non-negative")
        with self._lock:
            if self._used + count > self._limit:
                raise FESchemeError(
                    "partial decryption limit reached: {} of max_partial_decryptions = {} used, "
                    "{} requested; the KL flooding guarantee does not cover more".format(
                        self._used, self._limit, count
                    )
                )

    def consume(self, count: int = 1) -> None:
        """Account for *count* partial decryptions about to be produced.

        Raises:
            FESchemeError: If they would exceed the limit (nothing is consumed then).
        """
        with self._lock:  # check and increment as one step
            self.check(count)
            self._used += count


def server_budget(pp: dict, budget, flood: KLFlooding):
    """The budget a ``"kl"`` server counts against: *budget*, checked against *pp*.

    Raises:
        FESchemeError: If there is none (the deployment must decide where a
            server's count lives).
        FEValidationError: If it is not a :class:`PartialDecryptionBudget` or
            allows more than ``max_partial_decryptions``.
    """
    if budget is None:
        raise FESchemeError(
            "flooding_accounting='kl': a server needs a partial_budget (a PartialDecryptionBudget "
            "shared by everything that answers for it) in its crypto config"
        )
    if not isinstance(budget, PartialDecryptionBudget):
        raise FEValidationError("partial_budget must be a PartialDecryptionBudget")
    if budget.limit > flood.max_partial_decryptions:
        raise FEValidationError(
            "partial_budget allows {} partial decryptions, more than max_partial_decryptions = {}".format(
                budget.limit, flood.max_partial_decryptions
            )
        )
    return budget
