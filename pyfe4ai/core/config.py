"""
Typed configurations of the key generators, and the rules that combine
their options.

A key generator takes its configuration as a mapping of options. When it is
built, the mapping is converted into the config class of its scheme (a
:class:`dict` subclass, so a config is used as a dict everywhere) and the
security profile is applied (:func:`resolve`): unknown profiles, schemes the
``standard`` profile refuses (:data:`REFUSED`) and values weaker than its
minima (:data:`STANDARD_MINIMA`) are refused there, before ``__init__`` runs.
Each config class declares the options of its scheme (:class:`Option`: what
it holds, its default, the values it accepts), so ``config.ring_n`` reads an
option with its default, and the page *Configuration options* of the
documentation is rendered from these declarations (:func:`options_rst`).

The rules that combine the options of the threshold schemes -- the key
sharing for ``t < s``, the flooding accounting, the key-share layout -- are
here too (:func:`resolve_sharing`, :func:`flooding_config`,
:func:`check_kl_threshold`, :func:`check_kl_flooding`,
:func:`check_kl_sharing`, :func:`threshold_key_shares`,
:func:`check_no_flooding_options`); the schemes call them where their checks
were, so that a config is accepted or refused exactly as before, with the
same exception. Every other refusal stays with the code that sizes the
parameters (bounds against a modulus, ring and LWE dimensions under
``standard``, RNS factors against ``s``); :data:`COMBINATIONS` lists all of
them for the documentation.

Crypto objects (encryptors, servers, decryptors) take their config as given;
their options are documented with their scheme (``CRYPTO``).
"""

from __future__ import annotations

import warnings
from typing import Any, ClassVar

from pyfe4ai.core.profiles import MAX_STAT_PARAM
from pyfe4ai.core.profiles import PROFILES
from pyfe4ai.core.profiles import SCHEME_KIND
from pyfe4ai.core.profiles import STANDARD
from pyfe4ai.core.profiles import STANDARD_DCR_PRIME_BITS
from pyfe4ai.core.profiles import STANDARD_STAT_PARAM
from pyfe4ai.core.profiles import TOY
from pyfe4ai.core.exceptions import FESecurityWarning, FEValidationError

# ---------------------------------------------------------------------------
# option values shared by several schemes

#: Ways the LWE / Ring-LWE threshold layers share a functional key (option
#: ``sharing``): ``"shamir"``, ``"replicated"`` or the opt-in
#: ``"replicated-seeded"`` (see :mod:`pyfe4ai.schemes.threshold.sharing` and
#: :func:`resolve_sharing`).
SHARING_SCHEMES = ("shamir", "replicated", "replicated-seeded")

#: Accountings of the flooding noise (option ``flooding_accounting``; see
#: :mod:`pyfe4ai.schemes.threshold.kl`): per sample (the default) or the
#: opt-in KL bound of a run.
PER_SAMPLE = "per-sample"
KL = "kl"
FLOODING_ACCOUNTINGS = (PER_SAMPLE, KL)

#: Flooding distributions of the partial decryptions of the threshold MCFE
#: over Ring-LWE (option ``flooding``), and its default.
FLOODINGS = ("uniform", "gaussian")
DEFAULT_FLOODING = "uniform"

#: How the threshold MCFE over Ring-LWE issues key shares (option
#: ``key_shares``): ``"split"`` (the default for the Shamir and replicated
#: sharings) -- a weight part per weights and a label part per label -- or
#: ``"full"`` (whole key shares per label, as in versions <= 0.3.0; the only
#: kind of the replicated-seeded sharing).
KEY_SHARES = ("split", "full")

#: ``ct_format`` of the threshold MCFE over Ring-LWE when its config has none.
DEFAULT_CT_FORMAT = "binary"
#: ``ring_modulus`` of the threshold MCFE over Ring-LWE when its config has none.
DEFAULT_RING_MODULUS = "rns"

# ---------------------------------------------------------------------------
# options and config classes

_REQUIRED = object()


class Derived:
    """A default that the key generator derives (from other options, the
    environment or its parameters); *text* says how."""

    def __init__(self, text: str) -> None:
        self.text = text

    def __repr__(self) -> str:
        return self.text


class Option:
    """An option of a key generator's config.

    Args:
        doc: What the option holds.
        default: Its value when the config has none: a value, or a
            :class:`Derived` description.
        allowed: The values accepted: a tuple of choices or a description.
        standard: What the ``standard`` profile changes, beyond its minima
            and presets (:data:`STANDARD_MINIMA`, :data:`STANDARD_PRESETS`).
    """

    def __init__(self, doc: str, *, default: Any = None, allowed: Any = None, standard: str | None = None) -> None:
        self.doc = doc
        self.default = default
        self.allowed = allowed
        self.standard = standard
        self.name = ""

    def __set_name__(self, owner, name: str) -> None:
        self.name = name

    def __call__(self, **changes) -> "Option":
        """This option with another *default*, *allowed*, *standard* or *doc*."""
        out = Option(self.doc, default=self.default, allowed=self.allowed, standard=self.standard)
        for key, value in changes.items():
            if key not in ("doc", "default", "allowed", "standard"):
                raise TypeError("Option has no {!r}".format(key))
            setattr(out, key, value)
        return out

    def __get__(self, config, owner=None):
        if config is None:
            return self
        if isinstance(self.default, Derived):
            return config.get(self.name)
        return config.get(self.name, self.default)

    def __repr__(self) -> str:
        return "Option({!r}, default={!r})".format(self.name, self.default)


class SchemeConfig(dict):
    """The config of a key generator: a :class:`dict` of options whose
    declared options also read as attributes, with their defaults.

    Subclasses (one per scheme, with family bases) declare their options as
    :class:`Option` attributes and name the scheme in ``SCHEME``; ``CRYPTO``
    documents the options of the scheme's crypto objects.
    """

    #: ``(family, variant)`` of the scheme, or ``None`` for a base class.
    SCHEME: ClassVar[tuple[str, str] | None] = None
    #: Declared options, by name (every option of the class and its bases).
    OPTIONS: ClassVar[dict[str, Option]] = {}
    #: Options of the scheme's crypto objects, by name.
    CRYPTO: ClassVar[dict[str, Option]] = {}

    security_profile = Option(
        "``toy`` (small, fast, **not secure**) or ``standard`` (see :mod:`pyfe4ai.core.profiles`)",
        default=TOY, allowed=PROFILES,
    )
    sec_param = Option(
        "the DDH modulus bits, the Ring-LWE kappa, or only recorded (the scheme says which)", default=128,
    )
    stat_param = Option(
        "statistical parameter of the schemes that use one", default=Derived("unused"),
        allowed="an int of at most {}".format(MAX_STAT_PARAM),
    )
    n = Option("number of clients (slots)", default=None)
    s = Option("number of decryption servers (threshold schemes)", default=None)
    lst_nid = Option(
        "client ids", default=Derived("``nid_0`` ... ``nid_{n-1}`` when ``n`` is given"),
        allowed="``n`` distinct ids",
    )
    cache_dir = Option(
        "parameter cache directory", default=Derived("``PYFE4AI_CACHE_DIR`` or the per-user cache directory"),
        allowed="a non-empty path",
    )

    def __init_subclass__(cls, **kwargs) -> None:
        super().__init_subclass__(**kwargs)
        options: dict[str, Option] = {}
        for klass in reversed(cls.__mro__):
            for name, value in vars(klass).items():
                if isinstance(value, Option):
                    options[name] = value
        cls.OPTIONS = options
        if cls.SCHEME is not None and "SCHEME" in vars(cls):
            if cls.SCHEME in _CONFIGS:
                raise TypeError("two config classes for {}".format(cls.SCHEME))
            _CONFIGS[cls.SCHEME] = cls

    def __repr__(self) -> str:
        return "{}({})".format(type(self).__name__, dict.__repr__(self))


SchemeConfig.OPTIONS = {name: value for name, value in vars(SchemeConfig).items() if isinstance(value, Option)}

#: Config class of every registered scheme, by ``(family, variant)``.
_CONFIGS: dict[tuple[str, str], type[SchemeConfig]] = {}


def config_class(scheme: tuple | None) -> type[SchemeConfig]:
    """The config class of *scheme* (``(family, variant)``); :class:`SchemeConfig`
    for a key generator that is not a registered scheme."""
    return _CONFIGS.get(scheme, SchemeConfig)


def config_classes() -> dict[tuple[str, str], type[SchemeConfig]]:
    """The config class of every registered scheme, by ``(family, variant)``."""
    return dict(_CONFIGS)


# ---------------------------------------------------------------------------
# the security profiles

#: Minimum value of each size option under ``standard``, by scheme kind;
#: also its default there. ``stat_param`` is ignored by the schemes that do
#: not use it.
STANDARD_MINIMA = {
    "dl": {"sec_param": 3072, "modulus_length": 3072, "stat_param": STANDARD_STAT_PARAM},
    # ristretto255 is fixed (about 126-bit security); nothing to size
    "ec": {},
    "dcr": {"bit_length": STANDARD_DCR_PRIME_BITS, "stat_param": STANDARD_STAT_PARAM},
    "ring_lwe": {"sec_param": 128, "stat_param": STANDARD_STAT_PARAM},
    "lwe": {"stat_param": STANDARD_STAT_PARAM},
    "lwe_abdp15": {"stat_param": STANDARD_STAT_PARAM},
}
#: Options the ``standard`` profile sets when the config has none, by scheme
#: kind: the dimension is then derived from the 128-bit tables.
STANDARD_PRESETS = {"ring_lwe": {"ring_n": "auto"}, "lwe": {"lwe_n": "auto"}, "lwe_abdp15": {"lwe_n": "auto"}}
#: The pairing group of the ``standard`` profile (the only one it accepts).
STANDARD_PAIRING_GROUP = "BLS12-381"

#: Settings the ``standard`` profile fixes for single schemes, with the reason
#: (``{scheme: ({key: value}, reason)}``).
STANDARD_FIXED: dict = {}

#: Schemes (``(family, variant)``) and scheme kinds the ``standard`` profile
#: refuses, with the reason.
REFUSED = {
    ("mife", "lwe_threshold"): "read at the width sigma / (1 + 4 X sqrt(eta)) of its security "
    "reduction, the LWE of its flooding-enlarged q (stat_param 128) needs a dimension at the very "
    "end of the 128-bit table (5888 to 6144, a public matrix of 6.5 to 7.8*10^9 entries) for a few "
    "small bounds, e.g. X = Y in {1, 2, 3, 8, 10} at eta = 1, and beyond it for most others "
    "(X = Y = 4 to 6, X = Y >= 12, most eta >= 2; about 6500 for X = Y = 50); use "
    "mife/ddh_threshold, or mcfe/ring_lwe_threshold with labels",
    "lwe_matrix": "at a 128-bit dimension (lwe_n = 3328 to 4352 for X = Y <= 50, eta <= 10) its "
    "public matrix has 1 to 2*10^10 entries and every ciphertext m = 4 n log2 q = 2.9 to 4.5*10^6 "
    "coordinates of 219 to 258 bits (80 to 145 MB); use sife/fullysec_ring_lwe for adaptive "
    "security, or the Ring-LWE scheme of the same family",
    ("mcfe", "paillier_decentralized"): "its key shares would be combined under a DCR modulus "
    "of more than 15000 bits; use mcfe/ddh_decentralized",
}


def _copy(config):
    return type(config)(config) if isinstance(config, SchemeConfig) else dict(config)


def resolve(config: dict, scheme: tuple | None) -> dict:
    """Return *config* completed with the presets of its ``security_profile``.

    The key generators call it on their config, converted to the scheme's
    config class (see the module docstring).

    Args:
        config: A key generator's configuration; returned unchanged (the same
            object) for the ``toy`` profile, otherwise copied.
        scheme: ``(family, variant)`` of the key generator, or ``None`` if it
            is not a registered scheme.

    Raises:
        FEValidationError: For an unknown profile, a scheme the profile does
            not support, or an explicit value weaker than the profile's.
    """
    profile = config.get("security_profile", TOY)
    if profile not in PROFILES:
        raise FEValidationError("unknown security_profile {!r}; use one of {}".format(profile, PROFILES))
    if profile == TOY:
        return config
    kind = SCHEME_KIND.get(scheme)
    if kind is None:
        raise FEValidationError("no '{}' parameters for {}".format(profile, scheme))
    reason = REFUSED.get(scheme) or REFUSED.get(kind)
    if reason:
        raise FEValidationError(
            "no '{}' parameter set for {}/{}: {}".format(profile, scheme[0], scheme[1], reason)
        )
    out = _copy(config)
    if kind == "pairing":
        group = out.setdefault("pairing_group_param", STANDARD_PAIRING_GROUP)
        if group != STANDARD_PAIRING_GROUP:
            raise FEValidationError(
                "the '{}' profile uses pairing group {}, got {!r}".format(profile, STANDARD_PAIRING_GROUP, group)
            )
        warnings.warn(
            "{}/{}: BLS12-381 gives about 120 to 126-bit security after the exTNFS "
            "attacks, below the 128-bit target of the '{}' profile".format(*scheme, profile),
            FESecurityWarning,
            stacklevel=3,
        )
        return out
    for key, minimum in STANDARD_MINIMA[kind].items():
        value = out.setdefault(key, minimum)
        if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
            raise FEValidationError(
                "the '{}' profile needs {} >= {}, got {!r}".format(profile, key, minimum, value)
            )
    for key, value in STANDARD_PRESETS.get(kind, {}).items():
        out.setdefault(key, value)
    fixed, why = STANDARD_FIXED.get(scheme, ({}, ""))
    for key, value in fixed.items():
        if out.setdefault(key, value) != value:
            raise FEValidationError(
                "the '{}' profile needs {}={!r} for {}/{}: {}".format(
                    profile, key, value, scheme[0], scheme[1], why
                )
            )
    return out


# ---------------------------------------------------------------------------
# option combinations of the threshold schemes


def validate_sharing(sharing) -> str:
    """Return *sharing* if it is one of :data:`SHARING_SCHEMES`, else raise."""
    if sharing not in SHARING_SCHEMES:
        raise FEValidationError(
            "sharing must be one of {}, not {!r}".format(SHARING_SCHEMES, sharing)
        )
    return sharing


def resolve_sharing(config: dict, s, t, scheme: str) -> str:
    """The key sharing of a threshold LWE / Ring-LWE layer for *config*.

    ``config["sharing"]`` if given, else ``"replicated"`` for ``t < s`` and
    ``"shamir"`` for ``t = s`` (where the two are equally sound and Shamir
    gives the outputs of versions <= 0.3.0). An explicit ``"shamir"`` with
    ``t < s`` -- the sharing of versions <= 0.3.0, whose partial decryptions
    reveal the decryption noise modulo small integers (see
    :mod:`pyfe4ai.schemes.threshold.sharing`) -- is refused under the
    ``standard`` profile and draws a
    :class:`~pyfe4ai.core.exceptions.FESecurityWarning` otherwise.

    Args:
        config: The key generator's (profile-resolved) config.
        s: Number of servers.
        t: Threshold (checked later by
            :func:`~pyfe4ai.schemes.threshold.sharing.validate_threshold`).
        scheme: Name of the scheme, for messages.

    Raises:
        FEValidationError: For a threshold that is not an integer (also a
            ``bool``), an unknown sharing, or ``"shamir"`` with ``t < s``
            under ``standard``.
    """
    if isinstance(t, bool) or not isinstance(t, int):
        raise FEValidationError("invalid threshold t={!r}: need an integer".format(t))
    below = isinstance(s, int) and not isinstance(s, bool) and t < s
    sharing = config.get("sharing")
    if sharing is None:
        return "replicated" if below else "shamir"
    validate_sharing(sharing)
    if sharing == "shamir" and below:
        message = (
            "{}: Shamir sharing with t = {} < s = {} lets partial decryptions reveal the "
            "decryption noise modulo small integers (versions <= 0.3.0); use "
            "sharing='replicated' (the default for t < s)".format(scheme, t, s)
        )
        if config.get("security_profile") == STANDARD:
            raise FEValidationError(message)
        warnings.warn(message, FESecurityWarning, stacklevel=4)
    return sharing


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
    """Refuse ``"kl"`` for ``t < s`` (see :mod:`pyfe4ai.schemes.threshold.kl`).

    Raises:
        FEValidationError: Unless ``t == s``.
    """
    if t != s:
        raise FEValidationError(
            "flooding_accounting='kl' needs t = s (got t = {}, s = {}): for t < s the Lagrange "
            "coefficients of some enrolled sets make an honest partial decryption reveal the "
            "decryption noise modulo a small integer, whatever the flooding width".format(t, s)
        )


def check_kl_flooding(flooding: str, sharing: str) -> None:
    """Refuse ``"kl"`` (threshold MCFE over Ring-LWE) but for Gaussian
    flooding of Shamir key shares, which is what the KL bound accounts for.

    Raises:
        FEValidationError: Unless ``flooding == "gaussian"`` and ``sharing == "shamir"``.
    """
    if flooding != "gaussian" or sharing != "shamir":
        raise FEValidationError(
            "flooding_accounting='kl' accounts for Gaussian flooding of Shamir key shares "
            "(t = s): got flooding={!r}, sharing={!r}".format(flooding, sharing)
        )


def check_kl_sharing(sharing: str) -> None:
    """Refuse ``"kl"`` (threshold MIFE over LWE) but for Shamir key shares.

    Raises:
        FEValidationError: Unless ``sharing == "shamir"``.
    """
    if sharing != "shamir":
        raise FEValidationError(
            "flooding_accounting='kl' accounts for the flooding of Shamir key shares (t = s), "
            "not sharing={!r}".format(sharing)
        )


def threshold_key_shares(config: dict, sharing: str) -> str:
    """The ``key_shares`` of the threshold MCFE over Ring-LWE: the config's,
    else ``"full"`` for the replicated-seeded sharing and ``"split"`` otherwise.

    Raises:
        FEValidationError: For a value not in :data:`KEY_SHARES`, or
            ``"split"`` with the replicated-seeded sharing.
    """
    key_shares = config.get("key_shares", "full" if sharing == "replicated-seeded" else "split")
    if key_shares not in KEY_SHARES:
        raise FEValidationError(
            "key_shares must be one of {}, not {!r}".format(", ".join(KEY_SHARES), key_shares))
    if key_shares == "split" and sharing == "replicated-seeded":
        raise FEValidationError("split key shares need the sharing 'shamir' or 'replicated'; "
                                "use key_shares='full' with 'replicated-seeded'")
    return key_shares


def check_no_flooding_options(config: dict) -> None:
    """Refuse the flooding-accounting options for the threshold MCFE over LWE,
    whose flooding is part of an LWE argument, not a statistical step.

    Raises:
        FEValidationError: For ``flooding_accounting`` other than the default,
            or any ``max_partial_decryptions``.
    """
    if config.get("flooding_accounting", PER_SAMPLE) != PER_SAMPLE or "max_partial_decryptions" in config:
        raise FEValidationError(
            "mcfe/lwe_threshold has no statistical flooding: flooding_accounting and "
            "max_partial_decryptions do not apply"
        )


# ---------------------------------------------------------------------------
# the options, as templates the config classes specialise

ETA = Option("inputs per client (the length of each inner product)", default=1, allowed="a positive int")
BOUND_X = Option("bound X on the plaintext entries", default=4, allowed="a positive int")
BOUND_Y = Option("bound Y on the weights", default=4, allowed="a positive int")
BOUND = Option("bound on the plaintext entries and the weights", default=100, allowed="a positive int")
GROUP = Option("the group (fixed by the scheme)", default="modp", allowed=("modp",))
MODULUS_LENGTH = Option(
    "bits of the modulus ``p`` of the Damgard / MDDH group", default=Derived("``sec_param``"),
    allowed="an int >= 3 large enough that ``2 n eta bound^2 < q``",
)
BIT_LENGTH = Option(
    "bits of each safe prime of the Paillier modulus ``N``", default=128,
    allowed="an int >= 6 large enough that ``N`` exceeds the inner products",
)
LWE_N = Option("LWE dimension", default=32, allowed="``auto`` (the 128-bit table) or a positive int")
MATRIX_MODE = Option(
    "how the seeded public matrix ``A`` is held: expanded once, or per batch", default="memory",
    allowed=("memory", "stream"),
)
GAUSSIAN_SAMPLER = Option(
    "sampler of the Gaussian key noise", default="facct", allowed=("rounded", "facct"),
    standard="``rounded`` refused",
)
LIBRARY_SAMPLER = GAUSSIAN_SAMPLER(
    default=Derived("the library-wide sampler (``rounded`` unless ``PYFE4AI_GAUSSIAN_SAMPLER``, "
                    "``set_gaussian_sampler`` or a ``gaussian_sampler()`` block says otherwise); "
                    "``facct`` under ``standard``"),
)
RING_MODULUS = Option(
    "ring modulus: an NTT prime, or a product of NTT primes (RNS)", default="prime", allowed=("prime", "rns"),
)
RING_N = Option(
    "ring dimension", default=Derived("32 (also for ``None``)"),
    allowed="``auto`` (the 128-bit table) or a power of two >= 2",
)
LABEL_HASH = Option(
    "hash of the labels onto the key space", default="shake128-rejection",
    allowed=("shake256-wide", "shake256-rejection", "shake128-rejection"),
)
PAIRING_GROUP_PARAM = Option(
    "pairing group", default="BLS12-381",
    allowed="an asymmetric pairing group (other than BLS12-381: needs charm-crypto)",
)
T = Option("threshold: partial decryptions needed to decrypt", default=2, allowed="``1 <= t <= s``")
S = Option("number of decryption servers", default=Derived("``len(lst_sid)``, else 1"), allowed="``len(lst_sid)``")
LST_SID = Option(
    "server ids", default=Derived("``sid_0`` ... ``sid_{s-1}``"), allowed="``s`` distinct ids",
)
SHARING = Option(
    "how the functional keys are shared among the servers",
    default=Derived("``replicated`` for ``t < s``, else ``shamir``"), allowed=SHARING_SCHEMES,
    standard="``shamir`` with ``t < s`` refused",
)
FLOODING_ACCOUNTING = Option(
    "accounting of the flooding noise: per sample, or the opt-in KL bound of a run", default=PER_SAMPLE,
    allowed=FLOODING_ACCOUNTINGS,
)
MAX_PARTIAL_DECRYPTIONS = Option(
    "partial decryptions each server may produce (``flooding_accounting = 'kl'`` only)", default=None,
    allowed="a positive int with ``kl``; absent otherwise",
)
STAT_PARAM = Option(
    "statistical parameter (rounding, flooding, masks, key widths)", default=64,
    allowed="an int of at most {}".format(MAX_STAT_PARAM),
)


class DDHConfig(SchemeConfig):
    """Options of the DDH schemes over ``Z_p^*``."""

    sec_param = SchemeConfig.sec_param(doc="bits of the safe prime ``p`` (not a security level)")
    group = GROUP
    eta = ETA


class RistrettoConfig(SchemeConfig):
    """Options of the DDH schemes over ristretto255 (a fixed group)."""

    sec_param = SchemeConfig.sec_param(doc="recorded only (the group is fixed)")
    group = GROUP(default="ristretto255", allowed=("ristretto255",))
    eta = ETA


class DamgardDDHConfig(SchemeConfig):
    """Options of the Damgard / MDDH schemes."""

    modulus_length = MODULUS_LENGTH
    bound = BOUND
    eta = ETA


class PaillierConfig(SchemeConfig):
    """Options of the Paillier (DCR) schemes."""

    sec_param = SchemeConfig.sec_param(doc="recorded only (``bit_length`` sizes ``N``)")
    stat_param = STAT_PARAM
    bit_length = BIT_LENGTH
    bound_x = BOUND_X(default=100)
    bound_y = BOUND_Y(default=100)
    eta = ETA


class LWEConfig(SchemeConfig):
    """Options of the LWE schemes."""

    sec_param = SchemeConfig.sec_param(doc="recorded only (``lwe_n`` sets the hardness)")
    eta = ETA
    bound_x = BOUND_X(default=20)
    bound_y = BOUND_Y(default=20)
    lwe_n = LWE_N


class ABDP15Config(LWEConfig):
    """Options of the ABDP15 IPFE with a seeded public matrix (SIFE, MIFE and threshold MIFE over LWE)."""

    matrix_mode = MATRIX_MODE
    gaussian_sampler = LIBRARY_SAMPLER


class RingLWEConfig(SchemeConfig):
    """Options of the Ring-LWE schemes."""

    sec_param = SchemeConfig.sec_param(doc="kappa of the BMMS21 widths and noise bound")
    eta = ETA
    bound_x = BOUND_X
    bound_y = BOUND_Y
    ring_modulus = RING_MODULUS
    ring_n = RING_N
    gaussian_sampler = GAUSSIAN_SAMPLER


class PairingIPEConfig(SchemeConfig):
    """Options of the pairing-based inner-product schemes."""

    sec_param = SchemeConfig.sec_param(doc="unused (the pairing group fixes the security)")
    pairing_group_param = PAIRING_GROUP_PARAM
    eta = ETA
    bound_x = BOUND_X(default=16)
    bound_y = BOUND_Y(default=16)


class ThresholdConfig(SchemeConfig):
    """Options of the threshold schemes (with their scheme's options)."""

    t = T
    s = S
    lst_sid = LST_SID


class ThresholdSharingConfig(ThresholdConfig):
    """Options of the threshold LWE / Ring-LWE layers, which share keys in several ways."""

    sharing = SHARING


_DLOG = {"dlog_bound": Option(
    "bound on ``|<x, y>|`` (after scaling) of the final discrete log",
    default=Derived("``10^(precision + 2)``"), allowed="a non-negative int",
)}
_RING_CRYPTO = {"gaussian_sampler": Option(
    "sampler of the encryption noise", default=Derived("the public parameters' sampler, else ``facct``"),
    allowed=("rounded", "facct"),
)}
_CT_FORMAT = {"ct_format": Option(
    "ciphertext format: JSON lists, or the binary format 1", default="json", allowed=("json", "binary"),
)}
_MATRIX = {"matrix_mode": MATRIX_MODE}
_LEGACY_SHAMIR = {"legacy_shamir": Option(
    "a server's handling of a Shamir key share with ``t < s`` of versions <= 0.3.0",
    default="refuse", allowed=("refuse", "warn"), standard="refused whatever the option",
)}
_BUDGET = {"partial_budget": Option(
    "the server's budget of partial decryptions (``flooding_accounting = 'kl'``)", default=None,
    allowed="a ``PartialDecryptionBudget`` of at most the public parameters' limit",
)}


class CryptoConfig(dict):
    """Options of every crypto object (encryptor, server, decryptor); the
    config classes of the schemes list their own (``CRYPTO``)."""

    OPTIONS: ClassVar[dict[str, Option]] = {
        "keys": Option("``{'pp': public parameters}`` plus ``'sk'`` (encryptors) or a key", default=None,
                       allowed="a non-empty mapping"),
        "id": Option("the client or server id", default=None),
        "type": Option("unused", default=None),
        "precision": Option("decimal digits kept when values are scaled to integers", default=0,
                            allowed="a non-negative int"),
        "cache_dir": SchemeConfig.cache_dir,
    }


# ---------------------------------------------------------------------------
# SIFE


class SIFEDDHConfig(DDHConfig):
    """Config of :class:`~pyfe4ai.schemes.sife.ddh.SIFEDDHKeyGenerator`."""

    SCHEME = ("sife", "ddh")
    CRYPTO = _DLOG
    eta = ETA(default=5)


class SIFEDDHRistretto255Config(RistrettoConfig):
    """Config of :class:`~pyfe4ai.schemes.sife.ddh_ristretto255.SIFEDDHRistretto255KeyGenerator`."""

    SCHEME = ("sife", "ddh_ristretto255")
    CRYPTO = _DLOG
    eta = ETA(default=5)


class SIFELWEConfig(ABDP15Config):
    """Config of :class:`~pyfe4ai.schemes.sife.lwe.SIFELWEKeyGenerator`."""

    SCHEME = ("sife", "lwe")
    CRYPTO = _MATRIX
    eta = ETA(default=5)
    bound_x = BOUND_X(default=20, allowed="a non-negative int")


class SIFEPaillierConfig(PaillierConfig):
    """Config of :class:`~pyfe4ai.schemes.sife.paillier.SIFEPaillierKeyGenerator`."""

    SCHEME = ("sife", "paillier")
    eta = ETA(default=5)


class SIFEDamgardConfig(DamgardDDHConfig):
    """Config of :class:`~pyfe4ai.schemes.sife.damgard_ddh.SIFEDamgardKeyGenerator`."""

    SCHEME = ("sife", "damgard")
    eta = ETA(default=5)


class SIFERingLWEConfig(RingLWEConfig):
    """Config of :class:`~pyfe4ai.schemes.sife.ring_lwe.SIFERingLWEKeyGenerator`."""

    SCHEME = ("sife", "ring_lwe")
    CRYPTO = _RING_CRYPTO
    eta = ETA(default=5)
    ring_modulus = RING_MODULUS(allowed=("prime",))


class SIFEFullySecRingLWEConfig(RingLWEConfig):
    """Config of :class:`~pyfe4ai.schemes.sife.fullysec_ring_lwe.SIFEFullySecRingLWEKeyGenerator`."""

    SCHEME = ("sife", "fullysec_ring_lwe")
    CRYPTO = _RING_CRYPTO
    eta = ETA(default=5)
    ring_modulus = RING_MODULUS(allowed=("prime",))
    m = Option("number of key polynomials", default="auto",
               allowed="``auto`` (the smallest admitting a modulus) or an int >= 5")
    ndarray_layout = Option("ndarray ciphertexts: one per element, or packed (one per ``ring_n`` elements)",
                            default="elementwise", allowed=("elementwise", "packed"))


class SIFEFullySecLWEConfig(LWEConfig):
    """Config of :class:`~pyfe4ai.schemes.sife.fullysec_lwe.SIFEFullySecLWEKeyGenerator`."""

    SCHEME = ("sife", "fullysec_lwe")
    eta = ETA(default=5)
    lwe_n = LWE_N(default=64, allowed="an int >= 2")
    bound_x = BOUND_X(default=20, allowed="a non-negative int")
    bound_y = BOUND_Y(default=20, allowed="a non-negative int")


class SIFEDDHDynamicConfig(DDHConfig):
    """Config of :class:`~pyfe4ai.schemes.sife.ddh_dynamic.SIFEDDHDynamicKeyGenerator`."""

    SCHEME = ("sife", "ddh_dynamic")
    CRYPTO = _DLOG
    eta = ETA(default=5, doc="the largest number of inputs (keys may cover a prefix)")


class SIFEFHIPEPairingConfig(PairingIPEConfig):
    """Config of :class:`~pyfe4ai.schemes.sife.fh_ipe_pairing.SIFEFHIPEPairingKeyGenerator`."""

    SCHEME = ("sife", "fh_ipe_pairing")
    eta = ETA(default=5)


class SIFEPartFHIPEPairingConfig(SchemeConfig):
    """Config of :class:`~pyfe4ai.schemes.sife.part_fh_ipe_pairing.SIFEPartFHIPEPairingKeyGenerator`."""

    SCHEME = ("sife", "part_fh_ipe_pairing")
    sec_param = PairingIPEConfig.sec_param
    pairing_group_param = PAIRING_GROUP_PARAM
    eta = ETA(default=5, allowed="an int above ``subspace_dim``")
    bound = BOUND(default=Derived("``bound_x``, else 16"), doc="bound on the plaintext entries and the weights")
    bound_x = BOUND_X(default=16, doc="the default of ``bound``")
    subspace_dim = Option("dimension ``k`` of the plaintext subspace", default=1, allowed="``1 <= k < eta``")
    coeff_bound = Option("bound on the subspace coefficients of a public encryption", default=1,
                         allowed="a positive int with ``k coeff_bound <= bound``")


# ---------------------------------------------------------------------------
# MIFE


class MIFEDDHConfig(DDHConfig):
    """Config of :class:`~pyfe4ai.schemes.mife.ddh.MIFEDDHKeyGenerator`."""

    SCHEME = ("mife", "ddh")
    CRYPTO = _DLOG
    eta = ETA(allowed="a positive int, or one per client (a dict keyed by ``lst_nid``)")


class MIFEDDHRistretto255Config(RistrettoConfig):
    """Config of :class:`~pyfe4ai.schemes.mife.ddh_ristretto255.MIFEDDHRistretto255KeyGenerator`."""

    SCHEME = ("mife", "ddh_ristretto255")
    CRYPTO = _DLOG
    eta = ETA(allowed="a positive int, or one per client (a dict keyed by ``lst_nid``)")


class MIFELWEConfig(ABDP15Config):
    """Config of :class:`~pyfe4ai.schemes.mife.lwe.MIFELWEKeyGenerator`."""

    SCHEME = ("mife", "lwe")
    CRYPTO = _MATRIX
    bound_x = BOUND_X(default=20, allowed="a non-negative int")


class MIFEPaillierConfig(PaillierConfig):
    """Config of :class:`~pyfe4ai.schemes.mife.paillier.MIFEPaillierKeyGenerator`."""

    SCHEME = ("mife", "paillier")


class MIFEDamgardConfig(DamgardDDHConfig):
    """Config of :class:`~pyfe4ai.schemes.mife.damgard_ddh.MIFEDamgardKeyGenerator`."""

    SCHEME = ("mife", "damgard")


class MIFERingLWEConfig(RingLWEConfig):
    """Config of :class:`~pyfe4ai.schemes.mife.ring_lwe.MIFERingLWEKeyGenerator`."""

    SCHEME = ("mife", "ring_lwe")
    CRYPTO = {**_RING_CRYPTO, **_CT_FORMAT}


class MIFEFullySecLWEConfig(LWEConfig):
    """Config of :class:`~pyfe4ai.schemes.mife.fullysec_lwe.MIFEFullySecLWEKeyGenerator`."""

    SCHEME = ("mife", "fullysec_lwe")
    lwe_n = LWE_N(default=64, allowed="an int >= 2")
    bound_x = BOUND_X(default=20, allowed="a non-negative int")
    bound_y = BOUND_Y(default=20, allowed="a non-negative int")


class MIFEFHIPEPairingConfig(PairingIPEConfig):
    """Config of :class:`~pyfe4ai.schemes.mife.fh_ipe_pairing.MIFEFHIPEPairingKeyGenerator`."""

    SCHEME = ("mife", "fh_ipe_pairing")


class MIFEFHMultiIPEPairingConfig(PairingIPEConfig):
    """Config of :class:`~pyfe4ai.schemes.mife.fh_multi_ipe_pairing.MIFEFHMultiIPEPairingKeyGenerator`."""

    SCHEME = ("mife", "fh_multi_ipe_pairing")
    n = SchemeConfig.n(allowed="an int >= 2")
    sec_level = Option("``k`` of the k-Lin assumption", default=2, allowed="a positive int")


class ThresholdMIFEDDHConfig(MIFEDDHConfig, ThresholdConfig):
    """Config of :class:`~pyfe4ai.schemes.mife.ddh_threshold.ThresholdMIFEDDHKeyGenerator`."""

    SCHEME = ("mife", "ddh_threshold")


class ThresholdMIFELWEConfig(MIFELWEConfig, ThresholdSharingConfig):
    """Config of :class:`~pyfe4ai.schemes.mife.lwe_threshold.ThresholdMIFELWEKeyGenerator`."""

    SCHEME = ("mife", "lwe_threshold")
    CRYPTO = {**_MATRIX, **_LEGACY_SHAMIR, **_BUDGET}
    stat_param = STAT_PARAM
    flooding_accounting = FLOODING_ACCOUNTING
    max_partial_decryptions = MAX_PARTIAL_DECRYPTIONS


# ---------------------------------------------------------------------------
# MCFE


class MCFEDDHConfig(DDHConfig):
    """Config of :class:`~pyfe4ai.schemes.mcfe.ddh.MCFEDDHKeyGenerator`."""

    SCHEME = ("mcfe", "ddh")
    CRYPTO = _DLOG
    eta = ETA(allowed="a positive int, or one per client (a dict of ``n`` entries)")


class MCFEDDHRistretto255Config(RistrettoConfig):
    """Config of :class:`~pyfe4ai.schemes.mcfe.ddh_ristretto255.MCFEDDHRistretto255KeyGenerator`."""

    SCHEME = ("mcfe", "ddh_ristretto255")
    CRYPTO = _DLOG
    eta = ETA(allowed="a positive int, or one per client (a dict keyed by ``lst_nid``)")


class MCFELWEConfig(LWEConfig):
    """Config of :class:`~pyfe4ai.schemes.mcfe.lwe.MCFELWEKeyGenerator`."""

    SCHEME = ("mcfe", "lwe")
    stat_param = STAT_PARAM
    gaussian_sampler = GAUSSIAN_SAMPLER
    label_hash = LABEL_HASH


class MCFEPaillierConfig(PaillierConfig):
    """Config of :class:`~pyfe4ai.schemes.mcfe.paillier.MCFEPaillierKeyGenerator`."""

    SCHEME = ("mcfe", "paillier")


class MCFEDamgardConfig(DamgardDDHConfig):
    """Config of :class:`~pyfe4ai.schemes.mcfe.damgard_ddh.MCFEDamgardKeyGenerator`."""

    SCHEME = ("mcfe", "damgard")
    mddh_k = Option("``k`` of the MDDH assumption", default=1, allowed="a positive int")


class MCFERingLWEConfig(RingLWEConfig):
    """Config of :class:`~pyfe4ai.schemes.mcfe.ring_lwe.MCFERingLWEKeyGenerator`."""

    SCHEME = ("mcfe", "ring_lwe")
    CRYPTO = {**_RING_CRYPTO, **_CT_FORMAT}
    label_hash = LABEL_HASH


class MCFEFullySecLWEConfig(MCFELWEConfig):
    """Config of :class:`~pyfe4ai.schemes.mcfe.fullysec_lwe.MCFEFullySecLWEKeyGenerator`."""

    SCHEME = ("mcfe", "fullysec_lwe")


class MCFEFHMultiIPEPairingConfig(PairingIPEConfig):
    """Config of :class:`~pyfe4ai.schemes.mcfe.fh_multi_ipe_pairing.MCFEFHMultiIPEPairingKeyGenerator`."""

    SCHEME = ("mcfe", "fh_multi_ipe_pairing")
    bound_x = BOUND_X(default=10)
    bound_y = BOUND_Y(default=10)


class ThresholdMCFEFHMultiIPEPairingConfig(MCFEFHMultiIPEPairingConfig, ThresholdConfig):
    """Config of :class:`~pyfe4ai.schemes.mcfe.fh_multi_ipe_pairing_threshold.ThresholdMCFEFHMultiIPEPairingKeyGenerator`."""

    SCHEME = ("mcfe", "fh_multi_ipe_pairing_threshold")


class DecentralizedMCFEFHMultiIPEPairingConfig(MCFEFHMultiIPEPairingConfig):
    """Config of :class:`~pyfe4ai.schemes.mcfe.fh_multi_ipe_pairing_decentralized.DecentralizedMCFEFHMultiIPEPairingKeyGenerator`."""

    SCHEME = ("mcfe", "fh_multi_ipe_pairing_decentralized")
    n = SchemeConfig.n(allowed="an int >= 2 (checked by ``setup``)")


class ThresholdMCFEDDHConfig(MCFEDDHConfig, ThresholdConfig):
    """Config of :class:`~pyfe4ai.schemes.mcfe.ddh_threshold.ThresholdMCFEDDHKeyGenerator`."""

    SCHEME = ("mcfe", "ddh_threshold")
    eta = ETA
    s = S(default=Derived("``len(lst_sid)``, else 3"))


class DecentralizedMCFEDDHConfig(MCFEDDHConfig):
    """Config of :class:`~pyfe4ai.schemes.mcfe.ddh_decentralized.DecentralizedMCFEDDHKeyGenerator`."""

    SCHEME = ("mcfe", "ddh_decentralized")
    n = SchemeConfig.n(allowed="an int >= 2")
    stat_param = STAT_PARAM
    eta = ETA(allowed="a positive int (one for every client)")


class ThresholdMCFELWEConfig(MCFELWEConfig, ThresholdSharingConfig):
    """Config of :class:`~pyfe4ai.schemes.mcfe.lwe_threshold.ThresholdMCFELWEKeyGenerator`."""

    SCHEME = ("mcfe", "lwe_threshold")
    CRYPTO = _LEGACY_SHAMIR
    s = S(default=Derived("``len(lst_sid)``, else 3"))
    flooding_accounting = FLOODING_ACCOUNTING(allowed=(PER_SAMPLE,))
    max_partial_decryptions = MAX_PARTIAL_DECRYPTIONS(allowed="none (the option is refused)")


class DecentralizedMCFELWEConfig(MCFELWEConfig):
    """Config of :class:`~pyfe4ai.schemes.mcfe.lwe_decentralized.DecentralizedMCFELWEKeyGenerator`."""

    SCHEME = ("mcfe", "lwe_decentralized")


class ThresholdMCFERingLWEConfig(MCFERingLWEConfig, ThresholdSharingConfig):
    """Config of :class:`~pyfe4ai.schemes.mcfe.ring_lwe_threshold.ThresholdMCFERingLWEKeyGenerator`."""

    SCHEME = ("mcfe", "ring_lwe_threshold")
    CRYPTO = {**_RING_CRYPTO, "ct_format": _CT_FORMAT["ct_format"](
        default=Derived("the public parameters' format"), doc="ciphertext format (must be the public parameters')",
    ), **_LEGACY_SHAMIR, **_BUDGET}
    stat_param = STAT_PARAM
    ring_modulus = RING_MODULUS(default=DEFAULT_RING_MODULUS)
    flooding_accounting = FLOODING_ACCOUNTING
    max_partial_decryptions = MAX_PARTIAL_DECRYPTIONS
    flooding = Option("distribution of the flooding noise of the partial decryptions",
                      default=Derived("``uniform``; ``gaussian`` with ``flooding_accounting = 'kl'``"),
                      allowed=FLOODINGS)
    ct_format = Option("ciphertext format (published in the public parameters): binary format 2, or JSON lists",
                       default=DEFAULT_CT_FORMAT, allowed=("json", "binary"))
    key_shares = Option("key shares: a weight part and a label part, or whole key shares per label",
                        default=Derived("``full`` with ``replicated-seeded``, else ``split``"), allowed=KEY_SHARES)


class DecentralizedMCFERingLWEConfig(MCFERingLWEConfig):
    """Config of :class:`~pyfe4ai.schemes.mcfe.ring_lwe_decentralized.DecentralizedMCFERingLWEKeyGenerator`."""

    SCHEME = ("mcfe", "ring_lwe_decentralized")
    n = SchemeConfig.n(allowed="an int >= 2")


class DecentralizedMCFEPaillierConfig(MCFEPaillierConfig):
    """Config of :class:`~pyfe4ai.schemes.mcfe.paillier_decentralized.DecentralizedMCFEPaillierKeyGenerator`."""

    SCHEME = ("mcfe", "paillier_decentralized")
    n = SchemeConfig.n(allowed="an int >= 2")


# ---------------------------------------------------------------------------
# quadratic FE


class QuadraticQuadConfig(SchemeConfig):
    """Config of :class:`~pyfe4ai.schemes.quadratic.quad.QuadraticQuadKeyGenerator`."""

    SCHEME = ("quadratic", "quad")
    sec_param = PairingIPEConfig.sec_param
    pairing_group_param = PAIRING_GROUP_PARAM
    n = SchemeConfig.n(doc="length of ``x``", default=2, allowed="a positive int")
    m = Option("length of ``y``", default=Derived("``n``"), allowed="a positive int")
    bound = BOUND(default=16, doc="bound on the entries of ``x``, ``y`` and the function matrix")


class QuadraticSGPConfig(SchemeConfig):
    """Config of :class:`~pyfe4ai.schemes.quadratic.sgp.QuadraticSGPKeyGenerator`."""

    SCHEME = ("quadratic", "sgp")
    sec_param = PairingIPEConfig.sec_param
    pairing_group_param = PAIRING_GROUP_PARAM
    n = SchemeConfig.n(doc="length of ``x`` and ``y``", default=Derived("``eta``, else 2"), allowed="a positive int")
    eta = ETA(default=2, doc="the default of ``n``")
    bound = BOUND(default=16, doc="bound on the entries of ``x``, ``y`` and the function matrix")


class QuadraticMIQFEConfig(SchemeConfig):
    """Config of :class:`~pyfe4ai.schemes.quadratic.miqfe.QuadraticMIQFEKeyGenerator`."""

    SCHEME = ("quadratic", "miqfe")
    sec_param = PairingIPEConfig.sec_param
    pairing_group_param = PAIRING_GROUP_PARAM
    n = SchemeConfig.n(doc="number of users", allowed="an int >= 2 (required)")
    m = Option("inputs per user", default=2, allowed="a positive int")
    bound_x = BOUND_X(default=8)
    bound_c = Option("bound on the coefficients of the quadratic function", default=8, allowed="a positive int")
    mddh_k = Option("``k`` of the bilateral MDDH assumption", default=2, allowed="an int >= 2")


# ---------------------------------------------------------------------------
# refused and constrained combinations (documentation)


class Combination:
    """A combination of options a key generator or crypto object refuses (or
    accepts with a warning), for the documentation.

    Args:
        schemes: The schemes (``family/variant``, or a kind such as ``ring_lwe``).
        when: The combination.
        outcome: What happens (the exception, or the warning), and when.
    """

    def __init__(self, schemes: str, when: str, outcome: str) -> None:
        self.schemes = schemes
        self.when = when
        self.outcome = outcome


#: The combinations of options that are refused or constrained, besides the
#: profile's minima and refusals (:data:`STANDARD_MINIMA`, :data:`REFUSED`).
COMBINATIONS = (
    Combination("pairing schemes", "``security_profile = 'standard'`` with ``pairing_group_param`` "
                "other than ``BLS12-381``", "``FEValidationError``; with ``BLS12-381`` a "
                "``FESecurityWarning`` (about 120 to 126-bit security)"),
    Combination("``sife/lwe``, ``mife/lwe``, the MCFE LWE schemes (``mcfe/lwe``, ``fullysec_lwe``, "
                "``lwe_threshold``, ``lwe_decentralized``), the Ring-LWE schemes",
                "``security_profile = 'standard'`` with ``gaussian_sampler = 'rounded'`` (also a crypto "
                "object's, for public parameters made under ``standard``)", "``FEValidationError``"),
    Combination("Ring-LWE schemes", "``security_profile = 'standard'`` with an explicit ``ring_n`` whose "
                "modulus falls below the 128-bit table, or bounds too large for any tabulated dimension",
                "``FEValidationError``"),
    Combination("LWE schemes", "``security_profile = 'standard'`` with an explicit ``lwe_n`` below the "
                "128-bit table, or bounds too large for any tabulated dimension", "``FEValidationError``"),
    Combination("``sife/ring_lwe``, ``sife/fullysec_ring_lwe``", "``ring_modulus`` other than ``prime``",
                "``FEValidationError``"),
    Combination("``mife/ring_lwe``, ``mcfe/ring_lwe``, ``mcfe/ring_lwe_decentralized``", "``ring_modulus = "
                "'rns'`` with a modulus that two NTT primes below 2^62 cannot hold (above about 124 bits; large "
                "bounds)", "``FEValidationError``"),
    Combination("``sife/fullysec_ring_lwe``", "``m <= 4``", "``FEValidationError``"),
    Combination("``mife/lwe_threshold``, ``mcfe/lwe_threshold``, ``mcfe/ring_lwe_threshold``",
                "``sharing = 'shamir'`` with ``t < s``", "``FEValidationError`` under ``standard``, "
                "else a ``FESecurityWarning``"),
    Combination("``mife/lwe_threshold``, ``mcfe/ring_lwe_threshold``", "``flooding_accounting = 'kl'`` "
                "with ``t != s``, or without a positive ``max_partial_decryptions``; "
                "``max_partial_decryptions`` with ``per-sample``", "``FEValidationError``"),
    Combination("``mife/lwe_threshold``", "``flooding_accounting = 'kl'`` with a sharing other than "
                "``shamir``", "``FEValidationError``"),
    Combination("``mcfe/ring_lwe_threshold``", "``flooding_accounting = 'kl'`` with "
                "``flooding = 'uniform'`` or a sharing other than ``shamir``", "``FEValidationError``"),
    Combination("``mcfe/ring_lwe_threshold``", "``key_shares = 'split'`` with "
                "``sharing = 'replicated-seeded'``", "``FEValidationError``"),
    Combination("``mcfe/ring_lwe_threshold``", "``ring_modulus = 'rns'`` with a factor of the modulus "
                "not above ``s``", "``FEValidationError``"),
    Combination("``mcfe/ring_lwe_threshold`` (crypto)", "``ct_format`` other than the public "
                "parameters'", "``FEValidationError``"),
    Combination("``mcfe/lwe_threshold``", "any ``flooding_accounting`` other than ``per-sample``, or "
                "``max_partial_decryptions``", "``FEValidationError``"),
    Combination("threshold schemes", "``t`` outside ``1 .. s`` (also the defaults ``t = 2``, ``s = 1``), "
                "or ``s`` other than ``len(lst_sid)``", "``FEValidationError``"),
    Combination("threshold LWE / Ring-LWE servers (crypto)", "a Shamir key share with ``t < s`` (versions "
                "<= 0.3.0) without ``legacy_shamir = 'warn'``, or under ``standard``",
                "``FESchemeError`` at the partial decryption"),
    Combination("``mife/lwe_threshold``, ``mcfe/ring_lwe_threshold`` servers (crypto)",
                "``flooding_accounting = 'kl'`` public parameters without a ``partial_budget``, or one above "
                "their ``max_partial_decryptions``", "``FESchemeError`` / ``FEValidationError``"),
    Combination("DDH schemes", "``group`` other than the scheme's (``modp``, or ``ristretto255``)",
                "``FEValidationError``"),
    Combination("``mcfe/ddh``", "``eta`` as a dict whose length is not ``n``", "``FEValidationError``"),
    Combination("``mcfe/ddh_decentralized``", "``eta`` as a dict", "``FEValidationError``"),
    Combination("decentralised MCFE (DDH, DCR, LWE, Ring-LWE, FH)", "``n < 2``",
                "``FEValidationError`` at ``setup``"),
    Combination("``mife/fh_multi_ipe_pairing``", "``n = 1``", "``FEKeyError``"),
    Combination("``quadratic/miqfe``", "no ``n``", "``FEValidationError``"),
    Combination("Damgard and Paillier schemes", "bounds too large for the modulus (``2 n eta bound^2`` "
                "not below ``q``, or ``N``): raise ``modulus_length`` / ``bit_length``",
                "``FEValidationError`` (``ValueError`` for ``sife/paillier``)"),
    Combination("``mcfe/damgard``", "``mddh_k < 1``", "``FEValidationError``"),
    Combination("``sife/part_fh_ipe_pairing``", "``subspace_dim`` not in ``1 .. eta - 1``; "
                "``bound < subspace_dim * coeff_bound`` (at a public encryption)", "``FEValidationError``"),
    Combination("pairing and quadratic schemes", "a discrete-log bound above ``2^48`` (e.g. "
                "``eta bound_x bound_y``, ``n m bound^3``)", "``ValueError`` at decryption (quadratic "
                "schemes: ``None`` and a warning)"),
)


# ---------------------------------------------------------------------------
# the documentation page


def _value(value) -> str:
    if isinstance(value, Derived):
        return value.text
    if value is None:
        return "--"
    return "``{!r}``".format(value)


def _allowed(allowed) -> str:
    if allowed is None:
        return ""
    if isinstance(allowed, tuple):
        return ", ".join("``{}``".format(a) for a in allowed)
    return allowed


def _standard(cls: type[SchemeConfig], name: str, option: Option) -> str:
    kind = SCHEME_KIND.get(cls.SCHEME)
    parts = []
    if kind in STANDARD_MINIMA and name in STANDARD_MINIMA[kind]:
        parts.append(">= ``{}`` (its default)".format(STANDARD_MINIMA[kind][name]))
    if name in STANDARD_PRESETS.get(kind, {}):
        parts.append("``{!r}``".format(STANDARD_PRESETS[kind][name]))
    if kind == "pairing" and name == "pairing_group_param":
        parts.append("``{}`` only".format(STANDARD_PAIRING_GROUP))
    if option.standard:
        parts.append(option.standard)
    return "; ".join(parts)


def _table(rows: list[tuple[str, ...]], header: tuple[str, ...]) -> list[str]:
    out = [".. list-table::", "   :header-rows: 1", ""]
    for row in [header] + rows:
        out.append("   * - " + row[0])
        out.extend("     - " + cell for cell in row[1:])
    out.append("")
    return out


def _heading(text: str, mark: str) -> list[str]:
    return [text, mark * len(text), ""]


def options_rst() -> str:
    """The page *Configuration options* of the documentation, from the
    declarations of this module (``docs/configuration.rst``)."""
    lines = _heading("Configuration options", "=")
    lines += [
        ".. This page is generated by pyfe4ai.core.config.options_rst(); "
        "tests/test_config.py checks that it is current.",
        "",
        "A key generator takes its configuration as a mapping of options; it is converted into the config "
        "class of its scheme (:mod:`pyfe4ai.core.config`) and completed by its security profile before the key "
        "generator is built. Crypto objects take their config as given. Options not listed are ignored.",
        "",
    ]
    lines += _heading("Security profiles", "-")
    rows = []
    for kind in sorted({k for k in SCHEME_KIND.values()}):
        schemes = ", ".join("``{}/{}``".format(*s) for s, k in SCHEME_KIND.items() if k == kind)
        if REFUSED.get(kind) or (kind not in STANDARD_MINIMA and kind != "pairing"):
            standard = "refused"
        elif kind == "pairing":
            standard = "``pairing_group_param = '{}'`` only, with a ``FESecurityWarning``".format(STANDARD_PAIRING_GROUP)
        else:
            standard = "; ".join(["``{}`` >= ``{}``".format(k, v) for k, v in STANDARD_MINIMA[kind].items()]
                                 + ["``{} = {!r}``".format(k, v) for k, v in STANDARD_PRESETS.get(kind, {}).items()]
                                 ) or "nothing to set"
        rows.append((kind, schemes, standard))
    lines += [
        "``security_profile = 'toy'`` (the default) uses the config as given. ``'standard'`` sets the minima "
        "below where the config has no value and refuses weaker ones (``FEValidationError``); see "
        ":mod:`pyfe4ai.core.profiles`.",
        "",
    ]
    lines += _table(rows, ("kind", "schemes", "under ``standard``"))
    lines += ["Refused under ``standard`` (``FEValidationError``):", ""]
    for key, reason in REFUSED.items():
        name = "``{}/{}``".format(*key) if isinstance(key, tuple) else ", ".join(
            "``{}/{}``".format(*s) for s, k in SCHEME_KIND.items() if k == key)
        lines.append("* {}: {}.".format(name, reason))
    lines.append("")
    lines += _heading("Refused and constrained combinations", "-")
    lines += _table([(c.schemes, c.when, c.outcome) for c in COMBINATIONS], ("schemes", "options", "outcome"))
    lines += _heading("Options of every key generator", "-")
    lines += _table([(name, o.doc, _value(o.default), _allowed(o.allowed)) for name, o in SchemeConfig.OPTIONS.items()],
                    ("option", "meaning", "default", "accepted"))
    lines += ["Crypto objects take:", ""]
    lines += _table([(name, o.doc, _value(o.default), _allowed(o.allowed)) for name, o in CryptoConfig.OPTIONS.items()],
                    ("option", "meaning", "default", "accepted"))
    lines += _heading("Options of each scheme", "-")
    lines += ["The options of every key generator above, where a scheme changes them, and the scheme's own.", ""]
    for scheme, cls in _CONFIGS.items():
        lines += _heading("``{}/{}``".format(*scheme), "~")
        lines += [":class:`~pyfe4ai.core.config.{}`; kind ``{}``.".format(cls.__name__, SCHEME_KIND.get(scheme)), ""]
        rows = []
        for name, option in cls.OPTIONS.items():
            if option is SchemeConfig.OPTIONS.get(name) and not _standard(cls, name, option):
                continue
            rows.append((name, option.doc, _value(option.default), _standard(cls, name, option),
                         _allowed(option.allowed)))
        lines += _table(rows, ("option", "meaning", "default", "``standard``", "accepted"))
        if cls.CRYPTO:
            lines += ["Crypto objects also take:", ""]
            lines += _table([(name, o.doc, _value(o.default), _allowed(o.allowed)) for name, o in cls.CRYPTO.items()],
                            ("option", "meaning", "default", "accepted"))
    return "\n".join(lines).rstrip() + "\n"
