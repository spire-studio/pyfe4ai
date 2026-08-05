"""Base classes for DDH-family key generators.

Provides shared parameter caching logic (group generation, verification,
serialisation) so that concrete SIFE / MIFE / MCFE DDH schemes only need
to declare which configuration fields they use.
"""

from __future__ import annotations

import json
import logging
from abc import abstractmethod

import gmpy2 as gp

from pyfe4ai.schemes.ipfe import IPFEAbsKeyGenerator, ParameterCacheMixin
from pyfe4ai.utils.crypto_utils import _random, group_generator_fe

logger = logging.getLogger(__name__)


class DDHKeyGeneratorBase(IPFEAbsKeyGenerator, ParameterCacheMixin):
    """Base for standard DDH key generators sharing a (p, q, r, g) group.

    Subclasses must set ``_scheme_type`` and ``_verify_keys``, and may
    override ``_extra_param_fields`` to persist additional configuration.
    """

    _verify_keys: tuple[str, ...] = ()

    def _apply_parameters(self, param: dict) -> None:
        self.p = gp.mpz(param["group"]["p"])
        self.q = gp.mpz(param["group"]["q"])
        self.r = gp.mpz(param["group"]["r"])
        self.g = gp.mpz(param["g"])

    def _param_verification(self, param: dict) -> bool:
        """Return ``True`` if *param* matches the current configuration keys."""
        return all(param.get(k) == getattr(self, k) for k in self._verify_keys)

    def _extra_param_fields(self) -> dict:
        """Return extra fields to include in the saved parameter dict."""
        return {}

    def _generate_and_save(self, param_file: str) -> None:
        """Generate a fresh DDH group (p, q, r, g) and persist to *param_file*."""
        self.p, self.q, self.r, self.g = group_generator_fe(self.sec_param)
        _param = {
            "sec_param": self.sec_param,
            "g": gp.digits(self.g),
            "eta": self.eta,
            "group": {
                "p": gp.digits(self.p),
                "q": gp.digits(self.q),
                "r": gp.digits(self.r),
            },
            **self._extra_param_fields(),
        }
        with open(param_file, "w") as f:
            json.dump(_param, f)


class DamgardDDHKeyGeneratorBase(IPFEAbsKeyGenerator, ParameterCacheMixin):
    """Base for Damgard DDH key generators sharing a (p, q, g, h) group.

    Subclasses must set ``_scheme_type`` and ``_verify_keys``, implement
    ``_validate_bounds``, and may override ``_extra_param_fields``.
    """

    _verify_keys: tuple[str, ...] = ()

    def _apply_parameters(self, param: dict) -> None:
        """Assign group elements (p, q, g, h) from the loaded parameter dict."""
        self.p = gp.mpz(param["group"]["p"])
        self.q = gp.mpz(param["group"]["q"])
        self.g = gp.mpz(param["g"])
        self.h = gp.mpz(param["h"])

    def _param_verification(self, param: dict) -> bool:
        """Return ``True`` if *param* matches the current configuration keys."""
        return all(param.get(k) == getattr(self, k) for k in self._verify_keys)

    def _extra_param_fields(self) -> dict:
        """Return extra fields to include in the saved parameter dict."""
        return {}

    @abstractmethod
    def _validate_bounds(self) -> None:
        """Validate that the bound constraints hold for the group order."""
        raise NotImplementedError

    def _generate_and_save(self, param_file: str) -> None:
        """Generate a fresh Damgard group (p, q, g, h) and persist to *param_file*."""
        self.p, self.q, _, self.g = group_generator_fe(self.modulus_length)
        self._validate_bounds()
        while True:
            exponent = _random(self.q, self.modulus_length)
            if exponent <= 1:
                continue
            self.h = gp.powmod(self.g, exponent, self.p)
            if self.h not in (1, self.p - 1):
                break
        _param = {
            "sec_param": self.sec_param,
            "eta": self.eta,
            "modulus_length": self.modulus_length,
            "bound": self.bound,
            "group": {"p": gp.digits(self.p), "q": gp.digits(self.q)},
            "g": gp.digits(self.g),
            "h": gp.digits(self.h),
            **self._extra_param_fields(),
        }
        with open(param_file, "w", encoding="utf-8") as f:
            json.dump(_param, f)
