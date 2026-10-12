"""Base classes for DDH-family key generators.

Provides shared parameter caching logic (group generation, verification,
serialisation) so that concrete SIFE / MIFE / MCFE DDH schemes only need
to declare which configuration fields they use.
"""

from __future__ import annotations

import json
import logging
import secrets
from abc import abstractmethod

import gmpy2 as gp

from pyfe4ai.schemes.ipfe import IPFEAbsKeyGenerator, ParameterCacheMixin
from pyfe4ai.utils.crypto_utils import group_generator_fe, is_dl_group
from pyfe4ai.utils.crypto_utils import hash_to_qr_group

logger = logging.getLogger(__name__)



def _damgard_h(seed: str, p) -> gp.mpz:
    """The Damgard element ``h``, hashed into the order-``q`` subgroup from *seed*."""
    bytes.fromhex(seed)  # a hex string
    return hash_to_qr_group(seed, gp.mpz(p), domain="pyfe4ai/damgard/h")

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

    def _loaded_parameters_valid(self) -> bool:
        # every DDH group here is a safe-prime group (r = 2); a cached r would
        # let p = r q + 1 with a small q through
        return int(self.r) == 2 and is_dl_group(self.p, self.q, self.g, self.sec_param)

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
        self.h_seed = param["h_seed"]
        self.h = _damgard_h(self.h_seed, self.p)

    def _param_verification(self, param: dict) -> bool:
        """Return ``True`` if *param* matches the current configuration keys."""
        return all(param.get(k) == getattr(self, k) for k in self._verify_keys)

    def _loaded_parameters_valid(self) -> bool:
        return (
            is_dl_group(self.p, self.q, self.g, self.modulus_length)
            and 1 < self.h < self.p
            and gp.powmod(self.h, self.q, self.p) == 1
        )

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
        # h hashed into the group from a seed: nobody (nor a planted cache)
        # knows its discrete logarithm to the base g
        while True:
            self.h_seed = secrets.token_hex(32)
            self.h = _damgard_h(self.h_seed, self.p)
            if self.h != 1:
                break
        _param = {
            "sec_param": self.sec_param,
            "eta": self.eta,
            "modulus_length": self.modulus_length,
            "bound": self.bound,
            "group": {"p": gp.digits(self.p), "q": gp.digits(self.q)},
            "g": gp.digits(self.g),
            "h_seed": self.h_seed,
            **self._extra_param_fields(),
        }
        with open(param_file, "w", encoding="utf-8") as f:
            json.dump(_param, f)
