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

from pyfe4ai.core.base import IPFEAbsKeyGenerator, ParameterCacheMixin
from pyfe4ai.primitives.groups.integer_groups import group_generator_fe, is_dl_group
from pyfe4ai.primitives.hashing import hash_to_qr_group
from pyfe4ai.core.exceptions import FEValidationError
from pyfe4ai.core.formats.dl import DamgardRecord
from pyfe4ai.core.formats.dl import DDHRecord
from pyfe4ai.core.formats.dl import DLGroupRecord
from pyfe4ai.core.versions import record_matches

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

    def __init__(self, config: dict, **kwargs) -> None:
        super().__init__(config, **kwargs)
        group = config.get("group", "modp")
        if group != "modp":
            # the elliptic-curve instantiation is a scheme of its own
            raise FEValidationError(
                "{} runs in a Z_p^* subgroup (group='modp'), got group={!r}; ristretto255 is "
                "the variant ddh_ristretto255".format(type(self).__name__, group)
            )

    def _apply_parameters(self, param: dict) -> None:
        record = DDHRecord.from_dict(param)
        self.p = gp.mpz(DLGroupRecord(record.group).p)
        self.q = gp.mpz(DLGroupRecord(record.group).q)
        self.r = gp.mpz(DLGroupRecord(record.group).r)
        self.g = gp.mpz(record.g)

    def _param_verification(self, param: dict) -> bool:
        """Return ``True`` if *param* matches the current configuration keys."""
        return record_matches(param, {k: getattr(self, k) for k in self._verify_keys})

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
        _param = DDHRecord.new(
            sec_param=self.sec_param,
            g=gp.digits(self.g),
            eta=self.eta,
            group=DLGroupRecord.new(p=gp.digits(self.p), q=gp.digits(self.q), r=gp.digits(self.r)).to_dict(),
            **self._extra_param_fields(),
        )
        with open(param_file, "w") as f:
            json.dump(_param.to_dict(), f)


class DamgardDDHKeyGeneratorBase(IPFEAbsKeyGenerator, ParameterCacheMixin):
    """Base for Damgard DDH key generators sharing a (p, q, g, h) group.

    Subclasses must set ``_scheme_type`` and ``_verify_keys``, implement
    ``_validate_bounds``, and may override ``_extra_param_fields``.
    """

    _verify_keys: tuple[str, ...] = ()

    def _apply_parameters(self, param: dict) -> None:
        """Assign group elements (p, q, g, h) from the loaded parameter dict."""
        record = DamgardRecord.from_dict(param)
        self.p = gp.mpz(DLGroupRecord(record.group).p)
        self.q = gp.mpz(DLGroupRecord(record.group).q)
        self.g = gp.mpz(record.g)
        self.h_seed = record.h_seed
        self.h = _damgard_h(self.h_seed, self.p)

    def _param_verification(self, param: dict) -> bool:
        """Return ``True`` if *param* matches the current configuration keys."""
        return record_matches(param, {k: getattr(self, k) for k in self._verify_keys})

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
        _param = DamgardRecord.new(
            sec_param=self.sec_param,
            eta=self.eta,
            modulus_length=self.modulus_length,
            bound=self.bound,
            group=DLGroupRecord.new(p=gp.digits(self.p), q=gp.digits(self.q)).to_dict(),
            g=gp.digits(self.g),
            h_seed=self.h_seed,
            **self._extra_param_fields(),
        )
        with open(param_file, "w", encoding="utf-8") as f:
            json.dump(_param.to_dict(), f)
