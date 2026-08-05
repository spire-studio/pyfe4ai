"""
Shared base classes for Ring-LWE-family inner-product FE schemes.

Extracts duplicated parameter handling and crypto initialisation that is
identical across SIFE Ring-LWE, MIFE Ring-LWE and MCFE Ring-LWE.
"""

from __future__ import annotations

import gmpy2 as gp

from pyfe4ai.schemes.ipfe import IPFEAbsCrypto, IPFEAbsKeyGenerator, ParameterCacheMixin


class RingLWEKeyGeneratorBase(IPFEAbsKeyGenerator, ParameterCacheMixin):
    """Common key-generator base for Ring-LWE schemes.

    Subclasses MUST set:

    - ``_scheme_type`` -- ``CryptoCONST.TYPE_*``
    - ``_verify_keys`` -- tuple of config attribute names checked by
      ``_param_verification`` (e.g. ``("sec_param", "eta", "bound_x", "bound_y")``)

    Subclasses MAY override:

    - ``_extra_verify(param)`` -- additional param-cache checks (default True)
    - ``_extra_param_fields()`` -- extra fields for the param JSON (default {})
    """

    _verify_keys: tuple[str, ...] = ()

    # ------------------------------------------------------------------
    # Shared parameter loading
    # ------------------------------------------------------------------
    def _apply_parameters(self, param: dict) -> None:
        """Assign loaded parameters to instance attributes."""
        self.p = gp.mpz(param["p"])
        self.q = gp.mpz(param["q"])
        self.ring_n = int(param["ring_n"])
        self.sigma1 = float(param["sigma1"])
        self.sigma2 = float(param["sigma2"])
        self.sigma3 = float(param["sigma3"])
        self.A = [gp.mpz(v) for v in param["A"]]

    # ------------------------------------------------------------------
    # Shared verification
    # ------------------------------------------------------------------
    def _param_verification(self, param: dict) -> bool:
        """Return ``True`` if cached parameters match the current config."""
        for key in self._verify_keys:
            val = getattr(self, key)
            # bound_x / bound_y are stored as digit-strings in param cache
            if isinstance(val, gp.mpz):
                if param.get(key) != gp.digits(val):
                    return False
            else:
                if param.get(key) != val:
                    return False
        # ring_n may be None (auto-derive), only check if set
        if self.ring_n is not None and param.get("ring_n") != self.ring_n:
            return False
        return self._extra_verify(param)

    def _extra_verify(self, param: dict) -> bool:  # noqa: ARG002
        return True

    def _extra_param_fields(self) -> dict:
        """Return additional fields to include in the parameter file."""
        return {}


class RingLWECryptoBase(IPFEAbsCrypto):
    """Common crypto base for Ring-LWE schemes.

    Provides the shared ``__init__`` that loads ``pp``, ``A`` and ``sk``.
    """

    def __init__(self, config: dict, **kwargs) -> None:
        """Initialise with scheme configuration.

        Args:
            config: Scheme configuration dict.
        """
        super().__init__(config, **kwargs)
        self.pp = self.keys["pp"]
        self.A = [gp.mpz(v) for v in self.pp["A"]]
        if self._has_private_keys():
            self.sk = self.keys["sk"]
