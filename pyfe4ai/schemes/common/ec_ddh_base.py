"""Shared parts of the DDH schemes over ristretto255 (``*/ddh_ristretto255``).

The constructions are those of the ``*/ddh`` modules, which are stated for
any prime-order group where DDH is hard; here the group is ristretto255
(RFC 9496, order ``L ~ 2^252``, about 126-bit security against Pollard's rho)
instead of a 3072-bit ``Z_p^*`` subgroup. There are no group parameters to
generate or cache. Group elements are serialised as 64 hex digits, scalars
as decimal strings. Select these schemes with the variant
``ddh_ristretto255`` or, through :class:`~pyfe4ai.ml.FESchemeWrapper`,
with ``group="ristretto255"`` on the ``ddh`` variant.
"""

from __future__ import annotations

from pyfe4ai.core.base import IPFEAbsCrypto, IPFEAbsKeyGenerator
from pyfe4ai.primitives.groups import ec_group
from pyfe4ai.core.exceptions import FEKeyError, FEValidationError


def check_group(config: dict) -> str:
    """The ``group`` entry of a config, which must name ristretto255 if given."""
    group = config.get("group", ec_group.GROUP)
    if group != ec_group.GROUP:
        raise FEValidationError(
            "this scheme runs in {}, got group={!r}".format(ec_group.GROUP, group)
        )
    return group


class ECDDHKeyGeneratorBase(IPFEAbsKeyGenerator):
    """Key generator base: checks the ``group`` entry of the config."""

    def __init__(self, config: dict, **kwargs) -> None:
        super().__init__(config, **kwargs)
        self.group = check_group(config)


class ECDDHCryptoBase(IPFEAbsCrypto):
    """Crypto base: public parameters must name the group; ``dlog_bound``
    (default ``10^(precision + 2)``) bounds the results decryption recovers."""

    def __init__(self, config: dict, **kwargs) -> None:
        super().__init__(config, **kwargs)
        self.pp = self.keys["pp"]
        if self.pp.get("group") != ec_group.GROUP:
            raise FEKeyError("public parameters of another group: {!r}".format(self.pp.get("group")))
        if self._has_private_keys():
            self.sk = self.keys["sk"]
        self.dlog_bound = int(config.get("dlog_bound", pow(10, self.precision + 2)))

    def _solve_dlog(self, point) -> int | None:
        return ec_group.dlog(point, self.dlog_bound)
