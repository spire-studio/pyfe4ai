"""
Decentralised multi-client inner-product FE over DCR.

| Abdalla, Bourse, Marival, Pointcheval, Soleimanian, Waldner:
| "Multi-Client Inner-Product Functional Encryption in the Random-Oracle Model"
| SCN 2020, https://eprint.iacr.org/2020/788 -- Appendix D.1, Fig. 14
| (DMCFE from MCFE) with MCFE1 = their DCR MCFE (Fig. 3,
| :mod:`pyfe4ai.schemes.mcfe.paillier`) and MCFE2 = the same construction
| over a larger modulus ``N2`` (:mod:`pyfe4ai.schemes.mcfe.decentralize`) --
| the instantiation Appendix D.1 discusses.

* type:     decentralised multi-client encryption (DMCFE, DCR, random oracle)
* setting:  Integer based (Z_{N^2}^* for data, Z_{N2^2}^* for key shares)

Encryption and decryption are those of the DCR MCFE. For weights ``y`` every
client ``i`` publishes an MCFE2 encryption of its contribution
``k_i = sum_j y_{i,j} s_{i,j}`` (an integer: the MCFE1 key cannot be reduced
modulo anything public) under the label ``H(y)``; the zero-sum MCFE2 keys make
the product of all clients' shares decrypt to ``k = sum_i k_i``, the MCFE1
functional key. ``N2`` is sized for ``|k_i| <= eta * Y * 12 sigma``, with
``sigma`` the MCFE1 key width bounded through ``N < 2^(2 bit_length)``. Every
client must publish a share. Secure against static corruptions
(Theorem 38). Versions <= 0.2.0 used an ad-hoc label scalar, per-client key
components and brute-forceable keys instead.
"""

from __future__ import annotations

import json
import logging

import gmpy2 as gp

from pyfe4ai.schemes.mcfe.decentralize import combine_contributions
from pyfe4ai.schemes.mcfe.decentralize import encrypt_contribution
from pyfe4ai.schemes.mcfe.decentralize import generate_mcfe2_modulus
from pyfe4ai.schemes.mcfe.decentralize import key_label
from pyfe4ai.schemes.mcfe.decentralize import mcfe2_key_sigma
from pyfe4ai.schemes.mcfe.decentralize import mcfe2_prime_bits
from pyfe4ai.schemes.mcfe.decentralize import zero_sum_keys
from pyfe4ai.schemes.mcfe.paillier import MCFEPaillier
from pyfe4ai.schemes.mcfe.paillier import MCFEPaillierKeyGenerator
from pyfe4ai.schemes.mcfe.paillier import dcr_sigma
from pyfe4ai.schemes.threshold.threshold_utils import fusion_weight_id
from pyfe4ai.primitives.groups.integer_groups import generate_group_primes
from pyfe4ai.core.exceptions import FEKeyError, FESchemeError, FEValidationError
from pyfe4ai.primitives.lattice.lwe_utils import GAUSSIAN_TAIL_CUT
from pyfe4ai.primitives.groups.integer_groups import is_dcr_modulus
from pyfe4ai._validation import strict_int

logger = logging.getLogger(__name__)

TYPE_DMCFE_PAILLIER = "dMCFE_PAILLIER"
_CONSTRUCTION = "abmpsw20-fig14-dcr-dcr/v1"


class DecentralizedMCFEPaillierKeyGenerator(MCFEPaillierKeyGenerator):
    """Sets up DCR MCFE client keys and zero-sum MCFE2 keys (ABMPSW20 Fig. 14)."""

    _scheme_type = TYPE_DMCFE_PAILLIER

    def __init__(self, config: dict, **kwargs) -> None:
        """Initialise the key generator.

        Args:
            config: As for
                :class:`~pyfe4ai.schemes.mcfe.paillier.MCFEPaillierKeyGenerator`
                (``n``, ``lst_nid``, ``eta``, ``bit_length``, ``bound_x``,
                ``bound_y``, ``stat_param``).
        """
        eta = config.get("eta", 1)
        n = config.get("n") or len(config.get("lst_nid", []))
        kappa = config.get("stat_param", 64)
        # sigma of the MCFE1 keys for the largest N < 2^(2 bit_length)
        n_max = gp.mpz(1) << (2 * config.get("bit_length", 128))
        sigma_max = dcr_sigma(n_max, n * eta, config.get("bound_x", 100), kappa)
        self.contribution_bound = (
            eta * gp.mpz(config.get("bound_y", 100)) * GAUSSIAN_TAIL_CUT * sigma_max
        )
        self.n2_prime_bits = mcfe2_prime_bits(n, self.contribution_bound)
        super().__init__(config, **kwargs)

    def _param_verification(self, param: dict) -> bool:
        return (
            param.get("construction") == _CONSTRUCTION
            and param.get("bit_length") == self.bit_length
            and param.get("n2_prime_bits") == self.n2_prime_bits
        )

    def _apply_parameters(self, param: dict) -> None:
        super()._apply_parameters(param)
        self.n2_mod = gp.mpz(param["n2_mod"])

    def _loaded_parameters_valid(self) -> bool:
        return super()._loaded_parameters_valid() and is_dcr_modulus(self.n2_mod, self.n2_prime_bits)

    def _generate_and_save(self, param_file: str) -> None:
        # factorizations are decryption trapdoors: keep them local, never persist them
        p, _ = generate_group_primes(self.bit_length)
        while True:
            q, _ = generate_group_primes(self.bit_length)
            if q != p:
                break
        self.n_mod = p * q
        del p, q
        self.n_square = self.n_mod * self.n_mod
        self.n2_mod = generate_mcfe2_modulus(self.n2_prime_bits)
        with open(param_file, "w", encoding="utf-8") as f:
            json.dump(
                {
                    "construction": _CONSTRUCTION,
                    "bit_length": self.bit_length,
                    "n2_prime_bits": self.n2_prime_bits,
                    "n_mod": gp.digits(self.n_mod),
                    "n2_mod": gp.digits(self.n2_mod),
                },
                f,
            )

    def setup(self) -> None:
        """Sample the DCR MCFE client keys and the zero-sum MCFE2 keys."""
        super().setup()
        sigma2 = mcfe2_key_sigma(
            self.n2_mod, len(self.lst_nid), self.contribution_bound, self.stat_param
        )
        self.msk["s2"] = zero_sum_keys(self.lst_nid, 1, sigma2)

    def get_public_parameters(self) -> dict:
        """DCR MCFE public parameters plus ``n2_mod``, ``lst_nid`` and the bound."""
        pp = super().get_public_parameters()
        pp.update(
            {
                "n2_mod": gp.digits(self.n2_mod),
                "lst_nid": list(self.lst_nid),
                "contribution_bound": gp.digits(self.contribution_bound),
            }
        )
        return pp

    def get_private_keys(self, nid: str, **kwargs) -> dict | None:
        """Return client *nid*'s keys: DCR MCFE ``s`` and MCFE2 ``s2``."""
        sk = super().get_private_keys(nid)
        if sk is None:
            return None
        sk["s2"] = [gp.digits(v) for v in self.msk["s2"][nid]]
        return sk

    def get_decryption_keys(self, sid: str, **kwargs) -> dict | None:
        """Not available: functional keys are derived by the clients."""
        raise NotImplementedError(
            "decentralised MCFE has no central functional key; combine the "
            "clients' derive_function_decryption_key_share() outputs instead"
        )


class DecentralizedMCFEPaillier(MCFEPaillier):
    """DCR MCFE encryption / decryption with decentralised key derivation."""

    def derive_function_decryption_key_share(self, fusion_weight: dict) -> dict:
        """Publish this client's encrypted contribution ``sum_j y_{i,j} s_{i,j}``.

        Args:
            fusion_weight: The whole weight mapping ``y`` (absent clients
                contribute zero).

        Returns:
            Dict with ``c`` (MCFE2 ciphertext), ``nid`` and ``weights_id``.
        """
        if not self._has_private_keys() or "s2" not in self.sk:
            raise FEKeyError("need the client's keys to derive a key share")
        if self.id not in self.pp["lst_nid"]:
            raise FESchemeError("unknown client id {}".format(self.id))
        weights = fusion_weight.get(self.id, [])
        if len(weights) > len(self.sk["s"]):
            raise FEValidationError("fusion weight of {} is longer than eta".format(self.id))
        if any(abs(strict_int(v, "fusion weight")) > self.pp["bound_y"] for v in weights):
            raise FEValidationError("fusion weight exceeds configured bound_y")
        k_i = sum(gp.mpz(int(y)) * gp.mpz(s) for y, s in zip(weights, self.sk["s"]))
        return {
            "c": encrypt_contribution(
                self.pp["n2_mod"], self.sk["s2"], [gp.mpz(k_i)], key_label(fusion_weight),
                gp.mpz(self.pp["contribution_bound"]),
            ),
            "nid": self.id,
            "weights_id": fusion_weight_id(fusion_weight),
        }

    def combine_function_decryption_key_share(self, dct_dk_shares: dict) -> dict:
        """Combine every client's share into the DCR MCFE key ``{k}``.

        Raises:
            FESchemeError: If a client's share is missing or the shares were
                derived for different weights.
        """
        if len({share["weights_id"] for share in dct_dk_shares.values()}) != 1:
            raise FESchemeError("key shares were derived for different weights")
        (k,) = combine_contributions(
            self.pp["n2_mod"],
            {nid: share["c"] for nid, share in dct_dk_shares.items()},
            self.pp["lst_nid"],
        )
        return {"k": gp.digits(k)}
