"""
Decentralised multi-client inner-product FE over LWE.

| Abdalla, Bourse, Marival, Pointcheval, Soleimanian, Waldner:
| "Multi-Client Inner-Product Functional Encryption in the Random-Oracle Model"
| SCN 2020, https://eprint.iacr.org/2020/788 -- Appendix D.1, Fig. 14
| (DMCFE from MCFE), instantiated with MCFE1 = their LWE MCFE (Fig. 4,
| :mod:`pyfe4ai.schemes.mcfe.lwe`) and MCFE2 = their DCR MCFE (Fig. 3,
| :mod:`pyfe4ai.schemes.mcfe.decentralize`).

* type:     decentralised multi-client encryption (DMCFE, LWE + DCR, random oracle)
* setting:  Integer based (Z_q with a rounding map to Z_q0; Z_{N2^2}^* for key shares)

Encryption and decryption are those of the LWE MCFE. For weights ``y`` every
client ``i`` publishes an MCFE2 encryption of its contribution
``sum_j y_{i,j} Z_{i,j}`` over the integers (a vector of length ``n0 + m0``)
under the label ``H(y)``; the zero-sum MCFE2 keys make the product of all
clients' shares decrypt to ``sk_y = sum_{i,j} y_{i,j} Z_{i,j}`` in
``Z^(n0 + m0)``, the functional key of Fig. 4. Versions before this one
reduced each contribution mod ``q``, so the decrypted integer sum also told
how many ``y_i t_i`` were negative in every coordinate. Every client
must publish a share. Secure against static corruptions (Theorem 38).
Versions <= 0.2.0 used an ad-hoc label scalar with per-client key
components instead. A key share costs ``n0 + m0`` DCR encryptions of one
label hash (computed with a fixed-base method).

``N2`` only has to hold the shares (``2 n2 X2^2 < N2``, a few hundred bits
for ``X2 = q``); under the ``standard`` security profile each of its primes
has at least ``STANDARD_DCR_PRIME_BITS`` = 1536 bits. Key derivation thus
also rests on DCR, which is not post-quantum, whatever the LWE parameters.
"""

from __future__ import annotations

import json
import logging

import gmpy2 as gp

from pyfe4ai.core.base import ParameterCacheMixin
from pyfe4ai.schemes.mcfe.decentralize import combine_contributions
from pyfe4ai.schemes.mcfe.decentralize import encrypt_contribution
from pyfe4ai.schemes.mcfe.decentralize import generate_mcfe2_modulus
from pyfe4ai.schemes.mcfe.decentralize import key_label
from pyfe4ai.schemes.mcfe.decentralize import mcfe2_key_sigma
from pyfe4ai.schemes.mcfe.decentralize import MIN_PRIME_BITS
from pyfe4ai.schemes.mcfe.decentralize import mcfe2_prime_bits
from pyfe4ai.schemes.mcfe.decentralize import zero_sum_keys
from pyfe4ai.schemes.mcfe.lwe import MCFELWE
from pyfe4ai.schemes.mcfe.lwe import MCFELWEKeyGenerator
from pyfe4ai.core.profiles import STANDARD
from pyfe4ai.core.profiles import STANDARD_DCR_PRIME_BITS
from pyfe4ai.schemes.threshold.binding import fusion_weight_id
from pyfe4ai.core.constants import CryptoCONST
from pyfe4ai.core.exceptions import FEKeyError, FESchemeError, FEValidationError
from pyfe4ai.primitives.groups.integer_groups import is_dcr_modulus
from pyfe4ai._validation import strict_int

logger = logging.getLogger(__name__)

_CONSTRUCTION = "abmpsw20-fig14-lwe-dcr/v2"


class DecentralizedMCFELWEKeyGenerator(MCFELWEKeyGenerator, ParameterCacheMixin):
    """Sets up LWE MCFE client keys and zero-sum MCFE2 keys (ABMPSW20 Fig. 14)."""

    _holds_dcr_modulus = True  # never cached under standard (see ParameterCacheMixin)

    _scheme_type = CryptoCONST.TYPE_DMCFE_LWE

    def __init__(self, config: dict, **kwargs) -> None:
        """Initialise the key generator.

        Args:
            config: As for :class:`~pyfe4ai.schemes.mcfe.lwe.MCFELWEKeyGenerator`
                (``n``, ``lst_nid``, ``eta``, ``lwe_n``, ``bound_x``, ``bound_y``,
                ``stat_param``).
        """
        super().__init__(config, **kwargs)
        # contributions sum_j y_j Z_{i,j} over Z: |.| <= eta Y q (the s part)
        self.contribution_bound = gp.mpz(self.eta) * self.bound_y * self.q
        # N2 only has to hold the key shares (a few hundred bits); the profile
        # sets the factoring-hardness floor
        floor = STANDARD_DCR_PRIME_BITS if self.security_profile == STANDARD else MIN_PRIME_BITS
        self.n2_prime_bits = mcfe2_prime_bits(
            len(self.lst_nid) * self.dim, self.contribution_bound, min_bits=floor
        )
        self._load_parameters()  # caches only N2; the LWE parameters are derived

    def _param_verification(self, param: dict) -> bool:
        return (
            param.get("construction") == _CONSTRUCTION
            and param.get("n2_prime_bits") == self.n2_prime_bits
        )

    def _apply_parameters(self, param: dict) -> None:
        self.n2_mod = gp.mpz(param["n2_mod"])

    def _loaded_parameters_valid(self) -> bool:
        return is_dcr_modulus(self.n2_mod, self.n2_prime_bits)

    def _generate_and_save(self, param_file: str) -> None:
        self.n2_mod = generate_mcfe2_modulus(self.n2_prime_bits)
        with open(param_file, "w", encoding="utf-8") as f:
            json.dump(
                {
                    "construction": _CONSTRUCTION,
                    "n2_prime_bits": self.n2_prime_bits,
                    "n2_mod": gp.digits(self.n2_mod),
                },
                f,
            )

    def setup(self) -> None:
        """Sample the LWE MCFE client keys and the zero-sum MCFE2 keys."""
        super().setup()
        sigma2 = mcfe2_key_sigma(
            self.n2_mod, len(self.lst_nid) * self.dim, self.contribution_bound, self.stat_param
        )
        self.msk["s2"] = zero_sum_keys(self.lst_nid, self.dim, sigma2)

    def get_public_parameters(self) -> dict:
        """LWE MCFE public parameters plus ``n2_mod`` and ``lst_nid``."""
        pp = super().get_public_parameters()
        pp.update({
            "n2_mod": gp.digits(self.n2_mod),
            "lst_nid": list(self.lst_nid),
            "contribution_bound": gp.digits(self.contribution_bound),
        })
        return pp

    def get_private_keys(self, nid: str, **kwargs) -> dict | None:
        """Return client *nid*'s keys: LWE MCFE ``z`` and MCFE2 ``s2``."""
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


class DecentralizedMCFELWE(MCFELWE):
    """LWE MCFE encryption / decryption with decentralised key derivation."""

    def derive_function_decryption_key_share(self, fusion_weight: dict) -> dict:
        """Publish this client's encrypted contribution ``sum_j y_{i,j} Z_{i,j}`` (over Z).

        Args:
            fusion_weight: The whole weight mapping ``y`` (absent clients
                contribute zero).

        Returns:
            Dict with ``c`` (MCFE2 ciphertexts), ``nid`` and ``weights_id``.
        """
        if not self._has_private_keys() or "s2" not in self.sk:
            raise FEKeyError("need the client's keys to derive a key share")
        if self.id not in self.pp["lst_nid"]:
            raise FESchemeError("unknown client id {}".format(self.id))
        weights = fusion_weight.get(self.id, [])
        if len(weights) > len(self.sk["z"]):
            raise FEValidationError("fusion weight of {} is longer than eta".format(self.id))
        if any(abs(strict_int(v, "fusion weight")) > self.pp["bound_y"] for v in weights):
            raise FEValidationError("fusion weight exceeds configured bound_y")
        contribution = [gp.mpz(0)] * self.dim
        for y_j, z_j in zip(weights, self.sk["z"]):
            y_j = gp.mpz(int(y_j))
            contribution = [acc + y_j * gp.mpz(z) for acc, z in zip(contribution, z_j)]
        # sent over Z as in Fig. 14 (sk_y = sum y_i Z_i in Z^(n0 + m0)); reducing it
        # mod q first would reveal, per coordinate, how many y_i t_i are negative
        return {
            "c": encrypt_contribution(
                self.pp["n2_mod"], self.sk["s2"], contribution, key_label(fusion_weight),
                gp.mpz(self.pp["contribution_bound"]),
            ),
            "nid": self.id,
            "weights_id": fusion_weight_id(fusion_weight),
        }

    def combine_function_decryption_key_share(self, dct_dk_shares: dict) -> dict:
        """Combine every client's share into the LWE MCFE key ``{sk_y}``.

        Raises:
            FESchemeError: If a client's share is missing or the shares were
                derived for different weights.
        """
        if len({share["weights_id"] for share in dct_dk_shares.values()}) != 1:
            raise FESchemeError("key shares were derived for different weights")
        sums = combine_contributions(
            self.pp["n2_mod"],
            {nid: share["c"] for nid, share in dct_dk_shares.items()},
            self.pp["lst_nid"],
        )
        return {"sk_y": [gp.digits(v % self.q) for v in sums]}
