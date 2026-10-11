"""
Decentralised multi-client inner-product FE over DDH.

| Abdalla, Bourse, Marival, Pointcheval, Soleimanian, Waldner:
| "Multi-Client Inner-Product Functional Encryption in the Random-Oracle Model"
| SCN 2020, https://eprint.iacr.org/2020/788 -- Appendix D.1, Fig. 14
| (DMCFE from MCFE), instantiated with
| MCFE1 = CDGPP18 (ASIACRYPT 2018, :mod:`pyfe4ai.schemes.mcfe.ddh`) and
| MCFE2 = the DCR MCFE of ABMPSW20 Fig. 3 (:mod:`pyfe4ai.schemes.mcfe.decentralize`).

* type:     decentralised multi-client encryption (DMCFE, DDH + DCR, random oracle)
* setting:  Integer based (order-q subgroup of Z_p^*; Z_{N2^2}^* for key shares)

Encryption and decryption are those of CDGPP18 (``c = g^x [H(l)]^s``,
``g^{<x,y>} = prod c^y / [H(l)]^d``). There is no authority for functional
keys: for weights ``y`` every client ``i`` publishes an MCFE2 encryption of
its contribution ``c_i = (sum_j y_{i,j} s_{i,j} mod q) + q rho_i`` (a pair)
under the label ``H(y)``; the zero-sum MCFE2 keys make the product of all
clients' shares decrypt to ``sum_i c_i``, whose residue mod ``q`` is the
CDGPP18 functional key ``d``. Every client must publish a share (with zero
weights if absent from ``y``).

The carry mask ``rho_i`` is our own adaptation. ABMPSW20 states Fig. 14 for
MCFEs that need no discrete logarithm when decrypting (DCR, LWE): with
CDGPP18 the secrets are uniform mod ``q``, so ``sum_i y_i s_i`` is not small.
Reducing each contribution mod ``q`` (as versions before this one did) makes
the integer sum ``d + q carry`` with ``carry`` in ``0..n-1``, which anyone
holding the published shares reads off: a comparison oracle on the clients'
reduced contributions, from which a series of keys reveals their secrets.
Here ``rho_i`` is uniform below ``2^(stat_param + bits(n))`` and derived
from a per-client key and ``H(y)`` (repeating a derivation repeats the share,
so masks cannot be averaged out), and ``carry + sum_i rho_i`` is within
statistical distance ``2^-stat_param`` of a carry-independent distribution.
The argument is ours, hence the scheme's *experimental* status; Theorem 38
covers the MCFE2 layer only.
"""

from __future__ import annotations

import hashlib
import logging
import secrets

import gmpy2 as gp

from pyfe4ai.schemes.mcfe.ddh import MCFE
from pyfe4ai.schemes.mcfe.ddh import MCFEKeyGenerator
from pyfe4ai.schemes.mcfe.decentralize import combine_contributions
from pyfe4ai.schemes.mcfe.decentralize import encrypt_contribution
from pyfe4ai.schemes.mcfe.decentralize import generate_mcfe2_modulus
from pyfe4ai.schemes.mcfe.decentralize import key_label
from pyfe4ai.schemes.mcfe.decentralize import mcfe2_key_sigma
from pyfe4ai.schemes.mcfe.decentralize import mcfe2_prime_bits
from pyfe4ai.schemes.mcfe.decentralize import zero_sum_keys
from pyfe4ai.schemes.threshold_utils import fusion_weight_id
from pyfe4ai.utils.crypto_constants import CryptoCONST
from pyfe4ai.utils.exceptions import FEKeyError, FESchemeError, FEValidationError

logger = logging.getLogger(__name__)

_CONSTRUCTION = "abmpsw20-fig14-cdgpp18-dcr-carry-mask/v2"
_KEY_DIM = 2  # a CDGPP18 functional key is a pair in Z_q^2


def _carry_mask(key_hex: str, label: str, index: int, bits: int) -> gp.mpz:
    """``rho < 2^bits`` from SHAKE-256 keyed with the client's mask key (a PRF of the label)."""
    n_bytes = (bits + 7) // 8
    material = "|".join(["pyfe4ai/dmcfe-ddh/carry-mask", str(index), label]).encode("utf-8")
    digest = hashlib.shake_256(bytes.fromhex(key_hex) + material).digest(n_bytes)
    return gp.mpz(int.from_bytes(digest, "big") >> (8 * n_bytes - bits))


class DecentralizedMCFEKeyGenerator(MCFEKeyGenerator):
    """Sets up CDGPP18 client keys and zero-sum MCFE2 keys (ABMPSW20 Fig. 14)."""

    _scheme_type = CryptoCONST.TYPE_DMCFE
    _verify_keys = ("sec_param", "eta", "n", "construction", "n2_prime_bits")

    def __init__(self, config: dict, **kwargs) -> None:
        """Initialise the key generator.

        Args:
            config: As for :class:`MCFEKeyGenerator` (``sec_param``, ``eta``,
                ``n``, ``lst_nid``), plus ``stat_param`` for MCFE2.
        """
        self.construction = _CONSTRUCTION
        self.stat_param = int(config.get("stat_param", 64))
        sec_param = config.get("sec_param", CryptoCONST.SEC_PARAM)
        n = config.get("n") or len(config.get("lst_nid", []))
        # contributions (s . y mod q) + q rho with q < 2^sec_param, rho < 2^mask_bits
        self.mask_bits = self.stat_param + max(1, int(n).bit_length())
        self.contribution_bound = gp.mpz(1) << (sec_param + self.mask_bits + 1)
        self.n2_prime_bits = mcfe2_prime_bits(n * _KEY_DIM, self.contribution_bound)
        super().__init__(config, **kwargs)
        if isinstance(self.eta, dict):
            raise FEValidationError("decentralised MCFE needs an integer eta")

    def _extra_param_fields(self) -> dict:
        return dict(
            super()._extra_param_fields(),
            construction=self.construction,
            n2_prime_bits=self.n2_prime_bits,
            n2_mod=gp.digits(self.n2_mod),
        )

    def _apply_parameters(self, param: dict) -> None:
        super()._apply_parameters(param)
        self.n2_mod = gp.mpz(param["n2_mod"])

    def _generate_and_save(self, param_file: str) -> None:
        self.n2_mod = generate_mcfe2_modulus(self.n2_prime_bits)
        super()._generate_and_save(param_file)

    def setup(self) -> None:
        """Sample the CDGPP18 client secrets and the zero-sum MCFE2 keys."""
        super().setup()
        sigma2 = mcfe2_key_sigma(
            self.n2_mod, len(self.lst_nid) * _KEY_DIM, self.contribution_bound, self.stat_param
        )
        self.msk["s2"] = zero_sum_keys(self.lst_nid, _KEY_DIM, sigma2)
        self.msk["mask_key"] = {nid: secrets.token_hex(32) for nid in self.lst_nid}

    def get_public_parameters(self) -> dict:
        """CDGPP18 public parameters plus ``n2_mod``, ``lst_nid`` and the contribution bound."""
        pp = super().get_public_parameters()
        pp.update(
            {
                "n2_mod": gp.digits(self.n2_mod),
                "lst_nid": list(self.lst_nid),
                "contribution_bound": gp.digits(self.contribution_bound),
                "mask_bits": self.mask_bits,
            }
        )
        return pp

    def get_private_keys(self, nid: str) -> dict | None:
        """Return client *nid*'s keys: CDGPP18 ``s``, MCFE2 ``s2`` and the carry-mask key."""
        sk = super().get_private_keys(nid)
        if sk is None:
            return None
        sk["s2"] = [gp.digits(v) for v in self.msk["s2"][nid]]
        sk["mask_key"] = self.msk["mask_key"][nid]
        return sk

    def get_decryption_keys(self, sid: str, **kwargs) -> dict | None:
        """Not available: functional keys are derived by the clients."""
        raise NotImplementedError(
            "decentralised MCFE has no central functional key; combine the "
            "clients' derive_function_decryption_key_share() outputs instead"
        )


class DecentralizedMCFE(MCFE):
    """CDGPP18 encryption / decryption with decentralised key derivation."""

    def derive_function_decryption_key_share(self, fusion_weight: dict) -> dict:
        """Publish this client's encrypted key contribution for weights ``y``.

        Args:
            fusion_weight: Mapping of client IDs to integer weight lists (the
                whole ``y``; clients that are absent contribute zero).

        Returns:
            Dict with ``c`` (MCFE2 ciphertexts), ``nid`` and ``weights_id``.
        """
        if not self._has_private_keys() or "s2" not in self.sk:
            raise FEKeyError("need the client's keys to derive a key share")
        if self.id not in self.pp["lst_nid"]:
            raise FESchemeError("unknown client id {}".format(self.id))
        weights = fusion_weight.get(self.id, [])
        if len(weights) > len(self.sk["s"]):
            raise FEValidationError("fusion weight of {} is longer than eta".format(self.id))
        q = (gp.mpz(self.pp["p"]) - 1) // 2
        contribution = [gp.mpz(0)] * _KEY_DIM
        for y_j, s_j in zip(weights, self.sk["s"]):
            contribution = [acc + gp.mpz(int(y_j)) * gp.mpz(s) for acc, s in zip(contribution, s_j)]
        label = key_label(fusion_weight)
        contribution = [
            v % q + q * _carry_mask(self.sk["mask_key"], label, j, int(self.pp["mask_bits"]))
            for j, v in enumerate(contribution)
        ]
        return {
            "c": encrypt_contribution(
                self.pp["n2_mod"], self.sk["s2"], contribution,
                label, gp.mpz(self.pp["contribution_bound"]),
            ),
            "nid": self.id,
            "weights_id": fusion_weight_id(fusion_weight),
        }

    def combine_function_decryption_key_share(self, dct_dk_shares: dict) -> dict:
        """Combine every client's share into the CDGPP18 key ``{d}``.

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
        q = (gp.mpz(self.pp["p"]) - 1) // 2
        return {"d": [gp.digits(v % q) for v in sums]}
