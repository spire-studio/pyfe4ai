"""
LWE-based multi-client inner-product FE, adaptive-security parameters.

| Abdalla, Bourse, Marival, Pointcheval, Soleimanian, Waldner:
| "Multi-Client Inner-Product Functional Encryption in the Random-Oracle Model"
| SCN 2020, https://eprint.iacr.org/2020/788 -- Fig. 4 with the adaptive
| parameter setting of Appendix C.3 (Lemma 33: ``m0 log2(4/3) >=
| log2 q + 2 kappa + 4 n log2 P``). The construction is the MCFE counterpart
| of the fully secure LWE IPFE of Agrawal, Libert, Stehle (CRYPTO 2016).

* type:     secret-key multi-client encryption (MCFE, LWE, random oracle)
* setting:  Integer based (Z_q with a rounding map to Z_q0)

The algorithms are those of :mod:`pyfe4ai.schemes.mcfe.lwe`; only the choice
of ``m0`` differs (larger, so that the statistical step of the proof also
covers adaptively chosen challenges). See that module for the construction,
the signed-message deviation and the one-ciphertext-per-label rule.
"""

from __future__ import annotations

from pyfe4ai.schemes.mcfe.lwe import MCFELWE
from pyfe4ai.schemes.mcfe.lwe import MCFELWEKeyGenerator
from pyfe4ai.utils.crypto_constants import CryptoCONST


class MCFEFullySecLWEKeyGenerator(MCFELWEKeyGenerator):
    """Key generator for the adaptively secure parameter set of ABMPSW20 Fig. 4."""

    _scheme_type = CryptoCONST.TYPE_MCFE_FULLYSEC_LWE
    _security = "adaptive"


class MCFEFullySecLWE(MCFELWE):
    """Crypto operations for ABMPSW20 Fig. 4 (adaptive parameters)."""
