"""Multi-input functional encryption schemes: the key generator and crypto class of every
scheme of the family (the registry: :mod:`pyfe4ai.core.registry`)."""

from pyfe4ai.schemes.mife.ddh import MIFEDDH, MIFEDDHKeyGenerator
from pyfe4ai.schemes.mife.ddh_ristretto255 import MIFEDDHRistretto255, MIFEDDHRistretto255KeyGenerator
from pyfe4ai.schemes.mife.lwe import MIFELWE, MIFELWEKeyGenerator
from pyfe4ai.schemes.mife.paillier import MIFEPaillier, MIFEPaillierKeyGenerator
from pyfe4ai.schemes.mife.damgard_ddh import MIFEDamgard, MIFEDamgardKeyGenerator
from pyfe4ai.schemes.mife.ring_lwe import MIFERingLWE, MIFERingLWEKeyGenerator
from pyfe4ai.schemes.mife.fullysec_lwe import MIFEFullySecLWE, MIFEFullySecLWEKeyGenerator
from pyfe4ai.schemes.mife.fh_ipe_pairing import MIFEFHIPEPairing, MIFEFHIPEPairingKeyGenerator
from pyfe4ai.schemes.mife.fh_multi_ipe_pairing import MIFEFHMultiIPEPairing, MIFEFHMultiIPEPairingKeyGenerator
from pyfe4ai.schemes.mife.ddh_threshold import ThresholdMIFEDDH, ThresholdMIFEDDHKeyGenerator
from pyfe4ai.schemes.mife.lwe_threshold import ThresholdMIFELWE, ThresholdMIFELWEKeyGenerator

__all__ = [
    "MIFEDDH",
    "MIFEDDHKeyGenerator",
    "MIFEDDHRistretto255",
    "MIFEDDHRistretto255KeyGenerator",
    "MIFELWE",
    "MIFELWEKeyGenerator",
    "MIFEPaillier",
    "MIFEPaillierKeyGenerator",
    "MIFEDamgard",
    "MIFEDamgardKeyGenerator",
    "MIFERingLWE",
    "MIFERingLWEKeyGenerator",
    "MIFEFullySecLWE",
    "MIFEFullySecLWEKeyGenerator",
    "MIFEFHIPEPairing",
    "MIFEFHIPEPairingKeyGenerator",
    "MIFEFHMultiIPEPairing",
    "MIFEFHMultiIPEPairingKeyGenerator",
    "ThresholdMIFEDDH",
    "ThresholdMIFEDDHKeyGenerator",
    "ThresholdMIFELWE",
    "ThresholdMIFELWEKeyGenerator",
]
