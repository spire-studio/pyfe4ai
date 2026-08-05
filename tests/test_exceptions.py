"""Tests for the FE exception hierarchy and common error paths.

Validates that the custom exceptions from :mod:`utils.exceptions` are raised
in the expected situations and conform to the inheritance hierarchy.
"""

from __future__ import annotations

import pytest

from pyfe4ai.utils.exceptions import (
    FEConfigError,
    FEError,
    FEKeyError,
    FESchemeError,
    FEValidationError,
)


# ---------------------------------------------------------------------------
# Exception hierarchy
# ---------------------------------------------------------------------------

class TestExceptionHierarchy:
    """All custom exceptions inherit from FEError (and thus Exception)."""

    def test_fe_error_is_exception(self):
        assert issubclass(FEError, Exception)

    @pytest.mark.parametrize("exc_cls", [FEConfigError, FEKeyError, FEValidationError, FESchemeError])
    def test_subclass_of_fe_error(self, exc_cls):
        assert issubclass(exc_cls, FEError)

    def test_validation_error_is_value_error(self):
        assert issubclass(FEValidationError, ValueError)

    def test_scheme_error_is_runtime_error(self):
        assert issubclass(FESchemeError, RuntimeError)

    def test_catch_fe_error_catches_subtypes(self):
        for cls in (FEConfigError, FEKeyError, FEValidationError, FESchemeError):
            with pytest.raises(FEError):
                raise cls("test")


# ---------------------------------------------------------------------------
# Key error paths — missing keys / credentials
# ---------------------------------------------------------------------------

class TestKeyErrors:
    """Crypto classes must reject missing keys at construction time."""

    def test_crypto_init_no_keys(self):
        """IPFEAbsCrypto raises FEKeyError when config has no 'keys'."""
        from pyfe4ai.schemes.sife.ddh import SIFE

        with pytest.raises(FEKeyError, match="no keys"):
            SIFE({"keys": None})

    def test_sife_dk_no_credentials(self):
        """SIFEKeyGenerator.get_decryption_keys raises on missing credentials."""
        from pyfe4ai.schemes.sife.ddh import SIFEKeyGenerator

        kg = SIFEKeyGenerator({"sec_param": 128, "eta": 3})
        kg.setup()
        with pytest.raises(FEKeyError, match="credentials"):
            kg.get_decryption_keys("sid_1")


# ---------------------------------------------------------------------------
# Validation error paths — bad inputs
# ---------------------------------------------------------------------------

class TestValidationErrors:
    """Schemes must reject invalid inputs with FEValidationError."""

    def test_mife_invalid_eta_type(self):
        """MIFEKeyGenerator rejects non-int/non-dict eta."""
        from pyfe4ai.schemes.mife.ddh import MIFEKeyGenerator

        with pytest.raises(FEValidationError, match="invalid parameter"):
            MIFEKeyGenerator({"sec_param": 128, "eta": "bad", "n": 2, "s": 1,
                              "lst_nid": ["a", "b"]})

    def test_sife_dk_invalid_fusion_weights(self):
        """SIFEKeyGenerator.get_decryption_keys rejects non-list fusion weights."""
        from pyfe4ai.schemes.sife.ddh import SIFEKeyGenerator

        kg = SIFEKeyGenerator({"sec_param": 128, "eta": 3})
        kg.setup()
        with pytest.raises(FEValidationError, match="fusion weights"):
            kg.get_decryption_keys("sid_1", credentials={"fusion_weight": "not_a_list"})

    def test_sife_dk_fusion_weight_length_mismatch(self):
        """SIFEKeyGenerator.get_decryption_keys rejects wrong-length fusion weights."""
        from pyfe4ai.schemes.sife.ddh import SIFEKeyGenerator

        kg = SIFEKeyGenerator({"sec_param": 128, "eta": 3})
        kg.setup()
        with pytest.raises(FEValidationError, match="length mismatch"):
            kg.get_decryption_keys("sid_1", credentials={"fusion_weight": [1, 2]})

    def test_sife_encrypt_plaintext_length(self):
        """SIFE.encrypt rejects plaintext with wrong length."""
        from pyfe4ai.schemes.sife.ddh import SIFE, SIFEKeyGenerator

        kg = SIFEKeyGenerator({"sec_param": 128, "eta": 3})
        kg.setup()
        pp = kg.get_public_parameters()
        sk = kg.get_private_keys()
        crypto = SIFE({"keys": {"pp": pp, "sk": sk}})
        with pytest.raises(FEValidationError, match="invalid size"):
            crypto.encrypt([1, 2, 3, 4])  # eta=3 but 4 elements
