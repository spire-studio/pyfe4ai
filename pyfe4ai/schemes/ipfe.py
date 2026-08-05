from __future__ import annotations

import json
import logging
import os
from abc import ABC, abstractmethod
from os import path

from pyfe4ai.utils.crypto_constants import CryptoCONST
from pyfe4ai.utils.exceptions import FEKeyError

logger = logging.getLogger(__name__)


class ParameterCacheMixin:
    """Mixin that provides JSON-based parameter caching with validation.

    Subclasses must set ``_scheme_type`` (a CryptoCONST string) and implement
    :meth:`_param_verification`, :meth:`_apply_parameters`, and
    :meth:`_generate_and_save`.
    """

    config_folder: str
    _scheme_type: str

    def _load_parameters(self) -> None:
        """Load cached parameters from disk, or generate and save fresh ones.

        Looks for ``param.json`` in the scheme-specific config folder.
        If found and valid (per :meth:`_param_verification`), applies them
        via :meth:`_apply_parameters`; otherwise calls :meth:`_generate_and_save`.
        """
        folder = os.path.join(self.config_folder, self._scheme_type)
        os.makedirs(folder, exist_ok=True)
        param_file = os.path.join(folder, "param.json")
        if os.path.exists(param_file):
            logger.debug("load param file: %s", param_file)
            with open(param_file, "r") as f:
                param = json.load(f)
            if self._param_verification(param):
                self._apply_parameters(param)
                logger.debug("load param file successfully")
                return
            logger.error("param file is invalid, regenerate...")
        self._generate_and_save(param_file)

    @abstractmethod
    def _param_verification(self, param: dict) -> bool:
        """Return True if *param* matches the current configuration."""
        raise NotImplementedError

    @abstractmethod
    def _apply_parameters(self, param: dict) -> None:
        """Assign cryptographic attributes from the loaded *param* dict."""
        raise NotImplementedError

    @abstractmethod
    def _generate_and_save(self, param_file: str) -> None:
        """Generate fresh parameters, assign them to *self*, and persist to *param_file*."""
        raise NotImplementedError


class IPFEAbsKeyGenerator(ABC):
    """Abstract base for all inner-product FE key generators.

    Subclasses implement :meth:`setup` (generate keys), :meth:`get_public_parameters`,
    :meth:`get_private_keys`, and :meth:`get_decryption_keys`.

    Args:
        config: Scheme configuration dict.  Expected keys vary by scheme;
            common keys include ``sec_param``, ``eta``, ``n``, ``s``, ``lst_nid``.
    """

    def __init__(self, config: dict, **kwargs) -> None:
        """Perform the __init__ operation.

            Args:
                config: Scheme configuration dict.
        """
        super().__init__()
        self.sec_param = config.get("sec_param", CryptoCONST.SEC_PARAM)
        self.n = config.get("n", None)
        self.s = config.get("s", None)
        self.config_folder = path.join("config", "authority")
        if self.n is not None:
            self.lst_nid = config.get(
                "lst_nid", ["nid_{}".format(i) for i in range(self.n)]
            )

    @abstractmethod
    def setup(self) -> None:
        """Generate master public and secret keys for the scheme."""
        raise NotImplementedError()

    @abstractmethod
    def get_public_parameters(self, **kwargs) -> None | dict:
        """Return serialisable public parameters.

        Returns:
            A dict of public parameters, or ``None`` on error.
        """
        raise NotImplementedError()

    @abstractmethod
    def get_private_keys(self, nid: str, **kwargs) -> None | dict:
        """Return the private (encryption) keys for client *nid*.

        Args:
            nid: A client/node identifier.

        Returns:
            A dict of private keys, or ``None`` if *nid* is invalid.
        """
        raise NotImplementedError()

    @abstractmethod
    def get_decryption_keys(self, sid: str, **kwargs) -> None | dict:
        """Derive a functional decryption key for session *sid*.

        Args:
            sid: A session identifier.
            **kwargs: Must include ``credentials`` with a ``fusion_weight`` entry.

        Returns:
            A dict containing the decryption key material.

        Raises:
            FEKeyError: If credentials are missing.
            FEValidationError: If fusion weights are invalid.
        """
        raise NotImplementedError()


class IPFEAbsCrypto(ABC):
    """Abstract base for all inner-product FE crypto operations.

    Subclasses implement :meth:`encrypt_lst_ndarray`, :meth:`compute_lst_ndarray_ct`,
    and :meth:`decrypt_lst_ndarray_ct`.

    Args:
        config: Runtime configuration dict containing at least ``keys``
            (a dict with ``pp``, ``sk``, and/or ``dk`` sub-dicts) and
            optionally ``id``, ``type``, and ``precision``.
    """

    def __init__(self, config: dict, **kwargs) -> None:
        """Perform the __init__ operation.

            Args:
                config: Scheme configuration dict.
        """
        super().__init__()
        self.keys = config.get("keys", None)
        if not self.keys:
            raise FEKeyError("no keys provided to initialize crypto system")
        self.id = config.get("id", None)
        self.type = config.get("type", None)
        self.precision = config.get("precision", 0)
        self.config_folder = path.join("config", "crypto")

    @abstractmethod
    def encrypt_lst_ndarray(self, lst_ndarray: list, **kwargs) -> list | None:
        """Encrypt a list of ndarray-like plaintexts.

        Args:
            lst_ndarray: List of plaintext arrays to encrypt.

        Returns:
            List of ciphertext dicts, or ``None`` on error.
        """
        raise NotImplementedError()

    @abstractmethod
    def compute_lst_ndarray_ct(self, dict_ndarray_ct: dict, **kwargs) -> list | None:
        """Aggregate ciphertexts from multiple clients.

        Args:
            dict_ndarray_ct: Mapping of client IDs to their ciphertext lists.

        Returns:
            Aggregated ciphertext list, or ``None`` on error.
        """
        raise NotImplementedError()

    @abstractmethod
    def decrypt_lst_ndarray_ct(self, dict_ndarray_ct: dict, **kwargs) -> list | None:
        """Decrypt aggregated ndarray ciphertexts.

        Args:
            dict_ndarray_ct: Mapping of client IDs to their ciphertext lists.

        Returns:
            List of decrypted inner-product results, or ``None`` on error.
        """
        raise NotImplementedError()

    def _has_public_parameters(self) -> bool:
        """Check whether public parameters are available."""
        return "pp" in self.keys

    def _has_private_keys(self) -> bool:
        """Check whether private (encryption) keys are available."""
        return "sk" in self.keys

    def _has_decryption_keys(self) -> bool:
        """Check whether decryption keys are available."""
        return "dk" in self.keys
