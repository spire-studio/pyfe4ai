"""Abstract bases of the circuit FE schemes (the bounded-key family ``bcfe``).

The inner-product bases of :mod:`pyfe4ai.core.base` describe keys for a
weight vector and ciphertexts of vectors (``get_decryption_keys``,
``encrypt_lst_ndarray``). A circuit FE scheme computes a public boolean
circuit ``C(w, x)`` (:mod:`pyfe4ai.circuits`): a functional key fixes the
secret parameter ``w`` and decrypts ``C(w, x)`` from a ciphertext of ``x``.
Its two bases are here, beside the inner-product ones, which stay as they
are:

* :class:`CircuitFEAbsKeyGenerator` -- the authority: ``setup()``,
  ``get_public_parameters()`` and the issuance of keys (each scheme names
  its own issuance calls). Its metaclass is that of the inner-product key
  generators (``_ProfiledKeyGeneratorMeta``), so a config goes through
  :func:`pyfe4ai.core.config.resolve` exactly as theirs: converted to the
  scheme's config class, its option names checked, its security profile
  applied before ``__init__`` runs;
* :class:`CircuitFEAbsCrypto` -- the encryptor and the decryptor, built from
  ``{"keys": {"pp": public parameters}}``: ``encrypt(x)`` and
  ``decrypt(ct, key)``.

There is no ``get_private_keys`` (encryption needs the public parameters
only) and no ``get_decryption_keys``: the schemes issue keys with their own
calls (``issue_key`` and friends), which take the requester's identity.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Mapping
from os import path

from pyfe4ai.core import profiles
from pyfe4ai.core.base import _ProfiledKeyGeneratorMeta, cache_dir
from pyfe4ai.core.constants import CryptoCONST
from pyfe4ai.core.exceptions import FEKeyError
from pyfe4ai.core.status import warn_if_known_insecure


class CircuitFEAbsKeyGenerator(metaclass=_ProfiledKeyGeneratorMeta):
    """Abstract base of the key generators (authorities) of the circuit FE schemes.

    Args:
        config: The scheme's configuration (a mapping, converted to the
            scheme's config class of :mod:`pyfe4ai.core.config` and completed
            by its ``security_profile`` before ``__init__`` runs). Common
            entries: ``security_profile``, ``sec_param`` (recorded only),
            ``cache_dir`` (the directory of the authority's state, see
            :func:`pyfe4ai.core.base.cache_dir`).
    """

    def __init__(self, config: dict, **kwargs) -> None:
        super().__init__()
        warn_if_known_insecure(type(self).__module__)
        self.security_profile = config.get("security_profile", profiles.TOY)
        self.sec_param = config.get("sec_param", CryptoCONST.SEC_PARAM)
        #: The ``authority`` folder of the cache directory, where the schemes keep their state.
        self.config_folder = path.join(cache_dir(config.get("cache_dir")), "authority")

    @abstractmethod
    def setup(self) -> None:
        """Create a new instance of the scheme (its public parameters and the
        authority's state)."""
        raise NotImplementedError()

    @abstractmethod
    def get_public_parameters(self) -> dict:
        """The public parameters as they are now (a JSON-able dict)."""
        raise NotImplementedError()


class CircuitFEAbsCrypto(ABC):
    """Abstract base of the crypto objects (encryptors, decryptors) of the circuit FE schemes.

    Args:
        config: ``{"keys": {"pp": public parameters}}``, plus the crypto
            options of :class:`pyfe4ai.core.config.CryptoConfig`.

    Raises:
        FEKeyError: Without public parameters.
    """

    def __init__(self, config: dict, **kwargs) -> None:
        super().__init__()
        warn_if_known_insecure(type(self).__module__)
        keys = config.get("keys", None) if isinstance(config, Mapping) else None
        if not isinstance(keys, Mapping) or not keys or "pp" not in keys:
            raise FEKeyError("no public parameters provided: give {'keys': {'pp': public parameters}}")
        self.keys = keys
        self.id = config.get("id", None)
        self.config_folder = path.join(cache_dir(config.get("cache_dir")), "crypto")

    @abstractmethod
    def encrypt(self, x, **kwargs) -> dict:
        """A ciphertext of the data values *x*."""
        raise NotImplementedError()

    @abstractmethod
    def decrypt(self, ct: dict, key: dict, **kwargs) -> dict:
        """``C(w, x)``, decoded, from a ciphertext of ``x`` and a key for ``w``."""
        raise NotImplementedError()
