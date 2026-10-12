"""
Full-hiding multi-input inner-product FE over pairings (bounded slots)
| "Full-Hiding (Unbounded) Multi-Input Inner Product Functional Encryption
|  from the k-Linear Assumption"
| By Pratish Datta, Tatsuaki Okamoto, Junichi Tomida, PKC 2018 (ePrint
| 2018/061) -- Section 3.1, the bounded scheme from k-LIN with dual pairing
| vector spaces; structure as in GoFE's ``innerprod/fullysec/fh_multi_ipe.go``.

* type:     private-key multi-input encryption with function-hiding keys
* setting:  asymmetric pairing group (default MNT224), ``k = sec_level``

Per slot dual orthogonal bases ``B_i, B*_i`` of dimension ``2 eta + 2k + 1``
(``B_i B*_i^T = mu I``), ``g_T = e(g1, g2)^mu`` public.

* Enc:    ``c_i = [(x_i, 0^eta, 1, 0^(k-1), phi, 0) B_i]_1`` (``phi <- Z_p^k``).
* KeyGen: ``k_i = [(y_i, 0^eta, r_i, gamma_i, 0^k, 0) B*_i]_2`` with fresh
  ``sum_i r_i = 0`` and ``gamma_i <- Z_p^(k-1)``.
* Dec:    ``prod_i e(c_i, k_i) = g_T^{sum <x_i, y_i>}`` and one discrete log.

Each slot's pairing is masked by ``r_i``; slot keys of different functional
keys do not combine; keys hold group elements only and decryption does not
need ``y``. Deviations: no SKE layer (Remark B.1), so hiding assumes every
slot contributes a ciphertext; slot ``i``'s client holds ``B^_i`` in ``Z_p``
and could decode its own slot of a key (outside the model). ``n >= 2`` is
required (the proof needs the zero sharing ``r_i`` over at least two slots;
with ``n = 1`` and ``k = 1`` keys were deterministic). As in every MIFE there
are no labels.
"""

from __future__ import annotations

import json
import logging

import gmpy2 as gp
import numpy as np

from pyfe4ai.schemes.ipfe import IPFEAbsCrypto
from pyfe4ai.schemes.ipfe import decoded_layer
from pyfe4ai.schemes.ipfe import IPFEAbsKeyGenerator
from pyfe4ai.schemes.ipfe import ParameterCacheMixin
from pyfe4ai.utils.pairing_utils import bounded_discrete_log_gt
from pyfe4ai.utils.pairing_utils import matrix_inverse_mod
from pyfe4ai.utils.pairing_utils import random_nonzero_zr
from pyfe4ai.utils.pairing_utils import to_zr
from pyfe4ai.utils.pairing_utils import transpose_mod
from pyfe4ai.utils.pairing_utils import vector_add_mod
from pyfe4ai.utils.pairing_utils import vector_scalar_mod
from pyfe4ai.utils.pairing_utils import serialize_matrix_g
from pyfe4ai.utils.crypto_constants import CryptoCONST
from pyfe4ai.utils.pairing_backend import G1, G2, GT, ZR, PairingGroup, pair
from pyfe4ai.utils.pairing_backend import DEFAULT_PAIRING_GROUP, check_asymmetric
from pyfe4ai.utils.pairing_backend import require_pairing_backend
from pyfe4ai.utils.exceptions import FEKeyError, FESchemeError, FEValidationError
from pyfe4ai.utils.validation import int_vector
from pyfe4ai.utils.validation import scale_to_int

logger = logging.getLogger(__name__)


def _random_ob_matrix(
    group: PairingGroup, dim: int, mu: gp.mpz, order: gp.mpz
) -> tuple[list[list[gp.mpz]], list[list[gp.mpz]]]:
    """Perform the _random_ob_matrix operation.

        Args:
            group: Pairing group instance.
            dim: Vector dimension.
            mu: Scalar of the dual bases (``B B*^T = mu I``).
            order: Group order (or exponent order for dequantization).
    """
    while True:
        b = [
            [gp.mpz(int(group.random(ZR))) for _ in range(dim)]
            for _ in range(dim)
        ]
        try:
            b_inv, _ = matrix_inverse_mod(b, order)
            break
        except ValueError:
            continue
    b_star = transpose_mod(b_inv)
    b_star = [[(mu * entry) % order for entry in row] for row in b_star]
    return b, b_star


class MIFEFHMultiIPEKeyGenerator(IPFEAbsKeyGenerator, ParameterCacheMixin):
    """Key generator for function-hiding multi-slot multi-input IPE over pairings."""

    _scheme_type = CryptoCONST.TYPE_MIFE_FH_MULTI_IPE
    def __init__(self, config: dict, **kwargs) -> None:
        """Initialise the MIFE FH-Multi-IPE key generator.

        Args:
            config: Scheme configuration. Recognised keys: ``n`` (number of
                clients, at least 2), ``lst_nid``, ``eta``, ``sec_level``
                (``k`` of k-LIN, default 2), ``bound_x``, ``bound_y`` and
                ``pairing_group_param`` (asymmetric group, default
                ``"MNT224"``).
        """
        super().__init__(config, **kwargs)
        require_pairing_backend()
        self.sec_level = int(config.get("sec_level", 2))
        self.eta = int(config.get("eta", CryptoCONST.MIFE_ETA))
        self.bound_x = gp.mpz(config.get("bound_x", 16))
        self.bound_y = gp.mpz(config.get("bound_y", 16))
        self.pairing_group_param = config.get("pairing_group_param", DEFAULT_PAIRING_GROUP)
        check_asymmetric(self.pairing_group_param)
        if not self.n or self.n < 2:
            # the zero sharing r_i needs two slots; with n = 1 (and k = 1) keys
            # were deterministic and not function hiding
            raise FEKeyError("DOT18 needs at least two slots (n >= 2)")
        if len(self.lst_nid) != self.n:
            raise FEValidationError("lst_nid must list n = {} clients".format(self.n))
        if self.sec_level < 1:
            raise FEValidationError("sec_level (k of k-LIN) must be >= 1")
        if self.eta < 1 or self.bound_x < 0 or self.bound_y < 0:
            raise FEValidationError("need eta >= 1 and non-negative bounds")
        self._load_parameters()

    def _apply_parameters(self, param: dict) -> None:
        self.group = PairingGroup(self.pairing_group_param)
        self.order = gp.mpz(int(self.group.order()))
        self.dim = 2 * self.eta + 2 * self.sec_level + 1

    def _param_verification(self, param: dict) -> bool:
        return (
            param.get("sec_param") == self.sec_param
            and param.get("sec_level") == self.sec_level
            and param.get("eta") == self.eta
            and param.get("n") == self.n
            and param.get("bound_x") == gp.digits(self.bound_x)
            and param.get("bound_y") == gp.digits(self.bound_y)
            and param.get("pairing_group_param") == self.pairing_group_param
        )

    def _generate_and_save(self, param_file: str) -> None:
        with open(param_file, "w", encoding="utf-8") as f:
            json.dump(
                {
                    "sec_param": self.sec_param,
                    "sec_level": self.sec_level,
                    "eta": self.eta,
                    "n": self.n,
                    "bound_x": gp.digits(self.bound_x),
                    "bound_y": gp.digits(self.bound_y),
                    "pairing_group_param": self.pairing_group_param,
                },
                f,
            )
        self.group = PairingGroup(self.pairing_group_param)
        self.order = gp.mpz(int(self.group.order()))
        self.dim = 2 * self.eta + 2 * self.sec_level + 1

    def setup(self) -> None:
        mu = gp.mpz(int(random_nonzero_zr(self.group)))
        g1 = self.group.random(G1)
        g2 = self.group.random(G2)
        pub = pair(g1, g2) ** to_zr(self.group, mu)

        bhat = {}
        bstar_hat = {}
        for nid in self.lst_nid:
            b, b_star = _random_ob_matrix(self.group, self.dim, mu, self.order)
            b_hat_rows = []
            b_star_hat_rows = []
            for j in range(self.eta + self.sec_level + 1):
                if j < self.eta:
                    b_hat_rows.append(b[j])
                    b_star_hat_rows.append(b_star[j])
                elif j == self.eta:
                    b_hat_rows.append(b[j + self.eta])
                    b_star_hat_rows.append(b_star[j + self.eta])
                elif j < self.eta + self.sec_level:
                    b_hat_rows.append(b[j - 1 + self.eta + self.sec_level])
                    b_star_hat_rows.append(b_star[j + self.eta])
                else:
                    b_hat_rows.append(b[j - 1 + self.eta + self.sec_level])

            bhat[nid] = [[gp.digits(v) for v in row] for row in b_hat_rows]
            bstar_hat[nid] = [[gp.digits(v) for v in row] for row in b_star_hat_rows]

        self.msk = {
            "bhat": bhat,
            "bstar_hat": bstar_hat,
        }
        self.pp = {
            "sec_level": self.sec_level,
            "eta": self.eta,
            "n": self.n,
            "lst_nid": list(self.lst_nid),
            "bound_x": gp.digits(self.bound_x),
            "bound_y": gp.digits(self.bound_y),
            "pairing_group_param": self.pairing_group_param,
            "g1": self.group.serialize(g1).decode("utf-8"),
            "g2": self.group.serialize(g2).decode("utf-8"),
            "pub": self.group.serialize(pub).decode("utf-8"),
        }
        logger.info("MIFE FHMultiIPE setup successfully")

    def get_public_parameters(self) -> dict:
        return self.pp

    def get_private_keys(self, nid: str, **kwargs) -> dict | None:
        """Return client *nid*'s encryption key (its basis ``bhat``).

        Args:
            nid: Client identifier.

        Returns:
            The key dict, or ``None`` for an unknown client.
        """
        if nid not in self.msk["bhat"]:
            return None
        return {
            "bhat": self.msk["bhat"][nid],
        }

    def get_decryption_keys(self, sid: str, **kwargs) -> dict | None:
        """Derive a functional decryption key for the given fusion weights.

        Args:
            sid: Session identifier.
            **kwargs: Must include ``credentials`` with ``fusion_weight`` (dict
                ``{nid: list of int}`` covering every client).
        """
        credentials = kwargs.get("credentials")
        if not credentials:
            raise FEKeyError("need credentials for FHMultiIPE decryption key generation")
        fusion_weight = credentials.get("fusion_weight")
        if not isinstance(fusion_weight, dict) or set(fusion_weight) != set(self.lst_nid):
            raise FEValidationError("fusion weights must cover every client")

        gamma = [
            [gp.mpz(int(self.group.random(ZR))) for _ in range(self.n)]
            for _ in range(self.sec_level)
        ]
        # r_i = gamma[0][i] with sum_i r_i = 0 (n >= 2 is enforced)
        gamma[0][self.n - 1] = (-sum(gamma[0][: self.n - 1])) % self.order

        dk = {}
        for client_index, nid in enumerate(self.lst_nid):
            y = int_vector(
                fusion_weight[nid], "fusion weight of " + nid, length=self.eta, bound=int(self.bound_y)
            )

            bstar_hat = [
                [gp.mpz(v) for v in row]
                for row in self.msk["bstar_hat"][nid]
            ]
            key_vec = [gp.mpz(0) for _ in range(self.dim)]
            for row_index in range(self.eta + self.sec_level):
                if row_index < self.eta:
                    scalar = gp.mpz(y[row_index])
                else:
                    scalar = gamma[row_index - self.eta][client_index]
                contribution = vector_scalar_mod(
                    bstar_hat[row_index], scalar, self.order
                )
                key_vec = vector_add_mod(key_vec, contribution, self.order)
            g2 = self.group.deserialize(self.pp["g2"].encode("utf-8"))
            g2_row = [g2 ** to_zr(self.group, value) for value in key_vec]
            dk[nid] = serialize_matrix_g(self.group, [g2_row])[0]

        return {"dk": dk}


class MIFEFHMultiIPE(IPFEAbsCrypto):
    """Crypto operations for function-hiding multi-slot multi-input IPE over pairings."""

    def __init__(self, config: dict, **kwargs) -> None:
        """Initialise the MIFE FH-Multi-IPE crypto object.

        Args:
            config: Must contain ``keys`` with ``pp`` (public parameters) and
                optionally ``sk`` (encryption keys).
        """
        super().__init__(config, **kwargs)
        require_pairing_backend()
        self.pp = self.keys["pp"]
        self.group = PairingGroup(self.pp["pairing_group_param"])
        check_asymmetric(self.group)
        self.dim = 2 * self.pp["eta"] + 2 * self.pp["sec_level"] + 1
        if self._has_private_keys():
            self.sk = self.keys["sk"]

    def encrypt(self, lst_pt: list) -> dict:
        if not self._has_private_keys():
            raise FEKeyError("no encryption key provided")
        x = [
            gp.mpz(v)
            for v in int_vector(lst_pt, "plaintext", length=self.pp["eta"], bound=int(self.pp["bound_x"]))
        ]
        bhat = [[gp.mpz(v) for v in row] for row in self.sk["bhat"]]
        phi = [gp.mpz(int(self.group.random(ZR))) for _ in range(self.pp["sec_level"])]

        order = gp.mpz(int(self.group.order()))
        key_vec = [gp.mpz(0) for _ in range(self.dim)]
        for row_index in range(self.pp["eta"] + self.pp["sec_level"] + 1):
            if row_index < self.pp["eta"]:
                scalar = x[row_index]
            elif row_index == self.pp["eta"]:
                scalar = gp.mpz(1)
            else:
                scalar = phi[row_index - self.pp["eta"] - 1]
            contribution = vector_scalar_mod(bhat[row_index], scalar, order)
            key_vec = vector_add_mod(key_vec, contribution, order)

        g1 = self.group.deserialize(self.pp["g1"].encode("utf-8"))
        ct = [g1 ** to_zr(self.group, value) for value in key_vec]
        return {"ct": [self.group.serialize(value).decode("utf-8") for value in ct]}

    def decrypt(self, dct_ct: dict, dk: dict, fusion_weight: dict | None = None):
        """``log_{g_T} prod_i e(c_i, k_i)``; ``None`` if out of bound.

        *fusion_weight* is accepted for API compatibility and ignored (the key
        encodes ``y``).

        Raises:
            FESchemeError: Unless there is one ciphertext and one key part per slot.
        """
        if not dk or "dk" not in dk:
            raise FEKeyError("no decryption key provided")
        lst_nid = self.pp["lst_nid"]
        if not set(dct_ct) == set(dk["dk"]) == set(lst_nid):
            raise FESchemeError("need one ciphertext and one key part for every slot")
        pub = self.group.deserialize(self.pp["pub"].encode("utf-8"))
        acc = self.group.init(GT, 1)
        for nid in lst_nid:
            ct = [self.group.deserialize(v.encode("utf-8")) for v in dct_ct[nid]["ct"]]
            key = [self.group.deserialize(v.encode("utf-8")) for v in dk["dk"][nid]]
            if len(ct) != self.dim or len(key) != self.dim:
                raise FESchemeError("ciphertext or key of {} has the wrong length".format(nid))
            for lhs, rhs in zip(ct, key):
                acc *= pair(lhs, rhs)

        bound = int(
            self.pp["n"]
            * self.pp["eta"]
            * gp.mpz(self.pp["bound_x"])
            * gp.mpz(self.pp["bound_y"])
        )
        try:
            return bounded_discrete_log_gt(self.group, pub, acc, bound)
        except ValueError:
            logger.error("inner product out of bound (wrong key or ciphertexts?)")
            return None

    def encrypt_lst_ndarray(self, lst_ndarray: list, **kwargs) -> list | None:
        """Encrypt ndarrays element-wise (``eta = 1``).

        Every element is a separate ciphertext of the slot, and MIFE has no
        labels: the holder of one key can combine elements of different
        positions. Use for one input per key epoch, not for aggregation.
        """
        if self.pp["eta"] != 1:
            raise NotImplementedError("ndarray helper currently expects eta=1")
        lst_ndarray_ct = []
        for ary_src in lst_ndarray:
            ary = scale_to_int(ary_src, self.precision)
            ary_ct = np.empty(ary.shape, dtype=object)
            for i, w in np.ndenumerate(ary):
                ary_ct[i] = self.encrypt([int(w)])
            lst_ndarray_ct.append(ary_ct)
        return lst_ndarray_ct

    def compute_lst_ndarray_ct(self, dict_ndarray_ct: dict, **kwargs) -> list | None:
        """Compute inner products on encrypted ndarray ciphertexts.

            Args:
                dict_ndarray_ct: Encrypted ndarray ciphertexts (list or dict).
        """
        dk = kwargs.get("dk", None)
        if dk is None:
            raise FEKeyError("need to provide decryption key")
        return self.decrypt_lst_ndarray_ct(dict_ndarray_ct, dk=dk)

    def decrypt_lst_ndarray_ct(self, dict_ndarray_ct: dict, **kwargs) -> list | None:
        """Decrypt encrypted ndarray ciphertexts element-wise.

            Args:
                dict_ndarray_ct: Encrypted ndarray ciphertexts (list or dict).
        """
        dk = kwargs.get("dk", None)
        if self.pp["eta"] != 1:
            raise NotImplementedError("ndarray helper currently expects eta=1")
        if dk is None:
            raise FEKeyError("need to provide decryption key")

        sample = next(iter(dict_ndarray_ct.values()))
        lst_ndarray = []
        for l in range(len(sample)):
            ary = sample[l]
            ary_dec = np.empty(ary.shape, dtype=object)
            for i, _ in np.ndenumerate(ary):
                dct_ct = {nid: dict_ndarray_ct[nid][l][i] for nid in dict_ndarray_ct}
                value = self.decrypt(dct_ct, dk)
                ary_dec[i] = value
            lst_ndarray.append(decoded_layer(ary_dec, l, 10 ** self.precision))
        return lst_ndarray
