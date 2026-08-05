# Parameter & Naming Glossary

> Quick reference for the parameter names and conventions used across
> PyFE4AI scheme implementations.

---

## Scheme Taxonomy

| Prefix | Full Name | Clients | Description |
|--------|-----------|---------|-------------|
| **SIFE** | Single-Input FE | 1 encryptor, 1 decryptor | Basic inner-product FE |
| **MIFE** | Multi-Input FE | *n* encryptors, 1 decryptor | Each client encrypts independently |
| **MCFE** | Multi-Client FE | *n* encryptors, 1 decryptor (label-aware) | Ciphertexts tied to a common label |
| **dMCFE** | Decentralized MCFE | *n* encryptors, no central authority | Key generation distributed |
| **TMIFE / TMCFE** | Threshold MIFE / MCFE | *n* encryptors, *t*-of-*n* decryption | Threshold aggregation |

---

## Core Parameters

| Name | Type | Meaning |
|------|------|---------|
| `sec_param` | `int` | Security parameter (bit length), default 128 |
| `eta` | `int` or `dict[str, int]` | Dimension of each client's input vector (slot count) |
| `n` | `int` | Number of clients / input sources |
| `s` | `int` | Number of sessions or functional-key slots |
| `bound` | `int` | Plaintext bound – inputs must satisfy \|x_i\| ≤ bound |
| `modulus_length` | `int` | Bit length of the group modulus (Damgard schemes) |
| `lst_nid` | `list[str]` | Ordered list of client node IDs, length *n* |
| `precision` | `int` | Decimal precision for float→int quantisation (×10^precision) |

---

## Key Artifacts

| Name | Role | Typical Contents |
|------|------|-----------------|
| `mpk` / `pp` | Master public key / public parameters | Group elements, generators |
| `msk` | Master secret key | Secret exponents / vectors |
| `sk` | Per-client private (encryption) key | Derived from msk |
| `dk` | Functional decryption key | Derived from msk + fusion weights |
| `fusion_weight` | Functional key weights | Integer vector *y* s.t. decrypt recovers ⟨x, y⟩ |

---

## Cryptographic Group Variables

### Standard DDH (p, q, r, g)

| Variable | Description |
|----------|-------------|
| `p` | Safe-prime modulus |
| `q` | Prime order of the subgroup |
| `r` | Cofactor such that p = q·r + 1 |
| `g` | Generator of the order-q subgroup mod p |

### Damgard DDH (p, q, g, h)

| Variable | Description |
|----------|-------------|
| `p` | Prime modulus |
| `q` | Prime order of the subgroup |
| `g` | First generator |
| `h` | Second generator (random power of g) |

### Paillier

| Variable | Description |
|----------|-------------|
| `p`, `q` | Secret primes |
| `n_sq` | n² where n = p·q |
| `lam` | λ(n) = lcm(p−1, q−1) |

### LWE / Ring-LWE

| Variable | Description |
|----------|-------------|
| `q` | Ciphertext modulus |
| `p` | Plaintext modulus |
| `m` | Number of LWE samples |
| `sigma` | Gaussian noise parameter |
| `poly_degree` | Ring polynomial degree (Ring-LWE) |

### Pairing-Based

| Variable | Description |
|----------|-------------|
| `group` | `PairingGroup` instance (e.g., `'MNT224'`) |
| `order` | Group order |

---

## API Method Naming

| Method | Class | Purpose |
|--------|-------|---------|
| `setup()` | KeyGenerator | Generate mpk / msk |
| `get_public_parameters()` | KeyGenerator | Serialise mpk/pp for distribution |
| `get_private_keys(nid)` | KeyGenerator | Serialise per-client encryption key |
| `get_decryption_keys(sid, credentials)` | KeyGenerator | Derive functional decryption key |
| `encrypt(plaintext, ...)` | Crypto | Encrypt a plaintext vector |
| `decrypt(ciphertext, dk, ...)` | Crypto | Decrypt using a functional key |
| `encrypt_lst_ndarray(lst_ndarray, ...)` | Crypto | Batch encrypt numpy arrays |
| `compute_lst_ndarray_ct(...)` | Crypto | Aggregate ciphertexts |
| `decrypt_lst_ndarray_ct(...)` | Crypto | Batch decrypt aggregated ciphertexts |
