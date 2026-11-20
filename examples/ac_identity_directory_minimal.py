"""Identity directory: identity-based access control on lattices (experimental).

A key centre runs one instance of ``sife/fullysec_ring_lwe`` per identity and
publishes a directory of the identities' encryption keys. An encryptor checks
the directory against the digest it obtained over a trusted channel and
encrypts for an identity; only functional keys of that identity's instance
decrypt the ciphertext.

The key centre holds every identity's master key (key escrow) and
authenticates nobody: whoever calls ``get_decryption_keys`` decides who the
identity is. Ciphertexts name their identity in the clear. Read *Identity
directory* in ``docs/security.rst`` before use.
"""

import pyfe4ai
from pyfe4ai.ac import IdentityDirectory, IdentityKeyCentre


def main() -> None:
    x = [3, -1, 2]
    y = [1, 2, 4]

    # --- the key centre: one instance per identity, and the directory to publish ---
    centre = IdentityKeyCentre({"eta": len(x), "bound_x": 10, "bound_y": 10})
    for identity in ("alice", "bob"):
        centre.add_identity(identity)
    record = centre.directory()
    digest = record["digest"]  # the encryptor obtains this over a trusted channel
    print("identities:", centre.identities())

    # --- an encryptor: the directory, checked against the digest ---
    directory = IdentityDirectory(record, expected_digest=digest)
    ct = directory.encrypt("alice", x)
    print("x =", x, " y =", y, " plain <x, y> =", sum(a * b for a, b in zip(x, y)))
    print("ciphertext for:", ct["identity"])

    # --- decryption: keys of alice's instance only ---
    dk_alice = centre.get_decryption_keys("alice", y)
    print("key of alice: <x, y> =", directory.decrypt(ct, dk_alice, y))
    dk_bob = centre.get_decryption_keys("bob", y)
    try:
        directory.decrypt(ct, dk_bob, y)
    except pyfe4ai.FEKeyError:
        print("key of bob: refused (issued to another identity)")

    # --- adding an identity changes the directory's digest, not earlier ciphertexts and keys ---
    centre.add_identity("carol")
    record = centre.directory()
    try:
        IdentityDirectory(record, expected_digest=digest)
    except pyfe4ai.FEValidationError:
        print("directory with carol, checked against the old digest: refused")
    directory = IdentityDirectory(record, expected_digest=record["digest"])
    print("identities:", directory.identities())
    print("earlier ciphertext, key of alice: <x, y> =", directory.decrypt(ct, dk_alice, y))


if __name__ == "__main__":
    main()
