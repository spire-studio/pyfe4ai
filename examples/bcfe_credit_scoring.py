"""A credit decision with bounded-key FE (``bcfe/gc_ristretto255``, experimental).

A bank's scoring service holds a logistic-regression model: weights ``w`` and
a bias. An applicant (the data owner) encrypts its features ``x``. With the
key for its model, the scoring service decrypts the decision
``[w . x + bias >= 0]`` (the sign of the logit, the decision of the logistic
regression at probability 1/2) and learns nothing else about ``x``. A key
authority issues the key: it checks the requester's attributes against its
issuance policy and issues at most ``max_keys`` keys in all, one slot each.

Read before use (the documentation, ``docs/security.rst``, "Bounded-key FE"):

* the key holder learns its own ``w``: the key carries it. The scheme hides
  ``x`` from the key holder and ``w`` from the data owners, not ``w`` from
  the key holder or from the authority;
* a key decrypts only ciphertexts made after its slot existed: issue the
  keys first, then hand the public parameters to the data owners;
* the issuance policy is the authority's decision, not a cryptographic
  access control, and the library cannot authenticate a requester id: pass
  the id that your application has authenticated;
* the encryptor is assumed honest: do not send the decision, or whether the
  decryption worked, back to it;
* classical security only: a ciphertext recorded today gives ``x`` up to a
  future quantum computer.

Features and weights are integers: scale real values to fixed point first
(here 8-bit signed numbers; the bias has 16 bits).
"""

import tempfile

import pyfe4ai
from pyfe4ai import circuits
from pyfe4ai.policy import parse

#: The features of the model, in order.
FEATURES = ("income band", "debt ratio", "years employed", "late payments")
SCORING = {"org:bank", "role:scoring"}


def main() -> None:
    # the public circuit C(w, x) = [w . x + bias >= 0] for 4 features of 8 bits
    model = circuits.linear_threshold(n=len(FEATURES), bw=8, bx=8)

    with tempfile.TemporaryDirectory() as state:
        # --- the key authority. Its state (the slot list and the private records of the slots) lives in
        # <cache_dir>/authority/bcfe/<setup digest>; a real authority keeps that directory, here it is temporary.
        kg, BCFE = pyfe4ai.create(
            "bcfe/gc_ristretto255",
            circuit=model,
            max_keys=4,
            issuance_policy=parse("and(org:bank, role:scoring)"),
            cache_dir=state,
        )
        kg.setup()

        # --- issuance (h1, the default): the authority makes the key of w for an authenticated requester.
        # The requester id is the one your application authenticated (a login, a client certificate, ...);
        # the same requester asking again for the same w gets the same key, another requester a slot of its own.
        scoring_service = "svc-scoring-01"
        w = {"weights": [12, -20, 6, -30], "bias": 40}
        key = kg.issue_key(w, requester=scoring_service, attributes=SCORING, label="credit-v3")

        # attributes that do not satisfy the issuance policy: refused before any slot is taken
        try:
            kg.issue_key(w, requester="svc-marketing-07", attributes={"org:bank", "role:marketing"})
        except pyfe4ai.FEKeyError:
            pass
        else:
            raise SystemExit("the issuance policy was not applied")

        # the public parameters now list the key's slot (not w): hand them to the data owners
        pp = kg.get_public_parameters()

        # --- the data owners encrypt their features to the slots they accept (here the slot "credit-v3")
        applicants = {
            "applicant A": [9, 2, 7, 0],
            "applicant B": [3, 8, 1, 4],
        }
        cts = {
            name: BCFE({"keys": {"pp": pp}}).encrypt({"x": x}, slots=["credit-v3"])
            for name, x in applicants.items()
        }

        # --- the scoring service decrypts the decision, and nothing else about x. It knows w, its own model:
        # the key carries it (no function privacy).
        scorer = BCFE({"keys": {"pp": pp}})
        for name, ct in cts.items():
            decision = scorer.decrypt(ct, key)["decision"]
            print("{}: {}".format(name, "approve" if decision else "decline"))

        # --- a new key does not open the ciphertexts made before its slot existed
        key_v4 = kg.issue_key(
            {"weights": [10, -18, 8, -25], "bias": 30},
            requester=scoring_service,
            attributes=SCORING,
            label="credit-v4",
        )
        scorer = BCFE({"keys": {"pp": kg.get_public_parameters()}})
        try:
            scorer.decrypt(cts["applicant A"], key_v4)
        except pyfe4ai.FEKeyError:
            pass  # "the ciphertext is older than the key": the applicant encrypts again under the new pp
        else:
            raise SystemExit("a key opened a ciphertext made before its slot existed")


if __name__ == "__main__":
    main()
