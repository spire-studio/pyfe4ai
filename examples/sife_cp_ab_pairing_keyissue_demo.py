"""Threshold key issuance for ``sife/cp_ab_pairing`` (experimental).

A dealer runs the setup of the scheme, shares its master key among three key
servers and hands each server its share once; any two of them issue a
functional key together, and none of them holds the master key. Each server
answers a request only if its operator's approval rule accepts it; the
requester multiplies the two partial keys, checks the product against the
public parameters and keeps it as an ordinary key of the scheme.

Here one process plays every party, so the requester's identity is a plain
string. In a deployment each server runs on its own, and every request and
partial key travels over a mutually authenticated, confidential channel
(mTLS or an equivalent): ``authenticated_requester`` is the peer identity
that channel authenticated, never a value read from the request. Read
*Threshold key issuance* in ``docs/security.rst`` before use.
"""

import pyfe4ai
from pyfe4ai.schemes.sife.cp_ab_pairing import SIFECPABPairing
from pyfe4ai.schemes.sife.cp_ab_pairing_keyissue import (
    SIFECPABPairingDealer,
    SIFECPABPairingKeyCombiner,
    SIFECPABPairingKeyServer,
)

#: What each requester may obtain: the attributes it holds and the weight
#: vectors it may use. The rule must decide on the weights too: a rule that
#: checked the attributes only would let a user take keys of unit vectors and
#: read every coordinate of the plaintexts its attributes open.
GRANTS = {
    "alice": {"attributes": {"org:A", "role:auditor"}, "weights": [[1, 2, 4]]},
}


def approve(request, requester, history):
    """The operators' approval rule (the same at every server here)."""
    grant = GRANTS.get(requester)
    if grant is None:
        return False
    weights = [int(v) for v in request["y"]]
    return set(request["attributes"]) <= grant["attributes"] and weights in grant["weights"]


def main() -> None:
    x = [3, -1, 2]
    y = [1, 2, 4]
    config = {"eta": len(x), "bound_x": 10, "bound_y": 10,
              "universe": ["org:A", "org:B", "role:auditor"]}

    # --- the dealer: setup, then one share per server, delivered once ---
    dealer = SIFECPABPairingDealer(config, ["ks1", "ks2", "ks3"], threshold=2)
    dealer.setup()
    pp = dealer.get_public_parameters()
    key_servers = dealer.key_servers()  # public, like pp
    servers = {name: SIFECPABPairingKeyServer(pp, key_servers, dealer.server_share(name), approve)
               for name in key_servers["servers"]}
    print("key servers:", sorted(servers), " threshold:", key_servers["threshold"])

    # --- the requester asks two of the three servers ---
    requester = SIFECPABPairingKeyCombiner(pp, key_servers)
    request = requester.new_request("alice", y, ["org:A", "role:auditor"], ["ks1", "ks3"])
    partials = [servers[name].issue(request, authenticated_requester="alice") for name in request["servers"]]
    dk = requester.combine(request, partials)  # checked against pp; an ordinary key of the scheme

    # --- encryption and decryption are those of sife/cp_ab_pairing ---
    crypto = SIFECPABPairing({"keys": {"pp": pp}})
    ct = crypto.encrypt(x, policy="and(org:A, role:auditor)")
    print("x =", x, " y =", y, " plain <x, y> =", sum(a * b for a, b in zip(x, y)))
    print("key issued by ks1 and ks3: <x, y> =", crypto.decrypt(ct, dk))

    # --- the approval rule refuses weights it did not grant ---
    unit = requester.new_request("alice", [1, 0, 0], ["org:A", "role:auditor"], ["ks1", "ks2"])
    try:
        servers["ks1"].issue(unit, authenticated_requester="alice")
    except pyfe4ai.FEKeyError:
        print("request for y = [1, 0, 0]: refused by the approval rule")

    # --- a cheating server makes the issuance fail, it cannot hand out a wrong key ---
    request = requester.new_request("alice", y, ["org:A", "role:auditor"], ["ks2", "ks3"])
    partials = [servers[name].issue(request, authenticated_requester="alice") for name in request["servers"]]
    partials[0] = dict(partials[0], k0=list(reversed(partials[0]["k0"])))  # ks2 answers wrongly
    try:
        requester.combine(request, partials)
    except pyfe4ai.FEKeyError:
        print("partial keys of ks2 (altered) and ks3: refused by the check against pp")


if __name__ == "__main__":
    main()
