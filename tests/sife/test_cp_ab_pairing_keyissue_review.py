"""A review of the threshold key issuance (6724816): failing tests for the findings.

Each test states what the code should do; on 6724816 they failed.
"""

import copy

import pytest

from tests.pairing_group import skip_unless_available  # noqa: E402

skip_unless_available()

from pyfe4ai.core.exceptions import FEKeyError, FEValidationError  # noqa: E402
from pyfe4ai.schemes.sife import cp_ab_pairing_keyissue as ki  # noqa: E402
from tests.sife.test_cp_ab_pairing_keyissue import _resign, _system  # noqa: E402

pytestmark = pytest.mark.usefixtures("own_parameter_cache")


@pytest.mark.parametrize("verdict", ["refused", {"ok": False}, [False], 1])
def test_m1b_1_the_approval_rule_must_answer_true_to_approve(verdict):
    """M1b-1: a rule that answers anything but True (a reason string, a dict, a list) does not approve (fail
    closed); `if not approve(...)` approves every truthy answer."""
    _, _, _, servers, comb, _ = _system(approve=lambda request, requester, history: verdict)
    request = comb.new_request("alice", [1, 0, 0], ["org:A"], ["ks1", "ks2", "ks3"])
    with pytest.raises(FEKeyError):
        servers["ks1"].issue(request, "alice")


def test_m1b_2_a_request_with_fields_its_digest_does_not_bind_is_refused():
    """M1b-2: the digest binds the declared fields only, but the approval rule gets the whole dict. A field the
    digest does not cover must not reach the rule (here the rule approves on it): refuse unknown fields."""
    seen = []

    def approve(request, requester, history):
        seen.append(request.get("purpose"))
        return request.get("purpose") == "audit"

    _, _, _, servers, comb, _ = _system(approve=approve)
    request = comb.new_request("alice", [1, 0, 0], ["org:A"], ["ks1", "ks2", "ks3"])
    tagged = _resign(dict(copy.deepcopy(request), purpose="audit"))
    assert tagged["digest"] == request["digest"]          # the extra field is not bound
    with pytest.raises(FEValidationError):
        servers["ks1"].issue(tagged, "alice")


def test_m1b_3_a_sid_that_is_not_encodable_is_a_validation_error():
    """M1b-3: a sid with a lone surrogate is refused with FEValidationError, not UnicodeEncodeError from the
    digest's UTF-8 encoding (callers catch the library's errors)."""
    _, pp, ks, servers, comb, _ = _system()
    with pytest.raises(FEValidationError):
        comb.new_request("al\ud800ice", [1, 0, 0], ["org:A"], ["ks1", "ks2", "ks3"])
    request = comb.new_request("alice", [1, 0, 0], ["org:A"], ["ks1", "ks2", "ks3"])
    bad = copy.deepcopy(request)
    bad["sid"] = "al\ud800ice"
    with pytest.raises(FEValidationError):
        servers["ks1"].issue(bad, "al\ud800ice")


def test_m1b_4_the_history_given_to_the_rule_is_a_copy():
    """M1b-4 (Info): the rule gets the server's records; a rule that changes them must not change what later
    calls see (the docstring asks it not to; the server should not depend on that).

    Since the history is given as read-only views (08_history_copy_followup recommendation B), the careless
    rule's write raises inside the rule, which refuses that request; the stored history stays as it was."""
    calls = []

    def approve(request, requester, history):
        calls.append([copy.deepcopy(h[1]["y"]) for h in history])
        for _, earlier in history:
            earlier["y"] = ["0", "0", "0"]                     # a careless rule
        return True

    _, _, _, servers, comb, _ = _system(approve=approve)
    first = comb.new_request("alice", [1, 0, 0], ["org:A"], ["ks1", "ks2", "ks3"])
    second = comb.new_request("bob", [0, 2, 0], ["org:B"], ["ks1", "ks2", "ks3"])
    servers["ks1"].issue(first, first["sid"])                  # empty history: nothing to write
    with pytest.raises(FEKeyError, match="raised TypeError"):
        servers["ks1"].issue(second, second["sid"])
    assert calls == [[], [first["y"]]]
    assert [entry["y"] for _, entry in servers["ks1"]._history] == [first["y"]]
    assert ki.request_digest(first) == first["digest"]


# -- a later review (3120d70, read-only history views): checks the new tests do not make --------------------------


def test_r3_a_request_the_rule_keeps_stays_equal_and_is_the_stored_entry():
    kept = []

    def approve(request, requester, history):
        kept.append(request)
        return True

    _, _, _, servers, comb, _ = _system(approve=approve)
    first = comb.new_request("alice", [1, 0, 0], ["org:A"], ["ks1", "ks2", "ks3"])
    servers["ks1"].issue(first, first["sid"])
    for sid in ("bob", "carol"):
        r = comb.new_request(sid, [0, 1, 0], ["org:B"], ["ks1", "ks2", "ks3"])
        servers["ks1"].issue(r, sid)
    assert kept[0] == first and kept[0] is servers["ks1"]._history[0][1]


def test_r3_a_base_exception_of_the_rule_propagates_and_leaves_no_record():
    def approve(request, requester, history):
        raise KeyboardInterrupt

    _, _, _, servers, comb, _ = _system(approve=approve)
    server = servers["ks1"]
    with pytest.raises(KeyboardInterrupt):
        server.issue(comb.new_request("alice", [1, 0, 0], ["org:A"], ["ks1", "ks2", "ks3"]), "alice")
    assert not server._contents and not server._ids and not server._history and not server._lock.locked()


def test_r3_a_request_refused_by_a_raising_rule_is_asked_again_and_a_repeat_is_not():
    calls, mode = [], {"raise": True}

    def approve(request, requester, history):
        calls.append(request["sid"])
        if mode["raise"]:
            raise ValueError("operator bug")
        return True

    _, _, _, servers, comb, _ = _system(approve=approve)
    r = comb.new_request("alice", [1, 0, 0], ["org:A"], ["ks1", "ks2", "ks3"])
    with pytest.raises(FEKeyError, match="raised ValueError"):
        servers["ks1"].issue(r, "alice")
    mode["raise"] = False
    first = servers["ks1"].issue(r, "alice")
    mode["raise"] = True                       # a repeat of the accepted request does not ask the rule
    assert servers["ks1"].issue(r, "alice") == first and calls == ["alice", "alice"]
