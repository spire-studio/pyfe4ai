"""
Caches and stores of the threshold schemes, and their locks.

* The sums of a replicated key share's pieces for an enrolled set
  (:func:`~pyfe4ai.schemes.threshold.sharing.share_for_enrolled`), per
  scheme object (:func:`share_cache`, :data:`SHARE_CACHE_SIZE`).
* The threshold MCFE over Ring-LWE
  (:mod:`pyfe4ai.schemes.mcfe.ring_lwe_threshold`): the key generator's
  sharings of ``sk_y`` per weights (:data:`WEIGHT_SHARINGS_KEPT`) with the
  label parts issued for each (:data:`LABEL_PARTS_KEPT`; a label part is
  handed out only while its sharing is the weights' current one, checked
  under the lock), a server's weight key share parts and its cache of
  ``NTT(sk_y)``.

The stores are attributes of the scheme objects (the pickled state of those
objects is unchanged); the module locks guard the several steps of each
lookup and eviction, never the sharing or the arithmetic. The locks are
process-wide, not per object: a shallow copy (``copy.copy``) of a key
generator or a server shares its stores with the original, and one lock per
store container keeps the two consistent. A forked child replaces each lock
(the parent's may be held by another thread; :func:`pyfe4ai.core.forksafe.renew_lock`).
"""

from __future__ import annotations

import sys
import threading
from collections import OrderedDict

from pyfe4ai.core import forksafe
from pyfe4ai.core.exceptions import FESchemeError

#: Entries kept by a cache of
#: :func:`~pyfe4ai.schemes.threshold.sharing.share_for_enrolled` (per scheme object).
SHARE_CACHE_SIZE = 16


#: Guards the caches of :func:`~pyfe4ai.schemes.threshold.sharing.share_for_enrolled`
#: (lookups and evictions are several steps of an OrderedDict; the sums themselves
#: run outside it).
_SHARE_CACHE_LOCK = threading.Lock()
forksafe.renew_lock(sys.modules[__name__], "_SHARE_CACHE_LOCK")  # a forked child: the parent's may be held


def share_cache(obj) -> OrderedDict:
    """The :func:`~pyfe4ai.schemes.threshold.sharing.share_for_enrolled` cache
    of scheme object *obj* (created on first use)."""
    cache = getattr(obj, "_enrolled_share_cache", None)
    if cache is None:
        cache = obj._enrolled_share_cache = OrderedDict()
    return cache


def cached_share(cache: OrderedDict, key: tuple, dk: dict):
    """The sum kept in *cache* under *key* for the key share *dk* (the same
    object, not only the same id), made the most recently used; ``None`` if
    there is none."""
    with _SHARE_CACHE_LOCK:  # get and move_to_end at once (threads share a server object)
        hit = cache.get(key)
        if hit is not None and hit[0] is dk:
            cache.move_to_end(key)
            return hit[1]
    return None


def keep_share(cache: OrderedDict, key: tuple, dk: dict, out) -> None:
    """Keep the sum *out* of key share *dk* in *cache* under *key*; the least
    recently used beyond :data:`SHARE_CACHE_SIZE` are dropped."""
    with _SHARE_CACHE_LOCK:
        cache[key] = (dk, out)  # dk kept alive: its id is not reused while cached
        while len(cache) > SHARE_CACHE_SIZE:
            cache.popitem(last=False)


#: Guards the weight-share stores and the NTT(sk_y) cache (key generators and
#: servers; several steps of a dict each). The sharing and the NTTs run outside it.
_STORE_LOCK = threading.Lock()
forksafe.renew_lock(sys.modules[__name__], "_STORE_LOCK")  # a forked child: the parent's may be held


#: Weights whose shares of ``sk_y`` a key generator keeps (least recently used
#: out): a label part for weights no longer kept comes with a new share set,
#: which servers holding the old weight part refuse until given the new one.
WEIGHT_SHARINGS_KEPT = 64


#: Labels whose label parts a key generator keeps per sharing of ``sk_y``
#: (least recently used out; the servers of one label get the same shares while
#: it is kept). A label part asked for again after that is a new sharing of the
#: same terms (same share set and noise key) with another ``sharing_id``:
#: combiners refuse partial decryptions of the two sharings
#: (:class:`~pyfe4ai.core.versions.MixedSharingsError`), and the servers then fetch the label's parts
#: again, all from the new sharing.
LABEL_PARTS_KEPT = 16


def weight_entry(owner, weights_id: str, make) -> dict:
    """The key generator *owner*'s sharing of ``sk_y`` for the weights
    *weights_id* (``owner.dict_weight_shares``), made by ``make()`` on first
    use and made the most recently used; the least recently used beyond
    :data:`WEIGHT_SHARINGS_KEPT` are dropped with their label parts. Of two
    threads making it, both return the one kept."""
    with _STORE_LOCK:
        store = owner.__dict__.get("dict_weight_shares")
        if not isinstance(store, OrderedDict):
            store = owner.dict_weight_shares = OrderedDict(store or {})
        entry = store.get(weights_id)
        if entry is not None:
            store.move_to_end(weights_id)
            return entry
    made = make()
    with _STORE_LOCK:  # of two threads making it, both return the one kept
        entry = store.setdefault(weights_id, made)
        while len(store) > WEIGHT_SHARINGS_KEPT:  # with their label parts
            store.popitem(last=False)
    return entry


def label_parts(owner, weights_id: str, ident: str, current_entry, make, attempts: int) -> dict:
    """The label parts *ident* of the weights' current sharing of ``sk_y``
    (``current_entry()``, :func:`weight_entry`): kept with that sharing
    (evicted with it, without a scan of the other keys; the last
    :data:`LABEL_PARTS_KEPT` labels), made by ``make(entry)`` on first use. A
    label part, cached or made, is handed out only while its sharing is the
    weights' current one (checked under the lock): one whose sharing another
    thread evicted meanwhile is made again for the weights' new share set, so
    none is issued for a sharing the key generator no longer hands out.

    Raises:
        FESchemeError: If the sharing was evicted during each of *attempts*
            (more than :data:`WEIGHT_SHARINGS_KEPT` weights in use at once).
    """
    for _ in range(attempts):
        entry = current_entry()
        with _STORE_LOCK:
            if owner.dict_weight_shares.get(weights_id) is not entry:
                continue  # evicted since current_entry returned it: the weights' new share set
            cached = entry["label_parts"]
            parts = cached.pop(ident, None)
            if parts is not None:
                cached[ident] = parts  # most recently used
                return parts
        made = make(entry)
        with _STORE_LOCK:  # still the weights' current share set: keep it (of two threads, one wins)
            if owner.dict_weight_shares.get(weights_id) is entry:
                cached = entry["label_parts"]
                parts = cached.setdefault(ident, made)
                while len(cached) > LABEL_PARTS_KEPT:
                    del cached[next(iter(cached))]
                return parts
    raise FESchemeError(
        "the sharing of these weights was evicted while their label part was made, {} times: more than "
        "WEIGHT_SHARINGS_KEPT = {} weights in use at once".format(attempts, WEIGHT_SHARINGS_KEPT))


def keep_weight_share(owner, key: tuple, share: dict, kept: int) -> None:
    """Keep the weight key share part *share* of server *owner* under *key*
    (``(weights_id, share_set_id)``); the least recently used beyond *kept*
    are dropped."""
    with _STORE_LOCK:
        store = owner.__dict__.setdefault("_weight_shares", OrderedDict())
        store[key] = share
        store.move_to_end(key)
        while len(store) > kept:
            store.popitem(last=False)


def has_weight_share(owner, key: tuple) -> bool:
    """Whether server *owner* keeps a weight key share part under *key*."""
    with _STORE_LOCK:
        return key in owner.__dict__.get("_weight_shares", {})


def weight_share(owner, key: tuple):
    """The weight key share part server *owner* keeps under *key*, or ``None``."""
    with _STORE_LOCK:
        return owner.__dict__.get("_weight_shares", {}).get(key)


#: ``NTT(sk_y)`` of the shares of kept weight parts a server keeps (:func:`cached_ntt`).
NTT_CACHE_SIZE = 64


def cached_ntt(owner, poly, ctx, make) -> bytes:
    """``NTT(poly)`` in the native ring *ctx*, kept by server *owner* across
    labels (``make()`` on first use; the last :data:`NTT_CACHE_SIZE`)."""
    key = (id(poly), id(ctx))
    with _STORE_LOCK:
        cache = owner.__dict__.setdefault("_hat_cache", OrderedDict())
        hit = cache.get(key)
        if hit is not None and hit[0] is poly and hit[1] is ctx:
            cache.move_to_end(key)
            return hit[2]
    hat = make()
    with _STORE_LOCK:
        cache[key] = (poly, ctx, hat)  # poly and ctx kept alive: their ids are not reused while cached
        while len(cache) > NTT_CACHE_SIZE:
            cache.popitem(last=False)
    return hat
