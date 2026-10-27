"""The NTT root caches and :func:`register_rns_modulus` under threads.

Registering the factors of an RNS modulus ``q`` for the first time drops
the cached entries of ``q`` (``_ntt_root_cache``, ``RingNTT._cache``),
which were computed without them. Done while other threads used those
caches it could fail: the scan of the keys raised ``RuntimeError``
(dictionary changed size during iteration) when another thread added an
entry meanwhile, and ``_get_ntt_root`` / ``RingNTT.get``, which stored their
result and then read it back by key, raised ``KeyError`` when the entry was
dropped in between. The scan now runs over a snapshot of the keys, and the
two functions return the value they computed. The tests replay those
interleavings with real threads (the other thread runs at the point where
it hurt), and run many threads on one new modulus.
"""

from __future__ import annotations

import threading

import gmpy2 as gp
import pytest

from pyfe4ai.primitives.lattice import ring_lwe_utils as rlu

N = 16
P = 1 << 10


@pytest.fixture
def fresh_q():
    """A new RNS modulus ``(q, factors)`` of this test (none of its cache entries survive the test)."""
    made = []

    def make(bound):
        q, factors = rlu.rns_ntt_modulus(bound, N, P)
        made.append(int(q))
        assert int(q) not in rlu._rns_factors
        return int(q), factors

    yield make
    for q in made:
        rlu._rns_factors.pop(q, None)
        for cache in (rlu._ntt_root_cache, rlu.RingNTT._cache):
            for key in [k for k in list(cache) if k[0] == q]:
                cache.pop(key, None)


def _in_another_thread(fn):
    """Run *fn* in another thread, to its end (the interleaving: it runs now)."""
    errors = []

    def run():
        try:
            fn()
        except BaseException as exc:  # noqa: BLE001 - reported to the test
            errors.append(exc)

    th = threading.Thread(target=run)
    th.start()
    th.join(30)
    assert not th.is_alive() and not errors, errors


class _RegisterAfterStore(dict):
    """A cache that, right after the first entry of *q* is stored, lets
    another thread register *q*'s factors (which drops that entry)."""

    def __init__(self, data, q, factors):
        super().__init__(data)
        self.q, self.factors, self.fired = q, factors, False

    def __setitem__(self, key, value):
        super().__setitem__(key, value)
        if key[0] == self.q and not self.fired:
            self.fired = True
            _in_another_thread(lambda: rlu.register_rns_modulus(self.q, self.factors))


@pytest.mark.parametrize("cache", ["_ntt_root_cache", "RingNTT._cache"])
def test_a_lookup_while_another_thread_registers_the_modulus(cache, fresh_q, monkeypatch):
    q, factors = fresh_q(2 ** 60)
    if cache == "_ntt_root_cache":
        hooked = _RegisterAfterStore(rlu._ntt_root_cache, q, factors)
        monkeypatch.setattr(rlu, "_ntt_root_cache", hooked)
        lookup = lambda: rlu._get_ntt_root(gp.mpz(q), N)  # noqa: E731
    else:
        hooked = _RegisterAfterStore(rlu.RingNTT._cache, q, factors)
        monkeypatch.setattr(rlu.RingNTT, "_cache", hooked)
        lookup = lambda: rlu.RingNTT.get(q, N)  # noqa: E731
    lookup()  # computed without the factors; the registration drops the entry before it returns
    assert hooked.fired and rlu.rns_factors(q) == factors
    # computed again, from the factors: the CRT of the factors' roots
    root = rlu._get_ntt_root(gp.mpz(q), N)
    assert root is not None and all(gp.powmod(root, N, f) == f - 1 for f in factors)
    ring = rlu.RingNTT.get(q, N)
    assert ring is not None and ring.root == root


class _Tripwire(int):
    """A key part whose comparison lets another thread add an entry to the
    cache being scanned (as a concurrent first use of another modulus would)."""

    def __eq__(self, other):
        if not self.done:
            self.done = True
            _in_another_thread(self.action)
        return int.__eq__(self, other)

    __hash__ = int.__hash__


@pytest.mark.parametrize("cache", ["_ntt_root_cache", "RingNTT._cache"])
def test_registering_while_another_thread_adds_an_entry(cache, fresh_q):
    q, factors = fresh_q(2 ** 60)
    other, _ = fresh_q(2 ** 70)
    target = rlu._ntt_root_cache if cache == "_ntt_root_cache" else rlu.RingNTT._cache
    trip = _Tripwire(other)
    trip.done = False
    trip.action = lambda: rlu.RingNTT.get(other, N)  # adds (other, N) to both caches
    target[(trip, 8)] = None  # scanned by the registration of q
    try:
        rlu.register_rns_modulus(q, factors)
    finally:
        target.pop((trip, 8), None)
    assert trip.done and rlu.rns_factors(q) == factors
    assert (other, N) in rlu._ntt_root_cache and (other, N) in rlu.RingNTT._cache


def test_many_threads_use_a_modulus_registered_meanwhile(fresh_q, monkeypatch):
    """Eight threads look up one new modulus for the first time at once (each
    finds no entry, computes and stores it), and one of them registers its
    factors before any returns; then sixteen threads register and use another
    new one at once. No exception, and every product is the negacyclic one."""
    import random

    q, factors = fresh_q(2 ** 60)
    threads = 8
    registered = threading.Event()

    def register():
        rlu.register_rns_modulus(q, factors)
        registered.set()

    barrier = threading.Barrier(threads, action=register, timeout=30)

    class FirstUseAtOnce(dict):
        """``RingNTT._cache``: until q is registered every thread finds no entry
        of ``(q, N)`` (they all look it up before any stores), and each one
        that stores it waits for the others; the last registers q."""

        def __contains__(self, key):
            return False if key == (q, N) and not registered.is_set() else super().__contains__(key)

        def __getitem__(self, key):
            if key == (q, N) and not registered.is_set():
                raise KeyError(key)
            return super().__getitem__(key)

        def __setitem__(self, key, value):
            super().__setitem__(key, value)
            if key == (q, N) and not registered.is_set():
                barrier.wait()  # every thread has stored; the barrier's action drops the entries

    monkeypatch.setattr(rlu.RingNTT, "_cache", FirstUseAtOnce(rlu.RingNTT._cache))
    rng = random.Random(1)
    a = [gp.mpz(rng.randrange(q)) for _ in range(N)]
    b = [gp.mpz(rng.randrange(q)) for _ in range(N)]
    want = rlu.poly_mul_negacyclic(a, b, gp.mpz(q))
    results, errors = [], []

    def use():
        try:
            rlu.RingNTT.get(q, N)
            ring = rlu.RingNTT.get(q, N)
            results.append(ring.inverse(ring.mul(ring.forward(a), ring.forward(b))))
        except BaseException as exc:  # noqa: BLE001
            errors.append(repr(exc))

    workers = [threading.Thread(target=use) for _ in range(threads)]
    for th in workers:
        th.start()
    for th in workers:
        th.join(60)
    assert not errors, errors
    assert len(results) == threads and all(r == want for r in results)

    q2, factors2 = fresh_q(2 ** 70)
    start = threading.Barrier(16, timeout=30)
    want2 = rlu.poly_mul_negacyclic(a, b, gp.mpz(q2))
    results.clear()

    def register_and_use():
        try:
            start.wait()
            rlu.register_rns_modulus(q2, factors2)
            ring = rlu.RingNTT.get(q2, N)
            results.append(ring.inverse(ring.mul(ring.forward(a), ring.forward(b))))
            results.append(rlu._poly_mul_reference(a, b, gp.mpz(q2), rlu._get_ntt_root(gp.mpz(q2), N)))
        except BaseException as exc:  # noqa: BLE001
            errors.append(repr(exc))

    workers = [threading.Thread(target=register_and_use) for _ in range(16)]
    for th in workers:
        th.start()
    for th in workers:
        th.join(60)
    assert not errors, errors
    assert len(results) == 32 and all(r == want2 for r in results)
