"""discrete_gaussian_bigint on the native backend returns the reference's integers.

Known answers (computed with the Python reference before the native sampler
existed), agreement with the Python code on the same bits at the widths of the
schemes that sample this way -- SIFE FullySec Ring-LWE, ALS16 FullySec LWE and
the flooding noise of tMIFE-LWE and tMCFE-Ring-LWE --, the reference formula for
the samples the fixed point leaves undecided, and the error bound of the fixed
point evaluation against MPFR.
"""

import random

import gmpy2 as gp
import pytest

from pyfe4ai.schemes.mcfe import ring_lwe_threshold as tmcfe
from pyfe4ai.schemes.mife import lwe_threshold as tmife
from pyfe4ai.utils import native_backend
from pyfe4ai.utils import sampling_utils as su
from tests.test_native_backend import ByteStream

pytestmark = pytest.mark.skipif(
    native_backend.ext() is None or not hasattr(native_backend.ext(), "gauss_bigint"),
    reason="Rust extension with the wide-Gaussian sampler not built",
)

WIDTHS = {
    # SIFE FullySec Ring-LWE, standard profile (eta 4, X = Y = 4): sigma2, sigma3
    "fullysec_rlwe_sigma2": 125999223627111.6,
    "fullysec_rlwe_sigma3": 519508107760402.0,
    # ALS16 FullySec LWE (eta 4, X = Y = 4): sigma2, and a 64-bit integer width
    "als16_sigma2": 1.3627024450553898e+16,
    "als16_64bit": 2 ** 63 + 12345,
    # sigma_f of tMIFE-LWE (sec_param 64, lwe_n 16 / standard profile, n = 2, s = 3, t = 2)
    "tmife_toy": 414933754069760176222717870080,
    "tmife_std": 41547121354216193517533187480826480086738402952552749465600,
    # sigma_f of tMCFE-Ring-LWE (standard profile: eta 2, X = Y = 10, n = 3, s = 3, t = 2;
    # eta 4, X = Y = 100, n = 10, s = 5, t = 3)
    "tmcfe_std": 447376781648190700506077024983381091686950567609395922862080,
    "tmcfe_wide": 186834914345066495681357996796918057666081031937484318459297792000,
    "small": 3.2,
    "just_over_2_40": 2 ** 40 + 1,
}

KAT_SEED = b"pyfe4ai/discrete_gaussian_bigint/kat"
KAT = {  # discrete_gaussian_bigint(sigma, 8, rng=XofRandom(KAT_SEED)) of the Python reference at bits(sigma) + 320
    'fullysec_rlwe_sigma2': [
        -172698322160239,
        9778769733971,
        72818331335403,
        -59119738206320,
        97693267092431,
        157895728532274,
        -73118669070681,
        -118766459714026,
    ],
    'fullysec_rlwe_sigma3': [
        -712053423633612,
        40318900501756,
        300237671577093,
        -243756925183521,
        402799659133770,
        651020766572418,
        -301475995783000,
        -489686658181596,
    ],
    'als16_sigma2': [
        -18677609201877856,
        1057590121789348,
        7875422982706069,
        -6393897476956508,
        10565688430453092,
        17076684216047351,
        -7907905005564876,
        -12844789069640893,
    ],
    'als16_64bit': [
        -12641830874597103013,
        715823706867219276,
        5330434122332712959,
        -4327672222902627029,
        7151324603046419170,
        11558254133321715136,
        -5352419390094099856,
        -8693920580689080578,
    ],
    'tmife_toy': [
        -568720672022281618341301039023,
        32202909820368889213101268374,
        239801347312374692758000072266,
        -194689889408901251834852594000,
        321718125676448357501852552218,
        519973580038622041790132147575,
        -240790403120815433872210963154,
        -391115211412398988458886424060,
    ],
    'tmife_std': [
        -56945733012572836727279020295798776551630538371704316064769,
        3224462192200462897940990527235466719200027982848656129528,
        24011195955913365730947564126067478614754473274595748876422,
        -19494206924295272009718516217832897853618126417350317869976,
        32213484389325333784102108215508176745801081295903604119367,
        52064709652950609494430957177340808657319998362670954531686,
        -24110229648150587936354419178250560458261751650495218106069,
        -39162181896869499743548061935325721990095074532677931179746,
    ],
    'tmcfe_std': [
        -613188060529171734530013216492990159927772638773593324509407,
        34720805463132842780968244302361128892587929371363331890321,
        258551043252734817203540824819165349572610754137244755713408,
        -209912390324728523410915532254472689786365670357341340091238,
        346872767643819368674307644220118386772646400194942019122829,
        560629508923144899133632690045314879758200406411506104582493,
        -259617431802981025211616653550818974579636200521443581749089,
        -421695904030774958912580745959730677035632539112977899904729,
    ],
    'tmcfe_wide': [
        -256081547961237560947358237059455704263334203852027897698953768530,
        14500213200150944284839443995876170556999097186059495435386959184,
        107976908953535217184051402941757172038993985602179247816001450860,
        -87664280032149562073220188250123275234882054430520170830714956005,
        144862108383472125805463302048974969487475542103327631705967555215,
        234131878487476451155474931524876043605850120244341755206267946902,
        -108422257531326381329146589929108969523476936942024186970922602847,
        -176109984561542022427415577407073891639953386793561850945554198603,
    ],
    'small': [
        -4,
        0,
        2,
        -2,
        2,
        4,
        -2,
        -3,
    ],
    'just_over_2_40': [
        -1507023677183,
        85332835539,
        635437264721,
        -515898731101,
        852504325277,
        1377851263682,
        -638058112872,
        -1036396095836,
    ],
}


@pytest.fixture(autouse=True)
def _setup(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    with native_backend.backend("auto"), su.gaussian_sampler("rounded"):
        yield


class _Spy:
    """The extension, counting gauss_bigint calls and their undecided samples."""

    def __init__(self, ext) -> None:
        self._ext, self.calls, self.holes = ext, 0, 0

    def __getattr__(self, name):
        return getattr(self._ext, name)

    def gauss_bigint(self, *args, **kwargs):
        values, holes = self._ext.gauss_bigint(*args, **kwargs)
        self.calls += 1
        self.holes += len(holes)
        return values, holes


@pytest.fixture
def spy(monkeypatch):
    s = _Spy(native_backend.ext())
    monkeypatch.setattr(native_backend, "_ext", s)
    return s


class _BytesSystemRandom(random.Random):
    """``random.SystemRandom.getrandbits`` on a byte stream (big-endian, as os.urandom)."""

    def __init__(self, stream) -> None:
        self.stream = stream
        super().__init__(0)

    def getrandbits(self, k: int) -> int:
        nbytes = (k + 7) // 8
        return int.from_bytes(self.stream.take(nbytes), "big") >> (8 * nbytes - k)


def _both(fn):
    out = []
    for name in ("python", "native"):
        with native_backend.backend(name):
            out.append([int(v) for v in fn()])
    return out


@pytest.mark.parametrize("name", list(KAT))
def test_known_answers(name, spy):
    for backend in ("python", "native"):
        with native_backend.backend(backend):
            got = su.discrete_gaussian_bigint(WIDTHS[name], 8, rng=su.XofRandom(KAT_SEED))
        assert [int(v) for v in got] == KAT[name] and all(type(v) is type(gp.mpz(0)) for v in got)
    assert spy.calls == 1


@pytest.mark.parametrize("name", list(WIDTHS))
def test_xof_samples_and_counter_identical(name):
    sigma = WIDTHS[name]
    seed = name.encode() * 3

    def run():
        rng = su.XofRandom(seed)
        # one call for many samples, then one sample at a time (the tMIFE flooding loop)
        out = su.discrete_gaussian_bigint(sigma, 1200, rng=rng)
        out += [su.discrete_gaussian_bigint(sigma, 1, rng=rng)[0] for _ in range(5)]
        return out + [rng.getrandbits(64)]  # the stream continues where the reference leaves it

    py, nat = _both(run)
    assert py == nat


@pytest.mark.parametrize("name", list(WIDTHS))
def test_default_path_on_the_same_bytes(name, monkeypatch):
    """rng=None: the native sampler reads its bytes as getrandbits of os.urandom does."""
    sigma = WIDTHS[name]
    with native_backend.backend("python"):
        monkeypatch.setattr(su, "_CSPRNG", _BytesSystemRandom(ByteStream(7)))
        py = [int(v) for v in su.discrete_gaussian_bigint(sigma, 700)]
        monkeypatch.setattr(su, "_CSPRNG", su._SYSTEM_RANDOM)
    with native_backend.backend("native"):
        monkeypatch.setattr(native_backend, "_random_override", ByteStream(7).take)
        nat = [int(v) for v in su.discrete_gaussian_bigint(sigma, 700)]
    assert py == nat


def test_wide_matrix_on_the_same_bytes(monkeypatch):
    """discrete_gaussian_matrix above 2^40 (the FullySec Ring-LWE noise) samples through it."""
    sigma = WIDTHS["fullysec_rlwe_sigma3"]
    with native_backend.backend("python"):
        monkeypatch.setattr(su, "_CSPRNG", _BytesSystemRandom(ByteStream(3)))
        py = su.discrete_gaussian_matrix(7, 50, sigma)
        monkeypatch.setattr(su, "_CSPRNG", su._SYSTEM_RANDOM)
    with native_backend.backend("native"):
        monkeypatch.setattr(native_backend, "_random_override", ByteStream(3).take)
        nat = su.discrete_gaussian_matrix(7, 50, sigma)
    assert py == nat


@pytest.mark.parametrize("name", ["fullysec_rlwe_sigma2", "tmife_std", "small"])
def test_undecided_samples_take_the_reference_formula(name, spy, monkeypatch):
    # widen the margin 2^(max(s + 16 - 64 n, -100) + 1) to 2^-3: about half the samples are undecided
    _, _, mant, exp = _params(WIDTHS[name])
    s = mant.bit_length() + exp
    n = max(1, -(-(s + 64) // 64))
    monkeypatch.setattr(native_backend, "_bigint_widen", -3 - (max(s + 16 - 64 * n, -100) + 1))
    py, nat = _both(lambda: su.discrete_gaussian_bigint(WIDTHS[name], 600, rng=su.XofRandom(b"holes")))
    assert py == nat
    assert 100 < spy.holes < 500
    # and with every sample undecided
    monkeypatch.setattr(native_backend, "_bigint_widen", 200)
    spy.holes = 0
    py, nat = _both(lambda: su.discrete_gaussian_bigint(WIDTHS[name], 50, rng=su.XofRandom(b"all")))
    assert py == nat and spy.holes == 50


def test_scheme_flooding_noise_identical():
    sigma_m, sigma_c = WIDTHS["tmife_std"], WIDTHS["tmcfe_std"]
    py, nat = _both(lambda: [tmife._flooding_noise(sigma_m, su.XofRandom(b"k" * 32 + bytes([i]))) for i in range(40)])
    assert py == nat
    py, nat = _both(lambda: tmcfe._flooding_poly(sigma_c, 256, su.XofRandom(b"c" * 32)))
    assert py == nat


def _params(sigma):
    prec = su.bigint_precision(sigma)
    with gp.context(gp.get_context(), precision=prec):
        sig = gp.mpfr(sigma)
    mant, exp = (int(v) for v in sig.as_mantissa_exp())
    zeros = (mant & -mant).bit_length() - 1
    return prec, sig, mant >> zeros, exp + zeros


@pytest.mark.parametrize("name", ["fullysec_rlwe_sigma3", "als16_64bit", "tmcfe_wide", "small"])
def test_fixed_point_error_bound(name):
    """|native X - X| < 2^(s + 10 - 64 n) (sig < 2^s, n fraction words), the bound
    the decision margin 2^(s + 17 - 64 n) is built on, on random uniforms and on
    the edges of the evaluation (u1 near 1 and at the reductions' boundaries,
    u2 at the quadrant, octant and table boundaries)."""
    prec, sig, mant, exp = _params(WIDTHS[name])
    nb, P = (prec + 7) // 8, 2 ** prec
    pad = 8 * nb - prec
    rng = random.Random(name)
    edge1 = [0, 1, P - 2, P - 3, P - 2 ** (prec - 8) - 1, P - 2 ** (prec - 8), P - 2 ** (prec - 8) + 1,
             2 ** (prec - 1) - 1, 2 ** (prec - 1), P - 2 ** 40]
    edge2 = [0, 1, P - 1] + [k * P // 8 + d for k in range(8) for d in (-1, 0, 1)] + \
            [j * P // 2048 + d for j in (1, 255, 256, 257, 1023) for d in (-1, 0, 1)]
    pairs = [(a % P, b % P) for a in edge1 for b in edge2[::3]]
    pairs += [(rng.getrandbits(prec), rng.getrandbits(prec)) for _ in range(600)]
    pairs += [(P - 1 - rng.getrandbits(rng.randrange(1, prec - 8)), rng.getrandbits(prec)) for _ in range(200)]
    sbits = mant.bit_length() + exp
    mb = mant.to_bytes((mant.bit_length() + 7) // 8, "little")
    for b1, b2 in pairs:
        got = native_backend.ext().gauss_bigint_approx(prec, mb, exp, (b1 << pad).to_bytes(nb, "big"),
                                                       (b2 << pad).to_bytes(nb, "big"))
        if b1 == P - 1:  # u1 = 1: the sample is 0
            assert got is None
            continue
        prod, sh, neg, words = got
        with gp.context(gp.get_context(), precision=2 * prec + 256):
            x = gp.sqrt(-2 * gp.log((gp.mpfr(b1) + 1) / gp.mpfr(2) ** prec)) * \
                gp.cos(2 * gp.const_pi() * gp.mpfr(b2) / gp.mpfr(2) ** prec) * gp.mpfr(sig, prec)
            approx = gp.mpfr(prod) / gp.mpfr(2) ** sh
            err = abs((-approx if neg else approx) - x)
            assert err < gp.mpfr(2) ** (sbits + 10 - 64 * words), (b1, b2)


def test_threads_and_chunks_do_not_change_the_samples(monkeypatch):
    monkeypatch.setattr(native_backend, "_seed_override", bytes(range(32)))
    sigma = WIDTHS["fullysec_rlwe_sigma2"]
    out = {}
    for threads in (1, 3):
        monkeypatch.setattr(native_backend, "_threads", threads)
        out[threads] = su.discrete_gaussian_bigint(sigma, 1000)
    assert out[1] == out[3]
    monkeypatch.setattr(native_backend, "_seed_override", bytes(range(1, 33)))
    assert su.discrete_gaussian_bigint(sigma, 1000) != out[1]


def test_default_path_distribution():
    sigma = 2.0 ** 50
    v = [int(x) for x in su.discrete_gaussian_bigint(sigma, 20000)]
    mean = sum(v) / len(v)
    var = sum(x * x for x in v) / len(v)
    assert abs(mean) < 4 * sigma / len(v) ** 0.5
    assert abs(var / sigma ** 2 - 1) < 0.05


def test_python_path_where_the_native_one_does_not_apply(spy, monkeypatch):
    sigma = WIDTHS["tmife_toy"]
    su.discrete_gaussian_bigint(sigma, 3, rng=random.Random(1))  # another bit source
    su.discrete_gaussian_bigint(2 ** 450, 3)  # beyond 2^448
    su.discrete_gaussian_bigint(sigma, 3, sampler="facct")
    with native_backend.backend("python"):
        su.discrete_gaussian_bigint(sigma, 3)
    monkeypatch.setattr(su, "_CSPRNG", random.Random(2))  # a replaced OS source
    su.discrete_gaussian_bigint(sigma, 3)
    assert spy.calls == 0
    monkeypatch.setattr(su, "_CSPRNG", su._SYSTEM_RANDOM)
    # another rounding mode: the Python code (and its results) under that context
    with gp.context(gp.get_context(), round=gp.RoundDown):
        py, nat = _both(lambda: su.discrete_gaussian_bigint(sigma, 20, rng=su.XofRandom(b"rd")))
    assert py == nat and spy.calls == 0
    su.discrete_gaussian_bigint(sigma, 3)
    assert spy.calls == 1


# -- review 2026-10 (91d0369): inputs the reference accepts, shared XofRandom ------


@pytest.mark.parametrize("sigma, native", [(gp.xmpz(1000), True), (gp.xmpz(2 ** 200 + 7), True),
                                           (gp.mpfr("1e-300000"), True), (gp.mpfr("1e-400000"), False)],
                         ids=["xmpz", "xmpz-wide", "tiny-native", "tiny-beyond-2^20"])
def test_native_path_accepts_what_the_reference_accepts(sigma, native, spy):
    """An unhashable sigma (xmpz) is sampled natively; exponents beyond 2^20 keep the Python code."""
    py, nat = _both(lambda: su.discrete_gaussian_bigint(sigma, 5, rng=su.XofRandom(b"\x01" * 32)))
    assert py == nat and spy.calls == native


def test_context_and_count_gates(spy, monkeypatch):
    sigma = WIDTHS["tmife_toy"]
    for ctx in ({"emax": 2 ** 16}, {"emin": -(2 ** 16)}, {"subnormalize": True}):
        with gp.context(gp.get_context(), **ctx):
            py, nat = _both(lambda: su.discrete_gaussian_bigint(sigma, 4, rng=su.XofRandom(b"ctx")))
        assert py == nat
    assert spy.calls == 0
    monkeypatch.setattr(native_backend, "MAX_NATIVE_BIGINT_COUNT", 3)
    su.discrete_gaussian_bigint(sigma, 4)
    assert spy.calls == 0
    with pytest.raises(ValueError, match="too many"):
        native_backend.ext().gauss_bigint(2 ** 24 + 1, 160, b"\x01", 0)


def test_shared_xof_random_never_draws_a_block_twice():
    """XofRandom claims its counters atomically, and the native call claims all of
    its calls before it runs: a thread drawing from the same XofRandom meanwhile
    gets other blocks, and the native samples are the reference's from the claimed
    counter (found as the one gap the other thread's blocks leave)."""
    import hashlib
    import threading

    seed, count = b"\x02" * 32, 6000
    rng = su.XofRandom(seed)
    stop, outs = threading.Event(), []

    def other():
        while not stop.is_set():
            outs.append(rng.getrandbits(64))

    t = threading.Thread(target=other)
    with native_backend.backend("native"):
        t.start()
        native = su.discrete_gaussian_bigint(2 ** 100, count, rng=rng)
        stop.set()
        t.join()
    head = len(seed).to_bytes(8, "big") + seed
    block = {int.from_bytes(hashlib.shake_256(head + c.to_bytes(8, "big")).digest(8), "big"): c
             for c in range(rng._counter)}
    used = [block[v] for v in outs]
    assert len(set(used)) == len(used)
    free = sorted(set(range(rng._counter)) - set(used))
    assert free == list(range(free[0], free[0] + 2 * count))
    ref = su.XofRandom(seed)
    ref._counter = free[0]
    with native_backend.backend("python"):
        assert su.discrete_gaussian_bigint(2 ** 100, count, rng=ref) == native


def test_xof_random_threads_share_blocks_never():
    import threading

    rng = su.XofRandom(b"\x03" * 32)
    outs = [[] for _ in range(4)]

    def draw(out):
        for _ in range(3000):
            out.append(rng.getrandbits(128))

    threads = [threading.Thread(target=draw, args=(o,)) for o in outs]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    every = [v for o in outs for v in o]
    ref = su.XofRandom(b"\x03" * 32)
    assert sorted(every) == sorted(ref.getrandbits(128) for _ in range(12000)) and rng._counter == 12000
