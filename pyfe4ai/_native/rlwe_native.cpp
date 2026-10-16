// Native core of PyFE4AI's Ring-LWE schemes (BMMS21 MIFE / MCFE over
// Z_q[x] / (x^n + 1)); loaded by pyfe4ai.utils.native_backend, which keeps the
// Python implementation as reference and fallback.
//
// * negacyclic NTT with the root, tables and butterfly order of
//   pyfe4ai.utils.ring_lwe_utils.ntt_forward / ntt_inverse (values are the
//   same as the Python ones in both domains); Montgomery arithmetic with u64
//   words for q < 2^62 and u128 words for q < 2^125;
// * the rounded Gaussian of sampling_utils.discrete_gaussian_matrix: Box-Muller
//   as random.Random.gauss on SystemRandom uniforms (7 OS-random bytes each),
//   calling the C library's cos, sin, log and sqrt like CPython's math module;
// * SHAKE-256 and the reduction of crypto_utils.hash_to_zq_vector;
// * whole-tensor encryption / decryption / key-derivation loops of the MCFE
//   ndarray helpers, with the GIL released and optionally several threads.
//
// Inputs and outputs are Python ints, decimal strings (the ciphertext
// format) or packed little-endian words in bytes objects. Needs a 64-bit
// GCC or Clang (unsigned __int128); built by setup.py as an optional
// extension.
#define PY_SSIZE_T_CLEAN
#include <Python.h>

#include <algorithm>
#include <atomic>
#include <cerrno>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <exception>
#include <memory>
#include <new>
#include <string>
#include <thread>
#include <type_traits>
#include <vector>

#if defined(__linux__)
#include <sys/random.h>
#define RLWE_HAVE_GETRANDOM 1
#elif defined(__APPLE__) || defined(__FreeBSD__) || defined(__OpenBSD__)
#include <sys/random.h>
#include <unistd.h>
#define RLWE_HAVE_GETENTROPY 1
#endif

typedef uint64_t u64;
typedef unsigned __int128 u128;
typedef __int128 i128;

// ===================================================================== arithmetic

// Montgomery arithmetic, R = 2^64: a b R^-1 mod q for q < 2^62.
struct Arith64 {
    typedef u64 W;
    static const int BITS = 64;
    u64 q = 0, qinv = 0;  // qinv = -q^-1 mod 2^64
    inline W mul(W a, W b) const {
        u128 t = (u128)a * b;
        u64 m = (u64)t * qinv;
        u64 r = (u64)((t + (u128)m * q) >> 64);
        return r >= q ? r - q : r;
    }
    inline W add(W a, W b) const { W s = a + b; return s >= q ? s - q : s; }
    inline W sub(W a, W b) const { return a >= b ? a - b : a + q - b; }
};

static inline void mul_full(u128 a, u128 b, u128 &hi, u128 &lo) {
    u64 a0 = (u64)a, a1 = (u64)(a >> 64), b0 = (u64)b, b1 = (u64)(b >> 64);
    u128 p00 = (u128)a0 * b0, p01 = (u128)a0 * b1, p10 = (u128)a1 * b0, p11 = (u128)a1 * b1;
    u128 mid = (p00 >> 64) + (u64)p01 + (u64)p10;
    lo = (u128)(u64)p00 | (mid << 64);
    hi = p11 + (p01 >> 64) + (p10 >> 64) + (mid >> 64);
}

// Montgomery arithmetic, R = 2^128: a b R^-1 mod q for q < 2^125.
struct Arith128 {
    typedef u128 W;
    static const int BITS = 128;
    u128 q = 0, qinv = 0;  // qinv = -q^-1 mod 2^128
    inline W mul(W a, W b) const {
        u128 hi, lo, mh, ml;
        mul_full(a, b, hi, lo);
        u128 m = lo * qinv;
        mul_full(m, q, mh, ml);
        u128 r = hi + mh + (lo != 0);  // lo + ml is 0 or exactly 2^128
        return r >= q ? r - q : r;
    }
    inline W add(W a, W b) const { W s = a + b; return s >= q ? s - q : s; }
    inline W sub(W a, W b) const { return a >= b ? a - b : a + q - b; }
};

static inline u128 reduce_i128(i128 v, u128 q) {
    if (v >= 0) {
        u128 u = (u128)v;
        return u < q ? u : u % q;
    }
    u128 m = (u128)(-(v + 1)) + 1;  // |v| without overflow at INT128_MIN
    if (m >= q) m %= q;
    return m ? q - m : 0;
}

// ======================================================================== context

struct CtxBase {
    int bits = 0;  // word size: 64 or 128
    size_t n = 0;
    int logn = 0;
    u128 q = 0;
    int qbits = 0;
    PyObject *qobj = nullptr;  // q as a Python int
    virtual ~CtxBase() { Py_XDECREF(qobj); }
};

// Shoup multiplication: x w mod q in [0, 2q) for any word x, with wp = floor(w 2^BITS / q).
static inline u64 shoup_mul(u64 x, u64 w, u64 wp, u64 q) {
    u64 quot = (u64)(((u128)x * wp) >> 64);
    return x * w - quot * q;
}
static inline u128 shoup_mul(u128 x, u128 w, u128 wp, u128 q) {
    u128 hi, lo;
    mul_full(x, wp, hi, lo);
    return x * w - hi * q;
}
static inline u64 shoup_quot(u64 w, u64 q) { return (u64)(((u128)w << 64) / q); }
static inline u128 shoup_quot(u128 w, u128 q) {  // floor(w 2^128 / q), w < q < 2^126
    u128 r = w, quot = 0;
    for (int i = 0; i < 128; ++i) {
        r <<= 1;
        quot <<= 1;
        if (r >= q) { r -= q; quot |= 1; }
    }
    return quot;
}

template <class A>
struct Ctx : CtxBase {
    typedef typename A::W W;
    A ar;
    W r2 = 0;      // R^2 mod q
    W rmod = 0;    // R mod q (Montgomery form of 1)
    std::vector<uint32_t> br;
    // twiddles as plain values and their Shoup quotients
    std::vector<W> twist, twist_p, scale, scale_p, fwd, fwd_p, inv, inv_p;

    inline W to_mont(W x) const { return ar.mul(x, r2); }
    inline W from_mont(W x) const { return ar.mul(x, 1); }
    inline W mulp(W a, W b) const { return to_mont(ar.mul(a, b)); }  // plain a b mod q

    W powp(W b, u128 e) const {  // plain b^e mod q
        W r = rmod, x = to_mont(b);
        while (e) {
            if (e & 1) r = ar.mul(r, x);
            x = ar.mul(x, x);
            e >>= 1;
        }
        return from_mont(r);
    }

    // Cooley-Tukey stages with Harvey's lazy butterflies: values stay in [0, 4q)
    // (4q < 2^BITS), the same residues as ring_lwe_utils' (u + v, u - v) mod q.
    void butterflies(W *x, const W *tw, const W *twp) const {
        const W q = (W)this->q, q2 = q + q;
        size_t off = 0;
        for (size_t len = 2; len <= n; len <<= 1) {
            size_t half = len >> 1;
            const W *t = tw + off, *tp = twp + off;
            for (size_t s = 0; s < n; s += len) {
                W *lo = x + s, *hi = x + s + half;
                for (size_t j = 0; j < half; ++j) {
                    W u = lo[j];
                    u = u >= q2 ? u - q2 : u;
                    W v = shoup_mul(hi[j], t[j], tp[j], q);
                    lo[j] = u + v;
                    hi[j] = u - v + q2;
                }
            }
            off += half;
        }
    }
    static inline W reduce4(W v, W q) {  // [0, 4q) -> [0, q)
        W q2 = q + q;
        v = v >= q2 ? v - q2 : v;
        return v >= q ? v - q : v;
    }
    // ring_lwe_utils.ntt_forward: twist, bit-reverse, Cooley-Tukey stages (out in [0, q)).
    void forward(const W *in, W *out) const {
        const W q = (W)this->q;
        for (size_t i = 0; i < n; ++i) {
            uint32_t b = br[i];
            out[i] = shoup_mul(in[b], twist[b], twist_p[b], q);
        }
        butterflies(out, fwd.data(), fwd_p.data());
        for (size_t i = 0; i < n; ++i) out[i] = reduce4(out[i], q);
    }
    // ring_lwe_utils.ntt_inverse: bit-reverse, stages, then times n^-1 untwist[i] (out in [0, q)).
    void inverse(const W *in, W *out) const {
        const W q = (W)this->q;
        for (size_t i = 0; i < n; ++i) out[i] = in[br[i]];
        butterflies(out, inv.data(), inv_p.data());
        for (size_t i = 0; i < n; ++i) {
            W v = shoup_mul(out[i], scale[i], scale_p[i], q);
            out[i] = v >= q ? v - q : v;
        }
    }

    // Tables of ring_lwe_utils._get_twiddle_factors from the root of unity.
    bool build(u128 root) {
        W rt = (W)root, qw = (W)q;
        W root_inv = powp(rt, (u128)(2 * n - 1));
        if (powp(rt, (u128)n) != (W)(q - 1) || mulp(rt, root_inv) != 1) return false;
        W n_inv = powp((W)(n % q), q - 2);  // q is prime
        if (mulp((W)(n % q), n_inv) != 1) return false;
        twist.resize(n); twist_p.resize(n); scale.resize(n); scale_p.resize(n); br.resize(n);
        W t = 1, u = 1;
        for (size_t i = 0; i < n; ++i) {
            twist[i] = t;
            twist_p[i] = shoup_quot(t, qw);
            scale[i] = mulp(n_inv, u);
            scale_p[i] = shoup_quot(scale[i], qw);
            t = mulp(t, rt);
            u = mulp(u, root_inv);
            uint32_t r = 0, x = (uint32_t)i;
            for (int b = 0; b < logn; ++b) { r = (r << 1) | (x & 1); x >>= 1; }
            br[i] = r;
        }
        auto stages = [&](W base, std::vector<W> &out, std::vector<W> &outp) {
            W w2 = mulp(base, base);
            out.clear();
            outp.clear();
            for (size_t len = 2; len <= n; len <<= 1) {
                W step = powp(w2, (u128)(n / len)), w = 1;
                out.push_back(1);
                for (size_t j = 1; j < len / 2; ++j) {
                    w = mulp(w, step);
                    out.push_back(w);
                }
            }
            for (W v : out) outp.push_back(shoup_quot(v, qw));
        };
        stages(rt, fwd, fwd_p);
        stages(root_inv, inv, inv_p);
        return true;
    }
};

typedef Ctx<Arith64> Ctx64;
typedef Ctx<Arith128> Ctx128;

static const char *CTX_NAME = "pyfe4ai._rlwe_native.ctx";
static const char *KEY_NAME = "pyfe4ai._rlwe_native.enckey";

static void ctx_free(PyObject *cap) {
    CtxBase *c = (CtxBase *)PyCapsule_GetPointer(cap, CTX_NAME);
    delete c;
}

static CtxBase *get_ctx(PyObject *cap) { return (CtxBase *)PyCapsule_GetPointer(cap, CTX_NAME); }

// Dispatch a generic lambda on the word size of a context.
template <class F>
static auto with_ctx(CtxBase *c, F &&f) -> decltype(f(*(Ctx64 *)c)) {
    if (c->bits == 64) return f(*static_cast<Ctx64 *>(c));
    return f(*static_cast<Ctx128 *>(c));
}

// ============================================================ Python <-> integers

// Non-negative Python int below 2^128 -> u128 (public C API only).
static bool pylong_u128(PyObject *v, u128 *out) {
    unsigned long long lo = PyLong_AsUnsignedLongLongMask(v);
    if (lo == (unsigned long long)-1 && PyErr_Occurred()) return false;
    PyObject *sh = PyLong_FromLong(64);
    if (!sh) return false;
    PyObject *hi_o = PyNumber_Rshift(v, sh);
    Py_DECREF(sh);
    if (!hi_o) return false;
    unsigned long long hi = PyLong_AsUnsignedLongLong(hi_o);
    Py_DECREF(hi_o);
    if (hi == (unsigned long long)-1 && PyErr_Occurred()) return false;
    *out = ((u128)hi << 64) | lo;
    return true;
}

static PyObject *u128_pylong(u128 v) {
    if (!(v >> 64)) return PyLong_FromUnsignedLongLong((u64)v);
    PyObject *hi = PyLong_FromUnsignedLongLong((u64)(v >> 64));
    PyObject *lo = PyLong_FromUnsignedLongLong((u64)v);
    PyObject *sh = PyLong_FromLong(64);
    PyObject *res = nullptr;
    if (hi && lo && sh) {
        PyObject *t = PyNumber_Lshift(hi, sh);
        if (t) {
            res = PyNumber_Or(t, lo);
            Py_DECREF(t);
        }
    }
    Py_XDECREF(hi); Py_XDECREF(lo); Py_XDECREF(sh);
    return res;
}

// v mod q for an integer-like object (int, gmpy2.mpz, numpy integer: anything with __index__).
static bool index_mod(PyObject *o, PyObject *qobj, u128 *out) {
    PyObject *v = PyNumber_Index(o);
    if (!v) return false;
    PyObject *r = PyNumber_Remainder(v, qobj);
    Py_DECREF(v);
    if (!r) return false;
    bool ok = pylong_u128(r, out);
    Py_DECREF(r);
    return ok;
}

// int(str(o)) % q, the semantics of canonical_ring_ciphertext.
static bool str_int_mod(PyObject *o, PyObject *qobj, u128 *out) {
    PyObject *s = PyObject_Str(o);
    if (!s) return false;
    PyObject *v = PyLong_FromUnicodeObject(s, 10);
    Py_DECREF(s);
    if (!v) return false;
    PyObject *r = PyNumber_Remainder(v, qobj);
    Py_DECREF(v);
    if (!r) return false;
    bool ok = pylong_u128(r, out);
    Py_DECREF(r);
    return ok;
}

// Fast path of int(s) % q: optional '-', then 1..38 ASCII digits. false: use the slow path.
static inline bool parse_dec(const char *s, size_t len, u128 q, u128 *out) {
    size_t i = (len > 0 && s[0] == '-') ? 1 : 0;
    size_t nd = len - i;
    if (nd < 1 || nd > 38) return false;
    const char *d = s + i;
    size_t head = nd > 19 ? nd - 19 : 0;  // at most 19 digits per u64
    u64 hi = 0, lo = 0;
    for (size_t j = 0; j < head; ++j) {
        unsigned c = (unsigned char)d[j] - '0';
        if (c > 9) return false;
        hi = hi * 10 + c;
    }
    for (size_t j = head; j < nd; ++j) {
        unsigned c = (unsigned char)d[j] - '0';
        if (c > 9) return false;
        lo = lo * 10 + c;
    }
    u128 v = head ? (u128)hi * 10000000000000000000ULL + lo : (u128)lo;
    if (v >= q) v %= q;
    *out = (i && v) ? q - v : v;
    return true;
}

static const char DIGITS2[] =
    "00010203040506070809101112131415161718192021222324252627282930313233343536373839"
    "40414243444546474849505152535455565758596061626364656667686970717273747576777879"
    "8081828384858687888990919293949596979899";

// Decimal digits of v < 2^127 (gmpy2.digits of a non-negative value).
static size_t format_dec(u128 v, char *buf /* >= 40 bytes */) {
    char tmp[40];
    char *p = tmp + sizeof tmp;
    u64 lo;
    if (v >> 64) {  // v = hi 10^19 + rem with hi < 2^64: rem as exactly 19 digits
        const u64 TEN19 = 10000000000000000000ULL;
        u128 hi = v / TEN19;
        u64 rem = (u64)(v - hi * TEN19);
        for (int k = 0; k < 9; ++k) {
            u64 r = rem % 100;
            rem /= 100;
            p -= 2;
            memcpy(p, DIGITS2 + 2 * r, 2);
        }
        *--p = (char)('0' + rem);
        lo = (u64)hi;
    } else {
        lo = (u64)v;
    }
    if (lo || p == tmp + sizeof tmp) {
        while (lo >= 100) {
            u64 r = lo % 100;
            lo /= 100;
            p -= 2;
            memcpy(p, DIGITS2 + 2 * r, 2);
        }
        if (lo >= 10) {
            p -= 2;
            memcpy(p, DIGITS2 + 2 * lo, 2);
        } else {
            *--p = (char)('0' + lo);
        }
    }
    size_t len = (size_t)(tmp + sizeof tmp - p);
    memcpy(buf, p, len);
    return len;
}

static PyObject *ascii_str(const char *s, size_t len) {
    PyObject *o = PyUnicode_New((Py_ssize_t)len, 127);
    if (!o) return nullptr;
    memcpy(PyUnicode_1BYTE_DATA(o), s, len);
    return o;
}

// Decimal strings of n words as a new list.
template <class W>
static PyObject *dec_list(const W *v, size_t n) {
    PyObject *lst = PyList_New((Py_ssize_t)n);
    if (!lst) return nullptr;
    char buf[48];
    for (size_t i = 0; i < n; ++i) {
        size_t len = format_dec((u128)v[i], buf);
        PyObject *o = ascii_str(buf, len);
        if (!o) { Py_DECREF(lst); return nullptr; }
        PyList_SET_ITEM(lst, (Py_ssize_t)i, o);
    }
    return lst;
}

// Pre-formatted decimal strings (offsets into one buffer) as a new list.
static PyObject *dec_list_from(const char *buf, const uint32_t *off, size_t n) {
    PyObject *lst = PyList_New((Py_ssize_t)n);
    if (!lst) return nullptr;
    for (size_t i = 0; i < n; ++i) {
        PyObject *o = ascii_str(buf + off[i], off[i + 1] - off[i]);
        if (!o) { Py_DECREF(lst); return nullptr; }
        PyList_SET_ITEM(lst, (Py_ssize_t)i, o);
    }
    return lst;
}

// Words of a sequence of n integers / decimal strings, reduced mod q (with the GIL).
// str_semantics: int(str(v)) % q for every value (ciphertexts); else str -> int(s), others __index__.
template <class W>
static bool read_words(PyObject *seq_o, size_t n, PyObject *qobj, u128 q, W *out, bool str_semantics) {
    PyObject *seq = PySequence_Fast(seq_o, "expected a sequence of integers");
    if (!seq) return false;
    if ((size_t)PySequence_Fast_GET_SIZE(seq) != n) {
        Py_DECREF(seq);
        PyErr_Format(PyExc_ValueError, "expected %zu values", n);
        return false;
    }
    PyObject **items = PySequence_Fast_ITEMS(seq);
    for (size_t i = 0; i < n; ++i) {
        PyObject *o = items[i];
        u128 v;
        bool ok;
        if (PyUnicode_Check(o) && PyUnicode_IS_ASCII(o)
            && parse_dec((const char *)PyUnicode_DATA(o), (size_t)PyUnicode_GET_LENGTH(o), q, &v))
            ok = true;
        else if (str_semantics || PyUnicode_Check(o))
            ok = str_int_mod(o, qobj, &v);
        else
            ok = index_mod(o, qobj, &v);
        if (!ok) { Py_DECREF(seq); return false; }
        out[i] = (W)v;
    }
    Py_DECREF(seq);
    return true;
}

// ===================================================================== threads

template <class F>
static void parallel_for(size_t count, int threads, F &&f) {
    if (threads <= 1 || count <= 1) {
        for (size_t i = 0; i < count; ++i) f(i);
        return;
    }
    size_t nt = std::min<size_t>((size_t)threads, count);
    std::atomic<size_t> next(0);
    auto work = [&]() {
        for (size_t i; (i = next.fetch_add(1)) < count;) f(i);
    };
    std::vector<std::thread> pool;
    pool.reserve(nt - 1);
    for (size_t t = 1; t < nt; ++t) {
        try {
            pool.emplace_back(work);
        } catch (...) {
            break;  // fewer threads; the calling thread still finishes the work
        }
    }
    work();
    for (auto &th : pool) th.join();
}

// ================================================================ OS randomness

static bool os_random(unsigned char *buf, size_t len) {
#if defined(RLWE_HAVE_GETRANDOM)
    while (len) {
        ssize_t r = getrandom(buf, len, 0);
        if (r < 0) {
            if (errno == EINTR) continue;
            return false;
        }
        buf += r;
        len -= (size_t)r;
    }
    return true;
#elif defined(RLWE_HAVE_GETENTROPY)
    while (len) {
        size_t k = len < 256 ? len : 256;
        if (getentropy(buf, k) != 0) return false;
        buf += k;
        len -= k;
    }
    return true;
#else
    (void)buf;
    (void)len;
    return false;
#endif
}

static bool have_os_random() {
#if defined(RLWE_HAVE_GETRANDOM) || defined(RLWE_HAVE_GETENTROPY)
    return true;
#else
    return false;
#endif
}

// ===================================================================== Gaussian

// The C library's functions, called through pointers so that the compiler
// neither folds sin / cos into sincos nor substitutes its own versions: the
// results must be those of CPython's math module.
static double (*volatile c_cos)(double) = &::cos;
static double (*volatile c_sin)(double) = &::sin;
static double (*volatile c_log)(double) = &::log;

static inline double uniform53(const unsigned char *p) {  // SystemRandom.random()
    u64 x = 0;
    for (int i = 0; i < 7; ++i) x = (x << 8) | p[i];
    return (double)(x >> 3) * (1.0 / 9007199254740992.0);
}

static inline size_t gauss_bytes(size_t count) { return 14 * ((count + 1) / 2); }

// count samples of int(round(gauss(0.0, sigma))) from 14 bytes per pair (random.Random.gauss).
static void gauss_fill(const unsigned char *buf, size_t count, double sigma, int64_t *out) {
    const double TWOPI = 2.0 * 3.141592653589793;
    double (*fcos)(double) = c_cos, (*fsin)(double) = c_sin, (*flog)(double) = c_log;
    for (size_t i = 0; i < count; i += 2, buf += 14) {
        double x2pi = uniform53(buf) * TWOPI;
        double g2rad = std::sqrt(-2.0 * flog(1.0 - uniform53(buf + 7)));
        double z = fcos(x2pi) * g2rad;
        out[i] = (int64_t)std::nearbyint(0.0 + z * sigma);
        if (i + 1 < count) {
            double z2 = fsin(x2pi) * g2rad;
            out[i + 1] = (int64_t)std::nearbyint(0.0 + z2 * sigma);
        }
    }
}

// ===================================================================== SHAKE-256

static const u64 KECCAK_RC[24] = {
    0x0000000000000001ULL, 0x0000000000008082ULL, 0x800000000000808AULL, 0x8000000080008000ULL,
    0x000000000000808BULL, 0x0000000080000001ULL, 0x8000000080008081ULL, 0x8000000000008009ULL,
    0x000000000000008AULL, 0x0000000000000088ULL, 0x0000000080008009ULL, 0x000000008000000AULL,
    0x000000008000808BULL, 0x800000000000008BULL, 0x8000000000008089ULL, 0x8000000000008003ULL,
    0x8000000000008002ULL, 0x8000000000000080ULL, 0x000000000000800AULL, 0x800000008000000AULL,
    0x8000000080008081ULL, 0x8000000000008080ULL, 0x0000000080000001ULL, 0x8000000080008008ULL,
};

static inline u64 rotl(u64 x, int s) { return (x << s) | (x >> (64 - s)); }

// Keccak-f[1600] (FIPS 202), state as 25 little-endian lanes A[x + 5 y].
static void keccak_f(u64 *A) {
    for (int round = 0; round < 24; ++round) {
        u64 C0 = A[0] ^ A[5] ^ A[10] ^ A[15] ^ A[20];
        u64 C1 = A[1] ^ A[6] ^ A[11] ^ A[16] ^ A[21];
        u64 C2 = A[2] ^ A[7] ^ A[12] ^ A[17] ^ A[22];
        u64 C3 = A[3] ^ A[8] ^ A[13] ^ A[18] ^ A[23];
        u64 C4 = A[4] ^ A[9] ^ A[14] ^ A[19] ^ A[24];
        u64 D0 = C4 ^ rotl(C1, 1), D1 = C0 ^ rotl(C2, 1), D2 = C1 ^ rotl(C3, 1);
        u64 D3 = C2 ^ rotl(C4, 1), D4 = C3 ^ rotl(C0, 1);
        // theta, rho and pi: B[y, 2x + 3y] = rot(A[x, y] ^ D[x], r[x, y])
        u64 B[25];
        B[0] = A[0] ^ D0;
        B[10] = rotl(A[1] ^ D1, 1);
        B[20] = rotl(A[2] ^ D2, 62);
        B[5] = rotl(A[3] ^ D3, 28);
        B[15] = rotl(A[4] ^ D4, 27);
        B[16] = rotl(A[5] ^ D0, 36);
        B[1] = rotl(A[6] ^ D1, 44);
        B[11] = rotl(A[7] ^ D2, 6);
        B[21] = rotl(A[8] ^ D3, 55);
        B[6] = rotl(A[9] ^ D4, 20);
        B[7] = rotl(A[10] ^ D0, 3);
        B[17] = rotl(A[11] ^ D1, 10);
        B[2] = rotl(A[12] ^ D2, 43);
        B[12] = rotl(A[13] ^ D3, 25);
        B[22] = rotl(A[14] ^ D4, 39);
        B[23] = rotl(A[15] ^ D0, 41);
        B[8] = rotl(A[16] ^ D1, 45);
        B[18] = rotl(A[17] ^ D2, 15);
        B[3] = rotl(A[18] ^ D3, 21);
        B[13] = rotl(A[19] ^ D4, 8);
        B[14] = rotl(A[20] ^ D0, 18);
        B[24] = rotl(A[21] ^ D1, 2);
        B[9] = rotl(A[22] ^ D2, 61);
        B[19] = rotl(A[23] ^ D3, 56);
        B[4] = rotl(A[24] ^ D4, 14);
        // chi
        for (int y = 0; y < 25; y += 5) {
            u64 b0 = B[y], b1 = B[y + 1], b2 = B[y + 2], b3 = B[y + 3], b4 = B[y + 4];
            A[y] = b0 ^ (~b1 & b2);
            A[y + 1] = b1 ^ (~b2 & b3);
            A[y + 2] = b2 ^ (~b3 & b4);
            A[y + 3] = b3 ^ (~b4 & b0);
            A[y + 4] = b4 ^ (~b0 & b1);
        }
        A[0] ^= KECCAK_RC[round];  // iota
    }
}

static inline u64 load_le(const unsigned char *p) {
    u64 v = 0;
    for (int i = 7; i >= 0; --i) v = (v << 8) | p[i];
    return v;
}

static inline void store_le(u64 v, unsigned char *p) {
    for (int i = 0; i < 8; ++i) { p[i] = (unsigned char)v; v >>= 8; }
}

// SHAKE-256(msg) truncated to out_len bytes.
static void shake256(const unsigned char *msg, size_t len, unsigned char *out, size_t out_len) {
    const size_t RATE = 136;
    u64 A[25] = {0};
    while (len >= RATE) {
        for (size_t i = 0; i < RATE / 8; ++i) A[i] ^= load_le(msg + 8 * i);
        keccak_f(A);
        msg += RATE;
        len -= RATE;
    }
    unsigned char block[136] = {0};
    memcpy(block, msg, len);
    block[len] ^= 0x1F;
    block[RATE - 1] ^= 0x80;
    for (size_t i = 0; i < RATE / 8; ++i) A[i] ^= load_le(block + 8 * i);
    for (;;) {
        keccak_f(A);
        size_t take = out_len < RATE ? out_len : RATE;
        if (take == RATE) {
            for (size_t i = 0; i < RATE / 8; ++i) store_le(A[i], out + 8 * i);
        } else {
            unsigned char tmp[136];
            for (size_t i = 0; i < RATE / 8; ++i) store_le(A[i], tmp + 8 * i);
            memcpy(out, tmp, take);
        }
        out += take;
        out_len -= take;
        if (!out_len) break;
    }
}

// hash_to_zq_vector's block size: bits(q) + 128 bits.
static inline size_t zq_block(const CtxBase &c) { return (size_t)(c.qbits + 128 + 7) / 8; }

// hash_to_zq_vector after the SHAKE call: big-endian block k (nb bytes) mod q, as
// sum_k limb_k (2^(BITS k) mod q) with one Montgomery multiplication per word-sized limb
// (limb < R = 2^BITS, so each product is reduced mod q).
template <class A>
static void zq_reduce(const Ctx<A> &c, const unsigned char *s, size_t nb, size_t count, typename A::W *out) {
    typedef typename A::W W;
    const size_t LB = sizeof(W);
    const size_t limbs = (nb + LB - 1) / LB, first = nb % LB ? nb % LB : LB;
    W shift = (W)(A::BITS == 64 ? (((u128)1 << 64) % c.q) : ((((u128)-1) % c.q + 1) % c.q));  // R mod q
    std::vector<W> cm(limbs);  // Montgomery form of 2^(BITS k) mod q, k = 0 .. limbs - 1
    W pw = 1;
    for (size_t k = 0; k < limbs; ++k) {
        cm[k] = c.to_mont(pw);
        pw = c.mulp(pw, shift);
    }
    for (size_t b = 0; b < count; ++b, s += nb) {
        W acc = 0;
        const unsigned char *p = s;
        for (size_t k = limbs; k-- > 0;) {
            size_t take = k == limbs - 1 ? first : LB;
            W limb = 0;
            for (size_t j = 0; j < take; ++j) limb = (limb << 8) | p[j];
            p += take;
            acc = c.ar.add(acc, c.ar.mul(limb, cm[k]));
        }
        out[b] = acc;
    }
}

// ================================================================ capsule helpers

static bool buffer_words(PyObject *o, size_t count, size_t wsize, Py_buffer *view) {
    if (PyObject_GetBuffer(o, view, PyBUF_SIMPLE) < 0) return false;
    if ((size_t)view->len != count * wsize) {
        PyBuffer_Release(view);
        PyErr_Format(PyExc_ValueError, "expected %zu bytes of words", count * wsize);
        return false;
    }
    return true;
}

template <class W>
static PyObject *words_bytes(const W *v, size_t n) {
    return PyBytes_FromStringAndSize((const char *)v, (Py_ssize_t)(n * sizeof(W)));
}

// =============================================================== module: context

// ctx_new(q, n, root) -> capsule; root is ring_lwe_utils._get_ntt_root(q, n).
static PyObject *py_ctx_new(PyObject *, PyObject *args) {
    PyObject *qo, *rooto;
    Py_ssize_t n;
    if (!PyArg_ParseTuple(args, "OnO", &qo, &n, &rooto)) return nullptr;
    u128 q, root;
    PyObject *qi = PyNumber_Index(qo);
    if (!qi) return nullptr;
    if (!pylong_u128(qi, &q)) { Py_DECREF(qi); return nullptr; }
    PyObject *ri = PyNumber_Index(rooto);
    if (!ri || !pylong_u128(ri, &root)) { Py_XDECREF(ri); Py_DECREF(qi); return nullptr; }
    Py_DECREF(ri);
    if (n < 2 || (n & (n - 1)) || n > (1 << 24) || q < 3 || !(q & 1) || q >= ((u128)1 << 125)
        || root >= q || (q - 1) % (u128)(2 * n) != 0) {
        Py_DECREF(qi);
        PyErr_SetString(PyExc_ValueError, "unsupported (q, n): need q < 2^125 prime, q = 1 mod 2n");
        return nullptr;
    }
    int qbits = 0;
    for (u128 t = q; t; t >>= 1) ++qbits;
    CtxBase *base;
    bool ok;
    auto init = [&](auto *c) {
        typedef typename std::remove_pointer<decltype(c)>::type C;
        typedef typename C::W W;
        c->n = (size_t)n;
        while (((size_t)1 << c->logn) < c->n) ++c->logn;
        c->q = q;
        c->qbits = qbits;
        c->ar.q = (W)q;
        // qinv = -q^-1 mod 2^BITS by Newton's iteration
        W inv = (W)q;
        for (int i = 0; i < 7; ++i) inv *= (W)2 - (W)q * inv;
        c->ar.qinv = (W)0 - inv;
        // R mod q and R^2 mod q
        W rm;
        if (decltype(c->ar)::BITS == 64) rm = (W)(((u128)1 << 64) % q);
        else rm = (W)((((u128)-1) % q + 1) % q);
        c->rmod = rm;
        W r2 = rm;
        for (int i = 0; i < decltype(c->ar)::BITS; ++i) r2 = c->ar.add(r2, r2);  // R * 2^BITS = R^2 mod q
        c->r2 = r2;
        return c->build(root);
    };
    if (q < ((u128)1 << 62)) {
        Ctx64 *c = new Ctx64();
        c->bits = 64;
        base = c;
        ok = init(c);
    } else {
        Ctx128 *c = new Ctx128();
        c->bits = 128;
        base = c;
        ok = init(c);
    }
    if (!ok) {
        delete base;
        Py_DECREF(qi);
        PyErr_SetString(PyExc_ValueError, "root is not a primitive 2n-th root of unity mod q");
        return nullptr;
    }
    base->qobj = qi;  // owns the reference
    PyObject *cap = PyCapsule_New(base, CTX_NAME, ctx_free);
    if (!cap) delete base;
    return cap;
}

static PyObject *py_ctx_info(PyObject *, PyObject *args) {
    PyObject *cap;
    if (!PyArg_ParseTuple(args, "O", &cap)) return nullptr;
    CtxBase *c = get_ctx(cap);
    if (!c) return nullptr;
    return Py_BuildValue("{s:i,s:n,s:i}", "word_bits", c->bits, "n", (Py_ssize_t)c->n, "q_bits", c->qbits);
}

// pack(ctx, values) -> bytes: values (ints, mpz or decimal strings) mod q as words.
static PyObject *py_pack(PyObject *, PyObject *args) {
    PyObject *cap, *vals;
    if (!PyArg_ParseTuple(args, "OO", &cap, &vals)) return nullptr;
    CtxBase *c = get_ctx(cap);
    if (!c) return nullptr;
    Py_ssize_t len = PySequence_Size(vals);
    if (len < 0) return nullptr;
    return with_ctx(c, [&](auto &cc) -> PyObject * {
        typedef typename std::remove_reference<decltype(cc)>::type::W W;
        std::vector<W> w((size_t)len);
        if (!read_words<W>(vals, (size_t)len, cc.qobj, cc.q, w.data(), false)) return nullptr;
        return words_bytes(w.data(), w.size());
    });
}

// unpack(ctx, data, as_digits=False) -> list[int] or list[str]
static PyObject *py_unpack(PyObject *, PyObject *args) {
    PyObject *cap;
    Py_buffer buf;
    int digits = 0;
    if (!PyArg_ParseTuple(args, "Oy*|p", &cap, &buf, &digits)) return nullptr;
    CtxBase *c = get_ctx(cap);
    PyObject *res = nullptr;
    if (c) {
        res = with_ctx(c, [&](auto &cc) -> PyObject * {
            typedef typename std::remove_reference<decltype(cc)>::type::W W;
            if (buf.len % (Py_ssize_t)sizeof(W)) {
                PyErr_SetString(PyExc_ValueError, "length is not a multiple of the word size");
                return nullptr;
            }
            size_t n = (size_t)buf.len / sizeof(W);
            const W *v = (const W *)buf.buf;
            if (digits) return dec_list(v, n);
            PyObject *lst = PyList_New((Py_ssize_t)n);
            if (!lst) return nullptr;
            for (size_t i = 0; i < n; ++i) {
                PyObject *o = u128_pylong((u128)v[i]);
                if (!o) { Py_DECREF(lst); return nullptr; }
                PyList_SET_ITEM(lst, (Py_ssize_t)i, o);
            }
            return lst;
        });
    }
    PyBuffer_Release(&buf);
    return res;
}

// ntt(ctx, data, inverse=False) -> bytes (plain values in [0, q) both ways)
static PyObject *py_ntt(PyObject *, PyObject *args) {
    PyObject *cap, *data;
    int inverse = 0;
    if (!PyArg_ParseTuple(args, "OO|p", &cap, &data, &inverse)) return nullptr;
    CtxBase *c = get_ctx(cap);
    if (!c) return nullptr;
    return with_ctx(c, [&](auto &cc) -> PyObject * {
        typedef typename std::remove_reference<decltype(cc)>::type::W W;
        Py_buffer view;
        if (!buffer_words(data, cc.n, sizeof(W), &view)) return nullptr;
        std::vector<W> out(cc.n);
        if (inverse) cc.inverse((const W *)view.buf, out.data());
        else cc.forward((const W *)view.buf, out.data());
        PyBuffer_Release(&view);
        return words_bytes(out.data(), cc.n);
    });
}

// ntt_mont(ctx, data) -> bytes: NTT in Montgomery form, the key format of encryption / decryption.
static PyObject *py_ntt_mont(PyObject *, PyObject *args) {
    PyObject *cap, *data;
    if (!PyArg_ParseTuple(args, "OO", &cap, &data)) return nullptr;
    CtxBase *c = get_ctx(cap);
    if (!c) return nullptr;
    return with_ctx(c, [&](auto &cc) -> PyObject * {
        typedef typename std::remove_reference<decltype(cc)>::type::W W;
        Py_buffer view;
        if (!buffer_words(data, cc.n, sizeof(W), &view)) return nullptr;
        std::vector<W> out(cc.n);
        cc.forward((const W *)view.buf, out.data());
        PyBuffer_Release(&view);
        for (auto &v : out) v = cc.to_mont(v);
        return words_bytes(out.data(), cc.n);
    });
}

// poly_mul(ctx, a, b) -> bytes, the negacyclic product (for tests)
static PyObject *py_poly_mul(PyObject *, PyObject *args) {
    PyObject *cap, *a, *b;
    if (!PyArg_ParseTuple(args, "OOO", &cap, &a, &b)) return nullptr;
    CtxBase *c = get_ctx(cap);
    if (!c) return nullptr;
    return with_ctx(c, [&](auto &cc) -> PyObject * {
        typedef typename std::remove_reference<decltype(cc)>::type::W W;
        Py_buffer va, vb;
        if (!buffer_words(a, cc.n, sizeof(W), &va)) return nullptr;
        if (!buffer_words(b, cc.n, sizeof(W), &vb)) { PyBuffer_Release(&va); return nullptr; }
        std::vector<W> fa(cc.n), fb(cc.n), out(cc.n);
        cc.forward((const W *)va.buf, fa.data());
        cc.forward((const W *)vb.buf, fb.data());
        PyBuffer_Release(&va);
        PyBuffer_Release(&vb);
        for (size_t i = 0; i < cc.n; ++i) fa[i] = cc.mulp(fa[i], fb[i]);
        cc.inverse(fa.data(), out.data());
        return words_bytes(out.data(), cc.n);
    });
}

// lincomb(ctx, rows, coeffs) -> list[str]: sum_i coeffs[i] rows[i] mod q as decimal strings
// (rows: packed words; coeffs: integers, reduced mod q).
static PyObject *py_lincomb(PyObject *, PyObject *args) {
    PyObject *cap, *rows, *coeffs;
    if (!PyArg_ParseTuple(args, "OOO", &cap, &rows, &coeffs)) return nullptr;
    CtxBase *c = get_ctx(cap);
    if (!c) return nullptr;
    return with_ctx(c, [&](auto &cc) -> PyObject * {
        typedef typename std::remove_reference<decltype(cc)>::type::W W;
        Py_ssize_t m = PySequence_Size(rows);
        if (m < 0) return nullptr;
        if (PySequence_Size(coeffs) != m) {
            PyErr_SetString(PyExc_ValueError, "one coefficient per row");
            return nullptr;
        }
        std::vector<W> acc(cc.n, 0), y((size_t)m);
        if (m && !read_words<W>(coeffs, (size_t)m, cc.qobj, cc.q, y.data(), false)) return nullptr;
        for (Py_ssize_t r = 0; r < m; ++r) {
            PyObject *row = PySequence_GetItem(rows, r);
            if (!row) return nullptr;
            Py_buffer view;
            bool ok = buffer_words(row, cc.n, sizeof(W), &view);
            Py_DECREF(row);
            if (!ok) return nullptr;
            const W *v = (const W *)view.buf;
            W ym = cc.to_mont(y[(size_t)r]);
            for (size_t i = 0; i < cc.n; ++i) acc[i] = cc.ar.add(acc[i], cc.ar.mul(v[i], ym));
            PyBuffer_Release(&view);
        }
        return dec_list(acc.data(), cc.n);
    });
}

// ================================================================ module: kernels

// gauss(random_bytes | None, count, sigma) -> list[int]
static PyObject *py_gauss(PyObject *, PyObject *args) {
    PyObject *rnd;
    Py_ssize_t count;
    double sigma;
    if (!PyArg_ParseTuple(args, "Ond", &rnd, &count, &sigma)) return nullptr;
    if (count < 0) { PyErr_SetString(PyExc_ValueError, "count < 0"); return nullptr; }
    size_t need = gauss_bytes((size_t)count);
    std::vector<unsigned char> own;
    Py_buffer view;
    const unsigned char *src;
    bool have_view = false;
    if (rnd == Py_None) {
        own.resize(need);
        if (!os_random(own.data(), need)) { PyErr_SetFromErrno(PyExc_OSError); return nullptr; }
        src = own.data();
    } else {
        if (PyObject_GetBuffer(rnd, &view, PyBUF_SIMPLE) < 0) return nullptr;
        have_view = true;
        if ((size_t)view.len < need) {
            PyBuffer_Release(&view);
            PyErr_SetString(PyExc_ValueError, "not enough random bytes");
            return nullptr;
        }
        src = (const unsigned char *)view.buf;
    }
    std::vector<int64_t> s((size_t)count);
    gauss_fill(src, (size_t)count, sigma, s.data());
    if (have_view) PyBuffer_Release(&view);
    PyObject *lst = PyList_New(count);
    if (!lst) return nullptr;
    for (Py_ssize_t i = 0; i < count; ++i) {
        PyObject *o = PyLong_FromLongLong(s[(size_t)i]);
        if (!o) { Py_DECREF(lst); return nullptr; }
        PyList_SET_ITEM(lst, i, o);
    }
    return lst;
}

// shake256(data, length) -> bytes
static PyObject *py_shake256(PyObject *, PyObject *args) {
    Py_buffer msg;
    Py_ssize_t length;
    if (!PyArg_ParseTuple(args, "y*n", &msg, &length)) return nullptr;
    if (length < 0) { PyBuffer_Release(&msg); PyErr_SetString(PyExc_ValueError, "length < 0"); return nullptr; }
    PyObject *out = PyBytes_FromStringAndSize(nullptr, length);
    if (out) {
        unsigned char *o = (unsigned char *)PyBytes_AS_STRING(out);
        Py_BEGIN_ALLOW_THREADS
        shake256((const unsigned char *)msg.buf, (size_t)msg.len, o, (size_t)length);
        Py_END_ALLOW_THREADS
    }
    PyBuffer_Release(&msg);
    return out;
}

// hash_to_zq(ctx, material, count) -> bytes: crypto_utils.hash_to_zq_vector of the
// already encoded material ("domain|zq-vector|len|v" in UTF-8), as words.
static PyObject *py_hash_to_zq(PyObject *, PyObject *args) {
    PyObject *cap;
    Py_buffer mat;
    Py_ssize_t count;
    if (!PyArg_ParseTuple(args, "Oy*n", &cap, &mat, &count)) return nullptr;
    CtxBase *c = get_ctx(cap);
    PyObject *res = nullptr;
    if (c && count >= 0) {
        res = with_ctx(c, [&](auto &cc) -> PyObject * {
            typedef typename std::remove_reference<decltype(cc)>::type::W W;
            size_t nb = zq_block(cc);
            std::vector<unsigned char> stream(nb * (size_t)count);
            std::vector<W> out((size_t)count);
            Py_BEGIN_ALLOW_THREADS
            shake256((const unsigned char *)mat.buf, (size_t)mat.len, stream.data(), stream.size());
            zq_reduce(cc, stream.data(), nb, (size_t)count, out.data());
            Py_END_ALLOW_THREADS
            return words_bytes(out.data(), out.size());
        });
    } else if (c) {
        PyErr_SetString(PyExc_ValueError, "count < 0");
    }
    PyBuffer_Release(&mat);
    return res;
}

static PyObject *py_random_bytes(PyObject *, PyObject *args) {
    Py_ssize_t n;
    if (!PyArg_ParseTuple(args, "n", &n)) return nullptr;
    if (n < 0) { PyErr_SetString(PyExc_ValueError, "n < 0"); return nullptr; }
    PyObject *out = PyBytes_FromStringAndSize(nullptr, n);
    if (!out) return nullptr;
    bool ok;
    Py_BEGIN_ALLOW_THREADS
    ok = os_random((unsigned char *)PyBytes_AS_STRING(out), (size_t)n);
    Py_END_ALLOW_THREADS
    if (!ok) {
        Py_DECREF(out);
        if (errno) PyErr_SetFromErrno(PyExc_OSError);
        else PyErr_SetString(PyExc_OSError, "no OS random source in this build");
        return nullptr;
    }
    return out;
}

// ============================================================ encryption key

// Key of one MIFE / MCFE slot: NTT(a) and NTT(pk_row) in Montgomery form, the pads
// u_row, Delta = floor(q / p).
struct EncKeyBase {
    PyObject *ctx_cap = nullptr;
    size_t eta = 0;
    virtual ~EncKeyBase() { Py_XDECREF(ctx_cap); }
};

template <class A>
struct EncKey : EncKeyBase {
    typedef typename A::W W;
    const Ctx<A> *c = nullptr;
    std::vector<W> a_hat, pk_hat, u;  // pk_hat, u: eta rows of n
    W delta = 0, delta_m = 0;
    bool has_u = false;
};

static void key_free(PyObject *cap) { delete (EncKeyBase *)PyCapsule_GetPointer(cap, KEY_NAME); }

// enc_key(ctx, a, pk_rows, u_rows | None, delta) -> capsule; a, rows: integer / decimal sequences.
static PyObject *py_enc_key(PyObject *, PyObject *args) {
    PyObject *cap, *a, *pk, *u, *delta_o;
    if (!PyArg_ParseTuple(args, "OOOOO", &cap, &a, &pk, &u, &delta_o)) return nullptr;
    CtxBase *c = get_ctx(cap);
    if (!c) return nullptr;
    Py_ssize_t eta = PySequence_Size(pk);
    if (eta < 1) {
        if (!PyErr_Occurred()) PyErr_SetString(PyExc_ValueError, "pk needs at least one row");
        return nullptr;
    }
    if (u != Py_None && PySequence_Size(u) != eta) {
        if (!PyErr_Occurred()) PyErr_SetString(PyExc_ValueError, "one pad row per pk row");
        return nullptr;
    }
    EncKeyBase *kb = with_ctx(c, [&](auto &cc) -> EncKeyBase * {
        typedef typename std::remove_reference<decltype(cc)>::type CC;
        typedef typename CC::W W;
        typedef decltype(cc.ar) A;
        std::unique_ptr<EncKey<A>> k(new EncKey<A>());
        size_t n = cc.n;
        k->c = &cc;
        k->eta = (size_t)eta;
        std::vector<W> tmp(n);
        if (!read_words<W>(a, n, cc.qobj, cc.q, tmp.data(), false)) return nullptr;
        k->a_hat.resize(n);
        cc.forward(tmp.data(), k->a_hat.data());
        for (auto &v : k->a_hat) v = cc.to_mont(v);
        k->pk_hat.resize(n * (size_t)eta);
        for (Py_ssize_t r = 0; r < eta; ++r) {
            PyObject *row = PySequence_GetItem(pk, r);
            if (!row) return nullptr;
            bool ok = read_words<W>(row, n, cc.qobj, cc.q, tmp.data(), false);
            Py_DECREF(row);
            if (!ok) return nullptr;
            W *dst = k->pk_hat.data() + n * (size_t)r;
            cc.forward(tmp.data(), dst);
            for (size_t i = 0; i < n; ++i) dst[i] = cc.to_mont(dst[i]);
        }
        if (u != Py_None) {
            k->has_u = true;
            k->u.resize(n * (size_t)eta);
            for (Py_ssize_t r = 0; r < eta; ++r) {
                PyObject *row = PySequence_GetItem(u, r);
                if (!row) return nullptr;
                bool ok = read_words<W>(row, n, cc.qobj, cc.q, k->u.data() + n * (size_t)r, false);
                Py_DECREF(row);
                if (!ok) return nullptr;
            }
        }
        u128 d;
        if (!index_mod(delta_o, cc.qobj, &d)) return nullptr;
        k->delta = (W)d;
        k->delta_m = cc.to_mont((W)d);
        return k.release();
    });
    if (!kb) return nullptr;
    kb->ctx_cap = cap;
    Py_INCREF(cap);
    PyObject *res = PyCapsule_New(kb, KEY_NAME, key_free);
    if (!res) delete kb;
    return res;
}

// One chunk of BMMS21 encryption: r <- D(sigma2); ct0_row = pk_row r + e_row + Delta w_row
// with e_row <- D(sigma3); ct1 = a r + e with e <- D(sigma2). Random bytes are consumed
// for r, the e_row, e in that order (MIFERingLWE._encrypt_rows).
template <class A>
static void encrypt_chunk(const EncKey<A> &k, const typename A::W *w, double sigma2, double sigma3,
                          const unsigned char *rnd, typename A::W *ct0, typename A::W *ct1) {
    typedef typename A::W W;
    const Ctx<A> &c = *k.c;
    size_t n = c.n;
    u128 q = c.q;
    std::vector<int64_t> smp(n);
    std::vector<W> r(n), rh(n), t(n);
    gauss_fill(rnd, n, sigma2, smp.data());
    rnd += gauss_bytes(n);
    for (size_t i = 0; i < n; ++i) r[i] = (W)reduce_i128(smp[i], q);
    c.forward(r.data(), rh.data());
    for (size_t row = 0; row < k.eta; ++row) {
        const W *pkh = k.pk_hat.data() + n * row;
        for (size_t i = 0; i < n; ++i) t[i] = c.ar.mul(pkh[i], rh[i]);
        W *out = ct0 + n * row;
        c.inverse(t.data(), out);
        gauss_fill(rnd, n, sigma3, smp.data());
        rnd += gauss_bytes(n);
        const W *wr = w + n * row;
        for (size_t i = 0; i < n; ++i)
            out[i] = c.ar.add(c.ar.add(out[i], (W)reduce_i128(smp[i], q)), c.ar.mul(wr[i], k.delta_m));
    }
    for (size_t i = 0; i < n; ++i) t[i] = c.ar.mul(k.a_hat[i], rh[i]);
    c.inverse(t.data(), ct1);
    gauss_fill(rnd, n, sigma2, smp.data());
    for (size_t i = 0; i < n; ++i) ct1[i] = c.ar.add(ct1[i], (W)reduce_i128(smp[i], q));
}

static EncKeyBase *get_key(PyObject *cap) { return (EncKeyBase *)PyCapsule_GetPointer(cap, KEY_NAME); }

template <class F>
static PyObject *with_key(PyObject *kcap, F &&f) {
    EncKeyBase *kb = get_key(kcap);
    if (!kb) return nullptr;
    CtxBase *c = get_ctx(kb->ctx_cap);
    if (!c) return nullptr;
    if (c->bits == 64) return f(*static_cast<EncKey<Arith64> *>(kb));
    return f(*static_cast<EncKey<Arith128> *>(kb));
}

// Random bytes for `chunks` chunks of `per` bytes: the given buffer, or the OS source.
struct RandomSource {
    Py_buffer view;
    bool have_view = false;
    std::vector<unsigned char> own;
    const unsigned char *data = nullptr;
    bool from_os = false;
    ~RandomSource() { if (have_view) PyBuffer_Release(&view); }
    bool init(PyObject *rnd, size_t need) {
        if (rnd == Py_None) {
            if (!have_os_random()) {
                PyErr_SetString(PyExc_OSError, "no OS random source in this build; pass random bytes");
                return false;
            }
            own.resize(need);
            data = own.data();
            from_os = true;
            return true;
        }
        if (PyObject_GetBuffer(rnd, &view, PyBUF_SIMPLE) < 0) return false;
        have_view = true;
        if ((size_t)view.len < need) {
            PyErr_SetString(PyExc_ValueError, "not enough random bytes");
            return false;
        }
        data = (const unsigned char *)view.buf;
        return true;
    }
};

// Binary ciphertext format 1 (ring_lwe_utils.encode_binary_ciphertext):
// "PFRL", version 1, word bytes wb = ceil(bits(q) / 8), eta (u16 LE), log2 n, 0;
// then the eta rows of ct0 and ct1, n coefficients each, wb bytes little-endian.
static const size_t BIN_HEADER = 10;

static inline size_t bin_size(size_t eta, size_t n, size_t wb) { return BIN_HEADER + (eta + 1) * n * wb; }
static inline size_t bin_wb(const CtxBase &c) { return (size_t)(c.qbits + 7) / 8; }

template <class W>
static void bin_write(const CtxBase &c, size_t eta, const W *ct0, const W *ct1, unsigned char *out) {
    size_t n = c.n, wb = bin_wb(c);
    memcpy(out, "PFRL", 4);
    out[4] = 1;
    out[5] = (unsigned char)wb;
    out[6] = (unsigned char)(eta & 0xFF);
    out[7] = (unsigned char)(eta >> 8);
    out[8] = (unsigned char)c.logn;
    out[9] = 0;
    unsigned char *p = out + BIN_HEADER;
    for (size_t i = 0; i < (eta + 1) * n; ++i) {
        u128 v = (u128)(i < eta * n ? ct0[i] : ct1[i - eta * n]);
        for (size_t b = 0; b < wb; ++b) { *p++ = (unsigned char)v; v >>= 8; }
    }
}

// Coefficients of a format-1 ciphertext, reduced mod q (as int(v) % q); false if the
// header or the length does not match (eta, n, q).
template <class W>
static bool bin_read(const CtxBase &c, size_t eta, const unsigned char *in, size_t len, W *ct0, W *ct1) {
    size_t n = c.n, wb = bin_wb(c);
    if (len != bin_size(eta, n, wb) || memcmp(in, "PFRL", 4) || in[4] != 1 || in[5] != wb
        || (size_t)(in[6] | (in[7] << 8)) != eta || in[8] != c.logn || in[9] != 0)
        return false;
    const unsigned char *p = in + BIN_HEADER;
    for (size_t i = 0; i < (eta + 1) * n; ++i) {
        u128 v = 0;
        for (size_t b = wb; b-- > 0;) v = (v << 8) | p[b];
        p += wb;
        if (v >= c.q) v %= c.q;
        if (i < eta * n) ct0[i] = (W)v;
        else ct1[i - eta * n] = (W)v;
    }
    return true;
}

// encrypt(key, w_rows, sigma2, sigma3, random_bytes | None, binary=False) -> (ct0, ct1) | bytes
// MIFERingLWE._encrypt_rows for rows w (eta rows of n integers already reduced mod q):
// ct0 as eta lists and ct1 as one list of decimal strings.
static PyObject *py_encrypt(PyObject *, PyObject *args) {
    PyObject *kcap, *w_rows, *rnd;
    double s2, s3;
    int binary = 0;
    if (!PyArg_ParseTuple(args, "OOddO|p", &kcap, &w_rows, &s2, &s3, &rnd, &binary)) return nullptr;
    return with_key(kcap, [&](auto &k) -> PyObject * {
        typedef typename std::remove_reference<decltype(k)>::type K;
        typedef typename K::W W;
        const auto &c = *k.c;
        size_t n = c.n, eta = k.eta;
        if (PySequence_Size(w_rows) != (Py_ssize_t)eta) {
            if (!PyErr_Occurred()) PyErr_SetString(PyExc_ValueError, "one plaintext row per pk row");
            return nullptr;
        }
        std::vector<W> w(n * eta), ct0(n * eta), ct1(n);
        for (size_t r = 0; r < eta; ++r) {
            PyObject *row = PySequence_GetItem(w_rows, (Py_ssize_t)r);
            if (!row) return nullptr;
            bool ok = read_words<W>(row, n, c.qobj, c.q, w.data() + n * r, false);
            Py_DECREF(row);
            if (!ok) return nullptr;
        }
        size_t need = gauss_bytes(n) * (2 + eta);
        RandomSource src;
        if (!src.init(rnd, need)) return nullptr;
        bool ok = true;
        Py_BEGIN_ALLOW_THREADS
        if (src.from_os) ok = os_random(src.own.data(), need);
        if (ok) encrypt_chunk(k, w.data(), s2, s3, src.data, ct0.data(), ct1.data());
        Py_END_ALLOW_THREADS
        if (!ok) return PyErr_SetFromErrno(PyExc_OSError);
        if (binary) {
            PyObject *b = PyBytes_FromStringAndSize(nullptr, (Py_ssize_t)bin_size(eta, n, bin_wb(c)));
            if (!b) return nullptr;
            bin_write(c, eta, ct0.data(), ct1.data(), (unsigned char *)PyBytes_AS_STRING(b));
            return b;
        }
        PyObject *rows = PyList_New((Py_ssize_t)eta);
        if (!rows) return nullptr;
        for (size_t r = 0; r < eta; ++r) {
            PyObject *lst = dec_list(ct0.data() + n * r, n);
            if (!lst) { Py_DECREF(rows); return nullptr; }
            PyList_SET_ITEM(rows, (Py_ssize_t)r, lst);
        }
        PyObject *c1 = dec_list(ct1.data(), n);
        if (!c1) { Py_DECREF(rows); return nullptr; }
        return Py_BuildValue("(NN)", rows, c1);
    });
}

// Decimal strings of one chunk, formatted without the GIL.
struct DecChunk {
    std::vector<char> buf;
    std::vector<uint32_t> off;
    template <class W>
    void format(const W *v, size_t count) {
        buf.resize(count * 40);
        off.resize(count + 1);
        size_t pos = 0;
        for (size_t i = 0; i < count; ++i) {
            off[i] = (uint32_t)pos;
            pos += format_dec((u128)v[i], buf.data() + pos);
        }
        off[count] = (uint32_t)pos;
    }
};

// encrypt_tensor(key, x, pad_materials | None, sigma2, sigma3, random_bytes | None, threads)
//   -> list of (ct0, ct1, k), one per chunk of n values; with binary=True (data, k), data in
//   the binary format 1
// The MCFE ndarray helper (eta = 1) for one layer: x is an int64 buffer of L > 0 values,
// chunk c holds x[c n : (c + 1) n] (k values, zero-extended) and is encrypted as
// w = x + u + H_c mod q, H_c = hash_to_zq_vector of pad_materials[c] (n values).
// encrypt_tensor(key, x, pad_materials | None, sigma2, sigma3, random_bytes | None, threads, binary=False)
static PyObject *py_encrypt_tensor(PyObject *, PyObject *args) {
    PyObject *kcap, *xo, *mats, *rnd;
    double s2, s3;
    int threads, binary = 0;
    if (!PyArg_ParseTuple(args, "OOOddOi|p", &kcap, &xo, &mats, &s2, &s3, &rnd, &threads, &binary)) return nullptr;
    return with_key(kcap, [&](auto &k) -> PyObject * {
        typedef typename std::remove_reference<decltype(k)>::type K;
        typedef typename K::W W;
        const auto &c = *k.c;
        size_t n = c.n;
        if (k.eta != 1 || !k.has_u) {
            PyErr_SetString(PyExc_ValueError, "encrypt_tensor needs eta = 1 and the pad u");
            return nullptr;
        }
        Py_buffer xv;
        if (PyObject_GetBuffer(xo, &xv, PyBUF_SIMPLE) < 0) return nullptr;
        std::unique_ptr<Py_buffer, void (*)(Py_buffer *)> xguard(&xv, PyBuffer_Release);
        if (xv.len % 8 || xv.len == 0) {
            PyErr_SetString(PyExc_ValueError, "x must be a non-empty int64 buffer");
            return nullptr;
        }
        size_t L = (size_t)xv.len / 8;
        const int64_t *x = (const int64_t *)xv.buf;
        size_t chunks = (L + n - 1) / n;
        // pad materials (bytes), kept alive by the list
        std::vector<std::pair<const unsigned char *, size_t>> mat;
        PyObject *mseq = nullptr;
        if (mats != Py_None) {
            mseq = PySequence_Fast(mats, "pad materials must be a sequence of bytes");
            if (!mseq) return nullptr;
            if ((size_t)PySequence_Fast_GET_SIZE(mseq) != chunks) {
                Py_DECREF(mseq);
                PyErr_SetString(PyExc_ValueError, "one pad material per chunk");
                return nullptr;
            }
            for (size_t j = 0; j < chunks; ++j) {
                PyObject *m = PySequence_Fast_GET_ITEM(mseq, j);
                if (!PyBytes_Check(m)) {
                    Py_DECREF(mseq);
                    PyErr_SetString(PyExc_TypeError, "pad materials must be bytes");
                    return nullptr;
                }
                mat.emplace_back((const unsigned char *)PyBytes_AS_STRING(m), (size_t)PyBytes_GET_SIZE(m));
            }
        }
        size_t per = gauss_bytes(n) * 3;
        RandomSource src;
        if (!src.init(rnd, per * chunks)) { Py_XDECREF(mseq); return nullptr; }
        std::vector<W> ct((size_t)2 * n * chunks);
        std::vector<DecChunk> dec(2 * chunks);
        std::atomic<bool> rnd_fail(false);
        size_t nb = zq_block(c);
        Py_BEGIN_ALLOW_THREADS
        parallel_for(chunks, threads, [&](size_t j) {
            const unsigned char *rb = src.data + per * j;
            if (src.from_os && !os_random(src.own.data() + per * j, per)) {
                rnd_fail = true;
                return;
            }
            size_t k0 = j * n, kk = std::min(n, L - k0);
            std::vector<W> w(n);
            if (!mat.empty()) {
                std::vector<unsigned char> stream(nb * n);
                shake256(mat[j].first, mat[j].second, stream.data(), stream.size());
                zq_reduce(c, stream.data(), nb, n, w.data());
            } else {
                std::fill(w.begin(), w.end(), 0);
            }
            for (size_t i = 0; i < n; ++i) {
                W xi = i < kk ? (W)reduce_i128(x[k0 + i], c.q) : 0;
                w[i] = c.ar.add(c.ar.add(xi, k.u[i]), w[i]);
            }
            W *c0 = ct.data() + 2 * n * j, *c1 = c0 + n;
            encrypt_chunk(k, w.data(), s2, s3, rb, c0, c1);
            if (binary) {
                dec[2 * j].buf.resize(bin_size(1, n, bin_wb(c)));
                bin_write(c, 1, c0, c1, (unsigned char *)dec[2 * j].buf.data());
            } else {
                dec[2 * j].format(c0, n);
                dec[2 * j + 1].format(c1, n);
            }
        });
        Py_END_ALLOW_THREADS
        Py_XDECREF(mseq);
        if (rnd_fail) return PyErr_SetFromErrno(PyExc_OSError);
        PyObject *out = PyList_New((Py_ssize_t)chunks);
        if (!out) return nullptr;
        for (size_t j = 0; binary && j < chunks; ++j) {
            PyObject *item = Py_BuildValue("(y#n)", dec[2 * j].buf.data(), (Py_ssize_t)dec[2 * j].buf.size(),
                                           (Py_ssize_t)std::min(n, L - j * n));
            if (!item) { Py_DECREF(out); return nullptr; }
            PyList_SET_ITEM(out, (Py_ssize_t)j, item);
        }
        for (size_t j = 0; !binary && j < chunks; ++j) {
            PyObject *row = dec_list_from(dec[2 * j].buf.data(), dec[2 * j].off.data(), n);
            PyObject *c1 = row ? dec_list_from(dec[2 * j + 1].buf.data(), dec[2 * j + 1].off.data(), n) : nullptr;
            PyObject *item = c1 ? Py_BuildValue("([N]Nn)", row, c1, (Py_ssize_t)std::min(n, L - j * n)) : nullptr;
            if (!item) {
                if (!c1) Py_XDECREF(row);
                Py_DECREF(out);
                return nullptr;
            }
            PyList_SET_ITEM(out, (Py_ssize_t)j, item);
        }
        return out;
    });
}

// ============================================================ offline / online encryption

// Everything of an MCFE ndarray chunk that does not depend on the plaintext:
// P = pk r + e_0 + Delta (u + H_c) and ct1 = a r + e (encrypt_chunk with x = 0), so that
// online ct0 = P + Delta x mod q -- the same values as encrypting x directly with the same
// samples. One-time: a chunk is wiped when it is used, and every chunk on destruction.
static const char *PRE_NAME = "pyfe4ai._rlwe_native.precomputed";

struct PreBase {
    const void *key = nullptr;    // the EncKey it was made with
    PyObject *key_cap = nullptr;  // ... kept alive (so the address stays its own)
    size_t chunks = 0, n = 0;
    std::vector<uint8_t> used;
    virtual ~PreBase() { Py_XDECREF(key_cap); }
};

template <class W>
struct Pre : PreBase {
    std::vector<W> data;  // per chunk: P (n words), then ct1 (n words)
    void wipe(size_t j) {
        volatile W *p = data.data() + 2 * n * j;
        for (size_t i = 0; i < 2 * n; ++i) p[i] = 0;
    }
    ~Pre() override {
        for (size_t j = 0; j < chunks; ++j) wipe(j);
    }
};

static void pre_free(PyObject *cap) { delete (PreBase *)PyCapsule_GetPointer(cap, PRE_NAME); }

// precompute_tensor(key, pad_materials, sigma2, sigma3, random_bytes | None, threads) -> capsule
// One chunk per pad material (eta = 1), random bytes consumed per chunk as encrypt_tensor does.
static PyObject *py_precompute_tensor(PyObject *, PyObject *args) {
    PyObject *kcap, *mats, *rnd;
    double s2, s3;
    int threads;
    if (!PyArg_ParseTuple(args, "OOddOi", &kcap, &mats, &s2, &s3, &rnd, &threads)) return nullptr;
    return with_key(kcap, [&](auto &k) -> PyObject * {
        typedef typename std::remove_reference<decltype(k)>::type K;
        typedef typename K::W W;
        const auto &c = *k.c;
        size_t n = c.n;
        if (k.eta != 1 || !k.has_u) {
            PyErr_SetString(PyExc_ValueError, "precompute_tensor needs eta = 1 and the pad u");
            return nullptr;
        }
        PyObject *mseq = PySequence_Fast(mats, "pad materials must be a sequence of bytes");
        if (!mseq) return nullptr;
        std::unique_ptr<PyObject, void (*)(PyObject *)> g(mseq, [](PyObject *o) { Py_DECREF(o); });
        size_t chunks = (size_t)PySequence_Fast_GET_SIZE(mseq);
        std::vector<std::pair<const unsigned char *, size_t>> mat;
        for (size_t j = 0; j < chunks; ++j) {
            PyObject *m = PySequence_Fast_GET_ITEM(mseq, j);
            if (!PyBytes_Check(m)) {
                PyErr_SetString(PyExc_TypeError, "pad materials must be bytes");
                return nullptr;
            }
            mat.emplace_back((const unsigned char *)PyBytes_AS_STRING(m), (size_t)PyBytes_GET_SIZE(m));
        }
        size_t per = gauss_bytes(n) * 3;
        RandomSource src;
        if (!src.init(rnd, per * chunks)) return nullptr;
        std::unique_ptr<Pre<W>> pre(new Pre<W>());
        pre->key = &k;
        pre->chunks = chunks;
        pre->n = n;
        pre->used.assign(chunks, 0);
        pre->data.assign(2 * n * chunks, 0);
        std::atomic<bool> rnd_fail(false);
        size_t nb = zq_block(c);
        Py_BEGIN_ALLOW_THREADS
        parallel_for(chunks, threads, [&](size_t j) {
            if (src.from_os && !os_random(src.own.data() + per * j, per)) {
                rnd_fail = true;
                return;
            }
            std::vector<W> w(n);
            std::vector<unsigned char> stream(nb * n);
            shake256(mat[j].first, mat[j].second, stream.data(), stream.size());
            zq_reduce(c, stream.data(), nb, n, w.data());
            for (size_t i = 0; i < n; ++i) w[i] = c.ar.add(k.u[i], w[i]);
            W *p0 = pre->data.data() + 2 * n * j;
            encrypt_chunk(k, w.data(), s2, s3, src.data + per * j, p0, p0 + n);
        });
        Py_END_ALLOW_THREADS
        if (src.from_os) {  // the samples' bytes are no longer needed
            volatile unsigned char *b = src.own.data();
            for (size_t i = 0; i < src.own.size(); ++i) b[i] = 0;
        }
        if (rnd_fail) return PyErr_SetFromErrno(PyExc_OSError);
        pre->key_cap = kcap;
        Py_INCREF(kcap);
        PyObject *cap = PyCapsule_New(pre.get(), PRE_NAME, pre_free);
        if (cap) pre.release();
        return cap;
    });
}

// encrypt_precomputed(key, pre, first, x, threads, binary=False) -> as encrypt_tensor
// Online part for one layer: chunks first .. first + ceil(L / n) - 1 of `pre`, which are
// then wiped; a chunk can be used once.
static PyObject *py_encrypt_precomputed(PyObject *, PyObject *args) {
    PyObject *kcap, *pcap, *xo;
    Py_ssize_t first;
    int threads, binary = 0;
    if (!PyArg_ParseTuple(args, "OOnOi|p", &kcap, &pcap, &first, &xo, &threads, &binary)) return nullptr;
    PreBase *pb = (PreBase *)PyCapsule_GetPointer(pcap, PRE_NAME);
    if (!pb) return nullptr;
    return with_key(kcap, [&](auto &k) -> PyObject * {
        typedef typename std::remove_reference<decltype(k)>::type K;
        typedef typename K::W W;
        const auto &c = *k.c;
        size_t n = c.n;
        if (pb->key != (const void *)&k) {
            PyErr_SetString(PyExc_ValueError, "precomputed with another key");
            return nullptr;
        }
        Pre<W> &pre = *static_cast<Pre<W> *>(pb);
        Py_buffer xv;
        if (PyObject_GetBuffer(xo, &xv, PyBUF_SIMPLE) < 0) return nullptr;
        std::unique_ptr<Py_buffer, void (*)(Py_buffer *)> xguard(&xv, PyBuffer_Release);
        if (xv.len % 8 || xv.len == 0) {
            PyErr_SetString(PyExc_ValueError, "x must be a non-empty int64 buffer");
            return nullptr;
        }
        size_t L = (size_t)xv.len / 8, chunks = (L + n - 1) / n;
        const int64_t *x = (const int64_t *)xv.buf;
        if (first < 0 || (size_t)first + chunks > pre.chunks) {
            PyErr_SetString(PyExc_ValueError, "the precomputation does not cover these chunks");
            return nullptr;
        }
        for (size_t j = 0; j < chunks; ++j) {
            if (pre.used[(size_t)first + j]) {
                PyErr_SetString(PyExc_ValueError, "precomputed chunk already used");
                return nullptr;
            }
        }
        for (size_t j = 0; j < chunks; ++j) pre.used[(size_t)first + j] = 1;
        std::vector<DecChunk> dec(2 * chunks);
        Py_BEGIN_ALLOW_THREADS
        parallel_for(chunks, threads, [&](size_t j) {
            size_t k0 = j * n, kk = std::min(n, L - k0), jj = (size_t)first + j;
            W *p0 = pre.data.data() + 2 * n * jj, *c1 = p0 + n;
            for (size_t i = 0; i < kk; ++i)
                p0[i] = c.ar.add(p0[i], c.ar.mul((W)reduce_i128(x[k0 + i], c.q), k.delta_m));
            if (binary) {
                dec[2 * j].buf.resize(bin_size(1, n, bin_wb(c)));
                bin_write(c, 1, p0, c1, (unsigned char *)dec[2 * j].buf.data());
            } else {
                dec[2 * j].format(p0, n);
                dec[2 * j + 1].format(c1, n);
            }
            pre.wipe(jj);
        });
        Py_END_ALLOW_THREADS
        PyObject *out = PyList_New((Py_ssize_t)chunks);
        if (!out) return nullptr;
        for (size_t j = 0; j < chunks; ++j) {
            Py_ssize_t kk = (Py_ssize_t)std::min(n, L - j * n);
            PyObject *item;
            if (binary) {
                item = Py_BuildValue("(y#n)", dec[2 * j].buf.data(), (Py_ssize_t)dec[2 * j].buf.size(), kk);
            } else {
                PyObject *row = dec_list_from(dec[2 * j].buf.data(), dec[2 * j].off.data(), n);
                PyObject *c1 = row ? dec_list_from(dec[2 * j + 1].buf.data(), dec[2 * j + 1].off.data(), n) : nullptr;
                item = c1 ? Py_BuildValue("([N]Nn)", row, c1, kk) : nullptr;
                if (!c1) Py_XDECREF(row);
            }
            if (!item) { Py_DECREF(out); return nullptr; }
            PyList_SET_ITEM(out, (Py_ssize_t)j, item);
        }
        return out;
    });
}

static PyObject *py_precomputed_info(PyObject *, PyObject *args) {
    PyObject *pcap;
    if (!PyArg_ParseTuple(args, "O", &pcap)) return nullptr;
    PreBase *pb = (PreBase *)PyCapsule_GetPointer(pcap, PRE_NAME);
    if (!pb) return nullptr;
    size_t used = 0;
    for (auto u : pb->used) used += u;
    return Py_BuildValue("{s:n,s:n}", "chunks", (Py_ssize_t)pb->chunks, "used", (Py_ssize_t)used);
}

// ================================================================== decryption

// Values of one ciphertext polynomial: decimal strings copied with the GIL, parsed
// without it; anything else is handled with the GIL afterwards.
struct ParseJob {
    std::vector<char> text;          // concatenated ASCII
    std::vector<uint32_t> pos, len;  // per value; len == SLOW: not a plain str
    std::vector<PyObject *> objs;    // borrowed (kept alive by the caller's lists)
};

static const uint32_t SLOW = 0xFFFFFFFFu;

// Copy a sequence of n values for parsing; false (no exception) if it is not a list
// or tuple of length n, the caller then falls back to Python.
static bool gather(PyObject *seq, size_t n, ParseJob &job) {
    if (!PyList_Check(seq) && !PyTuple_Check(seq)) return false;
    if ((size_t)PySequence_Fast_GET_SIZE(seq) != n) return false;
    PyObject **items = PySequence_Fast_ITEMS(seq);
    job.pos.resize(n);
    job.len.resize(n);
    job.objs.assign(items, items + n);
    job.text.clear();
    job.text.reserve(n * 24);
    for (size_t i = 0; i < n; ++i) {
        PyObject *o = items[i];
        if (PyUnicode_CheckExact(o) && PyUnicode_IS_ASCII(o) && PyUnicode_GET_LENGTH(o) <= 39) {
            const char *s = (const char *)PyUnicode_DATA(o);
            job.pos[i] = (uint32_t)job.text.size();
            job.len[i] = (uint32_t)PyUnicode_GET_LENGTH(o);
            job.text.insert(job.text.end(), s, s + job.len[i]);
        } else {
            job.len[i] = SLOW;
        }
    }
    return true;
}

// Parse a gathered job (no GIL); entries that need Python are left marked in `slow`.
template <class W>
static void parse_job(const ParseJob &job, size_t n, u128 q, W *out, std::vector<uint8_t> &slow) {
    slow.assign(n, 0);
    for (size_t i = 0; i < n; ++i) {
        u128 v;
        if (job.len[i] != SLOW && parse_dec(job.text.data() + job.pos[i], job.len[i], q, &v)) out[i] = (W)v;
        else slow[i] = 1;
    }
}

// Python's int(str(v)) % q for the entries parse_job left (with the GIL).
template <class W>
static bool finish_job(const ParseJob &job, size_t n, PyObject *qobj, W *out, const std::vector<uint8_t> &slow) {
    for (size_t i = 0; i < n; ++i) {
        if (!slow[i]) continue;
        u128 v;
        if (!str_int_mod(job.objs[i], qobj, &v)) return false;
        out[i] = (W)v;
    }
    return true;
}

// decrypt_tensor(ctx, chunks, zs, ks, sk_hats, ys, delta, bound, threads) -> (int64 bytes, first_none)
//   chunks[c][s] = (ct0_rows, ct1): ciphertext of slot s for chunk c (decimal strings or ints),
//     or bytes in the binary format 1;
//   zs[c]: key term z of chunk c (n values); ks[c]: values to decode; sk_hats[s]: ntt_mont(sk_y_s);
//   ys[s]: eta weights. MIFERingLWE._decrypt_with_pad_term for every chunk: decode_scaled of
//   -Delta z + sum_s (<ct0_s, y_s> - ct1_s sk_s), with the products summed in the NTT domain.
// Returns the decoded values (int64, k per chunk) and a mask (uint8, 1: None) or None if
// every value decodes. Raises NotImplementedError for inputs it does not handle (the caller
// uses Python), ValueError("malformed ciphertext", slot, chunk) for a ciphertext value
// int(str(v)) cannot read and int()'s ValueError for such a value of z.
static PyObject *py_decrypt_tensor(PyObject *, PyObject *args) {
    PyObject *cap, *chunks_o, *zs_o, *ks_o, *hats_o, *ys_o, *delta_o, *bound_o;
    int threads;
    if (!PyArg_ParseTuple(args, "OOOOOOOOi", &cap, &chunks_o, &zs_o, &ks_o, &hats_o, &ys_o, &delta_o, &bound_o,
                          &threads))
        return nullptr;
    CtxBase *cb = get_ctx(cap);
    if (!cb) return nullptr;
    return with_ctx(cb, [&](auto &c) -> PyObject * {
        typedef typename std::remove_reference<decltype(c)>::type CC;
        typedef typename CC::W W;
        size_t n = c.n;
        u128 q = c.q;
        auto fallback = [&]() -> PyObject * {
            PyErr_SetString(PyExc_NotImplementedError, "input layout not handled natively");
            return nullptr;
        };
        PyObject *chunks = PySequence_Fast(chunks_o, "chunks");
        if (!chunks) return nullptr;
        std::unique_ptr<PyObject, void (*)(PyObject *)> g1(chunks, [](PyObject *o) { Py_DECREF(o); });
        size_t C = (size_t)PySequence_Fast_GET_SIZE(chunks);
        PyObject *hats = PySequence_Fast(hats_o, "sk_hats");
        if (!hats) return nullptr;
        std::unique_ptr<PyObject, void (*)(PyObject *)> g2(hats, [](PyObject *o) { Py_DECREF(o); });
        size_t S = (size_t)PySequence_Fast_GET_SIZE(hats);
        if (S < 1 || PySequence_Size(ys_o) != (Py_ssize_t)S || PySequence_Size(zs_o) != (Py_ssize_t)C
            || PySequence_Size(ks_o) != (Py_ssize_t)C) {
            if (!PyErr_Occurred()) PyErr_SetString(PyExc_ValueError, "inconsistent decryption inputs");
            return nullptr;
        }
        // weights (Montgomery form), eta per slot
        size_t eta = 0;
        std::vector<W> ym;
        for (size_t s = 0; s < S; ++s) {
            PyObject *y = PySequence_GetItem(ys_o, (Py_ssize_t)s);
            if (!y) return nullptr;
            Py_ssize_t e = PySequence_Size(y);
            if (e < 1 || (s && (size_t)e != eta)) { Py_DECREF(y); if (!PyErr_Occurred()) return fallback(); return nullptr; }
            eta = (size_t)e;
            std::vector<W> tmp(eta);
            bool ok = read_words<W>(y, eta, c.qobj, q, tmp.data(), false);
            Py_DECREF(y);
            if (!ok) return nullptr;
            for (auto v : tmp) ym.push_back(c.to_mont(v));
        }
        // NTT(sk_s), Montgomery form
        std::vector<Py_buffer> hv(S);
        size_t got = 0;
        struct Rel { std::vector<Py_buffer> *v; size_t *n; ~Rel() { for (size_t i = 0; i < *n; ++i) PyBuffer_Release(&(*v)[i]); } } rel{&hv, &got};
        for (size_t s = 0; s < S; ++s) {
            if (!buffer_words(PySequence_Fast_GET_ITEM(hats, s), n, sizeof(W), &hv[s])) return nullptr;
            ++got;
        }
        u128 delta;
        if (!index_mod(delta_o, c.qobj, &delta)) return nullptr;
        PyObject *bi = PyNumber_Index(bound_o);
        if (!bi) return nullptr;
        int overflow = 0;
        long long bound_ll = PyLong_AsLongLongAndOverflow(bi, &overflow);
        Py_DECREF(bi);
        if (bound_ll == -1 && PyErr_Occurred()) return nullptr;
        i128 bound = overflow > 0 ? ((i128)1 << 126) : overflow < 0 ? (i128)-1 : (i128)bound_ll;
        // gather: per chunk z, then per slot eta rows of ct0 and ct1
        size_t per_chunk = 1 + S * (eta + 1);
        std::vector<ParseJob> jobs(C * per_chunk);
        std::vector<size_t> ks(C);
        size_t total = 0;
        for (size_t j = 0; j < C; ++j) {
            PyObject *ko = PySequence_GetItem(ks_o, (Py_ssize_t)j);
            if (!ko) return nullptr;
            Py_ssize_t kv = PyLong_AsSsize_t(ko);
            Py_DECREF(ko);
            if (kv == -1 && PyErr_Occurred()) return nullptr;
            if (kv < 1 || (size_t)kv > n) return fallback();
            ks[j] = (size_t)kv;
            total += ks[j];
            PyObject *z = PySequence_GetItem(zs_o, (Py_ssize_t)j);
            if (!z) return nullptr;
            bool ok = gather(z, n, jobs[j * per_chunk]);
            Py_DECREF(z);
            if (!ok) return fallback();
            PyObject *slots = PySequence_Fast_GET_ITEM(chunks, j);
            if ((!PyList_Check(slots) && !PyTuple_Check(slots)) || (size_t)PySequence_Fast_GET_SIZE(slots) != S)
                return fallback();
            for (size_t s = 0; s < S; ++s) {
                PyObject *pair = PySequence_Fast_GET_ITEM(slots, s);
                size_t base = j * per_chunk + 1 + s * (eta + 1);
                if (PyBytes_Check(pair)) {  // binary format: decoded below, without the GIL
                    jobs[base].objs.assign(1, pair);
                    jobs[base].len.clear();
                    continue;
                }
                if (!PyTuple_Check(pair) || PyTuple_GET_SIZE(pair) != 2) return fallback();
                PyObject *c0 = PyTuple_GET_ITEM(pair, 0), *c1 = PyTuple_GET_ITEM(pair, 1);
                if ((!PyList_Check(c0) && !PyTuple_Check(c0)) || (size_t)PySequence_Fast_GET_SIZE(c0) != eta)
                    return fallback();
                for (size_t r = 0; r < eta; ++r)
                    if (!gather(PySequence_Fast_GET_ITEM(c0, r), n, jobs[base + r])) return fallback();
                if (!gather(c1, n, jobs[base + eta])) return fallback();
            }
        }
        std::vector<W> vals(jobs.size() * n);
        std::vector<std::vector<uint8_t>> slow(jobs.size());
        std::vector<uint8_t> bad_bin(jobs.size(), 0);
        std::vector<PyObject *> held;  // binary ciphertexts stay alive without the GIL
        for (auto &jb : jobs)
            if (jb.len.empty() && jb.objs.size() == 1) { Py_INCREF(jb.objs[0]); held.push_back(jb.objs[0]); }
        std::unique_ptr<std::vector<PyObject *>, void (*)(std::vector<PyObject *> *)> hold(
            &held, [](std::vector<PyObject *> *v) { for (PyObject *o : *v) Py_DECREF(o); });
        Py_BEGIN_ALLOW_THREADS
        parallel_for(jobs.size(), threads, [&](size_t i) {
            const ParseJob &jb = jobs[i];
            if (jb.len.empty() && jb.objs.size() == 1) {  // first job of a binary slot: all its eta + 1 rows
                PyObject *b = jb.objs[0];
                W *dst = vals.data() + i * n;
                if (!bin_read(c, eta, (const unsigned char *)PyBytes_AS_STRING(b), (size_t)PyBytes_GET_SIZE(b), dst,
                              dst + eta * n))
                    bad_bin[i] = 1;
                slow[i].assign(n, 0);
                return;
            }
            if (jb.len.empty()) {  // further rows of a binary slot
                slow[i].assign(n, 0);
                return;
            }
            parse_job(jb, n, q, vals.data() + i * n, slow[i]);
        });
        Py_END_ALLOW_THREADS
        for (size_t i = 0; i < jobs.size(); ++i) {
            if (bad_bin[i]) {
                size_t j = i / per_chunk, rest = i % per_chunk;
                PyObject *a = Py_BuildValue("(snn)", "malformed ciphertext", (Py_ssize_t)((rest - 1) / (eta + 1)),
                                            (Py_ssize_t)j);
                if (a) {
                    PyErr_SetObject(PyExc_ValueError, a);
                    Py_DECREF(a);
                }
                return nullptr;
            }
            if (jobs[i].len.empty()) continue;
            if (!finish_job(jobs[i], n, c.qobj, vals.data() + i * n, slow[i])) {
                size_t j = i / per_chunk, rest = i % per_chunk;
                if (rest == 0) return nullptr;  // z: ValueError of int(), as gmpy2.mpz(z) would raise
                PyErr_Clear();
                PyObject *a = Py_BuildValue("(snn)", "malformed ciphertext", (Py_ssize_t)((rest - 1) / (eta + 1)),
                                            (Py_ssize_t)j);
                if (a) {
                    PyErr_SetObject(PyExc_ValueError, a);
                    Py_DECREF(a);
                }
                return nullptr;
            }
        }
        // decrypt every chunk
        std::vector<int64_t> outv(total);
        std::vector<uint8_t> none(total, 0);
        std::vector<size_t> start(C);
        for (size_t j = 0, p = 0; j < C; ++j) { start[j] = p; p += ks[j]; }
        W delta_m = c.to_mont((W)delta);
        i128 den = 2 * (i128)delta;
        bool big = false;
        Py_BEGIN_ALLOW_THREADS
        parallel_for(C, threads, [&](size_t j) {
            std::vector<W> tot(n), acc(n, 0), xh(n), rhs(n);
            const W *z = vals.data() + j * per_chunk * n;
            for (size_t i = 0; i < n; ++i) tot[i] = c.ar.sub(0, c.ar.mul(z[i], delta_m));
            for (size_t s = 0; s < S; ++s) {
                const W *base = vals.data() + (j * per_chunk + 1 + s * (eta + 1)) * n;
                for (size_t r = 0; r < eta; ++r) {
                    W y = ym[s * eta + r];
                    const W *row = base + r * n;
                    for (size_t i = 0; i < n; ++i) tot[i] = c.ar.add(tot[i], c.ar.mul(row[i], y));
                }
                c.forward(base + eta * n, xh.data());
                const W *skh = (const W *)hv[s].buf;
                for (size_t i = 0; i < n; ++i) acc[i] = c.ar.add(acc[i], c.ar.mul(skh[i], xh[i]));
            }
            c.inverse(acc.data(), rhs.data());
            u128 half = q >> 1;
            for (size_t i = 0; i < ks[j]; ++i) {  // lwe_utils.decode_scaled
                u128 v = (u128)c.ar.sub(tot[i], rhs[i]);
                i128 sv = v > half ? (i128)v - (i128)q : (i128)v;
                i128 num = 2 * sv + (i128)delta, mu = num / den;
                if (num % den != 0 && num < 0) mu -= 1;  // floor division
                size_t p = start[j] + i;
                if (mu > bound || mu < -bound) none[p] = 1;
                else if (mu > INT64_MAX || mu < INT64_MIN) big = true;
                else outv[p] = (int64_t)mu;
            }
        });
        Py_END_ALLOW_THREADS
        if (big) return fallback();  // |values| beyond int64: let Python decode them
        bool any_none = std::find(none.begin(), none.end(), 1) != none.end();
        PyObject *data = PyBytes_FromStringAndSize((const char *)outv.data(), (Py_ssize_t)(total * 8));
        if (!data) return nullptr;
        if (!any_none) return Py_BuildValue("(NO)", data, Py_None);
        PyObject *mask = PyBytes_FromStringAndSize((const char *)none.data(), (Py_ssize_t)total);
        if (!mask) { Py_DECREF(data); return nullptr; }
        return Py_BuildValue("(NN)", data, mask);
    });
}

// pad_terms(ctx, base, materials, ys, eta, threads) -> list[list[str]]
//   For every label l: base + sum_s sum_row ys[s][row] H_{s,l}[row] mod q (decimal strings), with
//   H_{s,l} = hash_to_zq_vector of materials[l][s] (eta n values, row-major). The pad term
//   z'' of MCFERingLWEKeyGenerator.get_decryption_keys.
static PyObject *py_pad_terms(PyObject *, PyObject *args) {
    PyObject *cap, *base_o, *mats_o, *ys_o;
    Py_ssize_t eta_s;
    int threads;
    if (!PyArg_ParseTuple(args, "OOOOni", &cap, &base_o, &mats_o, &ys_o, &eta_s, &threads)) return nullptr;
    CtxBase *cb = get_ctx(cap);
    if (!cb) return nullptr;
    return with_ctx(cb, [&](auto &c) -> PyObject * {
        typedef typename std::remove_reference<decltype(c)>::type CC;
        typedef typename CC::W W;
        size_t n = c.n, eta = (size_t)eta_s;
        std::vector<W> base(n);
        if (eta < 1 || !read_words<W>(base_o, n, c.qobj, c.q, base.data(), false)) {
            if (!PyErr_Occurred()) PyErr_SetString(PyExc_ValueError, "eta < 1");
            return nullptr;
        }
        Py_ssize_t S = PySequence_Size(ys_o);
        if (S < 0) return nullptr;
        std::vector<W> ym;
        for (Py_ssize_t s = 0; s < S; ++s) {
            PyObject *y = PySequence_GetItem(ys_o, s);
            if (!y) return nullptr;
            std::vector<W> tmp(eta);
            bool ok = read_words<W>(y, eta, c.qobj, c.q, tmp.data(), false);
            Py_DECREF(y);
            if (!ok) return nullptr;
            for (auto v : tmp) ym.push_back(c.to_mont(v));
        }
        PyObject *mats = PySequence_Fast(mats_o, "materials");
        if (!mats) return nullptr;
        std::unique_ptr<PyObject, void (*)(PyObject *)> g(mats, [](PyObject *o) { Py_DECREF(o); });
        size_t Lc = (size_t)PySequence_Fast_GET_SIZE(mats);
        std::vector<std::pair<const unsigned char *, size_t>> mv;
        std::vector<PyObject *> keep;
        for (size_t l = 0; l < Lc; ++l) {
            PyObject *row = PySequence_Fast_GET_ITEM(mats, l);
            if ((!PyList_Check(row) && !PyTuple_Check(row)) || PySequence_Fast_GET_SIZE(row) != S) {
                PyErr_SetString(PyExc_ValueError, "one material per slot and label");
                return nullptr;
            }
            for (Py_ssize_t s = 0; s < S; ++s) {
                PyObject *m = PySequence_Fast_GET_ITEM(row, s);
                if (!PyBytes_Check(m)) {
                    PyErr_SetString(PyExc_TypeError, "materials must be bytes");
                    return nullptr;
                }
                mv.emplace_back((const unsigned char *)PyBytes_AS_STRING(m), (size_t)PyBytes_GET_SIZE(m));
            }
        }
        std::vector<DecChunk> dec(Lc);
        size_t nb = zq_block(c);
        Py_BEGIN_ALLOW_THREADS
        parallel_for(Lc, threads, [&](size_t l) {
            std::vector<W> z(base);
            std::vector<unsigned char> stream(nb * n * eta);
            std::vector<W> h(n * eta);
            for (size_t s = 0; s < (size_t)S; ++s) {
                const auto &m = mv[l * (size_t)S + s];
                shake256(m.first, m.second, stream.data(), stream.size());
                zq_reduce(c, stream.data(), nb, n * eta, h.data());
                for (size_t r = 0; r < eta; ++r) {
                    W y = ym[s * eta + r];
                    const W *hr = h.data() + r * n;
                    for (size_t i = 0; i < n; ++i) z[i] = c.ar.add(z[i], c.ar.mul(hr[i], y));
                }
            }
            dec[l].format(z.data(), n);
        });
        Py_END_ALLOW_THREADS
        PyObject *out = PyList_New((Py_ssize_t)Lc);
        if (!out) return nullptr;
        for (size_t l = 0; l < Lc; ++l) {
            PyObject *lst = dec_list_from(dec[l].buf.data(), dec[l].off.data(), n);
            if (!lst) { Py_DECREF(out); return nullptr; }
            PyList_SET_ITEM(out, (Py_ssize_t)l, lst);
        }
        return out;
    });
}

static PyObject *py_build_info(PyObject *, PyObject *) {
    return Py_BuildValue("{s:s,s:O,s:i}", "compiler",
#if defined(__clang__)
                         "clang " __clang_version__,
#elif defined(__GNUC__)
                         "gcc " __VERSION__,
#else
                         "unknown",
#endif
                         "os_random", have_os_random() ? Py_True : Py_False, "hardware_threads",
                         (int)std::thread::hardware_concurrency());
}

static PyMethodDef methods[] = {
    {"ctx_new", py_ctx_new, METH_VARARGS, "ctx_new(q, n, root) -> context"},
    {"ctx_info", py_ctx_info, METH_VARARGS, "ctx_info(ctx) -> dict"},
    {"pack", py_pack, METH_VARARGS, "pack(ctx, values) -> bytes (words mod q)"},
    {"unpack", py_unpack, METH_VARARGS, "unpack(ctx, data, as_digits=False) -> list"},
    {"ntt", py_ntt, METH_VARARGS, "ntt(ctx, data, inverse=False) -> bytes"},
    {"ntt_mont", py_ntt_mont, METH_VARARGS, "ntt_mont(ctx, data) -> bytes (Montgomery form)"},
    {"poly_mul", py_poly_mul, METH_VARARGS, "poly_mul(ctx, a, b) -> bytes"},
    {"lincomb", py_lincomb, METH_VARARGS, "lincomb(ctx, rows, coeffs) -> list[str]"},
    {"gauss", py_gauss, METH_VARARGS, "gauss(random_bytes | None, count, sigma) -> list[int]"},
    {"shake256", py_shake256, METH_VARARGS, "shake256(data, length) -> bytes"},
    {"hash_to_zq", py_hash_to_zq, METH_VARARGS, "hash_to_zq(ctx, material, count) -> bytes"},
    {"random_bytes", py_random_bytes, METH_VARARGS, "random_bytes(n) -> bytes from the OS source"},
    {"enc_key", py_enc_key, METH_VARARGS, "enc_key(ctx, a, pk_rows, u_rows, delta) -> key"},
    {"encrypt", py_encrypt, METH_VARARGS, "encrypt(key, w_rows, sigma2, sigma3, random_bytes) -> (ct0, ct1)"},
    {"encrypt_tensor", py_encrypt_tensor, METH_VARARGS, "MCFE ndarray encryption of one layer"},
    {"decrypt_tensor", py_decrypt_tensor, METH_VARARGS, "MIFE/MCFE decryption of the chunks of one layer"},
    {"pad_terms", py_pad_terms, METH_VARARGS, "MCFE key terms z'' for several labels"},
    {"precompute_tensor", py_precompute_tensor, METH_VARARGS, "offline part of MCFE ndarray encryption"},
    {"encrypt_precomputed", py_encrypt_precomputed, METH_VARARGS, "online part of MCFE ndarray encryption"},
    {"precomputed_info", py_precomputed_info, METH_VARARGS, "precomputed_info(pre) -> dict"},
    {"build_info", py_build_info, METH_NOARGS, "build_info() -> dict"},
    {nullptr, nullptr, 0, nullptr},
};

static struct PyModuleDef moddef = {PyModuleDef_HEAD_INIT, "_rlwe_native", nullptr, -1, methods};

PyMODINIT_FUNC PyInit__rlwe_native(void) { return PyModule_Create(&moddef); }
