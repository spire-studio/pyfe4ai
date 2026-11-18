//! Garbled circuits of the bounded-key FE schemes (bcfe/*): the
//! half-gates garbler and evaluator of the core's `garble` module (ZRE15
//! Fig. 2, the CIRC v1 encoding, the salted gate hash), on bytes: circuits as
//! their canonical encoding, labels and table entries as 16-byte strings,
//! bit strings packed low bit first. Mode 0 is "hardwired", 1 is "labels".
//! The garbling seed comes from the OS unless a test gives it; the seed, delta
//! and the labels are wiped in the core, the seed copy here. The GIL is
//! released around the garbling and the evaluation.
//!
//! The label KEM of a slot (SS10 Sec. 3.3 with hashed ElGamal over
//! ristretto255; byte for byte pyfe4ai.primitives.bcfe_kem): the slot's
//! public-key table (2m + 1 points: P[j,0] || P[j,1] for j < m, then P[m,0]
//! for the g bit), the
//! unselected points H_P = hash_to_ristretto255 (RFC 9380, the expand_message_xmd
//! of ec.rs), the label ciphertexts c[j,b] = L_j^b xor H_K(.., j, b, R, r P[j,b])
//! with H_K = SHAKE-256("PYFE4AI-V06-BCFE-KEM" || ..)[0:16], and the fused
//! encryption (garbling and label encryption in one call: delta, the seed, r
//! and the 0-labels never leave this module and are wiped). Everything that
//! depends on w is selected with masks, not branches. The per-position point
//! work runs on up to `threads` threads (pyfe4ai_core::parallel_for_joined:
//! the caller joins its workers instead of parking, so that a process forked
//! after a call can make the call again with several threads, on macOS too).

use curve25519_dalek::ristretto::{CompressedRistretto, RistrettoBasepointTable, RistrettoPoint};
use curve25519_dalek::scalar::Scalar;
use curve25519_dalek::traits::IsIdentity;
use pyfe4ai_core::garble::{self, Mode};
use pyfe4ai_core::keccak::{shake256, wipe_long};
use pyfe4ai_core::rlwe::wipe_bytes;
use pyfe4ai_core::sha512::expand_message_xmd_64;
use pyfe4ai_core::{osrand, parallel_for_joined};
use pyo3::exceptions::{PyOSError, PyValueError};
use pyo3::prelude::*;
use pyo3::types::PyBytes;
use std::sync::{Mutex, MutexGuard};
use zeroize::{Zeroize, Zeroizing};

fn mode_of(code: u8) -> PyResult<Mode> {
    Mode::from_code(code).ok_or_else(|| PyValueError::new_err("the garbling mode is 0 (hardwired) or 1 (labels)"))
}

type Garbled = (Py<PyBytes>, Py<PyBytes>, Py<PyBytes>, Py<PyBytes>, Py<PyBytes>);

/// Garble `circuit` (CIRC v1) for the packed data bits `x` with the 56-byte
/// instance `salt`: (delta, key0, data_active, tables, decode), the bytes of
/// the backend operation gc_garble. `seed` (32 bytes) is for the tests; None
/// draws it from the OS.
#[pyfunction]
#[pyo3(signature = (circuit, mode, salt, x, seed=None))]
fn gc_garble(
    py: Python<'_>, circuit: &[u8], mode: u8, salt: &[u8], x: &[u8], seed: Option<&[u8]>,
) -> PyResult<Garbled> {
    let mode = mode_of(mode)?;
    let mut s = Zeroizing::new([0u8; garble::SEED_LEN]);
    match seed {
        Some(b) => {
            if b.len() != garble::SEED_LEN {
                return Err(PyValueError::new_err("a garbling seed has 32 bytes"));
            }
            s.copy_from_slice(b);
        }
        None => osrand::fill(&mut s[..]).map_err(|e| PyOSError::new_err(e.to_string()))?,
    }
    let g = py.detach(|| garble::garble(circuit, mode, salt, x, &s[..])).map_err(PyValueError::new_err)?;
    Ok((
        PyBytes::new(py, &g.delta).unbind(),
        PyBytes::new(py, &g.key0).unbind(),
        PyBytes::new(py, &g.data_active).unbind(),
        PyBytes::new(py, &g.tables).unbind(),
        PyBytes::new(py, &g.decode).unbind(),
    ))
}

/// Evaluate a garbling of gc_garble with the active labels of the key wires
/// (16 bytes each) and of the data wires (mode "labels"; empty in mode
/// "hardwired"): the n_out output bits, packed.
#[pyfunction]
#[allow(clippy::too_many_arguments)]
fn gc_evaluate(
    py: Python<'_>, circuit: &[u8], mode: u8, salt: &[u8], tables: &[u8], decode: &[u8], key_active: &[u8],
    data_active: &[u8],
) -> PyResult<Py<PyBytes>> {
    let mode = mode_of(mode)?;
    let y = py
        .detach(|| garble::evaluate(circuit, mode, salt, tables, decode, key_active, data_active))
        .map_err(PyValueError::new_err)?;
    Ok(PyBytes::new(py, &garble::pack_bits(&y)).unbind())
}

// ------------------------------------------------------------------ the label KEM of a slot

const KEM_DST: &[u8; 20] = b"PYFE4AI-V06-BCFE-KEM";
const EPHEM_DST: &[u8] = b"PYFE4AI-V06-BCFE-EPHEM";
const UNSEL_PREFIX: &[u8; 10] = b"PFE6-UNSEL";
const UNSEL_DST: &[u8] = b"PYFE4AI-V06-BCFE-UNSEL-V01-CS02-with-ristretto255_XMD:SHA-512_R255MAP_RO_";
/// Bytes of a point encoding, a scalar, a digest; of randomness per scalar; of k.
const PT: usize = 32;
const WIDE: usize = 64;
const K_LEN: usize = 16;
const NONCE_LEN: usize = 16;
const LB: usize = garble::LABEL_LEN;
/// Below this many scalar multiplications by R, no table for R.
const TABLE_FROM: usize = 64;

fn lock<T>(m: &Mutex<T>) -> MutexGuard<'_, T> {
    m.lock().unwrap_or_else(|e| e.into_inner())
}

fn need(b: &[u8], n: usize, what: &str) -> Result<(), String> {
    if b.len() != n {
        return Err(format!("{what} has {n} bytes, not {}", b.len()));
    }
    Ok(())
}

/// The i-th `n`-byte piece of `b`.
fn piece<const N: usize>(b: &[u8], i: usize) -> [u8; N] {
    b[N * i..N * (i + 1)].try_into().unwrap()
}

/// `if1` where the low bit of `bit` is 1, else `if0` (masks, no branch).
fn select<const N: usize>(bit: u8, if0: &[u8; N], if1: &[u8; N]) -> [u8; N] {
    let mask = 0u8.wrapping_sub(bit & 1);
    let mut out = *if0;
    for (o, (a, b)) in out.iter_mut().zip(if0.iter().zip(if1.iter())) {
        *o ^= mask & (a ^ b);
    }
    out
}

/// The results of one parallel_for_joined, item by item, and the first failure (by
/// index). The items may be secret: wiped on drop.
struct Cells<const N: usize> {
    cells: Vec<Mutex<[u8; N]>>,
    failure: Mutex<Option<(usize, String)>>,
}

impl<const N: usize> Cells<N> {
    fn new(count: usize) -> Self {
        Cells { cells: (0..count).map(|_| Mutex::new([0u8; N])).collect(), failure: Mutex::new(None) }
    }

    fn put(&self, i: usize, v: &[u8; N]) {
        *lock(&self.cells[i]) = *v;
    }

    fn fail(&self, i: usize, msg: String) {
        let mut f = lock(&self.failure);
        if f.as_ref().map_or(true, |(k, _)| i < *k) {
            *f = Some((i, msg));
        }
    }

    fn check(&self) -> Result<(), String> {
        match lock(&self.failure).take() {
            Some((_, msg)) => Err(msg),
            None => Ok(()),
        }
    }

    fn get(&self, i: usize) -> [u8; N] {
        *lock(&self.cells[i])
    }

    /// All items concatenated (the caller wipes the copy if it is secret).
    fn concat(&self) -> Vec<u8> {
        let mut out = Vec::with_capacity(N * self.cells.len());
        for c in &self.cells {
            out.extend_from_slice(&*lock(c));
        }
        out
    }
}

impl<const N: usize> Drop for Cells<N> {
    fn drop(&mut self) {
        for c in &self.cells {
            wipe_bytes(&mut *lock(c));
        }
    }
}

/// One slot of one ciphertext: "PYFE4AI-V06-BCFE-KEM" || pp_digest || ct_nonce
/// || u64(slot_id), the prefix of H_K; its last 56 bytes are the gate hash's salt.
struct Instance {
    prefix: [u8; 20 + PT + NONCE_LEN + 8],
}

impl Instance {
    fn new(pp_digest: &[u8], ct_nonce: &[u8], slot_id: u64) -> Result<Instance, String> {
        need(pp_digest, PT, "the pp digest")?;
        need(ct_nonce, NONCE_LEN, "the ciphertext nonce")?;
        let mut prefix = [0u8; 76];
        prefix[..20].copy_from_slice(KEM_DST);
        prefix[20..52].copy_from_slice(pp_digest);
        prefix[52..68].copy_from_slice(ct_nonce);
        prefix[68..76].copy_from_slice(&slot_id.to_le_bytes());
        Ok(Instance { prefix })
    }

    fn salt(&self) -> &[u8] {
        &self.prefix[20..]
    }

    /// H_K(pp_digest, ct_nonce, slot_id, j, b, R, Z) = SHAKE-256("PYFE4AI-V06-BCFE-KEM"
    /// || pp_digest || ct_nonce || u64(slot_id) || u32(j) || u8(b) || R || Z)[0:16].
    fn hk(&self, j: u32, b: u8, r: &[u8; PT], z: &[u8; PT]) -> [u8; LB] {
        let mut msg = [0u8; 76 + 4 + 1 + PT + PT];
        msg[..76].copy_from_slice(&self.prefix);
        msg[76..80].copy_from_slice(&j.to_le_bytes());
        msg[80] = b;
        msg[81..113].copy_from_slice(r);
        msg[113..145].copy_from_slice(z);
        let mut out = [0u8; LB];
        shake256(&msg, &mut out);
        wipe_bytes(&mut msg);
        out
    }
}

/// H_P(setup_digest, slot_id, j, k) = hash_to_ristretto255("PFE6-UNSEL" || setup_digest
/// || u64(slot_id) || u32(j) || k, DST), encoded.
fn unselected_point(setup_digest: &[u8], slot_id: u64, j: u32, k: &[u8]) -> [u8; PT] {
    let mut msg = [0u8; 10 + PT + 8 + 4 + K_LEN];
    msg[..10].copy_from_slice(UNSEL_PREFIX);
    msg[10..42].copy_from_slice(setup_digest);
    msg[42..50].copy_from_slice(&slot_id.to_le_bytes());
    msg[50..54].copy_from_slice(&j.to_le_bytes());
    msg[54..70].copy_from_slice(k);
    let mut uniform = expand_message_xmd_64(&msg, UNSEL_DST);
    let p = RistrettoPoint::from_uniform_bytes(&uniform);
    wipe_bytes(&mut msg);
    wipe_bytes(&mut uniform);
    p.compress().to_bytes()
}

/// m, the key bits of C (guard(C) has m + 1 key wires): in [1, 2^26 - 1), so
/// that guard(C) stays below the wire bound of CIRC v1.
fn key_bits(m: usize) -> Result<usize, String> {
    if m == 0 || m as u64 + 1 >= garble::MAX_WIRES {
        return Err("m must be an integer in [1, 2^26 - 1), the key bits of the circuit".into());
    }
    Ok(m)
}

/// m and the packed w checked before any random byte is drawn: w must be
/// ceil(m / 8) bytes with zero padding, so that the bytes drawn for m are
/// backed by input bytes.
fn key_bits_and_w(m: usize, w: &[u8]) -> Result<usize, String> {
    let m = key_bits(m)?;
    let mut wb = garble::unpack_bits(w, m, "w")?;
    wipe_bytes(&mut wb);
    Ok(m)
}

/// m of guard(C) (the guard bit g is key wire m): its m' = m + 1 key wires minus the g bit, after
/// checking the shape guard() gives: m' >= 2; the gates end in INV(g) and then
/// AND(o_k, INV(g)) for the outputs in order, which are the outputs; no other
/// gate reads g.
fn guarded_key_bits(c: &garble::Circuit) -> Result<usize, String> {
    use garble::Op;
    let (mp, n, n_out) = (c.m, c.n, c.outputs.len());
    let g = (mp - 1) as u32;
    if mp < 2 || c.gates.len() < n_out + 1 {
        return Err("the circuit is not guard(C): no INV(g) before the output gates".into());
    }
    let inv = c.gates.len() - n_out - 1;
    if c.gates[inv] != (garble::Gate { op: Op::Inv, a: g, b: 0 }) {
        return Err("the circuit is not guard(C): no INV(g) before the output gates".into());
    }
    let ng = (mp + n + inv) as u32;
    for i in 0..n_out {
        let gt = c.gates[inv + 1 + i];
        if gt.op != Op::And || gt.b != ng || gt.a >= ng || c.outputs[i] != ng + 1 + i as u32 {
            return Err(format!("the circuit is not guard(C): output {i} is not AND(o, INV(g))"));
        }
    }
    for (i, gt) in c.gates.iter().enumerate() {
        if i != inv && (gt.a == g || (gt.op != Op::Inv && gt.b == g)) {
            return Err(format!("the circuit is not guard(C): gate {i} reads the g bit"));
        }
    }
    Ok(mp - 1)
}

/// The table of a slot: for j < m the point of the selected position (j, w_j)
/// at w_j and the unselected one at 1 - w_j, then the g bit's point.
fn assemble(selected: &Cells<PT>, unselected: &Cells<PT>, w: &[u8], m: usize) -> Vec<u8> {
    let mut pks = Vec::with_capacity(PT * (2 * m + 1));
    for (j, &bit) in w.iter().enumerate() {
        let (sel, uns) = (selected.get(j), unselected.get(j));
        pks.extend_from_slice(&select(bit, &sel, &uns)); // P[j,0]
        pks.extend_from_slice(&select(bit, &uns, &sel)); // P[j,1]
    }
    pks.extend_from_slice(&selected.get(m));
    pks
}

/// h1: the center's scalars s_j (64 bytes each of `rnd`, LE mod l, for j = 0 ..
/// m - 1 and g), k (the last 16 bytes), the table. Returns (scalars, pks).
fn new_slot(setup_digest: &[u8], slot_id: u64, m: usize, w: &[u8], rnd: &[u8], threads: usize)
    -> Result<(Zeroizing<Vec<u8>>, Vec<u8>), String>
{
    need(setup_digest, PT, "the setup digest")?;
    let m = key_bits(m)?;
    let mp = m + 1;
    let wb = Zeroizing::new(garble::unpack_bits(w, m, "w")?);
    need(rnd, WIDE * mp + K_LEN, "the randomness of a new slot")?;
    let k = &rnd[WIDE * mp..];
    let (scalars, selected, unselected) = (Cells::<PT>::new(mp), Cells::<PT>::new(mp), Cells::<PT>::new(m));
    parallel_for_joined(mp, threads, |j| {
        let mut wide: [u8; WIDE] = piece(rnd, j);
        let mut s = Scalar::from_bytes_mod_order_wide(&wide);
        wipe_bytes(&mut wide);
        if s == Scalar::ZERO {
            scalars.fail(j, "a drawn scalar is 0".into());
            return;
        }
        selected.put(j, &RistrettoPoint::mul_base(&s).compress().to_bytes());
        let mut sb = s.to_bytes();
        scalars.put(j, &sb);
        sb.zeroize();
        s.zeroize();
        if j < m {
            unselected.put(j, &unselected_point(setup_digest, slot_id, j as u32, k));
        }
    });
    scalars.check()?;
    let pks = assemble(&selected, &unselected, &wb, m);
    Ok((Zeroizing::new(scalars.concat()), pks))
}

/// h2: the holder's m + 1 public keys at the selected positions and the g bit,
/// the unselected points from k. Returns the table.
fn place_slot(setup_digest: &[u8], slot_id: u64, m: usize, w: &[u8], holder_pks: &[u8], k: &[u8], threads: usize)
    -> Result<Vec<u8>, String>
{
    need(setup_digest, PT, "the setup digest")?;
    let m = key_bits(m)?;
    let mp = m + 1;
    let wb = Zeroizing::new(garble::unpack_bits(w, m, "w")?);
    need(holder_pks, PT * mp, "the holder's public keys (m + 1 encodings)")?;
    need(k, K_LEN, "k")?;
    let (selected, unselected) = (Cells::<PT>::new(mp), Cells::<PT>::new(m));
    parallel_for_joined(mp, threads, |j| {
        let enc: [u8; PT] = piece(holder_pks, j);
        match CompressedRistretto(enc).decompress() {
            None => selected.fail(j, format!("holder public key {j} is not a ristretto255 encoding")),
            Some(p) if bool::from(p.is_identity()) => {
                selected.fail(j, format!("holder public key {j} is the identity"))
            }
            Some(_) => selected.put(j, &enc),
        }
        if j < m {
            unselected.put(j, &unselected_point(setup_digest, slot_id, j as u32, k));
        }
    });
    selected.check()?;
    Ok(assemble(&selected, &unselected, &wb, m))
}

/// CheckAuth, h' part: the table has 2m + 1 entries and every unselected point
/// is H_P(.., j, k) (a table of another length is false, not an error).
fn unselected_ok(setup_digest: &[u8], slot_id: u64, m: usize, w: &[u8], k: &[u8], pks: &[u8], threads: usize)
    -> Result<bool, String>
{
    need(setup_digest, PT, "the setup digest")?;
    need(k, K_LEN, "k")?;
    let m = key_bits(m)?;
    let wb = Zeroizing::new(garble::unpack_bits(w, m, "w")?);
    if pks.len() as u64 != PT as u64 * (2 * m as u64 + 1) {
        return Ok(false);
    }
    let diffs = Cells::<1>::new(m);
    parallel_for_joined(m, threads, |j| {
        let want = unselected_point(setup_digest, slot_id, j as u32, k);
        let got = select(wb[j], &piece::<PT>(pks, 2 * j + 1), &piece::<PT>(pks, 2 * j));
        diffs.put(j, &[want.iter().zip(got.iter()).fold(0u8, |d, (a, b)| d | (a ^ b))]);
    });
    Ok(diffs.concat().iter().fold(0u8, |d, &x| d | x) == 0)
}

/// The fused encryption of one (ciphertext, slot): garble guard(C) with the
/// seed's GBPRG stream and salt (pp_digest, ct_nonce, slot_id), r =
/// SHAKE-256("PYFE4AI-V06-BCFE-EPHEM" || salt || seed)[0:64] LE mod l (the salt
/// bound in: a seed used twice under two salts gives unrelated r; the
/// alternative was the seed alone), R = r B and the 2m + 1 label ciphertexts.
/// Returns (R,
/// label ciphertexts, the garbling); its delta and key0 are wiped on drop.
fn encrypt_slot(
    circuit: &[u8], mode: Mode, inst: &Instance, x: &[u8], pks: &[u8], seed: &[u8], threads: usize,
) -> Result<([u8; PT], Vec<u8>, garble::Garbling), String> {
    let m = guarded_key_bits(&garble::Circuit::parse(circuit)?)?;
    if pks.len() as u64 != PT as u64 * (2 * m as u64 + 1) {
        return Err(format!("the slot's table has {} public keys of 32 bytes for m = {m}", 2 * m + 1));
    }
    need(seed, garble::SEED_LEN, "the seed")?;
    let mut msg = Zeroizing::new([EPHEM_DST, inst.salt(), seed].concat());
    let mut wide = Zeroizing::new([0u8; WIDE]);
    shake256(&msg, &mut wide[..]);
    msg.zeroize();
    let mut r = Scalar::from_bytes_mod_order_wide(&wide);
    if r == Scalar::ZERO {
        return Err("the ephemeral scalar r is 0".into());
    }
    let big_r = RistrettoPoint::mul_base(&r).compress().to_bytes();
    let g = match garble::garble(circuit, mode, inst.salt(), x, seed) {
        Ok(g) => g,
        Err(e) => {
            r.zeroize();
            return Err(e);
        }
    };
    let mut delta = u128::from_le_bytes(g.delta);
    let cts = Cells::<LB>::new(2 * m + 1);
    {
        let (r, delta) = (&r, &delta);
        parallel_for_joined(2 * m + 1, threads, |i| {
            let (j, b) = (i / 2, (i % 2) as u8);
            let p = match CompressedRistretto(piece(pks, i)).decompress() {
                Some(p) if !bool::from(p.is_identity()) => p,
                Some(_) => return cts.fail(i, format!("public key {i} of the slot is the identity")),
                None => return cts.fail(i, format!("public key {i} of the slot is not a ristretto255 encoding")),
            };
            let mut zp = p * r;
            let mut z = zp.compress().to_bytes();
            let mut h = inst.hk(j as u32, b, &big_r, &z);
            // L_j^b = L_j^0 xor b delta (b is the position, public)
            let l0 = Zeroizing::new(piece::<LB>(&g.key0, j));
            let mut l = u128::from_le_bytes(*l0) ^ if b == 1 { *delta } else { 0 };
            let mut c = (l ^ u128::from_le_bytes(h)).to_le_bytes();
            cts.put(i, &c);
            zp.zeroize();
            z.zeroize();
            h.zeroize();
            l.zeroize();
            c.zeroize();
        });
    }
    r.zeroize();
    delta.zeroize();
    cts.check()?;
    Ok((big_r, cts.concat(), g))
}

/// Decrypt one (ciphertext, slot) with the holder's scalars (the selected
/// positions and g, in j order) and w: the active key labels from the label
/// ciphertexts, then the garbled circuit's output bits.
#[allow(clippy::too_many_arguments)]
fn decrypt_slot(
    circuit: &[u8], mode: Mode, inst: &Instance, r_enc: &[u8], label_cts: &[u8], tables: &[u8], decode: &[u8],
    data_active: &[u8], scalars: &[u8], w: &[u8], threads: usize,
) -> Result<Vec<bool>, String> {
    let m = guarded_key_bits(&garble::Circuit::parse(circuit)?)?;
    let mp = m + 1;
    need(r_enc, PT, "R")?;
    let big_r: [u8; PT] = r_enc.try_into().unwrap();
    let rp = CompressedRistretto(big_r).decompress().ok_or("R is not a ristretto255 encoding")?;
    if bool::from(rp.is_identity()) {
        return Err("R is the identity".into());
    }
    if label_cts.len() as u64 != LB as u64 * (2 * m as u64 + 1) {
        return Err(format!("{} label ciphertexts of 16 bytes expected for m = {m}", 2 * m + 1));
    }
    if scalars.len() as u64 != PT as u64 * mp as u64 {
        return Err(format!("{mp} scalars of 32 bytes expected"));
    }
    let wb = Zeroizing::new(garble::unpack_bits(w, m, "w")?);
    let table = if mp >= TABLE_FROM { Some(RistrettoBasepointTable::create(&rp)) } else { None };
    let keys = Cells::<LB>::new(mp);
    parallel_for_joined(mp, threads, |j| {
        let sb = Zeroizing::new(piece::<PT>(scalars, j));
        let mut s = match Scalar::from_canonical_bytes(*sb).into_option() {
            Some(s) => s,
            None => return keys.fail(j, format!("scalar {j} is not canonical (below the group order)")),
        };
        let mut zp = match &table {
            Some(t) => t * &s,
            None => rp * s,
        };
        let mut z = zp.compress().to_bytes();
        let (b, c) = if j < m {
            (wb[j], select(wb[j], &piece::<LB>(label_cts, 2 * j), &piece::<LB>(label_cts, 2 * j + 1)))
        } else {
            (0, piece::<LB>(label_cts, 2 * m))
        };
        let mut h = inst.hk(j as u32, b, &big_r, &z);
        let mut l = (u128::from_le_bytes(c) ^ u128::from_le_bytes(h)).to_le_bytes();
        keys.put(j, &l);
        s.zeroize();
        zp.zeroize();
        z.zeroize();
        h.zeroize();
        l.zeroize();
    });
    drop(table);
    keys.check()?;
    let mut key_active = keys.concat();
    let y = garble::evaluate(circuit, mode, inst.salt(), tables, decode, &key_active, data_active);
    wipe_long(&mut key_active);
    y
}

/// The bytes of the call's randomness: `given` (tests: exactly `n`) or `n` from the OS.
fn draw(given: Option<&[u8]>, n: usize) -> PyResult<Zeroizing<Vec<u8>>> {
    let mut v = Zeroizing::new(vec![0u8; n]);
    match given {
        Some(b) if b.len() == n => v.copy_from_slice(b),
        Some(_) => return Err(PyValueError::new_err(format!("the randomness of this call has {n} bytes"))),
        None => osrand::fill(&mut v[..]).map_err(|e| PyOSError::new_err(e.to_string()))?,
    }
    Ok(v)
}

/// h1: (scalars, pks, k) of a new slot for the m packed bits w; `rnd` (tests)
/// is the 64 (m + 1) + 16 bytes of randomness, else they come from the OS.
#[pyfunction]
#[pyo3(signature = (setup_digest, slot_id, m, w, rnd=None, threads=1))]
fn bcfe_new_slot(
    py: Python<'_>, setup_digest: &[u8], slot_id: u64, m: usize, w: &[u8], rnd: Option<&[u8]>, threads: usize,
) -> PyResult<(Py<PyBytes>, Py<PyBytes>, Py<PyBytes>)> {
    let m = key_bits_and_w(m, w).map_err(PyValueError::new_err)?; // m and w before the draw
    let bytes = draw(rnd, WIDE * (m + 1) + K_LEN)?;
    let (scalars, pks) =
        py.detach(|| new_slot(setup_digest, slot_id, m, w, &bytes, threads)).map_err(PyValueError::new_err)?;
    let k = &bytes[WIDE * (m + 1)..];
    Ok((PyBytes::new(py, &scalars).unbind(), PyBytes::new(py, &pks).unbind(), PyBytes::new(py, k).unbind()))
}

/// h2: (pks, k) with the holder's m + 1 public keys; `rnd` (tests) is k's 16 bytes.
#[pyfunction]
#[pyo3(signature = (setup_digest, slot_id, m, w, holder_pks, rnd=None, threads=1))]
#[allow(clippy::too_many_arguments)]
fn bcfe_place_slot(
    py: Python<'_>, setup_digest: &[u8], slot_id: u64, m: usize, w: &[u8], holder_pks: &[u8], rnd: Option<&[u8]>,
    threads: usize,
) -> PyResult<(Py<PyBytes>, Py<PyBytes>)> {
    key_bits_and_w(m, w).map_err(PyValueError::new_err)?; // m and w before the draw
    let k = draw(rnd, K_LEN)?;
    let pks = py
        .detach(|| place_slot(setup_digest, slot_id, m, w, holder_pks, &k, threads))
        .map_err(PyValueError::new_err)?;
    Ok((PyBytes::new(py, &pks).unbind(), PyBytes::new(py, &k).unbind()))
}

#[pyfunction]
#[pyo3(signature = (setup_digest, slot_id, m, w, k, pks, threads=1))]
#[allow(clippy::too_many_arguments)]
fn bcfe_unselected_ok(
    py: Python<'_>, setup_digest: &[u8], slot_id: u64, m: usize, w: &[u8], k: &[u8], pks: &[u8], threads: usize,
) -> PyResult<bool> {
    py.detach(|| unselected_ok(setup_digest, slot_id, m, w, k, pks, threads)).map_err(PyValueError::new_err)
}

/// The fused encryption: (R, label_cts, tables, decode, data_active); `seed`
/// (tests) is the 32 bytes of randomness, else they come from the OS.
#[pyfunction]
#[pyo3(signature = (circuit, mode, pp_digest, ct_nonce, slot_id, x, pks, seed=None, threads=1))]
#[allow(clippy::too_many_arguments)]
fn bcfe_encrypt_slot(
    py: Python<'_>, circuit: &[u8], mode: u8, pp_digest: &[u8], ct_nonce: &[u8], slot_id: u64, x: &[u8], pks: &[u8],
    seed: Option<&[u8]>, threads: usize,
) -> PyResult<Garbled> {
    let mode = mode_of(mode)?;
    let inst = Instance::new(pp_digest, ct_nonce, slot_id).map_err(PyValueError::new_err)?;
    let s = draw(seed, garble::SEED_LEN)?;
    let (big_r, cts, g) =
        py.detach(|| encrypt_slot(circuit, mode, &inst, x, pks, &s, threads)).map_err(PyValueError::new_err)?;
    Ok((
        PyBytes::new(py, &big_r).unbind(),
        PyBytes::new(py, &cts).unbind(),
        PyBytes::new(py, &g.tables).unbind(),
        PyBytes::new(py, &g.decode).unbind(),
        PyBytes::new(py, &g.data_active).unbind(),
    ))
}

/// The packed output bits of one (ciphertext, slot) for the holder of `scalars` and w.
#[pyfunction]
#[pyo3(signature = (circuit, mode, pp_digest, ct_nonce, slot_id, r, label_cts, tables, decode, data_active, scalars, w,
                    threads=1))]
#[allow(clippy::too_many_arguments)]
fn bcfe_decrypt_slot(
    py: Python<'_>, circuit: &[u8], mode: u8, pp_digest: &[u8], ct_nonce: &[u8], slot_id: u64, r: &[u8],
    label_cts: &[u8], tables: &[u8], decode: &[u8], data_active: &[u8], scalars: &[u8], w: &[u8], threads: usize,
) -> PyResult<Py<PyBytes>> {
    let mode = mode_of(mode)?;
    let inst = Instance::new(pp_digest, ct_nonce, slot_id).map_err(PyValueError::new_err)?;
    let y = py
        .detach(|| decrypt_slot(circuit, mode, &inst, r, label_cts, tables, decode, data_active, scalars, w, threads))
        .map_err(PyValueError::new_err)?;
    Ok(PyBytes::new(py, &garble::pack_bits(&y)).unbind())
}

pub fn register(m: &Bound<'_, PyModule>) -> PyResult<()> {
    for f in [
        wrap_pyfunction!(gc_garble, m)?, wrap_pyfunction!(gc_evaluate, m)?, wrap_pyfunction!(bcfe_new_slot, m)?,
        wrap_pyfunction!(bcfe_place_slot, m)?, wrap_pyfunction!(bcfe_unselected_ok, m)?,
        wrap_pyfunction!(bcfe_encrypt_slot, m)?, wrap_pyfunction!(bcfe_decrypt_slot, m)?,
    ] {
        m.add_function(f)?;
    }
    Ok(())
}
