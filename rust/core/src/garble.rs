//! Garbled boolean circuits of the bounded-key FE schemes (bcfe/*):
//! the canonical circuit encoding CIRC v1 with all its rejection rules, and the
//! half-gates garbling scheme of Zahur, Rosulek and Evans (ZRE15, Eurocrypt
//! 2015, Fig. 2) with free XOR and point-and-permute, plus the generator
//! half-gate for an AND gate with one wire the garbler knows (mode
//! "hardwired"). Byte for byte the values of the Python reference
//! (pyfe4ai.primitives.garbling): labels are 16-byte strings, held here as
//! u128 read little-endian, so that lsb(X) = X[0] & 1 is the low bit.
//!
//! * Randomness: delta and every 0-label from one 32-byte seed and the
//!   instance salt, SHAKE-256("PYFE4AI-V06-BCFE-GBPRG" || salt(56) || seed),
//!   read in the order delta (then delta[0] |= 1), the 0-labels of the key
//!   wires, those of the data wires (mode "labels" only). The salt is bound in
//!   so that a seed used twice under two salts still gives unrelated labels
//!   (defence in depth; the alternative was the seed alone).
//! * Gate hash: H_G(salt, idx, X) = SHAKE128("PFE6GC02" || salt(56) ||
//!   u64(idx) || X)[0:16], 88 input bytes in one block, one Keccak-f; the
//!   index idx is taken in gate order (NextIndex): two for an AND of two
//!   garbled wires, one for the generator half-gate, none otherwise.
//! * Wires that only the garbler knows (kind K: the data wires in mode
//!   "hardwired" and what depends on them alone) carry no labels and no table
//!   entries; whether a wire is K depends on the circuit and the mode only.
//!   Everything that depends on the values of x or on label bits is computed
//!   with masks, not branches.
//!
//! A circuit has no constant wires (the builder folds constants) and at most
//! 2^26 wires (m + n + n_gates, checked before anything is sized by the
//! counts). Delta, the seed, the 0-labels, the values of the garbler's wires
//! and the PRG state are wiped when a call ends.

use crate::keccak::{keccak_f, wipe_long, Sponge};

/// Magic of the canonical encoding.
pub const MAGIC: &[u8; 8] = b"PFE6CIRC";
/// Version byte of the canonical encoding.
pub const VERSION: u8 = 1;
/// Bytes of the header: magic, version, m, n, n_gates, n_out.
pub const HEADER_LEN: usize = 25;
/// Bytes of one gate: op, a, b.
pub const GATE_LEN: usize = 9;
/// Bytes of the instance salt: pp digest (32), ciphertext nonce (16), u64(slot id).
pub const SALT_LEN: usize = 56;
/// Bytes of the garbling seed.
pub const SEED_LEN: usize = 32;
/// Bytes of a label (and of a table entry).
pub const LABEL_LEN: usize = 16;
/// Domain separation of the seed expansion.
pub const PRG_DST: &[u8] = b"PYFE4AI-V06-BCFE-GBPRG";
/// Domain separation of the gate hash.
pub const GATE_DST: &[u8; 8] = b"PFE6GC02";
/// The wires (m + n + n_gates) must stay below this bound: 2^26 (64 Mi wires,
/// about 1 GiB of labels here), so that a short header cannot make a call
/// allocate without bound; the alternative was 2^31 with the per-wire state
/// sized by the labelled wires only.
pub const MAX_WIRES: u64 = 1 << 26;

#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum Op {
    Xor,
    And,
    Inv,
}

impl Op {
    /// The op byte of the encoding: 1 = XOR, 2 = AND, 3 = INV.
    pub fn code(self) -> u8 {
        match self {
            Op::Xor => 1,
            Op::And => 2,
            Op::Inv => 3,
        }
    }
}

/// A gate; its output wire is m + n + (its position). b is 0 for INV.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub struct Gate {
    pub op: Op,
    pub a: u32,
    pub b: u32,
}

/// Which wires carry labels: "hardwired" (the data wires are known to the
/// garbler only, kind K) or "labels" (the data wires are garbled too).
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum Mode {
    Hardwired,
    Labels,
}

impl Mode {
    /// 0: "hardwired", 1: "labels".
    pub fn from_code(code: u8) -> Option<Mode> {
        match code {
            0 => Some(Mode::Hardwired),
            1 => Some(Mode::Labels),
            _ => None,
        }
    }
}

/// A circuit C(w, x): key wires 0 .. m-1, data wires m .. m+n-1, the output of
/// gate i on wire m+n+i; inputs of a gate are earlier wires, two different
/// ones for XOR and AND (the builder folds XOR(a, a) and AND(a, a)).
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Circuit {
    pub m: usize,
    pub n: usize,
    pub gates: Vec<Gate>,
    pub outputs: Vec<u32>,
}

fn u32_at(b: &[u8], off: usize) -> u32 {
    u32::from_le_bytes(b[off..off + 4].try_into().unwrap())
}

/// A zeroed vector of `len` elements, or an error instead of aborting the
/// process when the allocation fails.
fn zeroed<T: Clone + Default>(len: usize) -> Result<Vec<T>, String> {
    let mut v: Vec<T> = Vec::new();
    v.try_reserve_exact(len).map_err(|_| "the circuit is too large to garble here".to_string())?;
    v.resize(len, T::default());
    Ok(v)
}

impl Circuit {
    /// Decode CIRC v1, refusing everything that is not a canonical encoding.
    pub fn parse(enc: &[u8]) -> Result<Circuit, String> {
        if enc.len() < HEADER_LEN {
            return Err("circuit encoding: shorter than its header".into());
        }
        if &enc[..8] != MAGIC {
            return Err("circuit encoding: not a CIRC encoding (magic)".into());
        }
        if enc[8] != VERSION {
            return Err("circuit encoding: unsupported version".into());
        }
        let (m, n, n_gates, n_out) = (u32_at(enc, 9), u32_at(enc, 13), u32_at(enc, 17), u32_at(enc, 21));
        if m == 0 {
            return Err("circuit encoding: no key wires (m = 0)".into());
        }
        if n_out == 0 {
            return Err("circuit encoding: no outputs".into());
        }
        let wires = m as u64 + n as u64 + n_gates as u64;
        if wires >= MAX_WIRES {
            return Err("circuit encoding: m + n + n_gates must be below 2^26".into());
        }
        let want = HEADER_LEN as u64 + GATE_LEN as u64 * n_gates as u64 + 4 * n_out as u64;
        if enc.len() as u64 != want {
            return Err("circuit encoding: the length does not match the counts".into());
        }
        let (m, n, n_gates, n_out) = (m as usize, n as usize, n_gates as usize, n_out as usize);
        let mut gates = Vec::with_capacity(n_gates);
        for i in 0..n_gates {
            let off = HEADER_LEN + GATE_LEN * i;
            let (a, b) = (u32_at(enc, off + 1), u32_at(enc, off + 5));
            let op = match enc[off] {
                1 => Op::Xor,
                2 => Op::And,
                3 => Op::Inv,
                _ => return Err(format!("circuit encoding: gate {i} has an unknown op")),
            };
            if op == Op::Inv && b != 0 {
                return Err(format!("circuit encoding: gate {i} is an INV with b != 0"));
            }
            if op != Op::Inv && a == b {
                return Err(format!("circuit encoding: gate {i} reads the same wire twice (a = b)"));
            }
            let o = (m + n + i) as u64;
            if a as u64 >= o || b as u64 >= o {
                return Err(format!("circuit encoding: gate {i} reads a wire that is not an earlier one"));
            }
            gates.push(Gate { op, a, b });
        }
        let base = HEADER_LEN + GATE_LEN * n_gates;
        let mut outputs = Vec::with_capacity(n_out);
        for k in 0..n_out {
            let o = u32_at(enc, base + 4 * k);
            if o as u64 >= wires {
                return Err(format!("circuit encoding: output {k} is not a wire of the circuit"));
            }
            outputs.push(o);
        }
        Ok(Circuit { m, n, gates, outputs })
    }

    /// The canonical encoding CIRC v1.
    pub fn encode(&self) -> Vec<u8> {
        let mut out = Vec::with_capacity(HEADER_LEN + GATE_LEN * self.gates.len() + 4 * self.outputs.len());
        out.extend_from_slice(MAGIC);
        out.push(VERSION);
        for v in [self.m, self.n, self.gates.len(), self.outputs.len()] {
            out.extend_from_slice(&(v as u32).to_le_bytes());
        }
        for g in &self.gates {
            out.push(g.op.code());
            out.extend_from_slice(&g.a.to_le_bytes());
            out.extend_from_slice(&g.b.to_le_bytes());
        }
        for o in &self.outputs {
            out.extend_from_slice(&o.to_le_bytes());
        }
        out
    }

    /// Number of wires: m + n + n_gates.
    pub fn wires(&self) -> usize {
        self.m + self.n + self.gates.len()
    }

    /// The kind of every wire, true for L (garbled), false for K (known to
    /// the garbler only): key wires L, data wires K in mode "hardwired" and L
    /// in mode "labels"; a gate's output is L if an input is (INV: its input's).
    pub fn kinds(&self, mode: Mode) -> Result<Vec<bool>, String> {
        let mut k: Vec<bool> = zeroed(self.wires())?;
        k[..self.m].fill(true);
        k[self.m..self.m + self.n].fill(mode == Mode::Labels);
        let base = self.m + self.n;
        for (i, g) in self.gates.iter().enumerate() {
            let (a, b) = (g.a as usize, g.b as usize);
            k[base + i] = match g.op {
                Op::Inv => k[a],
                _ => k[a] || k[b],
            };
        }
        Ok(k)
    }

    /// Table entries for these kinds: two per AND of two L wires, one per AND
    /// of an L and a K wire.
    pub fn table_entries(&self, kinds: &[bool]) -> usize {
        self.gates
            .iter()
            .filter(|g| g.op == Op::And)
            .map(|g| kinds[g.a as usize] as usize + kinds[g.b as usize] as usize)
            .sum()
    }

    /// Every output must be an L wire (always so after guard()).
    pub fn check_outputs(&self, kinds: &[bool]) -> Result<(), String> {
        match self.outputs.iter().position(|&o| !kinds[o as usize]) {
            Some(k) => Err(format!("output {k} is a wire known to the garbler only (kind K); it cannot be garbled")),
            None => Ok(()),
        }
    }

    /// C(w, x) in the clear (the tests' reference).
    pub fn eval_plain(&self, w: &[bool], x: &[bool]) -> Vec<bool> {
        let mut v = vec![false; self.wires()];
        v[..self.m].copy_from_slice(w);
        v[self.m..self.m + self.n].copy_from_slice(x);
        let base = self.m + self.n;
        for (i, g) in self.gates.iter().enumerate() {
            let (a, b) = (v[g.a as usize], v[g.b as usize]);
            v[base + i] = match g.op {
                Op::Xor => a ^ b,
                Op::And => a & b,
                Op::Inv => !a,
            };
        }
        self.outputs.iter().map(|&o| v[o as usize]).collect()
    }
}

/// Bits packed low bit first: bit i in bit i % 8 of byte i // 8.
pub fn pack_bits(bits: &[bool]) -> Vec<u8> {
    let mut out = vec![0u8; (bits.len() + 7) / 8];
    for (i, &b) in bits.iter().enumerate() {
        out[i / 8] |= (b as u8) << (i % 8);
    }
    out
}

/// The `count` bits of `packed` as 0 / 1 bytes; the length must be
/// ceil(count / 8) and the unused high bits of the last byte zero.
pub fn unpack_bits(packed: &[u8], count: usize, what: &str) -> Result<Vec<u8>, String> {
    if packed.len() != (count + 7) / 8 {
        return Err(format!("{what}: {count} bits take {} bytes, not {}", (count + 7) / 8, packed.len()));
    }
    if count % 8 != 0 && packed[count / 8] >> (count % 8) != 0 {
        return Err(format!("{what}: the unused high bits of the last byte must be 0"));
    }
    let mut out: Vec<u8> = zeroed(count)?;
    for (i, o) in out.iter_mut().enumerate() {
        *o = (packed[i / 8] >> (i % 8)) & 1;
    }
    Ok(out)
}

/// All ones if the low bit of v is 1, else zero (no branch).
#[inline(always)]
fn mask(v: u8) -> u128 {
    0u128.wrapping_sub((v & 1) as u128)
}

/// All ones if lsb(x), else zero (no branch).
#[inline(always)]
fn lsb_mask(x: u128) -> u128 {
    0u128.wrapping_sub(x & 1)
}

#[inline(always)]
fn label_at(b: &[u8], i: usize) -> u128 {
    u128::from_le_bytes(b[LABEL_LEN * i..LABEL_LEN * (i + 1)].try_into().unwrap())
}

fn wipe_labels(v: &mut [u128]) {
    for x in v.iter_mut() {
        unsafe { std::ptr::write_volatile(x, 0) };
    }
    std::sync::atomic::compiler_fence(std::sync::atomic::Ordering::SeqCst);
}

fn wipe_label(x: &mut u128) {
    unsafe { std::ptr::write_volatile(x, 0) };
    std::sync::atomic::compiler_fence(std::sync::atomic::Ordering::SeqCst);
}

/// The gate hash H_G(salt, idx, X) = SHAKE128("PFE6GC02" || salt || u64(idx) || X)[0:16]
/// of one instance: the 88 input bytes and the padding (0x1F at byte 88, 0x80
/// at byte 167) laid out once, idx and X put in per call, one Keccak-f. The
/// working state (it holds labels) is wiped on drop.
pub struct GateHash {
    tmpl: [u64; 25],
    st: [u64; 25],
}

impl GateHash {
    pub fn new(salt: &[u8]) -> Result<GateHash, String> {
        if salt.len() != SALT_LEN {
            return Err(format!("the salt has {SALT_LEN} bytes, not {}", salt.len()));
        }
        let mut t = [0u64; 25];
        t[0] = u64::from_le_bytes(*GATE_DST);
        for (i, lane) in salt.chunks_exact(8).enumerate() {
            t[1 + i] = u64::from_le_bytes(lane.try_into().unwrap());
        }
        // lane 8: idx; lanes 9, 10: X; byte 88 = lane 11, byte 0; byte 167 = lane 20, byte 7
        t[11] = 0x1F;
        t[20] = 0x80u64 << 56;
        Ok(GateHash { tmpl: t, st: [0u64; 25] })
    }

    #[inline(always)]
    pub fn h(&mut self, x: u128, idx: u64) -> u128 {
        self.st = self.tmpl;
        self.st[8] = idx;
        self.st[9] = x as u64;
        self.st[10] = (x >> 64) as u64;
        keccak_f(&mut self.st);
        (self.st[0] as u128) | ((self.st[1] as u128) << 64)
    }
}

impl Drop for GateHash {
    fn drop(&mut self) {
        for lane in self.st.iter_mut() {
            unsafe { std::ptr::write_volatile(lane, 0) };
        }
        std::sync::atomic::compiler_fence(std::sync::atomic::Ordering::SeqCst);
    }
}

/// What garble() returns, as the bytes of the backend operation gc_garble:
/// delta, the 0-labels of the key wires (16 bytes each), the active labels of
/// the data wires (mode "labels"; empty otherwise), the table entries (16
/// bytes each, in gate order) and the decoding bits (packed). Delta, key0 and
/// data_active are wiped on drop.
pub struct Garbling {
    pub delta: [u8; 16],
    pub key0: Vec<u8>,
    pub data_active: Vec<u8>,
    pub tables: Vec<u8>,
    pub decode: Vec<u8>,
}

/// Sizes only: no secret in a Debug output.
impl std::fmt::Debug for Garbling {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("Garbling")
            .field("key_labels", &(self.key0.len() / LABEL_LEN))
            .field("data_labels", &(self.data_active.len() / LABEL_LEN))
            .field("table_entries", &(self.tables.len() / LABEL_LEN))
            .finish_non_exhaustive()
    }
}

impl Drop for Garbling {
    fn drop(&mut self) {
        crate::rlwe::wipe_bytes(&mut self.delta);
        wipe_long(&mut self.key0);
        wipe_long(&mut self.data_active);
    }
}

/// Garble the circuit (CIRC v1 bytes) for the data bits x (n bits, packed)
/// with the randomness of `seed` (32 bytes) and the instance salt (56 bytes).
///
/// Runs with the processor's data-independent-timing mode set where it has one
/// (`dit::Guard`; aarch64 with FEAT_DIT): measured without it, the shape of the
/// garbling's time distribution depended on the seed. The result is the same.
pub fn garble(circuit: &[u8], mode: Mode, salt: &[u8], x: &[u8], seed: &[u8]) -> Result<Garbling, String> {
    let _dit = crate::dit::Guard::new();
    garble_unguarded(circuit, mode, salt, x, seed)
}

/// [`garble`] without its `dit::Guard`, for the timing harness only (`GARBLE_DIT=0` of the `ct_dudect` example),
/// which compares the garbling with and without DIT; the library calls [`garble`].
#[doc(hidden)]
pub fn garble_unguarded(circuit: &[u8], mode: Mode, salt: &[u8], x: &[u8], seed: &[u8]) -> Result<Garbling, String> {
    #[cfg(test)]
    tests::DIT_DURING_GARBLING.with(|c| c.set(Some(crate::dit::is_set())));
    let c = Circuit::parse(circuit)?;
    let mut hg = GateHash::new(salt)?;
    if seed.len() != SEED_LEN {
        return Err(format!("the seed has {SEED_LEN} bytes, not {}", seed.len()));
    }
    let kinds = c.kinds(mode)?;
    c.check_outputs(&kinds)?;
    let mut xb = unpack_bits(x, c.n, "x")?;
    let res = garble_parsed(&c, &kinds, mode, &mut hg, &xb, salt, seed);
    wipe_long(&mut xb);
    res
}

fn garble_parsed(
    c: &Circuit, kinds: &[bool], mode: Mode, hg: &mut GateHash, xb: &[u8], salt: &[u8], seed: &[u8],
) -> Result<Garbling, String> {
    let (m, n) = (c.m, c.n);
    let mut lab: Vec<u128> = zeroed(c.wires())?; // 0-labels of the L wires
    let mut val: Vec<u8> = zeroed(c.wires())?; // values of the K wires (0 / 1)
    let mut out = Garbling {
        delta: [0u8; 16],
        key0: Vec::new(),
        data_active: Vec::new(),
        tables: Vec::new(),
        decode: Vec::new(),
    };
    let entries = c.table_entries(kinds);
    let alloc = out
        .key0
        .try_reserve_exact(LABEL_LEN * m)
        .and_then(|_| out.tables.try_reserve_exact(LABEL_LEN * entries))
        .and_then(|_| out.data_active.try_reserve_exact(if mode == Mode::Labels { LABEL_LEN * n } else { 0 }));
    if alloc.is_err() {
        wipe_labels(&mut lab);
        return Err("the circuit is too large to garble here".into());
    }

    // delta, the key wires' 0-labels, the data wires' 0-labels (mode "labels")
    let mut msg = Vec::with_capacity(PRG_DST.len() + SALT_LEN + SEED_LEN);
    msg.extend_from_slice(PRG_DST);
    msg.extend_from_slice(salt);
    msg.extend_from_slice(seed);
    let mut prg = Sponge::new(&msg, 136);
    wipe_long(&mut msg);
    let mut buf = [0u8; LABEL_LEN];
    prg.squeeze(&mut buf);
    let mut delta = u128::from_le_bytes(buf) | 1;
    for l in lab[..m].iter_mut() {
        prg.squeeze(&mut buf);
        *l = u128::from_le_bytes(buf);
    }
    match mode {
        Mode::Labels => {
            for i in 0..n {
                prg.squeeze(&mut buf);
                let l0 = u128::from_le_bytes(buf);
                lab[m + i] = l0;
                out.data_active.extend_from_slice(&(l0 ^ (delta & mask(xb[i]))).to_le_bytes());
            }
        }
        Mode::Hardwired => val[m..m + n].copy_from_slice(xb),
    }
    crate::rlwe::wipe_bytes(&mut buf);
    drop(prg); // wipes the sponge

    let base = m + n;
    let mut idx: u64 = 0;
    for (i, g) in c.gates.iter().enumerate() {
        let (a, b, o) = (g.a as usize, g.b as usize, base + i);
        match g.op {
            Op::Xor => match (kinds[a], kinds[b]) {
                (true, true) => lab[o] = lab[a] ^ lab[b],
                (true, false) => lab[o] = lab[a] ^ (delta & mask(val[b])),
                (false, true) => lab[o] = lab[b] ^ (delta & mask(val[a])),
                (false, false) => val[o] = val[a] ^ val[b],
            },
            Op::Inv => {
                if kinds[a] {
                    lab[o] = lab[a] ^ delta;
                } else {
                    val[o] = val[a] ^ 1;
                }
            }
            Op::And => match (kinds[a], kinds[b]) {
                (true, true) => {
                    let (a0, b0) = (lab[a], lab[b]);
                    let (j, j1) = (idx, idx + 1);
                    idx += 2;
                    let ha0 = hg.h(a0, j);
                    let ha1 = hg.h(a0 ^ delta, j);
                    let hb0 = hg.h(b0, j1);
                    let hb1 = hg.h(b0 ^ delta, j1);
                    let tg = ha0 ^ ha1 ^ (delta & lsb_mask(b0));
                    let wg = ha0 ^ (tg & lsb_mask(a0));
                    let te = hb0 ^ hb1 ^ a0;
                    let we = hb0 ^ ((te ^ a0) & lsb_mask(b0));
                    out.tables.extend_from_slice(&tg.to_le_bytes());
                    out.tables.extend_from_slice(&te.to_le_bytes());
                    lab[o] = wg ^ we;
                }
                (true, false) | (false, true) => {
                    let (l, k) = if kinds[a] { (a, b) } else { (b, a) };
                    let l0 = lab[l];
                    let j = idx;
                    idx += 1;
                    let h0 = hg.h(l0, j);
                    let h1 = hg.h(l0 ^ delta, j);
                    let t = h0 ^ h1 ^ (delta & mask(val[k]));
                    out.tables.extend_from_slice(&t.to_le_bytes());
                    lab[o] = h0 ^ (t & lsb_mask(l0));
                }
                (false, false) => val[o] = val[a] & val[b],
            },
        }
    }
    debug_assert_eq!(out.tables.len(), LABEL_LEN * entries);

    for l in &lab[..m] {
        out.key0.extend_from_slice(&l.to_le_bytes());
    }
    let d: Vec<bool> = c.outputs.iter().map(|&o| lab[o as usize] & 1 == 1).collect();
    out.decode = pack_bits(&d);
    out.delta = delta.to_le_bytes();
    wipe_label(&mut delta);
    wipe_labels(&mut lab);
    wipe_long(&mut val);
    Ok(out)
}

/// Evaluate the garbled circuit with the active labels of the key wires (16
/// bytes each, m of them) and of the data wires (mode "labels": n of them;
/// mode "hardwired": none): the n_out output bits.
pub fn evaluate(
    circuit: &[u8], mode: Mode, salt: &[u8], tables: &[u8], decode: &[u8], key_active: &[u8], data_active: &[u8],
) -> Result<Vec<bool>, String> {
    let c = Circuit::parse(circuit)?;
    let mut hg = GateHash::new(salt)?;
    let kinds = c.kinds(mode)?;
    c.check_outputs(&kinds)?;
    let entries = c.table_entries(&kinds);
    if tables.len() as u64 != LABEL_LEN as u64 * entries as u64 {
        return Err(format!("the circuit has {entries} table entries of {LABEL_LEN} bytes; got {} bytes", tables.len()));
    }
    let d = unpack_bits(decode, c.outputs.len(), "decode")?;
    if key_active.len() as u64 != LABEL_LEN as u64 * c.m as u64 {
        return Err(format!("{} key labels of {LABEL_LEN} bytes expected; got {} bytes", c.m, key_active.len()));
    }
    let n_data = if mode == Mode::Labels { c.n } else { 0 };
    if data_active.len() as u64 != LABEL_LEN as u64 * n_data as u64 {
        return Err(format!("{n_data} data labels of {LABEL_LEN} bytes expected; got {} bytes", data_active.len()));
    }
    let (m, n) = (c.m, c.n);
    let mut lab: Vec<u128> = zeroed(c.wires())?;
    for i in 0..m {
        lab[i] = label_at(key_active, i);
    }
    for i in 0..n_data {
        lab[m + i] = label_at(data_active, i);
    }
    let base = m + n;
    let (mut idx, mut ti): (u64, usize) = (0, 0);
    for (i, g) in c.gates.iter().enumerate() {
        let (a, b, o) = (g.a as usize, g.b as usize, base + i);
        match g.op {
            Op::Xor => match (kinds[a], kinds[b]) {
                (true, true) => lab[o] = lab[a] ^ lab[b],
                (true, false) => lab[o] = lab[a],
                (false, true) => lab[o] = lab[b],
                (false, false) => {}
            },
            Op::Inv => {
                if kinds[a] {
                    lab[o] = lab[a];
                }
            }
            Op::And => match (kinds[a], kinds[b]) {
                (true, true) => {
                    let (wa, wb) = (lab[a], lab[b]);
                    let (tg, te) = (label_at(tables, ti), label_at(tables, ti + 1));
                    ti += 2;
                    let wg = hg.h(wa, idx) ^ (tg & lsb_mask(wa));
                    let we = hg.h(wb, idx + 1) ^ ((te ^ wa) & lsb_mask(wb));
                    idx += 2;
                    lab[o] = wg ^ we;
                }
                (true, false) | (false, true) => {
                    let l = if kinds[a] { a } else { b };
                    let wl = lab[l];
                    let t = label_at(tables, ti);
                    ti += 1;
                    lab[o] = hg.h(wl, idx) ^ (t & lsb_mask(wl));
                    idx += 1;
                }
                (false, false) => {}
            },
        }
    }
    let y = c.outputs.iter().zip(d.iter()).map(|(&o, &dk)| ((lab[o as usize] & 1) as u8 ^ dk) == 1).collect();
    wipe_labels(&mut lab);
    Ok(y)
}

#[cfg(test)]
mod tests {
    use super::*;

    /// xorshift64* (the tests' only randomness).
    struct Rng(u64);

    impl Rng {
        fn next(&mut self) -> u64 {
            self.0 ^= self.0 >> 12;
            self.0 ^= self.0 << 25;
            self.0 ^= self.0 >> 27;
            self.0.wrapping_mul(0x2545F4914F6CDD1D)
        }
        fn below(&mut self, k: usize) -> usize {
            (self.next() % k as u64) as usize
        }
        fn bits(&mut self, k: usize) -> Vec<bool> {
            (0..k).map(|_| self.next() & 1 == 1).collect()
        }
        fn bytes(&mut self, k: usize) -> Vec<u8> {
            (0..k).map(|_| self.next() as u8).collect()
        }
    }

    fn random_circuit(rng: &mut Rng, m: usize, n: usize, n_gates: usize, n_out: usize, mode: Mode) -> Circuit {
        let mut gates = Vec::with_capacity(n_gates);
        for i in 0..n_gates {
            let o = m + n + i;
            let op = if o < 2 { Op::Inv } else { [Op::Xor, Op::And, Op::Inv][rng.below(3)] };
            let a = rng.below(o) as u32;
            let b = if op == Op::Inv { 0 } else { ((a as usize + 1 + rng.below(o - 1)) % o) as u32 };
            gates.push(Gate { op, a, b });
        }
        let mut c = Circuit { m, n, gates, outputs: vec![] };
        let kinds = c.kinds(mode).unwrap();
        let l: Vec<u32> = (0..c.wires() as u32).filter(|&v| kinds[v as usize]).collect();
        c.outputs = (0..n_out).map(|_| l[rng.below(l.len())]).collect();
        c
    }

    /// Active key labels for w: key0[i] ^ w_i delta.
    fn key_active(g: &Garbling, w: &[bool]) -> Vec<u8> {
        let delta = u128::from_le_bytes(g.delta);
        let mut out = vec![];
        for (i, &b) in w.iter().enumerate() {
            out.extend_from_slice(&(label_at(&g.key0, i) ^ if b { delta } else { 0 }).to_le_bytes());
        }
        out
    }

    fn roundtrip(c: &Circuit, mode: Mode, salt: &[u8], seed: &[u8], w: &[bool], x: &[bool]) -> Vec<bool> {
        let enc = c.encode();
        let g = garble(&enc, mode, salt, &pack_bits(x), seed).unwrap();
        evaluate(&enc, mode, salt, &g.tables, &g.decode, &key_active(&g, w), &g.data_active).unwrap()
    }

    thread_local! {
        /// Whether DIT was set when `garble_unguarded` started on this thread (test builds only).
        pub(super) static DIT_DURING_GARBLING: std::cell::Cell<Option<bool>> = const { std::cell::Cell::new(None) };
    }

    fn dit_during(f: impl FnOnce()) -> bool {
        DIT_DURING_GARBLING.with(|c| c.set(None));
        f();
        DIT_DURING_GARBLING.with(|c| c.get()).expect("the garbling ran")
    }

    #[test]
    fn garble_runs_with_dit_where_supported() {
        let mut rng = Rng(29);
        let c = random_circuit(&mut rng, 6, 10, 80, 4, Mode::Hardwired);
        let enc = c.encode();
        let (salt, seed, x) = (rng.bytes(SALT_LEN), rng.bytes(SEED_LEN), pack_bits(&rng.bits(10)));
        assert!(!crate::dit::is_set());
        let guarded = dit_during(|| {
            garble(&enc, Mode::Hardwired, &salt, &x, &seed).unwrap();
        });
        assert_eq!(guarded, crate::dit::supported(), "garble() sets DIT where the processor has it");
        assert!(!crate::dit::is_set(), "and puts the previous value back");
        let unguarded = dit_during(|| {
            garble_unguarded(&enc, Mode::Hardwired, &salt, &x, &seed).unwrap();
        });
        assert!(!unguarded, "garble_unguarded leaves DIT as it was");
        // an error return restores DIT too
        assert!(garble(&enc, Mode::Hardwired, &salt, &x, &seed[..31]).is_err());
        assert!(!crate::dit::is_set());
    }

    #[test]
    fn the_garbling_is_the_same_with_and_without_dit() {
        let mut rng = Rng(23);
        for mode in [Mode::Labels, Mode::Hardwired] {
            for _ in 0..8 {
                let c = random_circuit(&mut rng, 6, 10, 80, 4, mode);
                let enc = c.encode();
                let salt = rng.bytes(SALT_LEN);
                let seed = rng.bytes(SEED_LEN);
                let x = pack_bits(&rng.bits(10));
                let a = garble(&enc, mode, &salt, &x, &seed).unwrap();
                assert!(!crate::dit::is_set(), "garble() restores DIT");
                let b = garble_unguarded(&enc, mode, &salt, &x, &seed).unwrap();
                assert_eq!(a.delta, b.delta);
                assert_eq!(a.key0, b.key0);
                assert_eq!(a.data_active, b.data_active);
                assert_eq!(a.tables, b.tables);
                assert_eq!(a.decode, b.decode);
            }
        }
    }

    #[test]
    fn gate_hash_is_shake128_of_the_88_input_bytes() {
        let mut rng = Rng(7);
        for _ in 0..20 {
            let salt = rng.bytes(SALT_LEN);
            let idx = rng.next();
            let x = u128::from_le_bytes(rng.bytes(16).try_into().unwrap());
            let mut msg = GATE_DST.to_vec();
            msg.extend_from_slice(&salt);
            msg.extend_from_slice(&idx.to_le_bytes());
            msg.extend_from_slice(&x.to_le_bytes());
            assert_eq!(msg.len(), 88);
            let mut want = [0u8; 16];
            Sponge::new(&msg, 168).squeeze(&mut want);
            assert_eq!(GateHash::new(&salt).unwrap().h(x, idx), u128::from_le_bytes(want));
        }
    }

    #[test]
    fn labels_and_delta_come_from_the_seed_in_order() {
        let mut rng = Rng(11);
        let c = random_circuit(&mut rng, 5, 9, 40, 3, Mode::Labels);
        let (salt, seed) = (rng.bytes(SALT_LEN), rng.bytes(SEED_LEN));
        let x = rng.bits(9);
        let mut msg = PRG_DST.to_vec();
        msg.extend_from_slice(&salt);
        msg.extend_from_slice(&seed);
        let mut stream = vec![0u8; 16 * (1 + 5 + 9)];
        crate::keccak::shake256(&msg, &mut stream);
        let delta = label_at(&stream, 0) | 1;
        for mode in [Mode::Labels, Mode::Hardwired] {
            let c = if mode == Mode::Labels { c.clone() } else { random_circuit(&mut rng, 5, 9, 40, 3, mode) };
            let g = garble(&c.encode(), mode, &salt, &pack_bits(&x), &seed).unwrap();
            assert_eq!(u128::from_le_bytes(g.delta), delta);
            assert_eq!(g.delta[0] & 1, 1);
            assert_eq!(g.key0, stream[16..16 * 6]);
            if mode == Mode::Labels {
                for i in 0..9 {
                    let want = label_at(&stream, 6 + i) ^ if x[i] { delta } else { 0 };
                    assert_eq!(label_at(&g.data_active, i), want);
                }
            } else {
                assert!(g.data_active.is_empty());
            }
        }
    }

    #[test]
    fn garbled_evaluation_is_the_plain_one_in_both_modes() {
        let mut rng = Rng(2024);
        for mode in [Mode::Hardwired, Mode::Labels] {
            for trial in 0..60 {
                let (m, n) = (1 + rng.below(8), rng.below(8));
                let (n_gates, n_out) = (rng.below(200), 1 + rng.below(6));
                let c = random_circuit(&mut rng, m, n, n_gates, n_out, mode);
                let (salt, seed) = (rng.bytes(SALT_LEN), rng.bytes(SEED_LEN));
                for _ in 0..4 {
                    let (w, x) = (rng.bits(m), rng.bits(n));
                    let want = c.eval_plain(&w, &x);
                    assert_eq!(roundtrip(&c, mode, &salt, &seed, &w, &x), want, "{mode:?} trial {trial}");
                }
            }
        }
    }

    #[test]
    fn every_gate_kind_evaluates_correctly() {
        // key 0, 1; data 2, 3 (K in mode hardwired): every op on every pair of kinds
        let mut gates = vec![];
        for op in [Op::Xor, Op::And] {
            for (a, b) in [(0, 1), (0, 2), (2, 0), (2, 3)] {
                gates.push(Gate { op, a, b });
            }
        }
        gates.push(Gate { op: Op::Inv, a: 0, b: 0 });
        gates.push(Gate { op: Op::Inv, a: 2, b: 0 });
        let k = gates.len() as u32;
        // make every gate output an L wire: AND it with key wire 1 (and XOR the INV(K))
        for i in 0..k {
            gates.push(Gate { op: Op::Xor, a: 4 + i, b: 1 });
        }
        let outputs = (4 + k..4 + 2 * k).collect();
        let c = Circuit { m: 2, n: 2, gates, outputs };
        let (salt, seed) = ([3u8; SALT_LEN], [9u8; SEED_LEN]);
        for mode in [Mode::Hardwired, Mode::Labels] {
            for v in 0..16u8 {
                let w = [v & 1 == 1, v & 2 == 2];
                let x = [v & 4 == 4, v & 8 == 8];
                assert_eq!(roundtrip(&c, mode, &salt, &seed, &w, &x), c.eval_plain(&w, &x), "{mode:?} {v}");
            }
        }
    }

    #[test]
    fn hash_indices_follow_gate_order() {
        // m = 2 (wires 0, 1), n = 1 (wire 2, K in mode hardwired):
        // g0 = AND(0, 2) -> 3 (L, K: idx 0), g1 = XOR(0, 1) -> 4 (no idx),
        // g2 = AND(4, 1) -> 5 (L, L: idx 1, 2), g3 = AND(2, 5) -> 6 (K, L: idx 3),
        // g4 = INV(6) -> 7, g5 = INV(2) -> 8 (K), g6 = AND(2, 8) -> 9 (K, K: none),
        // g7 = AND(7, 3) -> 10 (idx 4, 5)
        let gates = vec![
            Gate { op: Op::And, a: 0, b: 2 },
            Gate { op: Op::Xor, a: 0, b: 1 },
            Gate { op: Op::And, a: 4, b: 1 },
            Gate { op: Op::And, a: 2, b: 5 },
            Gate { op: Op::Inv, a: 6, b: 0 },
            Gate { op: Op::Inv, a: 2, b: 0 },
            Gate { op: Op::And, a: 2, b: 8 },
            Gate { op: Op::And, a: 7, b: 3 },
        ];
        let c = Circuit { m: 2, n: 1, gates, outputs: vec![10, 5] };
        let salt: Vec<u8> = (0..SALT_LEN as u8).collect();
        let seed = [0x5au8; SEED_LEN];
        for xv in [false, true] {
            let g = garble(&c.encode(), Mode::Hardwired, &salt, &pack_bits(&[xv]), &seed).unwrap();
            let d = u128::from_le_bytes(g.delta);
            let mut h = GateHash::new(&salt).unwrap();
            let sel = |b: bool, v: u128| if b { v } else { 0 };
            let lsb = |v: u128| v & 1 == 1;
            let (w0, w1) = (label_at(&g.key0, 0), label_at(&g.key0, 1));
            // g0: generator half-gate, idx 0
            let t0 = h.h(w0, 0) ^ h.h(w0 ^ d, 0) ^ sel(xv, d);
            let w3 = h.h(w0, 0) ^ sel(lsb(w0), t0);
            let w4 = w0 ^ w1;
            // g2: full AND, idx 1 and 2
            let tg = h.h(w4, 1) ^ h.h(w4 ^ d, 1) ^ sel(lsb(w1), d);
            let te = h.h(w1, 2) ^ h.h(w1 ^ d, 2) ^ w4;
            let w5 = h.h(w4, 1) ^ sel(lsb(w4), tg) ^ h.h(w1, 2) ^ sel(lsb(w1), te ^ w4);
            // g3: K input first, idx 3
            let t3 = h.h(w5, 3) ^ h.h(w5 ^ d, 3) ^ sel(xv, d);
            let w6 = h.h(w5, 3) ^ sel(lsb(w5), t3);
            let w7 = w6 ^ d;
            // g7: full AND, idx 4 and 5
            let tg6 = h.h(w7, 4) ^ h.h(w7 ^ d, 4) ^ sel(lsb(w3), d);
            let te6 = h.h(w3, 5) ^ h.h(w3 ^ d, 5) ^ w7;
            let w9 = h.h(w7, 4) ^ sel(lsb(w7), tg6) ^ h.h(w3, 5) ^ sel(lsb(w3), te6 ^ w7);
            let want: Vec<u8> = [t0, tg, te, t3, tg6, te6].iter().flat_map(|t| t.to_le_bytes()).collect();
            assert_eq!(g.tables, want);
            assert_eq!(g.decode, pack_bits(&[lsb(w9), lsb(w5)]));
        }
    }

    #[test]
    fn bits_pack_low_bit_first() {
        let bits = [true, false, false, true, false, false, false, false, true, true];
        assert_eq!(pack_bits(&bits), vec![0b0000_1001, 0b0000_0011]);
        let back = unpack_bits(&[0b0000_1001, 0b0000_0011], 10, "x").unwrap();
        assert_eq!(back, bits.iter().map(|&b| b as u8).collect::<Vec<_>>());
        assert!(unpack_bits(&[0, 0b0000_0100], 10, "x").is_err()); // high bit set
        assert!(unpack_bits(&[0], 10, "x").is_err());
        assert!(unpack_bits(&[0, 0, 0], 10, "x").is_err());
        assert_eq!(unpack_bits(&[], 0, "x").unwrap(), Vec::<u8>::new());
        assert!(unpack_bits(&[0], 0, "x").is_err());
    }

    fn header(m: u32, n: u32, n_gates: u32, n_out: u32) -> Vec<u8> {
        let mut v = MAGIC.to_vec();
        v.push(VERSION);
        for x in [m, n, n_gates, n_out] {
            v.extend_from_slice(&x.to_le_bytes());
        }
        v
    }

    fn gate(v: &mut Vec<u8>, op: u8, a: u32, b: u32) {
        v.push(op);
        v.extend_from_slice(&a.to_le_bytes());
        v.extend_from_slice(&b.to_le_bytes());
    }

    #[test]
    fn parse_refuses_non_canonical_encodings() {
        // valid: m = 1, n = 1, AND(0, 1) -> 2, INV(2) -> 3, outputs 3
        let mut ok = header(1, 1, 2, 1);
        gate(&mut ok, 2, 0, 1);
        gate(&mut ok, 3, 2, 0);
        ok.extend_from_slice(&3u32.to_le_bytes());
        let c = Circuit::parse(&ok).unwrap();
        assert_eq!(c.encode(), ok);
        let bad = |f: &dyn Fn(&mut Vec<u8>)| {
            let mut v = ok.clone();
            f(&mut v);
            Circuit::parse(&v).unwrap_err()
        };
        assert!(bad(&|v| v[0] = b'Q').contains("magic"));
        assert!(bad(&|v| v[8] = 2).contains("version"));
        assert!(bad(&|v| v.truncate(24)).contains("header"));
        assert!(bad(&|v| { v.pop(); }).contains("length"));
        assert!(bad(&|v| v.push(0)).contains("length"));
        assert!(bad(&|v| v[25] = 0).contains("unknown op"));
        assert!(bad(&|v| v[25] = 4).contains("unknown op"));
        assert!(bad(&|v| v[34 + 5] = 1).contains("INV"));
        assert!(bad(&|v| v[25 + 5] = 0).contains("same wire")); // AND(0, 0)
        assert!(bad(&|v| { v[25] = 1; v[25 + 1] = 1; v[25 + 5] = 1 }).contains("same wire")); // XOR(1, 1)
        let mut xor = ok.clone();
        xor[34] = 1; // INV(2) -> 3 becomes XOR(2, 0) -> 3: two different wires, fine
        assert!(Circuit::parse(&xor).is_ok());
        assert!(bad(&|v| v[25 + 5] = 2).contains("earlier")); // AND(0, 2) -> 2
        assert!(bad(&|v| v[25 + 1] = 7).contains("earlier"));
        assert!(bad(&|v| v[34 + 1] = 3).contains("earlier")); // INV(3) -> 3
        assert!(bad(&|v| v[43] = 4).contains("output")); // wires 0..3
        assert!(bad(&|v| v[9] = 0).contains("m = 0"));
        // n_out = 0 (and the length then off by 4: the count rule comes first)
        assert!(bad(&|v| v[21] = 0).contains("no outputs"));
        // m + n + n_gates >= 2^26, refused from the header alone (nothing sized by the counts)
        let mut big = header(1 << 25, 1 << 25, 0, 1);
        big.extend_from_slice(&0u32.to_le_bytes());
        assert!(Circuit::parse(&big).unwrap_err().contains("2^26"));
        let mut big = header((1 << 26) - 2, 1, 1, 1); // exactly 2^26 wires
        gate(&mut big, 3, 0, 0);
        big.extend_from_slice(&0u32.to_le_bytes());
        assert!(Circuit::parse(&big).unwrap_err().contains("2^26"));
        for (m, n) in [(u32::MAX, 0), (1, u32::MAX), (1 << 31, 1 << 31)] {
            let mut huge = header(m, n, 0, 1);
            huge.extend_from_slice(&0u32.to_le_bytes());
            assert!(Circuit::parse(&huge).unwrap_err().contains("2^26"));
        }
        let mut fits = header((1 << 26) - 3, 1, 1, 1); // 2^26 - 1 wires
        gate(&mut fits, 3, 0, 0);
        fits.extend_from_slice(&0u32.to_le_bytes());
        assert!(Circuit::parse(&fits).is_ok());
        // counts near 2^32 do not overflow the length check
        let huge = header(1, 0, 0, u32::MAX);
        assert!(Circuit::parse(&huge).is_err());
        // no gates, n = 0: an output on a key wire
        let mut tiny = header(1, 0, 0, 1);
        tiny.extend_from_slice(&0u32.to_le_bytes());
        assert!(Circuit::parse(&tiny).is_ok());
    }

    #[test]
    fn garbling_refuses_bad_inputs() {
        // m = 1, n = 2: XOR(1, 2) -> 3 (K in mode hardwired), AND(0, 3) -> 4
        let mut enc = header(1, 2, 2, 1);
        gate(&mut enc, 1, 1, 2);
        gate(&mut enc, 2, 0, 3);
        enc.extend_from_slice(&4u32.to_le_bytes());
        let (salt, seed) = ([0u8; SALT_LEN], [0u8; SEED_LEN]);
        assert!(garble(&enc, Mode::Hardwired, &salt, &[3], &seed).is_ok());
        assert!(garble(&enc, Mode::Hardwired, &salt[..55], &[3], &seed).unwrap_err().contains("salt"));
        assert!(garble(&enc, Mode::Hardwired, &salt, &[3], &seed[..31]).unwrap_err().contains("seed"));
        assert!(garble(&enc, Mode::Hardwired, &salt, &[4], &seed).is_err()); // a bit beyond n
        assert!(garble(&enc, Mode::Hardwired, &salt, &[3, 0], &seed).is_err());
        assert!(garble(&enc, Mode::Hardwired, &salt, &[], &seed).is_err());
        // an output of kind K: refused in mode hardwired, fine in mode labels
        let k = enc.len() - 4;
        enc[k] = 3;
        assert!(garble(&enc, Mode::Hardwired, &salt, &[3], &seed).unwrap_err().contains("kind K"));
        assert!(garble(&enc, Mode::Labels, &salt, &[3], &seed).is_ok());
        enc[k] = 2;
        assert!(garble(&enc, Mode::Hardwired, &salt, &[3], &seed).unwrap_err().contains("kind K"));
    }

    #[test]
    fn evaluation_refuses_wrong_counts() {
        let mut rng = Rng(5);
        for mode in [Mode::Hardwired, Mode::Labels] {
            let c = random_circuit(&mut rng, 3, 4, 50, 5, mode);
            let enc = c.encode();
            let salt = rng.bytes(SALT_LEN);
            let g = garble(&enc, mode, &salt, &[0b1010], &rng.bytes(SEED_LEN)).unwrap();
            let ka = key_active(&g, &[true, false, true]);
            let ev = |t: &[u8], d: &[u8], k: &[u8], x: &[u8]| evaluate(&enc, mode, &salt, t, d, k, x);
            assert!(ev(&g.tables, &g.decode, &ka, &g.data_active).is_ok());
            let mut longer = g.tables.clone();
            longer.extend_from_slice(&[0u8; 16]);
            assert!(ev(&longer, &g.decode, &ka, &g.data_active).unwrap_err().contains("table"));
            assert!(ev(&g.tables[..g.tables.len() - 1], &g.decode, &ka, &g.data_active).is_err());
            assert!(ev(&g.tables, &[], &ka, &g.data_active).is_err());
            assert!(ev(&g.tables, &[0xff], &ka, &g.data_active).is_err()); // 5 bits: high bits set
            assert!(ev(&g.tables, &g.decode, &ka[16..], &g.data_active).is_err());
            let mut more = ka.clone();
            more.extend_from_slice(&[0u8; 16]);
            assert!(ev(&g.tables, &g.decode, &more, &g.data_active).is_err());
            let mut data = g.data_active.clone();
            data.extend_from_slice(&[0u8; 16]);
            assert!(ev(&g.tables, &g.decode, &ka, &data).unwrap_err().contains("data labels"));
            assert!(evaluate(&enc, mode, &salt[1..], &g.tables, &g.decode, &ka, &g.data_active).is_err());
        }
    }

    #[test]
    fn the_salt_and_the_seed_change_everything() {
        let mut rng = Rng(99);
        let c = random_circuit(&mut rng, 4, 4, 100, 2, Mode::Hardwired);
        let enc = c.encode();
        let (salt, seed) = (rng.bytes(SALT_LEN), rng.bytes(SEED_LEN));
        let g = garble(&enc, Mode::Hardwired, &salt, &[5], &seed).unwrap();
        let again = garble(&enc, Mode::Hardwired, &salt, &[5], &seed).unwrap();
        assert_eq!((g.delta, &g.key0, &g.tables, &g.decode), (again.delta, &again.key0, &again.tables, &again.decode));
        let mut salt2 = salt.clone();
        salt2[55] ^= 1;
        let other = garble(&enc, Mode::Hardwired, &salt2, &[5], &seed).unwrap();
        assert_ne!(other.delta, g.delta); // the salt is in the seed expansion too
        assert_ne!(other.key0, g.key0);
        assert_ne!(other.tables, g.tables);
        let mut seed2 = seed.clone();
        seed2[0] ^= 1;
        let other = garble(&enc, Mode::Hardwired, &salt, &[5], &seed2).unwrap();
        assert_ne!(other.key0, g.key0);
    }
}
