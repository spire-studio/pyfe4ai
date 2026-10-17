//! Dense-LWE MCFE (ABMPSW20): <key, H(label)> per label, before reduction mod q.

use crate::keccak::shake256;
use crate::parallel_for;

/// acc (al limbs) += a * b, little-endian 64-bit limbs.
#[inline]
fn mac_limbs(acc: &mut [u64], a: &[u64], b: &[u64]) {
    for (i, &ai) in a.iter().enumerate() {
        if ai == 0 {
            continue;
        }
        let mut carry = 0u64;
        for (j, &bj) in b.iter().enumerate() {
            let t = ai as u128 * bj as u128 + acc[i + j] as u128 + carry as u128;
            acc[i + j] = t as u64;
            carry = (t >> 64) as u64;
        }
        let mut k = i + b.len();
        while carry != 0 && k < acc.len() {
            let t = acc[k] as u128 + carry as u128;
            acc[k] = t as u64;
            carry = (t >> 64) as u64;
            k += 1;
        }
    }
}

/// Limbs of each sum: key limbs + ceil(nb / 8) + 1.
pub fn sum_limbs(nb: usize, kl: usize) -> usize {
    kl + (nb + 7) / 8 + 1
}

/// For every material: sum_k key[k] B_k exactly, B_k = block k (nb bytes, big-endian)
/// of SHAKE-256(material) -- hash_to_zq_vector's blocks before their reduction mod q,
/// so the sum is congruent to <key, hash_to_zq_vector> mod q. `key` holds `count`
/// values of `kl` little-endian limbs; the result `materials.len()` sums of
/// sum_limbs(nb, kl) limbs.
pub fn dense_dots(materials: &[&[u8]], nb: usize, count: usize, key: &[u64], kl: usize, threads: usize) -> Vec<u64> {
    let hl = (nb + 7) / 8;
    let sl = sum_limbs(nb, kl);
    let out: Vec<std::sync::Mutex<Vec<u64>>> = (0..materials.len()).map(|_| std::sync::Mutex::new(Vec::new())).collect();
    parallel_for(materials.len(), threads, |m| {
        let mut stream = vec![0u8; nb * count];
        shake256(materials[m], &mut stream);
        let mut acc = vec![0u64; sl];
        let mut h = vec![0u64; hl];
        for k in 0..count {
            let p = &stream[k * nb..(k + 1) * nb];
            for (j, limb) in h.iter_mut().enumerate() {
                let hi = nb - 8 * j;
                let lo = hi.saturating_sub(8);
                let mut v = 0u64;
                for &b in &p[lo..hi] {
                    v = (v << 8) | b as u64;
                }
                *limb = v;
            }
            mac_limbs(&mut acc, &key[k * kl..(k + 1) * kl], &h);
        }
        *out[m].lock().unwrap() = acc;
    });
    out.into_iter().flat_map(|m| m.into_inner().unwrap()).collect()
}
