//! The replicated key sharing of the LWE threshold layers as
//! pyfe4ai.schemes.threshold.sharing.share_replicated and share_for_enrolled
//! compute it, over flat values mod q (no NTT ring): the random pieces drawn as
//! random.SystemRandom.randrange(0, q) draws them (shamir::below), piece by
//! piece and value by value, with the first piece the correction; and sums of
//! pieces, explicit or expanded from seeds (keccak::hash_to_zq, the
//! "shake256-wide" hash_to_zq_vector), as share_for_enrolled and
//! _combine_pieces add them up. Secret buffers are wiped.

use crate::arith::{Modulus, Word};
use crate::rlwe::wipe_words;

/// `[values - sum r_j, r_1, ..., r_k]` mod q: k = `n_random` pieces of
/// values.len() values each (values reduced mod q), drawn in that order from
/// `fill`. None (and nothing left unwiped) if `fill` runs out of bytes; the
/// caller wipes the output.
pub fn deal<W: Word, F: FnMut(&mut [u8]) -> bool>(
    m: &Modulus<W>, values: &[W], n_random: usize, mut fill: F,
) -> Option<Vec<Vec<W>>> {
    let nb = ((m.q.bits() + 7) / 8) as usize;
    let mut buf = vec![0u8; nb];
    let mut out: Vec<Vec<W>> = Vec::with_capacity(n_random + 1);
    out.push(values.to_vec());
    let mut ok = true;
    'pieces: for _ in 0..n_random {
        let mut piece = Vec::with_capacity(values.len());
        for _ in 0..values.len() {
            match crate::shamir::below(m.q, &mut buf, &mut fill) {
                Some(r) => piece.push(r),
                None => {
                    wipe_words(&mut piece);
                    ok = false;
                    break 'pieces;
                }
            }
        }
        for (c, &r) in out[0].iter_mut().zip(piece.iter()) {
            *c = m.sub(*c, r);
        }
        out.push(piece);
    }
    crate::rlwe::wipe_bytes(&mut buf);
    if ok {
        return Some(out);
    }
    for piece in out.iter_mut() {
        wipe_words(piece);
    }
    None
}

/// The sum mod q of the rows (`count` values each, reduced mod q) plus -- or,
/// with `subtract`, minus -- hash_to_zq(material, count) of every material.
pub fn sum<W: Word>(m: &Modulus<W>, count: usize, rows: &[Vec<W>], materials: &[&[u8]], subtract: bool) -> Vec<W> {
    let mut acc = vec![W::ZERO; count];
    for row in rows {
        assert_eq!(row.len(), count, "rows of another length");
        for (a, &v) in acc.iter_mut().zip(row.iter()) {
            *a = m.add(*a, v);
        }
    }
    let mut h = vec![W::ZERO; count];
    for material in materials {
        crate::keccak::hash_to_zq(m, material, &mut h);
        for (a, &v) in acc.iter_mut().zip(h.iter()) {
            *a = if subtract { m.sub(*a, v) } else { m.add(*a, v) };
        }
    }
    wipe_words(&mut h);
    acc
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::arith::U256;

    #[test]
    fn pieces_add_up_and_draw_as_below() {
        let q = (1u128 << 100) + 277; // 13 bytes per draw, 3 excess bits
        let m = Modulus::<u128>::new(q);
        let values = [5u128, q - 1, 0, 7];
        let mut stream = (0u8..=255).cycle();
        let out = deal(&m, &values, 3, |b: &mut [u8]| {
            b.iter_mut().for_each(|x| *x = stream.next().unwrap());
            true
        })
        .unwrap();
        assert_eq!(out.len(), 4);
        for i in 0..values.len() {
            let total = out.iter().fold(0u128, |acc, p| m.add(acc, p[i]));
            assert_eq!(total, values[i]);
        }
        // the draws are those of shamir::below on the same stream, in order
        let mut s2 = (0u8..=255).cycle();
        let mut buf = [0u8; 13];
        let mut fill = |b: &mut [u8]| {
            b.iter_mut().for_each(|x| *x = s2.next().unwrap());
            true
        };
        for p in 1..4 {
            for i in 0..values.len() {
                assert_eq!(Some(out[p][i]), crate::shamir::below(q, &mut buf, &mut fill));
            }
        }
        assert!(deal(&m, &values, 2, |_: &mut [u8]| false).is_none());
        assert_eq!(deal(&m, &values, 0, |_: &mut [u8]| false).unwrap(), vec![values.to_vec()]);
    }

    #[test]
    fn sums_add_rows_and_hashes() {
        let q = U256::from_u128((1u128 << 120) + 451);
        let m = Modulus::<U256>::new(q);
        let rows = vec![vec![U256::from_u128(3), q.wsub(U256::ONE)], vec![U256::from_u128(4), U256::from_u128(2)]];
        assert_eq!(sum(&m, 2, &rows, &[], false), vec![U256::from_u128(7), U256::ONE]);
        let mut h = vec![U256::ZERO; 2];
        crate::keccak::hash_to_zq(&m, b"material", &mut h);
        let plus = sum(&m, 2, &rows[..1], &[b"material"], false);
        let minus = sum(&m, 2, &rows[..1], &[b"material"], true);
        for i in 0..2 {
            assert_eq!(plus[i], m.add(rows[0][i], h[i]));
            assert_eq!(minus[i], m.sub(rows[0][i], h[i]));
        }
    }
}
