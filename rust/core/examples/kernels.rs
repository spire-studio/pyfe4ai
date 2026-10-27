//! Kernel timings (one thread): cargo run --release --example kernels
use pyfe4ai_core::rlwe::{self, EncKey, Randomness};
use pyfe4ai_core::{keccak, gauss, osrand, with_ring, AnyRing, Modulus, Word, U256};
use std::time::Instant;

fn time<F: FnMut()>(reps: u32, mut f: F) -> f64 {
    f();
    let t = Instant::now();
    for _ in 0..reps {
        f();
    }
    t.elapsed().as_secs_f64() * 1e3 / reps as f64
}

fn ntt_prime(bits: u32, n: usize) -> (U256, U256) {
    let m2 = U256::from_u128(2 * n as u128);
    let base = U256::ONE.shl(bits - 1);
    let mut c = base.wsub(base.rem(m2)).wadd(U256::ONE);
    let small = |v: u128| U256::from_u128(v);
    loop {
        let m = Modulus::<U256>::new(c);
        // Fermat test base 2, 3, 5 suffices for a timing example
        if [2u128, 3, 5].iter().all(|&a| m.pow(small(a), c.wsub(U256::ONE)) == U256::ONE) {
            let mut cof = c.wsub(U256::ONE);
            for _ in 0..(2 * n).trailing_zeros() {
                cof = cof.shr1();
            }
            for g in 2u128..200 {
                let r = m.pow(small(g), cof);
                if m.pow(r, small(n as u128)) == c.wsub(U256::ONE) {
                    return (c, r);
                }
            }
        }
        c = c.wadd(m2);
    }
}

fn main() {
    for (bits, n) in [(77u32, 4096usize), (60, 4096), (212, 8192)] {
        let (q, root) = ntt_prime(bits, n);
        let ring = AnyRing::new(q, n, root).unwrap();
        let delta = (0..10).fold(q, |d, _| d.shr1()); // about q / 2^10
        println!("q {} bits, {}-bit words", bits, ring.word_bits());
        with_ring!(&ring, r => {
            let a: Vec<_> = (0..n).map(|i| r.m.reduce_u128(i as u128 * 0x9e3779b97f4a7c15)).collect();
            let mut b = a.clone();
            println!("  ntt fwd      {:.4} ms", time(2000, || r.forward(&a, &mut b)));
            println!("  ntt inv      {:.4} ms", time(2000, || r.inverse(&a, &mut b)));
            let nb = keccak::zq_block(r.m.bits);
            let mut stream = vec![0u8; nb * n];
            println!("  shake {} blocks {:.4} ms", n, time(500, || keccak::shake256(b"label material", &mut stream)));
            let mut h = vec![Word::ZERO; n];
            println!("  hash_to_zq   {:.4} ms", time(500, || keccak::hash_to_zq(&r.m, b"label material", &mut h)));
            let mut rnd = vec![0u8; 3 * gauss::gauss_bytes(n)];
            println!("  os random {} KB {:.4} ms", rnd.len() / 1000, time(500, || osrand::fill(&mut rnd).unwrap()));
            let mut smp = vec![0i64; 3 * n];
            println!("  gauss {}  {:.4} ms", 3 * n, time(500, || gauss::gauss_fill(&rnd, 4.97e9, &mut smp)));
            let key = EncKey::new(r, &a, &[a.clone()], Some(&[a.clone()]), Word::from_u256(delta));
            let x: Vec<i64> = (0..10000).map(|i| (i % 201) as i64 - 100).collect();
            let mats: Vec<Vec<u8>> = (0..(10000 + n - 1) / n).map(|c| format!("material {}", c).into_bytes()).collect();
            let refs: Vec<&[u8]> = mats.iter().map(|m| m.as_slice()).collect();
            let t1 = time(50, || { rlwe::encrypt_tensor(r, &key, &x, Some(&refs), 4.97e9, 7.03e9, &rlwe::Noise::Rounded, Randomness::Os, 1, |_, _, _| {}).unwrap(); });
            println!("  encrypt 1e4 elements ({} chunks, OS random, words out) {:.3} ms", (10000 + n - 1) / n, t1);
            let slot: Vec<_> = a.iter().chain(a.iter()).cloned().collect();
            let slots = [&slot[..], &slot[..], &slot[..]];
            let ym = vec![r.m.to_mont(Word::ONE); 3];
            let hats = [&a[..], &a[..], &a[..]];
            let (mut out, mut none) = (vec![0i64; n], vec![0u8; n]);
            let t2 = time(200, || { rlwe::decrypt_chunk(r, 1, &a, &slots, &ym, &hats, Word::from_u256(delta), 300, &mut out, &mut none); });
            println!("  decrypt one chunk, 3 slots {:.4} ms", t2);
        });
    }
}
