//! Kernel timings (one thread): cargo run --release --example kernels
use pyfe4ai_core::rlwe::{self, EncKey, Randomness};
use pyfe4ai_core::{keccak, gauss, osrand, with_ring, AnyRing, Modulus, Word};
use std::time::Instant;

fn time<F: FnMut()>(reps: u32, mut f: F) -> f64 {
    f();
    let t = Instant::now();
    for _ in 0..reps {
        f();
    }
    t.elapsed().as_secs_f64() * 1e3 / reps as f64
}

fn ntt_prime(bits: u32, n: usize) -> (u128, u128) {
    let m2 = 2 * n as u128;
    let mut c = (1u128 << (bits - 1)) / m2 * m2 + 1;
    loop {
        let m = Modulus::<u128>::new(c | 1);
        // Fermat test base 2, 3, 5 suffices for a timing example
        if [2u128, 3, 5].iter().all(|&a| m.pow(a, c - 1) == 1) {
            for g in 2u128..200 {
                let r = m.pow(g, (c - 1) / m2);
                if m.pow(r, n as u128) == c - 1 {
                    return (c, r);
                }
            }
        }
        c += m2;
    }
}

fn main() {
    let n = 4096;
    for bits in [77u32, 60] {
        let (q, root) = ntt_prime(bits, n);
        let ring = AnyRing::new(q, n, root).unwrap();
        println!("q {} bits, {}-bit words", bits, ring.word_bits());
        with_ring!(&ring, r => {
            let a: Vec<_> = (0..n).map(|i| r.m.reduce_u128((i as u128 * 0x9e3779b97f4a7c15) % q)).collect();
            let mut b = a.clone();
            println!("  ntt fwd      {:.4} ms", time(2000, || r.forward(&a, &mut b)));
            println!("  ntt inv      {:.4} ms", time(2000, || r.inverse(&a, &mut b)));
            let nb = keccak::zq_block(r.m.bits);
            let mut stream = vec![0u8; nb * n];
            println!("  shake 4096 blocks {:.4} ms", time(500, || keccak::shake256(b"label material", &mut stream)));
            let mut h = vec![Word::ZERO; n];
            println!("  hash_to_zq   {:.4} ms", time(500, || keccak::hash_to_zq(&r.m, b"label material", &mut h)));
            let mut rnd = vec![0u8; 3 * gauss::gauss_bytes(n)];
            println!("  os random 86 KB {:.4} ms", time(500, || osrand::fill(&mut rnd).unwrap()));
            let mut smp = vec![0i64; 3 * n];
            println!("  gauss 12288  {:.4} ms", time(500, || gauss::gauss_fill(&rnd, 4.97e9, &mut smp)));
            let key = EncKey::new(r, &a, &[a.clone()], Some(&[a.clone()]), r.m.reduce_u128(q / 602));
            let x: Vec<i64> = (0..10000).map(|i| (i % 201) as i64 - 100).collect();
            let mats: Vec<Vec<u8>> = (0..3).map(|c| format!("material {}", c).into_bytes()).collect();
            let refs: Vec<&[u8]> = mats.iter().map(|m| m.as_slice()).collect();
            let t1 = time(50, || { rlwe::encrypt_tensor(r, &key, &x, Some(&refs), 4.97e9, 7.03e9, Randomness::Os, 1, |_, _, _| {}).unwrap(); });
            println!("  encrypt 1e4 elements (3 chunks, OS random, words out) {:.3} ms", t1);
            let slot: Vec<_> = a.iter().chain(a.iter()).cloned().collect();
            let slots = [&slot[..], &slot[..], &slot[..]];
            let ym = vec![r.m.to_mont(Word::ONE); 3];
            let hats = [&a[..], &a[..], &a[..]];
            let (mut out, mut none) = (vec![0i64; n], vec![0u8; n]);
            let t2 = time(200, || { rlwe::decrypt_chunk(r, 1, &a, &slots, &ym, &hats, r.m.reduce_u128(q / 602), 300, &mut out, &mut none); });
            println!("  decrypt one chunk, 3 slots {:.4} ms", t2);
        });
    }
}
