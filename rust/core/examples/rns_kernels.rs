//! Kernel timings of an 80-bit q at n = 4096 (one thread): a prime in u128
//! words against q = q1 q2 in two u64 residues.
//! cargo run --release --example rns_kernels
//! (the rounded sampler: the kernels' arithmetic, without FACCT's sampling cost)
use pyfe4ai_core::rlwe::{self, EncKey, Noise, Randomness};
use pyfe4ai_core::rns::{self, RnsKey, RnsRing};
use pyfe4ai_core::{gauss, AnyRing, Modulus, Word, U256};
use std::time::Instant;

fn time<F: FnMut()>(reps: u32, mut f: F) -> f64 {
    f();
    let t = Instant::now();
    for _ in 0..reps {
        f();
    }
    t.elapsed().as_secs_f64() * 1e3 / reps as f64
}

fn is_prime(p: u128) -> bool {
    let m = Modulus::<u128>::new(p);
    [2u128, 3, 5, 7, 11, 13].iter().all(|&a| m.pow(a, p - 1) == 1)
}

/// The first NTT prime above `start` and a primitive 2n-th root.
fn ntt_prime(start: u128, n: usize) -> (u128, u128) {
    let step = 2 * n as u128;
    let mut c = start / step * step + 1;
    loop {
        if c > start && is_prime(c) {
            let m = Modulus::<u128>::new(c);
            if let Some(r) = (2u128..200).map(|g| m.pow(g, (c - 1) / step)).find(|&r| m.pow(r, n as u128) == c - 1) {
                return (c, r);
            }
        }
        c += step;
    }
}

fn main() {
    let n = 4096usize;
    let (qp, rootp) = ntt_prime(1u128 << 79, n);
    let prime = match AnyRing::new(U256::from_u128(qp), n, U256::from_u128(rootp)).unwrap() {
        AnyRing::W128(r) => r,
        _ => unreachable!(),
    };
    let (q1, r1) = ntt_prime(1u128 << 39, n);
    let (q2, r2) = ntt_prime(q1 + 1, n);
    let rns = RnsRing::new(q1 * q2, n, [q1 as u64, q2 as u64], [r1 as u64, r2 as u64]).unwrap();
    println!("prime q {} bits (u128 words); rns q {} bits = {} x {} bits", 128 - qp.leading_zeros(), rns.qbits,
             128 - q1.leading_zeros(), 128 - q2.leading_zeros());

    let a: Vec<u128> = (0..n).map(|i| prime.m.reduce_u128(i as u128 * 0x9e37_79b9_7f4a_7c15_f39c)).collect();
    let ar: Vec<u128> = (0..n).map(|i| rns.m.reduce_u128(i as u128 * 0x9e37_79b9_7f4a_7c15_f39c)).collect();
    let mut b = a.clone();
    println!("ntt fwd   u128 {:.4} ms", time(2000, || prime.forward(&a, &mut b)));
    println!("ntt inv   u128 {:.4} ms", time(2000, || prime.inverse(&a, &mut b)));
    let mut x = vec![0u64; 2 * n];
    let mut y = vec![0u64; 2 * n];
    rns.split(&ar, &mut x);
    println!("ntt fwd   2 x u64 {:.4} ms", time(2000, || rns.forward(&x, &mut y)));
    println!("ntt inv   2 x u64 {:.4} ms", time(2000, || rns.inverse(&x, &mut y)));
    println!("split (to residues) {:.4} ms", time(2000, || rns.split(&ar, &mut x)));
    let mut back = ar.clone();
    println!("join (Garner)       {:.4} ms", time(2000, || rns.join(&x, &mut back)));

    // the label pad's reduction (blocks of bits(q) + 128 bits): mod q in u128, or mod q1 and q2
    let nb = pyfe4ai_core::keccak::zq_block(rns.qbits);
    let mut stream = vec![0u8; nb * n];
    pyfe4ai_core::keccak::shake256(b"label material", &mut stream);
    let mut h128 = vec![0u128; n];
    println!("zq_reduce u128  {:.4} ms", time(2000, || pyfe4ai_core::keccak::zq_reduce(&prime.m, &stream, nb, &mut h128)));
    let tz = time(2000, || {
        for i in 0..2 {
            pyfe4ai_core::keccak::zq_reduce(&rns.r[i].m, &stream, nb, &mut x[i * n..(i + 1) * n]);
        }
    });
    println!("zq_reduce 2 x u64 {:.4} ms", tz);
    println!("shake256 {} bytes {:.4} ms", nb * n, time(500, || pyfe4ai_core::keccak::shake256(b"label material", &mut stream)));

    // a client's 10^4-element encryption (3 chunks), the same random bytes
    let elems = 10_000usize;
    let xs: Vec<i64> = (0..elems).map(|i| (i % 201) as i64 - 100).collect();
    let chunks = (elems + n - 1) / n;
    let mut rnd = vec![0u8; rlwe::tensor_chunk_bytes(n) * chunks];
    pyfe4ai_core::osrand::fill(&mut rnd).unwrap();
    let mats: Vec<Vec<u8>> = (0..chunks).map(|c| format!("label material {}", c).into_bytes()).collect();
    let refs: Vec<&[u8]> = mats.iter().map(|v| v.as_slice()).collect();
    let (s2, s3) = (4.97e9, 9.94e9);
    let kp = EncKey::new(&prime, &a, &[a.clone()], Some(&[a.clone()]), qp / 602);
    let kr = RnsKey::new(&rns, &ar, &[ar.clone()], Some(&[ar.clone()]), rns.q / 602);
    let tp = time(200, || {
        rlwe::encrypt_tensor(&prime, &kp, &xs, Some(&refs), s2, s3, &Noise::Rounded, Randomness::Given(&rnd), 1,
                             |_, _, _| {}).unwrap();
    });
    let tr = time(200, || {
        rns::encrypt_tensor(&rns, &kr, &xs, Some(&refs), s2, s3, &Noise::Rounded, Randomness::Given(&rnd), 1,
                            |_, _, _| {}).unwrap();
    });
    println!("encrypt 10^4 elements: u128 {:.3} ms, rns {:.3} ms ({:+.0}%)", tp, tr, 100.0 * (tr / tp - 1.0));
    let mut smp = vec![0i64; 3 * n];
    let g = time(200, || gauss::gauss_fill(&rnd, s2, &mut smp)) * chunks as f64;
    println!("  of which Gaussians {:.3} ms", g);

    // one decryption chunk of 3 clients (eta = 1)
    let slot: Vec<u128> = a.iter().chain(a.iter()).copied().collect();
    let slotr: Vec<u128> = ar.iter().chain(ar.iter()).copied().collect();
    let slots: Vec<&[u128]> = vec![&slot, &slot, &slot];
    let slotsr: Vec<&[u128]> = vec![&slotr, &slotr, &slotr];
    let mut hp = vec![0u128; n];
    prime.forward_mont(&a, &mut hp);
    let hr = rns.ntt_mont_std(&ar);
    let hats: Vec<&[u128]> = vec![&hp, &hp, &hp];
    let hatsr: Vec<&[u64]> = vec![&hr, &hr, &hr];
    let ym = [prime.m.to_mont(1); 3];
    let ymr = [rns.m.to_mont(1); 3];
    let (mut out, mut none) = (vec![0i64; n], vec![0u8; n]);
    let dp = time(500, || {
        rlwe::decrypt_chunk(&prime, 1, &a, &slots, &ym, &hats, qp / 602, 300, &mut out, &mut none);
    });
    let dr = time(500, || {
        rns::decrypt_chunk(&rns, 1, &ar, &slotsr, &ymr, &hatsr, rns.q / 602, 300, &mut out, &mut none);
    });
    println!("decrypt 1 chunk, 3 clients: u128 {:.3} ms, rns {:.3} ms ({:+.0}%)", dp, dr, 100.0 * (dr / dp - 1.0));
    let _ = Word::to_u128(1u64);
}
