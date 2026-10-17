//! SHA-256 (FIPS 180-4) and RFC 9380 expand_message_xmd with it, for the
//! hash-to-curve suites BLS12381G1/G2_XMD:SHA-256_SSWU_RO_ of the pairing
//! backend.

const K: [u32; 64] = [
    0x428a2f98, 0x71374491, 0xb5c0fbcf, 0xe9b5dba5, 0x3956c25b, 0x59f111f1, 0x923f82a4, 0xab1c5ed5,
    0xd807aa98, 0x12835b01, 0x243185be, 0x550c7dc3, 0x72be5d74, 0x80deb1fe, 0x9bdc06a7, 0xc19bf174,
    0xe49b69c1, 0xefbe4786, 0x0fc19dc6, 0x240ca1cc, 0x2de92c6f, 0x4a7484aa, 0x5cb0a9dc, 0x76f988da,
    0x983e5152, 0xa831c66d, 0xb00327c8, 0xbf597fc7, 0xc6e00bf3, 0xd5a79147, 0x06ca6351, 0x14292967,
    0x27b70a85, 0x2e1b2138, 0x4d2c6dfc, 0x53380d13, 0x650a7354, 0x766a0abb, 0x81c2c92e, 0x92722c85,
    0xa2bfe8a1, 0xa81a664b, 0xc24b8b70, 0xc76c51a3, 0xd192e819, 0xd6990624, 0xf40e3585, 0x106aa070,
    0x19a4c116, 0x1e376c08, 0x2748774c, 0x34b0bcb5, 0x391c0cb3, 0x4ed8aa4a, 0x5b9cca4f, 0x682e6ff3,
    0x748f82ee, 0x78a5636f, 0x84c87814, 0x8cc70208, 0x90befffa, 0xa4506ceb, 0xbef9a3f7, 0xc67178f2,
];

const H0: [u32; 8] = [0x6a09e667, 0xbb67ae85, 0x3c6ef372, 0xa54ff53a, 0x510e527f, 0x9b05688c, 0x1f83d9ab, 0x5be0cd19];

fn compress(h: &mut [u32; 8], block: &[u8]) {
    let mut w = [0u32; 64];
    for i in 0..16 {
        w[i] = u32::from_be_bytes(block[4 * i..4 * i + 4].try_into().unwrap());
    }
    for i in 16..64 {
        let s0 = w[i - 15].rotate_right(7) ^ w[i - 15].rotate_right(18) ^ (w[i - 15] >> 3);
        let s1 = w[i - 2].rotate_right(17) ^ w[i - 2].rotate_right(19) ^ (w[i - 2] >> 10);
        w[i] = w[i - 16].wrapping_add(s0).wrapping_add(w[i - 7]).wrapping_add(s1);
    }
    let [mut a, mut b, mut c, mut d, mut e, mut f, mut g, mut hh] = *h;
    for i in 0..64 {
        let s1 = e.rotate_right(6) ^ e.rotate_right(11) ^ e.rotate_right(25);
        let ch = (e & f) ^ (!e & g);
        let t1 = hh.wrapping_add(s1).wrapping_add(ch).wrapping_add(K[i]).wrapping_add(w[i]);
        let s0 = a.rotate_right(2) ^ a.rotate_right(13) ^ a.rotate_right(22);
        let maj = (a & b) ^ (a & c) ^ (b & c);
        let t2 = s0.wrapping_add(maj);
        hh = g;
        g = f;
        f = e;
        e = d.wrapping_add(t1);
        d = c;
        c = b;
        b = a;
        a = t1.wrapping_add(t2);
    }
    for (x, v) in h.iter_mut().zip([a, b, c, d, e, f, g, hh]) {
        *x = x.wrapping_add(v);
    }
}

/// SHA-256 of the concatenation of `parts`.
pub fn sha256(parts: &[&[u8]]) -> [u8; 32] {
    let mut h = H0;
    let mut buf = [0u8; 64];
    let (mut fill, mut total) = (0usize, 0u64);
    for part in parts {
        total += part.len() as u64;
        let mut p: &[u8] = part;
        while !p.is_empty() {
            let take = (64 - fill).min(p.len());
            buf[fill..fill + take].copy_from_slice(&p[..take]);
            fill += take;
            p = &p[take..];
            if fill == 64 {
                compress(&mut h, &buf);
                fill = 0;
            }
        }
    }
    buf[fill] = 0x80;
    for b in buf[fill + 1..].iter_mut() {
        *b = 0;
    }
    if fill + 1 > 56 {
        compress(&mut h, &buf);
        buf = [0u8; 64];
    }
    buf[56..].copy_from_slice(&(total * 8).to_be_bytes());
    compress(&mut h, &buf);
    let mut out = [0u8; 32];
    for (i, v) in h.iter().enumerate() {
        out[4 * i..4 * i + 4].copy_from_slice(&v.to_be_bytes());
    }
    out
}

/// RFC 9380 Sec. 5.3.1 expand_message_xmd with SHA-256 to `len` bytes
/// (`len <= 8160`, `dst` at most 255 bytes).
pub fn expand_message_xmd(msg: &[u8], dst: &[u8], len: usize) -> Vec<u8> {
    assert!(dst.len() <= 255 && len <= 255 * 32 && len > 0);
    let ell = len.div_ceil(32);
    let dst_len = [dst.len() as u8];
    let b0 = sha256(&[&[0u8; 64], msg, &(len as u16).to_be_bytes(), &[0u8], dst, &dst_len]);
    let mut out = Vec::with_capacity(32 * ell);
    let mut bi = sha256(&[&b0, &[1u8], dst, &dst_len]);
    out.extend_from_slice(&bi);
    for i in 2..=ell {
        let mut x = [0u8; 32];
        for (k, v) in x.iter_mut().enumerate() {
            *v = b0[k] ^ bi[k];
        }
        bi = sha256(&[&x, &[i as u8], dst, &dst_len]);
        out.extend_from_slice(&bi);
    }
    out.truncate(len);
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    fn hex(b: &[u8]) -> String {
        b.iter().map(|x| format!("{:02x}", x)).collect()
    }

    #[test]
    fn known_answers() {
        assert_eq!(hex(&sha256(&[b"abc"])), "ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad");
        assert_eq!(hex(&sha256(&[b""])), "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855");
        let long: Vec<u8> = (0..1000u32).map(|i| (i * 7) as u8).collect();
        let (a, b) = long.split_at(333);
        assert_eq!(sha256(&[&long]), sha256(&[a, b]));
        // RFC 9380 K.1 (expand_message_xmd, SHA-256), DST "QUUX-V01-CS02-with-expander-SHA256-128"
        let dst = b"QUUX-V01-CS02-with-expander-SHA256-128";
        assert_eq!(
            hex(&expand_message_xmd(b"", dst, 32)),
            "68a985b87eb6b46952128911f2a4412bbc302a9d759667f87f7a21d803f07235"
        );
        assert_eq!(
            hex(&expand_message_xmd(b"abc", dst, 32)),
            "d8ccab23b5985ccea865c6c97b6e5b8350e794e603b4b97902f53a8a0d605615"
        );
        assert_eq!(
            hex(&expand_message_xmd(b"", dst, 0x80)),
            "af84c27ccfd45d41914fdff5df25293e221afc53d8ad2ac06d5e3e29485dadbe\
             e0d121587713a3e0dd4d5e69e93eb7cd4f5df4cd103e188cf60cb02edc3edf18\
             eda8576c412b18ffb658e3dd6ec849469b979d444cf7b26911a08e63cf31f9dc\
             c541708d3491184472c2c29bb749d4286b004ceb5ee6b9a7fa5b646c993f0ced"
        );
    }
}
