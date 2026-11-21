// Shared by the fork review tests (included, not a test binary of its own).
use std::time::Duration;

extern "C" {
    fn fork() -> i32;
    fn waitpid(pid: i32, status: *mut i32, options: i32) -> i32;
    fn _exit(code: i32) -> !;
}

/// Run `child` in a forked child; how it ended ("exit N" or "signal N").
#[allow(dead_code)]
fn in_child(child: impl FnOnce() -> i32) -> String {
    let pid = unsafe { fork() };
    assert!(pid >= 0, "fork failed");
    if pid == 0 {
        let code = child();
        unsafe { _exit(code) }
    }
    let mut status = 0i32;
    unsafe { waitpid(pid, &mut status, 0) };
    let sig = status & 0x7f;
    if sig != 0 { format!("signal {sig}") } else { format!("exit {}", (status >> 8) & 0xff) }
}

/// One Gaussian fill with the library's `bigauss` (builds its `TABLES` the first time, ~2.6 ms here).
#[allow(dead_code)]
fn gaussian_fill() {
    use pyfe4ai_core::bigauss;
    let p = bigauss::Params::new(138, &1000u64.to_le_bytes(), 0, 0).unwrap();
    let (nb, mw) = (p.nbytes(), p.mag_words());
    let bytes = vec![1u8; 2 * nb * 8];
    let (mut mags, mut flags) = (vec![0u64; mw * 8], vec![0u8; 8]);
    bigauss::fill(&p, &bytes, &mut mags, &mut flags);
}

/// Two threads make the library build `TABLES` at once; this one arrives while the other builds, so it waits in
/// the `OnceLock` (a park). Models two Python threads (GIL released) calling Gaussian sampling for the first time.
#[allow(dead_code)]
fn first_gaussian_fill_from_two_threads() {
    let other = std::thread::spawn(gaussian_fill);
    std::thread::sleep(Duration::from_micros(500));
    gaussian_fill();
    other.join().unwrap();
}
