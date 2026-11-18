//! Fork fix 820cd2a, review case R3: what the joined form does not cover. The parent's calling thread waits once in
//! a library static's `OnceLock` (two threads build `bigauss`'s `TABLES` at once: a park, not in `parallel_for`);
//! the child's calling thread later waits in another static that a second thread is building (in the library:
//! `pairing.rs`'s `KEY` or `GT_GEN`, first used in the child; here a stand-in, since they live in the extension
//! crate). That second park aborts the child on macOS (SIGTRAP), with or without the pid check. Building the
//! statics once, uncontended, at import closes it.
#![cfg(unix)]
include!("fork_review_common.inc");

#[test]
fn a_child_survives_a_second_static_after_the_parent_waited_on_the_first() {
    if std::env::var("REVIEW_WARM").is_ok() {
        gaussian_fill(); // control: TABLES built once, uncontended
    }
    first_gaussian_fill_from_two_threads(); // the parent: a wait in TABLES (unless warmed)
    let mut ends = Vec::new();
    for _ in 0..3 {
        ends.push(in_child(|| {
            static OTHER: std::sync::OnceLock<u64> = std::sync::OnceLock::new();
            let t = std::thread::spawn(|| {
                OTHER.get_or_init(|| {
                    std::thread::sleep(Duration::from_millis(30));
                    7
                });
            });
            std::thread::sleep(Duration::from_millis(5));
            let v = *OTHER.get_or_init(|| 8); // the calling thread waits for the other initialiser
            t.join().unwrap();
            if v == 7 { 0 } else { 2 }
        }));
    }
    assert_eq!(ends, vec!["exit 0"; 3], "the child must survive");
}
