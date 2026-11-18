//! Fork fix, FR-1 of its review: a first build of a lazy static must never wait.
//!
//! The parent's main thread waits in the first initialisation of a `OnceLock` of its own (another thread is building
//! it), so that thread has parked once in the parent. Nothing is recorded by `thread_limit` (no multi-threaded
//! `parallel_for`), and `parallel_for_joined` does not park, so neither guard sees this. In the child, two threads
//! build the library's `bigauss` tables for the first time at once; if that build made the late thread wait (a
//! `OnceLock`), the main thread would park a second time and the child would abort on macOS (SIGTRAP). The tables are
//! built without any wait (an atomic pointer: a late thread builds its own copy and the first one stored wins).
#![cfg(unix)]
include!("fork_review_common.inc");

use std::sync::OnceLock;

#[test]
fn a_child_builds_the_gaussian_tables_from_two_threads_after_its_parent_parked() {
    static PARENT_ONLY: OnceLock<u64> = OnceLock::new();
    let other = std::thread::spawn(|| {
        PARENT_ONLY.get_or_init(|| {
            std::thread::sleep(Duration::from_millis(20));
            1
        });
    });
    std::thread::sleep(Duration::from_millis(5));
    assert_eq!(*PARENT_ONLY.get_or_init(|| 2), 1); // the main thread waits for the other one: a park
    other.join().unwrap();
    assert_eq!(pyfe4ai_core::thread_limit_peek(4), 4);
    let mut ends = Vec::new();
    for _ in 0..3 {
        ends.push(in_child(|| {
            first_gaussian_fill_from_two_threads();
            0
        }));
    }
    assert_eq!(ends, vec!["exit 0"; 3], "the child must survive");
}
