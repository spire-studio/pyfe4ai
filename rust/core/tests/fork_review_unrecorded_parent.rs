//! Fork fix 2b1c685, review case R2: a parent that never ran `parallel_for` with several threads can still have
//! parked, so its child, which keeps its threads, aborts at its first multi-threaded `parallel_for`.
//!
//! The parent builds `TABLES` from two threads at once (two Python threads calling the single-threaded sampler for
//! the first time); its main thread waits in the `OnceLock` (a park) but nothing is recorded in `THREADED_PID`. The
//! child runs `parallel_for` with 4 threads, whose scope parks the same thread: SIGTRAP on macOS.
#![cfg(unix)]
include!("fork_review_common.inc");

#[test]
fn a_child_of_an_unrecorded_parent_survives_parallel_for() {
    first_gaussian_fill_from_two_threads(); // the parent: a park, but no multi-threaded parallel_for
    assert_eq!(pyfe4ai_core::thread_limit_peek(4), 4);
    let mut ends = Vec::new();
    for _ in 0..3 {
        ends.push(in_child(|| {
            pyfe4ai_core::parallel_for(16, 4, |_| std::thread::sleep(Duration::from_millis(2)));
            0
        }));
    }
    assert_eq!(ends, vec!["exit 0"; 3], "the child must survive");
}
