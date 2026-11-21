//! Fork fix 2b1c685, review case R1: a limited child still parks when two of its threads build a static at once.
//!
//! The parent runs `parallel_for` with 4 threads (it is the recorded process; its calling thread parks in the
//! scope). In the child `parallel_for` is limited to one thread, but two threads of the child (two Python threads
//! with the GIL released, here two Rust threads) call the Gaussian sampler for the first time; the calling thread
//! of the parent, which is the child's main thread, waits in `TABLES`'s `OnceLock` and the child aborts on macOS
//! (SIGTRAP). Passes where parking needs no libdispatch (Linux).
#![cfg(unix)]
include!("fork_review_common.inc");

#[test]
fn a_child_survives_two_threads_building_a_static() {
    // the parent: 4 threads; the workers' indices are slow, so the calling thread finishes first and parks
    let caller = std::thread::current().id();
    pyfe4ai_core::parallel_for(16, 4, |_| {
        if std::thread::current().id() != caller {
            std::thread::sleep(Duration::from_millis(5));
        }
    });
    assert_eq!(pyfe4ai_core::thread_limit_peek(4), 4);
    let mut ends = Vec::new();
    for _ in 0..3 {
        ends.push(in_child(|| {
            if pyfe4ai_core::thread_limit_peek(4) != 1 {
                return 2;
            }
            first_gaussian_fill_from_two_threads();
            0
        }));
    }
    assert_eq!(ends, vec!["exit 0"; 3], "the child must survive");
}
