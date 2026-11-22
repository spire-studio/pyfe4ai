//! The data-independent-timing mode (PSTATE.DIT, Arm FEAT_DIT) of the calling
//! thread, for code that works on secrets.
//!
//! With DIT set, the processor runs the instructions that Arm lists as
//! data-independent in a time that does not depend on their operands; on Apple
//! silicon from M3 on, DIT also turns off the data-memory-dependent
//! prefetcher. It does not make secret-dependent branches or memory accesses
//! constant-time. DIT is a per-thread bit: a `Guard` sets it for the current
//! thread and puts the previous value back when it is dropped, so a caller that
//! had set it keeps it. Where the processor has no FEAT_DIT, or the target is
//! not aarch64, a `Guard` does nothing.

/// Sets PSTATE.DIT for the current thread while it lives.
pub struct Guard {
    /// Whether DIT was clear before and has to be cleared again on drop.
    #[cfg(target_arch = "aarch64")]
    clear_on_drop: bool,
}

/// Whether this processor has FEAT_DIT (always false off aarch64).
pub fn supported() -> bool {
    #[cfg(target_arch = "aarch64")]
    {
        std::arch::is_aarch64_feature_detected!("dit")
    }
    #[cfg(not(target_arch = "aarch64"))]
    {
        false
    }
}

/// Whether PSTATE.DIT is set on the current thread (false where unsupported).
pub fn is_set() -> bool {
    #[cfg(target_arch = "aarch64")]
    {
        if !supported() {
            return false;
        }
        let v: u64;
        // Not `nomem`: the asm is a compiler barrier, so no access to secret
        // data moves across it.
        unsafe { core::arch::asm!(".arch_extension dit", "mrs {0}, DIT", out(reg) v, options(nostack, preserves_flags)) };
        v & (1 << 24) != 0
    }
    #[cfg(not(target_arch = "aarch64"))]
    {
        false
    }
}

impl Guard {
    /// Sets DIT for the current thread if the processor has it.
    ///
    /// A write to PSTATE.DIT with MSR (immediate) applies to the instructions
    /// after it in program order, with no context synchronization needed.
    #[allow(clippy::new_without_default)]
    pub fn new() -> Guard {
        #[cfg(target_arch = "aarch64")]
        {
            let clear_on_drop = supported() && !is_set();
            if clear_on_drop {
                unsafe { core::arch::asm!(".arch_extension dit", "msr DIT, #1", options(nostack, preserves_flags)) };
            }
            Guard { clear_on_drop }
        }
        #[cfg(not(target_arch = "aarch64"))]
        {
            Guard {}
        }
    }
}

impl Drop for Guard {
    fn drop(&mut self) {
        #[cfg(target_arch = "aarch64")]
        if self.clear_on_drop {
            unsafe { core::arch::asm!(".arch_extension dit", "msr DIT, #0", options(nostack, preserves_flags)) };
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_guard_sets_dit_and_restores_the_previous_value() {
        let before = is_set();
        {
            let _g = Guard::new();
            assert_eq!(is_set(), supported());
            {
                let _inner = Guard::new();
                assert_eq!(is_set(), supported());
            }
            // the inner guard found DIT set and leaves it set
            assert_eq!(is_set(), supported());
        }
        assert_eq!(is_set(), before);
    }

    /// Every Apple silicon processor (M1 on) has FEAT_DIT, so there the tests above set the bit for real.
    #[cfg(all(target_arch = "aarch64", target_os = "macos"))]
    #[test]
    fn apple_silicon_has_dit() {
        assert!(supported());
    }

    #[test]
    fn dit_is_per_thread() {
        let _g = Guard::new();
        let other = std::thread::spawn(is_set).join().unwrap();
        assert!(!other, "a new thread starts with DIT clear");
    }
}
