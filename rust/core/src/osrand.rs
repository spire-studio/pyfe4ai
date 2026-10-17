//! Random bytes from the operating system, as os.urandom reads them
//! (getrandom on Linux, getentropy on macOS and the BSDs).

#[cfg(target_os = "linux")]
extern "C" {
    fn getrandom(buf: *mut u8, len: usize, flags: u32) -> isize;
}

#[cfg(any(target_os = "macos", target_os = "ios", target_os = "freebsd", target_os = "openbsd"))]
extern "C" {
    fn getentropy(buf: *mut u8, len: usize) -> i32;
}

/// Whether this build has an OS random source.
pub const AVAILABLE: bool = cfg!(any(
    target_os = "linux",
    target_os = "macos",
    target_os = "ios",
    target_os = "freebsd",
    target_os = "openbsd"
));

/// Fill `buf` from the OS source.
pub fn fill(buf: &mut [u8]) -> std::io::Result<()> {
    #[cfg(target_os = "linux")]
    {
        let mut done = 0;
        while done < buf.len() {
            let r = unsafe { getrandom(buf[done..].as_mut_ptr(), buf.len() - done, 0) };
            if r < 0 {
                let e = std::io::Error::last_os_error();
                if e.kind() == std::io::ErrorKind::Interrupted {
                    continue;
                }
                return Err(e);
            }
            done += r as usize;
        }
        return Ok(());
    }
    #[cfg(any(target_os = "macos", target_os = "ios", target_os = "freebsd", target_os = "openbsd"))]
    {
        for chunk in buf.chunks_mut(256) {
            if unsafe { getentropy(chunk.as_mut_ptr(), chunk.len()) } != 0 {
                return Err(std::io::Error::last_os_error());
            }
        }
        return Ok(());
    }
    #[allow(unreachable_code)]
    {
        let _ = buf;
        Err(std::io::Error::new(std::io::ErrorKind::Unsupported, "no OS random source in this build"))
    }
}
