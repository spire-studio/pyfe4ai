//! Copies of secret inputs (pads, key pieces, pad materials) and the wipes that
//! overwrite them before they are freed, on every path out of a call.

use pyfe4ai_core::{rlwe, Word};
use pyo3::exceptions::{PyTypeError, PyValueError};
use pyo3::prelude::*;
use pyo3::types::PyBytes;

/// Overwrite words that may be secret (seed-expanded key pieces) before they are freed.
pub(crate) fn wipe_words<W: Word>(buf: &mut [W]) {
    for v in buf.iter_mut() {
        unsafe { std::ptr::write_volatile(v, W::ZERO) };
    }
    std::sync::atomic::compiler_fence(std::sync::atomic::Ordering::SeqCst);
}

/// The bytes of an iterable of bytes objects, copied; on a failure the copies
/// made so far are wiped (they may be secret: pads, key pieces).
pub(crate) fn secret_byte_rows(rows: &Bound<'_, PyAny>) -> PyResult<Vec<Vec<u8>>> {
    let mut out: Vec<Vec<u8>> = Vec::new();
    let res = (|| -> PyResult<()> {
        for r in rows.try_iter()? {
            out.push(r?.extract::<Vec<u8>>()?);
        }
        Ok(())
    })();
    match res {
        Ok(()) => Ok(out),
        Err(e) => {
            wipe_byte_rows(&mut out);
            Err(e)
        }
    }
}

pub(crate) fn wipe_byte_rows(rows: &mut [Vec<u8>]) {
    for row in rows.iter_mut() {
        rlwe::wipe_bytes(row);
    }
}

pub(crate) fn wipe_word_rows<W: Word>(rows: &mut [Vec<W>]) {
    for row in rows.iter_mut() {
        rlwe::wipe_words(row);
    }
}

/// The pad materials of pad_terms: per label one bytes object per slot (`slots`
/// of them). They hash a client's u' (and dMCFE's pairwise keys): on a failure
/// the copies made so far are wiped; the caller wipes them after use.
pub(crate) fn secret_materials(materials: &Bound<'_, PyAny>, slots: usize) -> PyResult<Vec<Vec<Vec<u8>>>> {
    let mut out: Vec<Vec<Vec<u8>>> = Vec::new();
    let res = (|| -> PyResult<()> {
        for row in materials.try_iter()? {
            let mut v: Vec<Vec<u8>> = Vec::new();
            let fill = || -> PyResult<()> {
                for m in row?.try_iter()? {
                    let b = m?;
                    let b = b.cast::<PyBytes>().map_err(|_| PyTypeError::new_err("materials must be bytes"))?;
                    v.push(b.as_bytes().to_vec());
                }
                if v.len() != slots {
                    return Err(PyValueError::new_err("one material per slot and label"));
                }
                Ok(())
            };
            let res = fill();
            out.push(v);
            res?;
        }
        Ok(())
    })();
    match res {
        Ok(()) => Ok(out),
        Err(e) => {
            wipe_materials(&mut out);
            Err(e)
        }
    }
}

pub(crate) fn wipe_materials(mats: &mut [Vec<Vec<u8>>]) {
    for row in mats.iter_mut() {
        wipe_byte_rows(row);
    }
}
