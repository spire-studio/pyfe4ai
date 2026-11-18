//! Garbled circuits of the bounded-collusion FE schemes (bcfe/*): the
//! half-gates garbler and evaluator of the core's `garble` module (ZRE15
//! Fig. 2, the CIRC v1 encoding, the salted gate hash), on bytes: circuits as
//! their canonical encoding, labels and table entries as 16-byte strings,
//! bit strings packed low bit first. Mode 0 is "hardwired", 1 is "labels".
//! The garbling seed comes from the OS unless a test gives it; the seed, delta
//! and the labels are wiped in the core, the seed copy here. The GIL is
//! released around the garbling and the evaluation.

use pyfe4ai_core::garble::{self, Mode};
use pyfe4ai_core::osrand;
use pyo3::exceptions::{PyOSError, PyValueError};
use pyo3::prelude::*;
use pyo3::types::PyBytes;

fn mode_of(code: u8) -> PyResult<Mode> {
    Mode::from_code(code).ok_or_else(|| PyValueError::new_err("the garbling mode is 0 (hardwired) or 1 (labels)"))
}

type Garbled = (Py<PyBytes>, Py<PyBytes>, Py<PyBytes>, Py<PyBytes>, Py<PyBytes>);

/// Garble `circuit` (CIRC v1) for the packed data bits `x` with the 56-byte
/// instance `salt`: (delta, key0, data_active, tables, decode), the bytes of
/// the backend operation gc_garble. `seed` (32 bytes) is for the tests; None
/// draws it from the OS.
#[pyfunction]
#[pyo3(signature = (circuit, mode, salt, x, seed=None))]
fn gc_garble(py: Python<'_>, circuit: &[u8], mode: u8, salt: &[u8], x: &[u8], seed: Option<&[u8]>) -> PyResult<Garbled> {
    let mode = mode_of(mode)?;
    let mut s = zeroize::Zeroizing::new([0u8; garble::SEED_LEN]);
    match seed {
        Some(b) => {
            if b.len() != garble::SEED_LEN {
                return Err(PyValueError::new_err("a garbling seed has 32 bytes"));
            }
            s.copy_from_slice(b);
        }
        None => osrand::fill(&mut s[..]).map_err(|e| PyOSError::new_err(e.to_string()))?,
    }
    let g = py.detach(|| garble::garble(circuit, mode, salt, x, &s[..])).map_err(PyValueError::new_err)?;
    Ok((
        PyBytes::new(py, &g.delta).unbind(),
        PyBytes::new(py, &g.key0).unbind(),
        PyBytes::new(py, &g.data_active).unbind(),
        PyBytes::new(py, &g.tables).unbind(),
        PyBytes::new(py, &g.decode).unbind(),
    ))
}

/// Evaluate a garbling of gc_garble with the active labels of the key wires
/// (16 bytes each) and of the data wires (mode "labels"; empty in mode
/// "hardwired"): the n_out output bits, packed.
#[pyfunction]
#[allow(clippy::too_many_arguments)]
fn gc_evaluate(
    py: Python<'_>, circuit: &[u8], mode: u8, salt: &[u8], tables: &[u8], decode: &[u8], key_active: &[u8],
    data_active: &[u8],
) -> PyResult<Py<PyBytes>> {
    let mode = mode_of(mode)?;
    let y = py
        .detach(|| garble::evaluate(circuit, mode, salt, tables, decode, key_active, data_active))
        .map_err(PyValueError::new_err)?;
    Ok(PyBytes::new(py, &garble::pack_bits(&y)).unbind())
}

pub fn register(m: &Bound<'_, PyModule>) -> PyResult<()> {
    for f in [wrap_pyfunction!(gc_garble, m)?, wrap_pyfunction!(gc_evaluate, m)?] {
        m.add_function(f)?;
    }
    Ok(())
}
