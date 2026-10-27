// A CPython extension resolves the interpreter's symbols when it is loaded
// (-undefined dynamic_lookup on macOS), whoever runs cargo (cargo, maturin).
// The Py_3_x cfgs select version-specific C API calls (src/sampling.rs, long_from_le).
fn main() {
    pyo3_build_config::add_extension_module_link_args();
    pyo3_build_config::use_pyo3_cfgs();
    pyo3_build_config::print_expected_cfgs();
}
