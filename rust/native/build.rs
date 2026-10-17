// A CPython extension resolves the interpreter's symbols when it is loaded
// (-undefined dynamic_lookup on macOS), whoever runs cargo (cargo, maturin).
fn main() {
    pyo3_build_config::add_extension_module_link_args();
}
