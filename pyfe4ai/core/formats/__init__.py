"""The serialized objects of the schemes: their layouts and byte codecs.

Every object that leaves a scheme -- public parameters, parameter records,
encryption keys, functional keys and key shares, ciphertexts, partial
decryptions -- is a dict (JSON-friendly, with bytes for the binary formats).
Its layout is declared once as a :class:`~pyfe4ai.core.formats.record.Record`:
the fields in wire order, what each holds, and what an absent field means
(versions and their markers: :mod:`pyfe4ai.core.versions`). The schemes build
these dicts with ``Record.new`` and read them through record views, so the
dicts themselves are what they were: a record is the dict, not a copy.

* :mod:`~pyfe4ai.core.formats.record` -- :class:`~pyfe4ai.core.formats.record.Record` and :class:`~pyfe4ai.core.formats.record.Field`;
* :mod:`~pyfe4ai.core.formats.binary` -- the byte codecs of the Ring-LWE ciphertexts and values;
* :mod:`~pyfe4ai.core.formats.lwe` -- the LWE schemes (ABDP15, FullySec, the ABMPSW20 MCFE);
* :mod:`~pyfe4ai.core.formats.ring_lwe` -- the Ring-LWE schemes;
* :mod:`~pyfe4ai.core.formats.dl` -- the discrete-log and DCR schemes;
* :mod:`~pyfe4ai.core.formats.threshold` -- key shares, replicated pieces and partial decryptions of the threshold schemes.
"""
