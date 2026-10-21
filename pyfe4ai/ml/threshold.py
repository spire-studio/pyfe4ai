"""Threshold MCFE in the ML adapter (``mcfe/ring_lwe_threshold``): the
decryption servers the wrapper runs, their partial decryptions
(:func:`partial_decrypt_gradients`, :class:`ThresholdPartial`) and the
combination (:func:`combine_partial_decryptions`)."""

from __future__ import annotations

import base64
from dataclasses import dataclass

import numpy as np

from pyfe4ai.core.formats.record import Field
from pyfe4ai.core.formats.record import Record
from pyfe4ai.core.formats.threshold import TMCFERingLWELayerPartials
from pyfe4ai.core.formats.threshold import TMCFERingLWEPartial
from pyfe4ai.ml.serialization import _BYTES, EncryptedGradient, _from_base64
from pyfe4ai.ml.wrapper import FESchemeWrapper, PreparedAggregation, _aggregated, _require_served_threshold

def _enrolled_servers(wrapper, servers) -> list:
    """The servers whose partial decryptions a threshold aggregation combines:
    *servers* (at least ``t`` distinct servers of the public parameters) or
    the first ``t`` of ``lst_sid``.

    Raises:
        ValueError: For other servers.
        NotImplementedError: Under ``flooding_accounting = "kl"``, whose
            servers need a partial-decryption budget of their own.
    """
    if wrapper.pp.get("flooding_accounting", "per-sample") != "per-sample":
        raise NotImplementedError(
            "flooding_accounting={!r}: every server needs a partial_budget of its own; run the servers "
            "with the scheme classes".format(wrapper.pp.get("flooding_accounting"))
        )
    lst_sid, t = list(wrapper.pp["lst_sid"]), int(wrapper.pp["t"])
    if servers is None:
        return lst_sid[:t]
    servers = list(servers)
    if len(servers) < t or len(set(servers)) != len(servers) or not set(servers) <= set(lst_sid):
        raise ValueError("servers must be at least t = {} distinct servers of {}".format(t, lst_sid))
    return servers


def _server_key_shares(wrapper, credentials: dict, servers: list) -> dict:
    """``{sid: key share}`` of the enrolled servers for the round (by default
    label key shares, with the chunk keys only), each server given its weight
    key share first where it lacks it: once per weights and share set."""
    out = {}
    for sid in servers:
        dk = wrapper.kg.get_decryption_keys(sid, credentials=dict(credentials, label_term=False))
        _serving(wrapper, sid, dk, credentials["fusion_weight"])
        out[sid] = dk
    return out


def _serving(wrapper, sid: str, dk: dict, fusion_weight: dict):
    """Server *sid* of the wrapper, given the weight key share that the label
    key share *dk* completes where it lacks it (first use, a copied wrapper's
    new server objects, or a weight part its store dropped)."""
    server = wrapper._get_server(sid)
    if not server.has_weight_key_share(dk):
        server.add_weight_key_share(wrapper.kg.get_weight_key_share(sid, credentials={"fusion_weight": fusion_weight}))
    return server


#: Fields of one partial decryption (one chunk) of a :class:`ThresholdPartial`.
_PARTIAL_FIELDS = frozenset(TMCFERingLWEPartial.FIELDS)


class ThresholdPartialWire(Record):
    """The wire form of a :class:`ThresholdPartial` (:meth:`ThresholdPartial.to_dict`)."""

    sid = Field("str, the server")
    label = Field("str, the round's label")
    format = Field("``binary`` (``w`` and ``c_y`` as base64) or ``json``")
    layers = Field(":class:`~pyfe4ai.core.formats.threshold.TMCFERingLWELayerPartials`")


@dataclass
class ThresholdPartial:
    """One server's partial decryptions of a round's gradients (threshold
    MCFE), from :func:`partial_decrypt_gradients` for
    :func:`combine_partial_decryptions`."""

    sid: str
    label: str
    #: :meth:`~pyfe4ai.schemes.mcfe.ring_lwe_threshold.ThresholdMCFERingLWE.compute_lst_ndarray_ct`
    #: of the clients' gradients: per layer ``{"shape", "parts"}``, one partial
    #: decryption per chunk (``w`` and ``c_y`` bytes for binary public
    #: parameters, lists of decimal strings for JSON ones; ``sharing_id``
    #: that of the server's key share, or ``None``). :meth:`to_dict` keeps the
    #: ``sharing_id``; a transport of its own must keep it as well, or the
    #: combiner can no longer tell partials of two sharings apart.
    layers: list

    def to_dict(self) -> dict:
        """Serialise for transfer: ``{"sid", "label", "format", "layers"}``,
        binary ``w`` and ``c_y`` (``format = "binary"``) as base64 strings.

        Raises:
            TypeError: If the partial decryptions mix binary and JSON values.
        """
        parts = [TMCFERingLWEPartial(part) for layer in self.layers for part in TMCFERingLWELayerPartials(layer).parts]
        binary = [isinstance(part.w, _BYTES) and isinstance(part.c_y, _BYTES) for part in parts]
        if any(binary) and not all(binary):
            raise TypeError("partial decryptions mixing binary and JSON values")
        fmt = "binary" if binary and all(binary) else "json"

        def written(part):
            view = TMCFERingLWEPartial(part)
            if fmt == "json":
                return TMCFERingLWEPartial.extend(part, w=list(view.w), c_y=list(view.c_y),
                                                  lst_sid_enrolled=list(view.lst_sid_enrolled)).to_dict()
            return TMCFERingLWEPartial.extend(part, w=base64.b64encode(bytes(view.w)).decode("ascii"),
                                              c_y=base64.b64encode(bytes(view.c_y)).decode("ascii"),
                                              lst_sid_enrolled=list(view.lst_sid_enrolled)).to_dict()

        return ThresholdPartialWire.new(
            sid=self.sid,
            label=self.label,
            format=fmt,
            layers=[TMCFERingLWELayerPartials.new(shape=list(layer.shape), parts=[written(p) for p in layer.parts]).to_dict()
                    for layer in map(TMCFERingLWELayerPartials, self.layers)],
        ).to_dict()

    @classmethod
    def from_dict(cls, d: dict) -> ThresholdPartial:
        """Deserialise from :meth:`to_dict`, checking the layout (the combiner
        checks the values).

        Raises:
            ValueError: For another layout, a partial decryption of another
                server, or ``w`` / ``c_y`` not of the format (strict base64
                for ``"binary"``, lists of strings for ``"json"``).
        """
        if not isinstance(d, dict) or set(d) != set(ThresholdPartialWire.FIELDS):
            raise ValueError("invalid partial decryption: keys sid, label, format and layers")
        wire = ThresholdPartialWire(d)
        sid, label, fmt, layers = wire.sid, wire.label, wire.format, wire.layers
        if not isinstance(sid, str) or not isinstance(label, str) or fmt not in ("binary", "json"):
            raise ValueError("invalid partial decryption: sid and label are strings, format binary or json")
        if not isinstance(layers, list) or not layers:
            raise ValueError("invalid partial decryption: layers must be a non-empty list")
        out = []
        for l, layer in enumerate(layers):
            if not isinstance(layer, dict) or set(layer) != set(TMCFERingLWELayerPartials.FIELDS):
                raise ValueError("invalid partial decryption: layer {} has keys shape and parts".format(l))
            layer = TMCFERingLWELayerPartials(layer)
            shape, _ = EncryptedGradient._checked_shape(layer.shape)
            if not isinstance(layer.parts, list) or not layer.parts:
                raise ValueError("invalid partial decryption: layer {} has no parts".format(l))
            parts = []
            for j, part in enumerate(layer.parts):
                where = "partial decryption ({}, chunk {})".format(l, j)
                if not isinstance(part, dict) or set(part) != _PARTIAL_FIELDS:
                    raise ValueError("invalid {}: keys {}".format(where, sorted(_PARTIAL_FIELDS)))
                view = TMCFERingLWEPartial(part)
                k, enrolled = view.k, view.lst_sid_enrolled
                if type(k) is not int or k < 1 or not isinstance(view.ct_digest, str) or view.sid != sid:
                    raise ValueError("invalid {}: k >= 1, a ct_digest and this server's sid".format(where))
                if not isinstance(enrolled, list) or not all(isinstance(v, str) for v in enrolled):
                    raise ValueError("invalid {}: lst_sid_enrolled is a list of server ids".format(where))
                sharing_id = view.sharing_id
                if sharing_id is not None and not (isinstance(sharing_id, str) and 0 < len(sharing_id) <= 128):
                    raise ValueError("invalid {}: sharing_id is a string (or null)".format(where))
                if fmt == "binary":
                    values = {"w": _from_base64(view.w, "{} w".format(where)),
                              "c_y": _from_base64(view.c_y, "{} c_y".format(where))}
                else:
                    if any(not isinstance(value, list) or not all(isinstance(v, str) for v in value)
                           for value in (view.w, view.c_y)):
                        raise ValueError("invalid {}: w and c_y are lists of decimal strings".format(where))
                    values = {"w": list(view.w), "c_y": list(view.c_y)}
                parts.append(TMCFERingLWEPartial.extend(part, lst_sid_enrolled=list(enrolled), **values).to_dict())
            out.append(TMCFERingLWELayerPartials.new(shape=list(shape), parts=parts).to_dict())
        return cls(sid=sid, label=label, layers=out)


def partial_decrypt_gradients(
    encrypted: dict[str, EncryptedGradient],
    wrapper: FESchemeWrapper,
    prepared: PreparedAggregation,
    *,
    server: str,
) -> ThresholdPartial:
    """Server *server*'s partial decryptions of the clients' gradients for a
    prepared threshold aggregation (``mcfe/ring_lwe_threshold``; the wrapper
    runs the server, with its key share of *prepared*, see
    :func:`prepare_aggregation`).

    Raises:
        ValueError: If *server* is not enrolled in *prepared*, or the clients
            are not those of its weights.
    """
    _require_served_threshold(wrapper, "partial_decrypt_gradients")
    if not prepared.servers or server not in prepared.servers:
        raise ValueError("server {!r} is not enrolled in the prepared aggregation ({})".format(server, prepared.servers))
    if set(encrypted) != set(prepared.fusion_weight):
        raise ValueError("the prepared key is for the clients {}".format(sorted(prepared.fusion_weight)))
    layers = _serving(wrapper, server, prepared.dk[server], prepared.fusion_weight).compute_lst_ndarray_ct(
        {nid: enc.ciphertexts for nid, enc in encrypted.items()},
        {"fusion_weight": prepared.fusion_weight, "label": prepared.label},
        prepared.dk[server],
        prepared.servers,
    )
    return ThresholdPartial(sid=server, label=prepared.label, layers=layers)


def combine_partial_decryptions(
    partials: dict[str, ThresholdPartial],
    wrapper: FESchemeWrapper,
    prepared: PreparedAggregation,
) -> np.ndarray:
    """Combine the enrolled servers' partial decryptions
    (:func:`partial_decrypt_gradients`) into the weighted sum of the
    gradients, as :func:`aggregate_gradients` returns it.

    Raises:
        ValueError: Unless *partials* are those of the prepared servers for
            its label.
        FESchemeError: If they do not combine (see
            :meth:`~pyfe4ai.schemes.mcfe.ring_lwe_threshold.ThresholdMCFERingLWE.decrypt_lst_ndarray_ct`;
            servers that decrypted different ciphertexts:
            :data:`~pyfe4ai.schemes.threshold.roles.DISAGREEING_PARTIALS`);
            :class:`~pyfe4ai.core.versions.MixedSharingsError`
            if their label parts are of different sharings (the key generator
            shared the label again between the servers' fetches): prepare the
            round again and have all servers decrypt with the new key shares
            (:func:`aggregate_gradients` does this once by itself).
    """
    _require_served_threshold(wrapper, "combine_partial_decryptions")
    if set(partials) != set(prepared.servers or []) or any(
            p.sid != sid or p.label != prepared.label for sid, p in partials.items()):
        raise ValueError("need the partial decryptions of the servers {} for label {!r}".format(
            prepared.servers, prepared.label))
    combiner = wrapper._get_decryptor(prepared.sid)
    return _aggregated(combiner.decrypt_lst_ndarray_ct({sid: p.layers for sid, p in partials.items()}),
                       prepared.weight_scale)
