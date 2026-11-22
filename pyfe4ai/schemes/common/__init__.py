"""Building blocks shared by several scheme families: the DDH and EC-DDH base
classes, the ABDP15 LWE layer, the configuration types, and what the
attribute-based pairing schemes share: the binding of their objects to their
public parameters (``ac_binding``), the readers of their untrusted records
(``ac_codec``), the group elements a crypto object decodes once
(``ac_decoded``) and the pieces of the key-policy schemes of NPP22
(``kp_ab``)."""
