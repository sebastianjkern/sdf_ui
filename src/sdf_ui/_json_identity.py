"""Serialize finite JSON identity values for checkpoints and environments."""

import json


def serialize_identity(identity, label):
    if identity is None:
        return None
    try:
        return json.dumps(identity, sort_keys=True, allow_nan=False, separators=(",", ":"))
    except (TypeError, ValueError) as error:
        raise ValueError(f"{label} identity must be finite JSON-serializable data") from error
