"""Shared JSON checkpoint input and output operations."""

import json
from pathlib import Path


class _CheckpointLoader:
    def __get__(self, instance, owner):
        if instance is None:
            return self

        def load(path):
            state = json.loads(Path(path).read_text(encoding="utf-8"))
            instance.load_state_dict(state)

        return load


class JsonCheckpoint:
    def save(self, path):
        Path(path).write_text(json.dumps(self.state_dict(), indent=2), encoding="utf-8")

    load = _CheckpointLoader()
