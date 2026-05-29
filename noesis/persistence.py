"""
Session persistence — save/load SelfModel state to JSON.

Atomic writes (tmp → rename) so an interrupted save never corrupts an
existing checkpoint. Restores the full SelfModel including session_id
and the attention_vector numpy array.

cycles_completed is stored alongside so AutoMode can enforce a max_cycles
budget correctly across warm restarts.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

_VERSION = 1


def save(self_model, path: str | Path, *, cycles_completed: int = 0) -> None:
    """
    Atomically save SelfModel to a JSON checkpoint.

    Writes to a .tmp file first, then renames — POSIX atomic on the same
    filesystem — so a mid-write crash never corrupts an existing checkpoint.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    data = {
        "version": _VERSION,
        "cycles_completed": cycles_completed,
        "identity": self_model.identity,
        "session_id": self_model.session_id,
        "current_intent": self_model.current_intent,
        "confidence": self_model.confidence,
        "attention_vector": self_model.attention_vector.tolist(),
        "session_wisdom": list(self_model.session_wisdom),
        "metacognitive_depth": self_model.metacognitive_depth,
        "action_history": list(self_model.action_history),
        "arousal": self_model.arousal,
        "coherence": self_model.coherence,
    }

    text = json.dumps(data, indent=2)
    tmp = Path(str(path) + ".tmp")
    tmp.write_text(text, encoding="utf-8")
    tmp.replace(path)


def load(path: str | Path) -> tuple:
    """
    Load SelfModel from a JSON checkpoint.

    Returns (self_model, cycles_completed).
    """
    from .self_model import SelfModel

    data = json.loads(Path(path).read_text(encoding="utf-8"))

    sm = SelfModel(
        identity=data["identity"],
        session_id=data["session_id"],
        current_intent=data["current_intent"],
        confidence=float(data["confidence"]),
        attention_vector=np.array(data["attention_vector"], dtype=float),
        session_wisdom=list(data["session_wisdom"]),
        metacognitive_depth=int(data["metacognitive_depth"]),
        action_history=list(data["action_history"]),
        arousal=float(data["arousal"]),
        coherence=float(data["coherence"]),
    )

    return sm, int(data.get("cycles_completed", 0))


def exists(path: str | Path) -> bool:
    """Return True if a checkpoint exists at path; False on missing file or OSError."""
    try:
        return Path(path).exists()
    except OSError:
        return False


class SessionPersistence:
    """Namespace wrapper exposing save/load/exists as static methods."""

    save = staticmethod(save)
    load = staticmethod(load)
    exists = staticmethod(exists)
