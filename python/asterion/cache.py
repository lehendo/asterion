from __future__ import annotations

import fcntl
import os
from pathlib import Path

from .io import ValidationError, atomic_json, digest, read


class Cache:
    def __init__(self, root):
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)

    @classmethod
    def default(cls):
        root = os.environ.get("ASTERION_CACHE_DIR")
        if not root:
            scratch = os.environ.get("ASTERION_SCRATCH")
            root = (
                (Path(scratch) / "cache" / "asterion")
                if scratch
                else Path(os.environ.get("XDG_CACHE_HOME") or Path.home() / ".cache") / "asterion"
            )
        return cls(root)

    def put(self, key, payload):
        if len(key) != 64 or any(c not in "0123456789abcdef" for c in key):
            raise ValidationError("invalid cache key")
        with (self.root / (key + ".lock")).open("a") as lock:
            fcntl.flock(lock, fcntl.LOCK_EX)
            atomic_json(
                self.root / (key + ".json"),
                {
                    "schema_version": 1,
                    "key": key,
                    "payload_hash": digest(payload),
                    "payload": payload,
                },
            )

    def get(self, key):
        if len(key) != 64 or any(c not in "0123456789abcdef" for c in key):
            raise ValidationError("invalid cache key")
        p = self.root / (key + ".json")
        if not p.exists():
            return None
        try:
            value = read(p)
            if (
                value["schema_version"] != 1
                or value["key"] != key
                or digest(value["payload"]) != value["payload_hash"]
            ):
                raise ValidationError("cache integrity mismatch")
            return value["payload"]
        except (ValueError, KeyError, TypeError) as exc:
            raise ValidationError(f"corrupt cache entry {p.name}: {exc}") from exc

    def stats(self):
        entries = list(self.root.glob("*.json"))
        return {
            "root": str(self.root),
            "entries": len(entries),
            "bytes": sum(p.stat().st_size for p in entries),
            "scope": "generated source metadata; device binaries are managed by Triton",
        }
