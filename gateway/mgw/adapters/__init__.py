"""Adapters: what a backend needs done to a request before it is sent, and to its answer.

An adapter is a Python file with any of these functions (all optional):

    def prepare(body: dict, headers: dict, ctx: dict) -> None
        Change the request in place. ctx: backend (config.Backend), session (str), path ("/v1/chat/completions" or
        "/v1/responses"), stream (bool). The model id and the backend's extra_body are already in `body`.

    def repair(data: bytes) -> bytes
        Fix the answer's bytes: the whole body, or one line of a stream.

The built-in ones are in this folder. A file of the same name in the config folder's `adapters/` replaces it, and is
re-read when it changes.
"""
import importlib.util
import logging
from pathlib import Path
from types import ModuleType

log = logging.getLogger("mgw")
BUILTIN = Path(__file__).parent


class Adapters:
    def __init__(self, folder: str | Path | None):
        self.folder = Path(folder) if folder else None
        self.loaded: dict[str, tuple[Path, float, ModuleType]] = {}

    def path(self, name: str) -> Path:
        own = self.folder / f"{name}.py" if self.folder else None
        return own if own and own.exists() else BUILTIN / f"{name}.py"

    def get(self, name: str) -> ModuleType:
        path = self.path(name)
        mtime = path.stat().st_mtime
        known = self.loaded.get(name)
        if known and known[0] == path and known[1] == mtime:
            return known[2]
        spec = importlib.util.spec_from_file_location(f"mgw_adapter_{name}", path)
        module = importlib.util.module_from_spec(spec)
        try:
            spec.loader.exec_module(module)
        except Exception:
            if known:  # a broken edit: keep the last good one
                log.exception("adapter %s at %s failed to load; keeping the previous one", name, path)
                return known[2]
            raise
        self.loaded[name] = (path, mtime, module)
        return module
