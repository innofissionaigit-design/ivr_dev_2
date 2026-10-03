"""Import main.py / main_pcm.py off-pod by stubbing the libraries only a pod has.

main.py and main_pcm.py import torch, torchaudio, NeMo, SpeechBrain and friends
at module level, which is why the orchestrator has had no off-pod tests at all.
None of the logic under test touches those libraries at import time, so an
import hook that answers for them with MagicMock modules is enough to load the
real modules and drive the real functions (`_speak`, `_interrupt_playback`,
`_decide_turn`, `CallSession.append`, ...) against a fake WebSocket.

Everything the hook installs is removed afterwards, so it cannot leak a mock
`torch` into any other test.

    with pod_stubs() as import_module:
        m = import_module("main_pcm")
"""

from __future__ import annotations

import contextlib
import importlib
import importlib.abc
import importlib.machinery
import os
import sys
from unittest import mock

STUBBED_ROOTS = (
    "torch",
    "torchaudio",
    "nemo",
    "speechbrain",
    "soundfile",
    "transformers",
    "omegaconf",
    "lightning",
    "pytorch_lightning",
    "onnxruntime",
    "TTS",
    "sentence_transformers",
)
# modules that import the stubs and so must not survive the context
_OWN_MODULES = (
    "main",
    "main_pcm",
    "agent.asr",
    "agent.lid",
    "agent.vad_stream",
    "agent.semantic_cache",
    "agent.pcm_buffer",
    "agent.tts",
    "agent.tts_router",
    "agent.channel_quality",
)


class _Finder(importlib.abc.MetaPathFinder, importlib.abc.Loader):
    def find_spec(self, name, path, target=None):
        if name.split(".")[0] in STUBBED_ROOTS:
            return importlib.machinery.ModuleSpec(name, self, is_package=True)
        return None

    def create_module(self, spec):
        m = mock.MagicMock(name=spec.name)
        m.__path__ = []
        m.__spec__ = spec
        m.__name__ = spec.name
        return m

    def exec_module(self, module):
        pass


@contextlib.contextmanager
def pod_stubs(repo_root: str):
    # `main` is ambiguous in this repo: the orchestrator here and clinic-api/main.py. Which one
    # `import main` finds depends on sys.path order, which other tests change; so for the duration of
    # the context put repo_root FIRST and set aside any `main` already imported from elsewhere.
    saved_path = list(sys.path)
    sys.path[:] = [repo_root] + [p for p in sys.path if p != repo_root]
    stale_main = sys.modules.get("main")
    if stale_main is not None and os.path.dirname(
        os.path.abspath(getattr(stale_main, "__file__", "") or "")
    ) != os.path.abspath(repo_root):
        del sys.modules["main"]
    else:
        stale_main = None
    before = set(sys.modules)
    finder = _Finder()
    sys.meta_path.insert(0, finder)
    try:
        yield importlib.import_module
    finally:
        sys.meta_path.remove(finder)
        for name in list(sys.modules):
            if name in before:
                continue
            if name.split(".")[0] in STUBBED_ROOTS or name in _OWN_MODULES:
                del sys.modules[name]
        sys.path[:] = saved_path
        if stale_main is not None:
            sys.modules["main"] = stale_main
