"""An external review flagged that tts_server._render writes the per-request speed onto a shared model
object. Two overlapping requests must each render at THEIR OWN speed.

    python -m pytest tests/test_tts_concurrency.py -v
"""

import os
import sys
import threading
import time

import numpy as np

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(REPO_ROOT, "tests"))

from _pod_stubs import pod_stubs


class _Model:
    length_scale = 1.0


class _Synth:
    """Records the speed in force at the moment of each synthesis, with a pause that lets another
    thread run in between if nothing stops it."""

    def __init__(self):
        self.tts_model = _Model()
        self.seen = []
        self.output_sample_rate = 22050

    def tts(self, text, speaker_name=None, split_sentences=False):
        time.sleep(0.01)
        self.seen.append((text, self.tts_model.length_scale))
        return np.zeros(100, dtype=np.float32)


def test_overlapping_requests_each_render_at_their_own_speed(tmp_path, monkeypatch):
    monkeypatch.setenv("TTS_CHECKPOINTS_ROOT", str(tmp_path))  # the module chdir()s into it on import
    monkeypatch.chdir(tmp_path)  # ...and pytest restores the directory after
    with pod_stubs(REPO_ROOT) as imp:
        ts = imp("tts_server")
        lang = next(iter(ts.SUPPORTED_LANGUAGES))
        synth = _Synth()
        ts.SYNTHESIZERS[lang] = synth
        ts.SAMPLE_RATES[lang] = 22050
        params = ts.DEFAULT_PROSODY

        def call(text, speed):
            ts._render(lang, text, "female", speed, False, params)

        threads = [threading.Thread(target=call, args=(f"t{i}", 0.8 if i % 2 else 1.3)) for i in range(16)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()
        assert len(synth.seen) == 16
        for text, scale in synth.seen:
            assert scale == (0.8 if int(text[1:]) % 2 else 1.3), (text, scale)


def test_the_speakers_endpoint_lists_each_languages_voices_for_choosing_one_by_ear(tmp_path, monkeypatch):
    """KCD-511: which voices exist, and which is in use, so the agent's voice can be chosen without a code change."""
    monkeypatch.setenv("TTS_CHECKPOINTS_ROOT", str(tmp_path))
    monkeypatch.chdir(tmp_path)
    with pod_stubs(REPO_ROOT) as imp:
        ts = imp("tts_server")

        class _Manager:
            name_to_id = {"female": 0, "male": 1, "elder_female": 2}

        class _Model:
            speaker_manager = _Manager()

        class _Multi:
            tts_model = _Model()

        class _Single:
            tts_model = object()  # a single-speaker model has no speaker manager

        ts.SYNTHESIZERS["bn"], ts.SYNTHESIZERS["hi"], ts.SYNTHESIZERS["en"] = _Multi(), _Multi(), _Single()
        out = ts.speakers()
        assert out["default"] == ts.DEFAULT_SPEAKER
        assert out["speakers"]["bn"] == ["elder_female", "female", "male"] and out["speakers"]["en"] == []
