import asyncio
import time
from pathlib import Path

import numpy as np
import pytest

from bark_detector_module.detector import (
    Detector,
    _pcm16_to_float32,
    _resample,
)
from bark_detector_module.yamnet import (
    DOG_CLASSES,
    YAMNET_SAMPLE_RATE_HZ,
    YAMNET_WINDOW_SAMPLES,
    dog_score,
)


class RecordingSensor:
    def __init__(self):
        self.commands: list[dict] = []

    async def do_command(self, cmd, **kwargs):
        self.commands.append(cmd)
        return {"ok": True}


class StubClassifier:
    """Returns a preset score vector so tests don't need a real model."""

    def __init__(self, scores: np.ndarray | None = None):
        self.scores = (
            scores
            if scores is not None
            else np.zeros(521, dtype=np.float32)
        )
        self.calls: list[np.ndarray] = []

    def classify(self, samples: np.ndarray) -> np.ndarray:
        self.calls.append(samples)
        return self.scores


def _make(tmp_path: Path, *, with_events: bool = True, threshold: float = 0.5,
          debounce_sec: float = 2.0) -> tuple[Detector, RecordingSensor | None]:
    d = Detector(name="bark")
    d._audio_in_name = "microphone"
    d._events_sensor = RecordingSensor() if with_events else None
    d._events_sensor_name = "events" if with_events else ""
    d._threshold = threshold
    d._debounce_sec = debounce_sec
    d._classifier = StubClassifier()
    return d, d._events_sensor


def _scores_with_bark(bark_score: float = 0.9) -> np.ndarray:
    """521-class score vector with the Bark class hot."""
    arr = np.zeros(521, dtype=np.float32)
    arr[69] = bark_score  # 69 == Bark
    return arr


# -- dog_score aggregation --------------------------------------------


def test_dog_score_takes_max_across_dog_classes():
    arr = np.zeros(521, dtype=np.float32)
    arr[69] = 0.6  # Bark
    arr[71] = 0.9  # Howl (highest)
    arr[100] = 0.99  # unrelated class — should NOT count
    score, per_class = dog_score(arr)
    assert score == pytest.approx(0.9)
    assert per_class["Howl"] == pytest.approx(0.9)
    assert "Dog" in per_class and per_class["Dog"] == pytest.approx(0.0)


def test_dog_score_rejects_wrong_shape():
    with pytest.raises(ValueError, match="521"):
        dog_score(np.zeros(100, dtype=np.float32))


def test_dog_score_zero_for_silence():
    score, _ = dog_score(np.zeros(521, dtype=np.float32))
    assert score == 0.0


def test_dog_class_map_expected_names():
    # Guardrails against accidentally shifting indices — these are the
    # AudioSet class names YAMNet ships with.
    assert DOG_CLASSES[69] == "Bark"
    assert DOG_CLASSES[68] == "Dog"


# -- threshold + debounce --------------------------------------------


async def test_bark_above_threshold_fires_event(tmp_path):
    d, events = _make(tmp_path, threshold=0.5)
    d._classifier = StubClassifier(_scores_with_bark(0.9))
    d._classify_and_emit(np.zeros(YAMNET_WINDOW_SAMPLES, dtype=np.float32))
    # give the fire-and-forget task a tick to complete
    await asyncio.sleep(0)
    await asyncio.sleep(0)
    assert events is not None
    assert len(events.commands) == 1
    pushed = events.commands[0]["event"]
    assert pushed["event_type"] == "bark_detected"
    assert pushed["top_class"] == "Bark"
    assert pushed["score"] == pytest.approx(0.9)


async def test_bark_below_threshold_no_event(tmp_path):
    d, events = _make(tmp_path, threshold=0.8)
    d._classifier = StubClassifier(_scores_with_bark(0.3))
    d._classify_and_emit(np.zeros(YAMNET_WINDOW_SAMPLES, dtype=np.float32))
    await asyncio.sleep(0)
    assert events is not None
    assert events.commands == []


async def test_debounce_collapses_repeated_barks(tmp_path):
    d, events = _make(tmp_path, threshold=0.5, debounce_sec=1.0)
    d._classifier = StubClassifier(_scores_with_bark(0.9))
    d._classify_and_emit(np.zeros(YAMNET_WINDOW_SAMPLES, dtype=np.float32))
    d._classify_and_emit(np.zeros(YAMNET_WINDOW_SAMPLES, dtype=np.float32))
    d._classify_and_emit(np.zeros(YAMNET_WINDOW_SAMPLES, dtype=np.float32))
    await asyncio.sleep(0)
    await asyncio.sleep(0)
    assert events is not None
    # Only the first crossing produced an event.
    assert len(events.commands) == 1


async def test_debounce_lifts_after_window_elapses(tmp_path):
    d, events = _make(tmp_path, threshold=0.5, debounce_sec=0.01)
    d._classifier = StubClassifier(_scores_with_bark(0.9))
    d._classify_and_emit(np.zeros(YAMNET_WINDOW_SAMPLES, dtype=np.float32))
    time.sleep(0.02)
    d._classify_and_emit(np.zeros(YAMNET_WINDOW_SAMPLES, dtype=np.float32))
    await asyncio.sleep(0)
    await asyncio.sleep(0)
    assert events is not None
    assert len(events.commands) == 2


async def test_no_events_when_no_events_sensor(tmp_path):
    d, _ = _make(tmp_path, with_events=False)
    d._classifier = StubClassifier(_scores_with_bark(0.9))
    d._classify_and_emit(np.zeros(YAMNET_WINDOW_SAMPLES, dtype=np.float32))
    await asyncio.sleep(0)
    # No exception raised, session counter still ticks.
    assert d._bark_count_session == 1


# -- get_readings shape ----------------------------------------------


async def test_get_readings_reports_current_state(tmp_path):
    d, _ = _make(tmp_path)
    d._classifier = StubClassifier(_scores_with_bark(0.9))
    d._classify_and_emit(np.zeros(YAMNET_WINDOW_SAMPLES, dtype=np.float32))
    r = await d.get_readings()
    assert r["kind"] == "bark_detector"
    assert r["bark_count_session"] == 1
    assert r["last_dog_score"] == pytest.approx(0.9)
    assert r["last_bark_at"]  # non-empty ISO string
    assert r["class_scores"]["Bark"] == pytest.approx(0.9)


async def test_get_readings_empty_before_any_classification(tmp_path):
    d, _ = _make(tmp_path)
    r = await d.get_readings()
    assert r["bark_count_session"] == 0
    assert r["last_bark_at"] == ""
    assert r["last_dog_score"] == 0.0


# -- audio helpers ---------------------------------------------------


def test_pcm16_to_float32_mono_round_trip():
    # A signed int16 sample at max positive value should map to ~1.0.
    data = np.array([0, 16384, -16384, 32767], dtype=np.int16).tobytes()
    out = _pcm16_to_float32(data, channels=1)
    assert out.shape == (4,)
    assert out[0] == pytest.approx(0.0)
    assert out[1] == pytest.approx(0.5, abs=0.01)
    assert out[2] == pytest.approx(-0.5, abs=0.01)
    assert out[3] == pytest.approx(1.0, abs=0.01)


def test_pcm16_to_float32_stereo_downmix():
    # Two channels interleaved (L, R, L, R). Downmix averages them.
    ints = np.array([1000, 3000, -1000, 1000], dtype=np.int16)
    out = _pcm16_to_float32(ints.tobytes(), channels=2)
    assert out.shape == (2,)
    assert out[0] == pytest.approx(2000 / (1 << 15))
    assert out[1] == pytest.approx(0.0)


def test_resample_noop_when_rates_match():
    x = np.array([1.0, 2.0, 3.0], dtype=np.float32)
    y = _resample(x, 16000, 16000)
    np.testing.assert_array_equal(x, y)


def test_resample_length_scales_with_ratio():
    x = np.zeros(44100, dtype=np.float32)  # 1s at 44.1kHz
    y = _resample(x, 44100, 16000)
    # Roughly 16000 samples; allow ±1 for rounding.
    assert abs(y.shape[0] - 16000) <= 1


def test_yamnet_window_samples_is_16k_x_0975():
    # Guardrail: if this constant drifts, YAMNet won't accept the input.
    assert int(round(YAMNET_SAMPLE_RATE_HZ * 0.975)) == YAMNET_WINDOW_SAMPLES
