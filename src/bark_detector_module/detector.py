"""Bark detector Sensor. See README for design + wiring."""
from __future__ import annotations

import asyncio
import contextlib
import logging
import time
from collections import deque
from collections.abc import Mapping, Sequence
from datetime import UTC, datetime
from typing import Any, ClassVar

import numpy as np
from viam.components.audio_in import AudioIn
from viam.components.sensor import Sensor
from viam.proto.app.robot import ComponentConfig
from viam.proto.common import ResourceName
from viam.resource.base import ResourceBase
from viam.resource.types import Model, ModelFamily
from viam.utils import struct_to_dict

from .yamnet import (
    YAMNET_SAMPLE_RATE_HZ,
    YAMNET_WINDOW_SAMPLES,
    YAMNetClassifier,
    dog_score,
    resolve_model_path,
)

LOGGER = logging.getLogger(__name__)

DEFAULT_THRESHOLD = 0.5
DEFAULT_DEBOUNCE_SEC = 2.0
DEFAULT_CODEC = "pcm16"
BARK_HISTORY_MAX = 500

# PCM16 is signed int16, so the divisor to normalize into [-1, 1] is 2**15.
PCM16_SCALE = float(1 << 15)


def _now_iso() -> str:
    return datetime.now(UTC).isoformat()


class Detector(Sensor):
    """Streams audio from an audio_in dep, classifies with YAMNet, and
    emits `bark_detected` events when the dog-vocalization score crosses
    a configurable threshold.

    Detection cadence matches YAMNet's native window: 0.975s per inference.
    Events are debounced so a sustained barking session produces one event
    per `debounce_sec`, not one per window.
    """

    MODEL: ClassVar[Model] = Model(ModelFamily("joseph", "bark-detector"), "sensor")

    def __init__(self, name: str) -> None:
        super().__init__(name)
        self._audio_in: AudioIn | None = None
        self._audio_in_name: str = ""
        self._events_sensor: Sensor | None = None
        self._events_sensor_name: str = ""
        self._threshold: float = DEFAULT_THRESHOLD
        self._debounce_sec: float = DEFAULT_DEBOUNCE_SEC
        self._classifier: YAMNetClassifier | None = None
        self._run_task: asyncio.Task | None = None
        # Rolling telemetry surfaced via get_readings.
        self._last_bark_at: str | None = None
        self._last_dog_score: float = 0.0
        self._last_class_scores: dict[str, float] = {}
        self._bark_count_session: int = 0
        # In-memory bark log for the chart. Cloud tabular-data isn't
        # reachable from the webapp's cookie-scoped API key, so the
        # dashboard reads history via get_history do_command instead.
        self._bark_history: deque[dict] = deque(maxlen=BARK_HISTORY_MAX)
        self._last_debounce_ts: float = 0.0

    # -- Viam lifecycle ------------------------------------------------

    @classmethod
    def new(
        cls,
        config: ComponentConfig,
        dependencies: Mapping[ResourceName, ResourceBase],
    ) -> Detector:
        d = cls(config.name)
        d.reconfigure(config, dependencies)
        return d

    @classmethod
    def validate_config(cls, config: ComponentConfig) -> tuple[Sequence[str], Sequence[str]]:
        attrs = struct_to_dict(config.attributes)
        audio_in = attrs.get("audio_in")
        if not isinstance(audio_in, str) or not audio_in:
            raise ValueError("`audio_in` is required")
        required = [audio_in]
        optional: list[str] = []
        events_sensor = attrs.get("events_sensor")
        if events_sensor is not None:
            if not isinstance(events_sensor, str) or not events_sensor:
                raise ValueError("`events_sensor` must be a non-empty string")
            required.append(events_sensor)
        threshold = attrs.get("threshold")
        if threshold is not None and (
            isinstance(threshold, bool)
            or not isinstance(threshold, int | float)
            or not 0.0 <= threshold <= 1.0
        ):
            raise ValueError("`threshold` must be a number between 0 and 1")
        debounce = attrs.get("debounce_sec")
        if debounce is not None and (
            isinstance(debounce, bool)
            or not isinstance(debounce, int | float)
            or debounce < 0
        ):
            raise ValueError("`debounce_sec` must be a non-negative number")
        model_path = attrs.get("model_path")
        if model_path is not None and (
            not isinstance(model_path, str) or not model_path
        ):
            raise ValueError("`model_path` must be a non-empty string")
        return required, optional

    def reconfigure(
        self,
        config: ComponentConfig,
        dependencies: Mapping[ResourceName, ResourceBase],
    ) -> None:
        attrs = struct_to_dict(config.attributes)
        self._audio_in_name = str(attrs["audio_in"])
        self._events_sensor_name = str(attrs.get("events_sensor") or "")
        self._threshold = float(attrs.get("threshold") or DEFAULT_THRESHOLD)
        self._debounce_sec = float(attrs.get("debounce_sec") or DEFAULT_DEBOUNCE_SEC)

        self._audio_in = None
        self._events_sensor = None
        for name, resource in dependencies.items():
            if name.name == self._audio_in_name and isinstance(resource, AudioIn):
                self._audio_in = resource
            elif (
                self._events_sensor_name
                and name.name == self._events_sensor_name
                and isinstance(resource, Sensor)
            ):
                self._events_sensor = resource
        if self._audio_in is None:
            raise RuntimeError(f"audio_in {self._audio_in_name!r} not found")
        if self._events_sensor_name and self._events_sensor is None:
            LOGGER.warning(
                "events_sensor %r not found among dependencies; "
                "bark events will not be pushed",
                self._events_sensor_name,
            )

        # Lazy model bootstrap: resolve the on-disk path (raises with a
        # helpful message if the operator hasn't placed the file yet).
        if self._classifier is None:
            model_path = resolve_model_path(attrs.get("model_path"))
            self._classifier = YAMNetClassifier(model_path)

        # Restart the streaming loop.
        if self._run_task and not self._run_task.done():
            self._run_task.cancel()
        with contextlib.suppress(RuntimeError):
            self._run_task = asyncio.create_task(self._run_loop())

    # -- Audio ingest + classification --------------------------------

    async def _run_loop(self) -> None:
        assert self._audio_in is not None
        try:
            props = await self._audio_in.get_properties()
        except Exception as e:
            LOGGER.warning("get_properties failed on audio_in: %s", e)
            props = None
        source_sample_rate = int(getattr(props, "sample_rate_hz", 0) or 0) or 44100
        source_channels = int(getattr(props, "num_channels", 0) or 0) or 1

        # We ask for a continuous stream (duration=0). Buffer PCM16 bytes
        # into a rolling float32 array at 16kHz until we have a full
        # YAMNet window, then classify and slide by a full window (no
        # overlap — keeps CPU predictable and matches YAMNet's training).
        buffer = np.zeros(0, dtype=np.float32)
        try:
            stream = await self._audio_in.get_audio(
                codec=DEFAULT_CODEC, duration_seconds=0, previous_timestamp_ns=0
            )
            async for chunk in stream:
                samples = _pcm16_to_float32(
                    chunk.audio.audio_data,
                    source_channels,
                )
                if source_sample_rate != YAMNET_SAMPLE_RATE_HZ:
                    samples = _resample(
                        samples, source_sample_rate, YAMNET_SAMPLE_RATE_HZ
                    )
                buffer = np.concatenate([buffer, samples])
                while buffer.shape[0] >= YAMNET_WINDOW_SAMPLES:
                    window = buffer[:YAMNET_WINDOW_SAMPLES]
                    buffer = buffer[YAMNET_WINDOW_SAMPLES:]
                    await asyncio.to_thread(self._classify_and_emit, window)
        except asyncio.CancelledError:
            return
        except Exception as e:
            LOGGER.error("bark detector loop crashed: %s", e)

    def _classify_and_emit(self, window: np.ndarray) -> None:
        """Runs on a worker thread — TFLite blocks."""
        assert self._classifier is not None
        scores = self._classifier.classify(window)
        score, per_class = dog_score(scores)
        self._last_dog_score = score
        self._last_class_scores = per_class
        now = time.monotonic()
        if score < self._threshold:
            return
        if now - self._last_debounce_ts < self._debounce_sec:
            return
        self._last_debounce_ts = now
        self._bark_count_session += 1
        at = _now_iso()
        self._last_bark_at = at
        top_class = max(per_class.items(), key=lambda kv: kv[1])[0]
        self._bark_history.append(
            {"at": at, "score": float(score), "top_class": top_class}
        )
        # Fire-and-forget the event push; get_readings picks up the
        # rolling state either way.
        asyncio.get_event_loop().create_task(
            self._push_bark_event(at, score, top_class, per_class)
        )

    async def _push_bark_event(
        self, at: str, score: float, top_class: str, per_class: dict[str, float]
    ) -> None:
        if self._events_sensor is None:
            return
        event = {
            "event_type": "bark_detected",
            "source": self.name,
            "at": at,
            "score": float(score),
            "top_class": top_class,
            "class_scores": per_class,
        }
        try:
            await self._events_sensor.do_command({"command": "push_event", "event": event})
        except Exception as e:
            LOGGER.warning("bark event push failed: %s", e)

    # -- Sensor readings ----------------------------------------------

    async def get_readings(
        self, *, extra: Mapping[str, Any] | None = None, timeout: float | None = None, **kwargs: Any
    ) -> Mapping[str, Any]:
        return {
            "kind": "bark_detector",
            "source": self.name,
            "last_dog_score": self._last_dog_score,
            "last_bark_at": self._last_bark_at or "",
            "bark_count_session": self._bark_count_session,
            "threshold": self._threshold,
            "debounce_sec": self._debounce_sec,
            "class_scores": dict(self._last_class_scores),
        }

    async def do_command(
        self,
        command: Mapping[str, Any],
        *,
        timeout: float | None = None,
        **kwargs: Any,
    ) -> Mapping[str, Any]:
        verb = command.get("command")
        if verb == "get_history":
            return {"history": list(self._bark_history)}
        raise ValueError(f"unknown command: {verb!r}")


def _pcm16_to_float32(data: bytes, channels: int) -> np.ndarray:
    """Decode PCM16 little-endian bytes to a mono float32 array in [-1, 1].

    If the stream is stereo (or more), average across channels — YAMNet
    only wants mono. Averaging is the cheapest downmix; on interleaved
    L/R data it matches a standard mono mixdown closely enough for
    classification.
    """
    if not data:
        return np.zeros(0, dtype=np.float32)
    ints = np.frombuffer(data, dtype=np.int16)
    if channels > 1:
        ints = ints.reshape(-1, channels).mean(axis=1)
    return (ints.astype(np.float32) / PCM16_SCALE).astype(np.float32)


def _resample(samples: np.ndarray, from_hz: int, to_hz: int) -> np.ndarray:
    """Linear-interpolation resample.

    Simpler than scipy.signal.resample and good enough for audio-event
    classification: YAMNet's mel-frontend is far more sensitive to
    spectral content than to fractional-sample timing accuracy. This
    keeps the module dep-light.
    """
    if from_hz == to_hz or samples.shape[0] == 0:
        return samples
    ratio = to_hz / from_hz
    new_len = int(round(samples.shape[0] * ratio))
    if new_len <= 0:
        return np.zeros(0, dtype=np.float32)
    src_idx = np.linspace(0, samples.shape[0] - 1, new_len, dtype=np.float64)
    return np.interp(src_idx, np.arange(samples.shape[0]), samples).astype(np.float32)
