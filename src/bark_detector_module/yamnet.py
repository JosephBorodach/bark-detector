"""YAMNet TFLite wrapper.

YAMNet is an audio event classifier trained on AudioSet. It takes 0.975s of
16kHz mono float32 audio (15600 samples) and returns scores for 521 classes.
We only care about the dog-vocalization slice.

The .tflite file is NOT auto-downloaded. The operator must place it at
DEFAULT_MODEL_PATH (or configure a `model_path` on the sensor). See the
README for sources and how to verify the file — pinning a hash to a URL
we haven't independently vetted would be a supply-chain footgun for
everyone using this module.
"""
from __future__ import annotations

import logging
import threading
from pathlib import Path

import numpy as np

LOGGER = logging.getLogger(__name__)

# AudioSet dog-vocalization ontology (0-indexed into YAMNet's 521 classes).
# Source: https://github.com/tensorflow/models/blob/master/research/audioset/yamnet/yamnet_class_map.csv
DOG_CLASSES: dict[int, str] = {
    68: "Dog",
    69: "Bark",
    70: "Yip",
    71: "Howl",
    72: "Bow-wow",
    73: "Growling",
    74: "Whimper (dog)",
}

YAMNET_SAMPLE_RATE_HZ = 16000
YAMNET_WINDOW_SAMPLES = 15600  # 0.975 seconds at 16 kHz

DEFAULT_MODEL_PATH = Path.home() / ".viam" / "bark-detector" / "yamnet.tflite"


def dog_score(class_scores: np.ndarray) -> tuple[float, dict[str, float]]:
    """Aggregate 521 class scores to a single dog-vocalization score.

    Returns (score, per_class) where per_class is a dict of the individual
    dog-related class scores. The overall score is max() across the slice —
    max is preferable to sum because a low sustained score across many
    classes shouldn't count as a bark; a peak on any single dog class
    should.
    """
    if class_scores.ndim != 1 or class_scores.shape[0] != 521:
        raise ValueError(
            f"expected class_scores shape (521,), got {class_scores.shape}"
        )
    per_class = {name: float(class_scores[idx]) for idx, name in DOG_CLASSES.items()}
    return max(per_class.values(), default=0.0), per_class


def resolve_model_path(configured: str | None = None) -> Path:
    """Return the on-disk YAMNet path or raise a helpful error.

    Falls back to DEFAULT_MODEL_PATH. Refuses to guess or download —
    operator supplies the file (see README for sources) and this just
    surfaces a clear error if it isn't there.
    """
    path = Path(configured).expanduser() if configured else DEFAULT_MODEL_PATH
    if not path.exists():
        raise RuntimeError(
            f"YAMNet model not found at {path}. "
            f"Download yamnet.tflite (see README for sources) and place it "
            f"at that path, or set `model_path` on the sensor config."
        )
    return path


class YAMNetClassifier:
    """Thin wrapper around a YAMNet TFLite interpreter.

    Loading and inference are both done under a lock — TFLite interpreters
    aren't thread-safe. Callers only need to pass a 15600-sample float32
    waveform to `classify`.
    """

    def __init__(self, model_path: Path):
        # Deferred import: tflite-runtime isn't available on macOS dev
        # machines without extra setup. This lets the test suite import
        # the module without requiring TFLite at collection time.
        try:
            from tflite_runtime.interpreter import Interpreter  # type: ignore
        except ImportError:
            from tensorflow.lite.python.interpreter import Interpreter  # type: ignore
        self._interp = Interpreter(model_path=str(model_path))
        self._interp.allocate_tensors()
        self._in_idx = self._interp.get_input_details()[0]["index"]
        # YAMNet returns three tensors: scores (521,), embeddings, log-mel.
        # We only need scores — the first output.
        self._out_idx = self._interp.get_output_details()[0]["index"]
        self._lock = threading.Lock()

    def classify(self, samples: np.ndarray) -> np.ndarray:
        """Run the model on one 0.975s window.

        Args:
            samples: shape (15600,), dtype float32, range [-1, 1], 16kHz.
        Returns:
            class_scores: shape (521,), float32 probabilities.
        """
        if samples.shape != (YAMNET_WINDOW_SAMPLES,) or samples.dtype != np.float32:
            raise ValueError(
                f"YAMNet expects ({YAMNET_WINDOW_SAMPLES},) float32; "
                f"got {samples.shape} {samples.dtype}"
            )
        with self._lock:
            self._interp.set_tensor(self._in_idx, samples)
            self._interp.invoke()
            out = self._interp.get_tensor(self._out_idx)
        # Model returns shape (1, 521) — squeeze the batch dim.
        return np.asarray(out, dtype=np.float32).reshape(-1)
