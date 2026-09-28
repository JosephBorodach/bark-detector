# bark-detector

Detect dog barks from an `audio_in` component using YAMNet, and publish
`bark_detected` events to a `viam:event-queue:sensor` for tabular capture
and dashboards.

Designed for the household use case — one dog, one mic, "did he bark, and
when." Not a professional acoustic-event pipeline. Detection cadence is
YAMNet's native 0.975s window; events are debounced so a sustained
barking session produces one event per configurable window, not one per
inference.

## Detector

**Model:** `joseph:bark-detector:sensor`
**API:** `rdk:component:sensor`

### Configuration

```json
{
  "name": "bark",
  "namespace": "rdk",
  "type": "sensor",
  "model": "joseph:bark-detector:sensor",
  "attributes": {
    "audio_in": "microphone",
    "events_sensor": "events",
    "threshold": 0.5,
    "debounce_sec": 2.0
  },
  "depends_on": ["microphone", "events"]
}
```

| Attribute       | Type   | Required | Description                                                                                     |
|-----------------|--------|----------|-------------------------------------------------------------------------------------------------|
| `audio_in`      | string | yes      | Name of an `rdk:component:audio_in` (e.g. `viam:system-audio:microphone`).                      |
| `events_sensor` | string | no       | Name of a `viam:event-queue:sensor` to receive `bark_detected` events. Omit → detection only.   |
| `threshold`     | number | no       | Bark-score threshold in `[0, 1]`. Defaults to `0.5`. Lower catches more; higher rejects more.   |
| `debounce_sec`  | number | no       | Minimum seconds between events. Defaults to `2.0`. Prevents one event per YAMNet window.        |
| `model_path`    | string | no       | Absolute path to `yamnet.tflite`. Defaults to `~/.viam/bark-detector/yamnet.tflite`.            |

### Detection

- The detector reads a continuous PCM16 stream from `audio_in`.
- Chunks are resampled to 16 kHz mono float32 (YAMNet's native input).
- Every 0.975 s window is classified into 521 AudioSet classes.
- The bark score is `max()` over the dog-vocalization slice: `Dog`, `Bark`,
  `Yip`, `Howl`, `Bow-wow`, `Growling`, `Whimper (dog)`. Max over the slice
  (rather than sum) so a low sustained score across many classes doesn't
  trigger; a peak on any single dog class does.
- If the score crosses `threshold` and `debounce_sec` has elapsed since the
  last event, a `bark_detected` event is pushed to `events_sensor`.

### Event shape

```json
{
  "event_type": "bark_detected",
  "source": "bark",
  "at": "2026-09-28T22:13:00Z",
  "score": 0.87,
  "top_class": "Bark",
  "class_scores": {
    "Dog": 0.12,
    "Bark": 0.87,
    "Yip": 0.04,
    "Howl": 0.01,
    "Bow-wow": 0.09,
    "Growling": 0.02,
    "Whimper (dog)": 0.00
  }
}
```

Configure `events_sensor` with data capture on `Readings` to build a
tabular history for triggers (e.g. "notify me when barks/min > N") and
dashboards.

### `get_readings`

Returns the current rolling state — safe to poll from a web app.

```json
{
  "kind": "bark_detector",
  "source": "bark",
  "last_dog_score": 0.87,
  "last_bark_at": "2026-09-28T22:13:00Z",
  "bark_count_session": 42,
  "threshold": 0.5,
  "debounce_sec": 2.0,
  "class_scores": { "...": "..." }
}
```

`bark_count_session` resets whenever the module reconfigures.

## Model

**The operator supplies the YAMNet TFLite file.** The module does not
auto-download — pinning a URL and hash we haven't independently vetted
would be a supply-chain footgun for every downstream user.

To install:

1. Obtain `yamnet.tflite` from a source you trust. Candidates (verify each
   yourself before using):
   - **Kaggle Models** (formerly TensorFlow Hub) — `google/yamnet/tfLite/classification-tflite`.
   - **MediaPipe** ships YAMNet as its audio classifier default.
   - Build it yourself from the [TensorFlow Models](https://github.com/tensorflow/models/tree/master/research/audioset/yamnet)
     repo and export to TFLite.
2. Verify the SHA-256 against the source's published hash.
3. Copy it to `~/.viam/bark-detector/yamnet.tflite` (the default) on the Pi,
   or set `model_path` on the sensor config to an alternate path.

If the file is missing at boot the module logs a clear error and refuses to
start, so it's easy to catch in the machine's error log.

## Wiring example

Minimum viable machine config on a Pi with a USB microphone:

```json
{
  "components": [
    {
      "name": "microphone",
      "api": "rdk:component:audio_in",
      "model": "viam:system-audio:microphone",
      "attributes": {
        "device_name": "plughw:CARD=USB,DEV=0",
        "num_channels": 1,
        "sample_rate": 16000
      }
    },
    {
      "name": "events",
      "namespace": "rdk",
      "type": "sensor",
      "model": "viam:event-queue:sensor",
      "attributes": { "queue_capacity": 1000 },
      "service_configs": [
        {
          "type": "data_manager",
          "attributes": {
            "capture_methods": [
              { "method": "Readings", "capture_frequency_hz": 1 }
            ]
          }
        }
      ]
    },
    {
      "name": "bark",
      "namespace": "rdk",
      "type": "sensor",
      "model": "joseph:bark-detector:sensor",
      "attributes": {
        "audio_in": "microphone",
        "events_sensor": "events"
      },
      "depends_on": ["microphone", "events"]
    }
  ]
}
```

Use `plughw:CARD=<name>,DEV=0` (ALSA card name, not index) so USB
enumeration on reboot doesn't break `device_name`. `arecord -L` on the Pi
lists the stable names.

## Development

```
python3 -m venv .venv
./.venv/bin/pip install -e '.[dev]'
./.venv/bin/pytest
```

`tflite-runtime` is under the `runtime` extra rather than a base dep so
dev machines without a prebuilt wheel can still install and run the
tests. Tests use a stub classifier — no real inference or model download.
