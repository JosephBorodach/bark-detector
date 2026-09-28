#!/usr/bin/env bash
# Viam module entrypoint. Bootstraps a venv on first run, then execs the
# module server.
set -euo pipefail

cd "$(dirname "$0")"

if [ ! -d ".venv" ] || [ ! -x "./.venv/bin/pip" ]; then
    # python3 -m venv occasionally succeeds while ensurepip fails silently,
    # leaving a venv with no bin/pip. Rebuild in that case rather than
    # crash-looping.
    rm -rf .venv
    python3 -m venv .venv
    ./.venv/bin/python -m ensurepip --upgrade
fi

if ! ./.venv/bin/python -c "import bark_detector_module" 2>/dev/null; then
    ./.venv/bin/pip install --upgrade pip
    # '.[runtime]' pulls tflite-runtime on the Pi. Kept as an extra
    # rather than a base dep because tflite-runtime doesn't ship wheels
    # for every dev platform, and pip fails resolve if a dep can't be
    # satisfied at all.
    ./.venv/bin/pip install '.[runtime]'
fi

exec ./.venv/bin/python -m bark_detector_module.main "$@"
