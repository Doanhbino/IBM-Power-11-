#!/usr/bin/env bash
# No sudo, no OS package changes, no source fallback. Supply an already approved Python.
set -euo pipefail
if [[ "${EUID}" -eq 0 ]]; then echo 'Run as an unprivileged account.' >&2; exit 2; fi
MODE="${1:-help}"
PYTHON_BIN="${PYTHON_BIN:-python3}"
VENV_DIR="${VENV_DIR:-$PWD/.venv-cpu}"
WHEEL_DIR="${WHEEL_DIR:-$PWD/wheelhouse}"
REQ_FILE="${REQ_FILE:-$PWD/requirements.cpu.lock}"
case "$MODE" in
  probe)
    # User must provide exact versions/hashes after checking official package provenance.
    test -f "$REQ_FILE" || { echo "Missing reviewed lock file: $REQ_FILE" >&2; exit 2; }
    mkdir -p "$WHEEL_DIR"
    "$PYTHON_BIN" -m pip debug --verbose > "$WHEEL_DIR/pip-platform.txt"
    "$PYTHON_BIN" -m pip download --only-binary=:all: --require-hashes --dest "$WHEEL_DIR" -r "$REQ_FILE"
    echo 'All required compatible wheels downloaded. A missing wheel stops this gate; no source build attempted.'
    ;;
  install)
    test -f "$REQ_FILE" || { echo "Missing reviewed lock file: $REQ_FILE" >&2; exit 2; }
    test -d "$WHEEL_DIR" || { echo 'Run probe or transfer a verified wheelhouse first.' >&2; exit 2; }
    test ! -e "$VENV_DIR" || { echo "Refusing to reuse/overwrite existing environment: $VENV_DIR" >&2; exit 2; }
    "$PYTHON_BIN" -m venv "$VENV_DIR"
    "$VENV_DIR/bin/python" -m pip install --no-index --find-links "$WHEEL_DIR" --only-binary=:all: --require-hashes -r "$REQ_FILE"
    "$VENV_DIR/bin/python" -m pip check
    "$VENV_DIR/bin/python" -m pip freeze > "$VENV_DIR/packages.freeze.txt"
    echo "Environment ready: $VENV_DIR"
    ;;
  *)
    echo 'PYTHON_BIN=/approved/python REQ_FILE=/reviewed/hash-pinned.lock WHEEL_DIR=/path/wheels VENV_DIR=/path/new-venv bash scripts/setup_env.sh probe|install'
    echo 'probe uses network through the currently configured pip index; install is offline. Does not change pip indexes, drivers, RHEL, SMT or system Python.'
    [[ "$MODE" == help ]] || exit 2
    ;;
esac
