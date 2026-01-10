#!/usr/bin/env bash
set -euo pipefail

# Creates .venv and installs both packages in editable mode.
#
#   scripts/bootstrap.sh                 # uses python3.12 if found, else python3
#   PYTHON_BIN=/path/to/python3.12 scripts/bootstrap.sh
#   PINNED=0 scripts/bootstrap.sh        # ignore constraints.txt
#
# The committed result bundles were produced with CPython 3.12.14 and the
# versions in constraints.txt. With Python 3.12 the constraints are applied by
# default so a rebuilt environment reproduces those bundles byte for byte. With
# another interpreter the latest compatible versions are installed and a
# warning is printed; the verifiers still pass, but a fresh run may differ.

WORKSPACE_DIR="$(CDPATH= cd -- "$(dirname -- "$0")/.." && pwd)"
VENV_DIR="$WORKSPACE_DIR/.venv"
CONSTRAINTS="$WORKSPACE_DIR/constraints.txt"

if [ -z "${PYTHON_BIN:-}" ]; then
  for candidate in python3.12 python3.13 python3.11 python3; do
    if command -v "$candidate" >/dev/null 2>&1; then PYTHON_BIN="$(command -v "$candidate")"; break; fi
  done
fi
: "${PYTHON_BIN:?no python3 interpreter found}"

PY_VERSION="$("$PYTHON_BIN" -c 'import sys; print("%d.%d" % sys.version_info[:2])')"
case "$PY_VERSION" in
  3.11|3.12|3.13|3.14) ;;
  *) echo "error: Python >= 3.11 required, found $PY_VERSION at $PYTHON_BIN" >&2; exit 1 ;;
esac

if [ -z "${PINNED:-}" ]; then
  if [ "$PY_VERSION" = "3.12" ]; then PINNED=1; else PINNED=0; fi
fi

if [ -e "$VENV_DIR" ]; then
  echo "error: $VENV_DIR already exists; remove or rename it first" >&2
  exit 1
fi

echo "Creating $VENV_DIR with $PYTHON_BIN (Python $PY_VERSION, pinned=$PINNED)"
"$PYTHON_BIN" -m venv "$VENV_DIR"
"$VENV_DIR/bin/python" -m pip install --quiet --upgrade pip setuptools wheel

PIP_ARGS=()
if [ "$PINNED" = "1" ]; then
  PIP_ARGS+=(-c "$CONSTRAINTS")
else
  echo "warning: not applying constraints.txt (Python $PY_VERSION); committed results used 3.12.14" >&2
fi

"$VENV_DIR/bin/python" -m pip install --quiet "${PIP_ARGS[@]}" \
  -e "$WORKSPACE_DIR/energy-ai-data-onboarding[dev]" \
  -e "$WORKSPACE_DIR/energy-demand-forecasting[dev]"

"$VENV_DIR/bin/python" -m pip check
"$VENV_DIR/bin/python" - <<'PY'
import sys, numpy, pandas, sklearn
print("Environment ready:")
print("  python      ", sys.version.split()[0])
print("  numpy       ", numpy.__version__)
print("  pandas      ", pandas.__version__)
print("  scikit-learn", sklearn.__version__)
PY
