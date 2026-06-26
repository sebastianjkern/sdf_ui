#!/usr/bin/env bash

set -e

VENV_DIR=".venv"

if [ -n "${1:-}" ]; then
    PYEXE="$1"
else
    PYEXE="python"
fi

# Check Python availability
if ! command -v "$PYEXE" >/dev/null 2>&1; then
    echo "Python executable '$PYEXE' was not found. Please install Python 3.10+ and retry."
    exit 1
fi

if ! "$PYEXE" --version >/dev/null 2>&1; then
    echo "Python executable '$PYEXE' was not found. Please install Python 3.10+ and retry."
    exit 1
fi

if [ -f "$VENV_DIR/bin/python" ]; then
    echo "Virtual environment already exists; skipping creation."
else
    echo "Creating virtual environment in $VENV_DIR..."
    "$PYEXE" -m venv "$VENV_DIR"
fi

# Use the venv Python directly for installation so this is reproducible.
echo "Using venv python executable to run pip commands so installation is reproducible."
echo "Upgrading pip, setuptools and wheel inside the venv..."
"$VENV_DIR/bin/python" -m pip install --upgrade pip setuptools wheel

echo "Installing project in editable mode into the venv..."
"$VENV_DIR/bin/python" -m pip install --no-build-isolation -e .[dev]

if [ "${BASH_SOURCE[0]}" != "$0" ]; then
    echo "Activating the virtual environment in this shell..."
    # shellcheck disable=SC1090
    source "$VENV_DIR/bin/activate"
else
    echo "To activate the virtual environment in your current shell, run:"
    echo "    source \"$VENV_DIR/bin/activate\""
fi

echo "Installation complete."
echo "The virtual environment is now ready."
echo "To run an example inside the venv, use:"
echo "    python examples.py render_api"
echo "To run tests inside the venv:"
echo "    python -m pytest"

exit 0
