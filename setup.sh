#!/bin/bash

# Default to example.py if no argument is given
INPUT_FILE=${1:-example.py}

echo "Installing 'code-analyser' in editable mode..."
pip install -e .

# Assume ast-metrics is in PATH or local venv bin
echo "Running ast-metrics on $INPUT_FILE..."
ast-metrics --file "$INPUT_FILE" --out metrics.json --verbose

# Optional: open metrics in text editor (commented for CI safety)
# xdg-open metrics.json  # Linux desktop
# open metrics.json      # macOS

echo "Done. Output written to metrics.json"
