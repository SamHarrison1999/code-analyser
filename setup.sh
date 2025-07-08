#!/bin/bash

# Default to example.py if no argument is given
INPUT_FILE="${1:-example.py}"

echo "🔧 Installing 'code-analyser' in editable mode..."
pip install -e . > /dev/null
if [ $? -ne 0 ]; then
  echo "❌ pip install failed. Are you in the project root?"
  exit 1
fi

# Check if ast-metrics is installed in PATH
if command -v ast-metrics &> /dev/null; then
  echo "🚀 Running ast-metrics on $INPUT_FILE..."
  ast-metrics --file "$INPUT_FILE" --out metrics.json --verbose
else
  echo "⚠️ 'ast-metrics' CLI not found. Falling back to 'python -m metrics.main'..."
  python -m metrics.main --file "$INPUT_FILE" --out metrics.json --verbose
fi

if [ -f metrics.json ]; then
  echo "✅ Output written to metrics.json"

  # Optional: auto-open based on OS
  case "$OSTYPE" in
    linux*)   xdg-open metrics.json >/dev/null 2>&1 & ;;
    darwin*)  open metrics.json ;;
  esac
else
  echo "❌ metrics.json not created."
  exit 1
fi
