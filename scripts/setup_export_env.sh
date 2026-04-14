#!/usr/bin/env bash
# Setup Python environment for Gemma 4 ExecuTorch export.
# Run on a Linux machine with CUDA (A100/H100 recommended).
#
# Usage:
#   bash scripts/setup_export_env.sh
#   source .venv/bin/activate
#   export HF_TOKEN=hf_...
#   python scripts/export_gemma4_e4b.py

set -euo pipefail

VENV_DIR=".venv"
PYTHON="python3"

echo "=== Gemma 4 ExecuTorch Export Environment Setup ==="
echo ""

# Python check
if ! command -v "$PYTHON" &>/dev/null; then
    echo "ERROR: python3 not found. Install Python 3.10+."
    exit 1
fi
PYTHON_VERSION=$("$PYTHON" -c "import sys; print(f'{sys.version_info.major}.{sys.version_info.minor}')")
echo "Python: $PYTHON_VERSION"

# Create venv
if [ ! -d "$VENV_DIR" ]; then
    echo "Creating virtualenv at $VENV_DIR ..."
    "$PYTHON" -m venv "$VENV_DIR"
fi
# shellcheck disable=SC1091
source "$VENV_DIR/bin/activate"

echo "Installing dependencies ..."
pip install --quiet --upgrade pip

# PyTorch (CUDA 12.1 build — adjust cu121 if your CUDA version differs)
pip install --quiet \
    "torch==2.5.1" \
    "torchvision==0.20.1" \
    --index-url https://download.pytorch.org/whl/cu121

# ExecuTorch (pip wheel covers the Python export pipeline)
pip install --quiet executorch==1.2.0

# Model tooling
pip install --quiet \
    torchtune \
    transformers \
    "huggingface_hub[cli]" \
    sentencepiece \
    protobuf

echo ""
echo "Setup complete. Activate with:"
echo "  source $VENV_DIR/bin/activate"
echo ""
echo "Then export with:"
echo "  export HF_TOKEN=hf_..."
echo "  python scripts/export_gemma4_e4b.py"
