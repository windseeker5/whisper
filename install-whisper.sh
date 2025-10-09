#!/bin/bash
# Install Whisper backend with timeout workaround
# This script downloads torch manually using wget (which supports resume)

set -e

echo "================================================"
echo "  Whisper Backend Installation (with timeout fix)"
echo "================================================"
echo ""
echo "WARNING: This will download ~900 MB of dependencies"
echo "Recommended for systems with 6GB+ RAM only"
echo ""
read -p "Continue? (y/n) " -n 1 -r
echo
if [[ ! $REPLY =~ ^[Yy]$ ]]; then
    echo "Installation cancelled"
    exit 1
fi

# Activate venv
if [ ! -d "venv" ]; then
    echo "ERROR: Virtual environment not found. Run ./install.sh first"
    exit 1
fi

source venv/bin/activate

# Get Python version
PYTHON_VERSION=$(python -c 'import sys; print(f"cp{sys.version_info.major}{sys.version_info.minor}")')
echo "Detected Python version: $PYTHON_VERSION"

# Torch wheel URL for Python 3.13
TORCH_URL="https://files.pythonhosted.org/packages/16/82/3948e54c01b2109238357c6f86242e6ecbf0c63a1af46906772902f82057/torch-2.8.0-cp313-cp313-manylinux_2_28_x86_64.whl"
TORCH_FILE="torch-2.8.0-cp313-cp313-manylinux_2_28_x86_64.whl"

# For other Python versions, let pip handle it
if [ "$PYTHON_VERSION" != "cp313" ]; then
    echo "Python version is not 3.13, using pip with extended timeout..."
    pip install --timeout=1000 --retries=10 torch
else
    # Download torch manually with wget (supports resume)
    echo "Downloading torch with wget (resume supported)..."
    echo "If download fails, just run this script again - it will resume"
    echo ""

    if [ -f "$TORCH_FILE" ]; then
        echo "Found partial download, resuming..."
    fi

    wget -c "$TORCH_URL" -O "$TORCH_FILE"

    echo ""
    echo "Installing torch from local file..."
    pip install "$TORCH_FILE"

    echo "Cleaning up..."
    rm -f "$TORCH_FILE"
fi

# Install whisper and other dependencies
echo ""
echo "Installing openai-whisper and dependencies..."
pip install --timeout=600 -r requirements-whisper.txt

echo ""
echo "================================================"
echo "  Whisper backend installed successfully!"
echo "================================================"
echo ""
echo "Configure whisper backend:"
echo "  python bin/setup_backend.py --backend whisper"
echo ""
