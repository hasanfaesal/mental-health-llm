#!/bin/bash
# Fast installation script using uv package manager

echo "====================================="
echo "Fast Setup with UV Package Manager"
echo "====================================="

# Check if uv is available
if ! command -v uv &> /dev/null; then
    echo "Error: uv package manager not found."
    echo "Installing uv..."
    curl -LsSf https://astral.sh/uv/install.sh | sh
    source $HOME/.cargo/env
fi

echo "UV version: $(uv --version)"

# Sync dependencies using uv (much faster than pip)
echo "Syncing dependencies with uv..."
uv sync

# Alternative: Install from requirements.txt if no pyproject.toml
if [ ! -f "pyproject.toml" ]; then
    echo "Installing from requirements.txt..."
    uv pip install -r requirements.txt
fi

echo "Dependencies installed successfully!"
echo "====================================="
echo "To activate environment: source .venv/bin/activate"
echo "To run training: ./run_training.sh"
echo "====================================="
