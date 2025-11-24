#!/bin/bash
set -e

echo "Setting up various-llm-benchmark..."

# Check for uv
if ! command -v uv &> /dev/null; then
    echo "uv not found. Installing uv..."
    curl -LsSf https://astral.sh/uv/install.sh | sh
    # Attempt to add uv to PATH for the current session
    export PATH="$HOME/.local/bin:$HOME/.cargo/bin:$PATH"
    
    if ! command -v uv &> /dev/null; then
        echo "Error: Failed to install uv or add it to PATH. Please install manually."
        exit 1
    fi
fi

# Install dependencies
echo "Creating virtual environment and installing dependencies..."
if [ ! -d .venv ]; then
    echo "Creating .venv using uv..."
    uv venv .venv
fi
# Activate the virtual environment
source .venv/bin/activate
uv sync --extra dev

# Setup .env
if [ ! -f .env ]; then
    echo "Creating .env from .env.example..."
    cp .env.example .env
    echo "Please edit .env to set your API keys."
else
    echo ".env already exists. Skipping."
fi

# Start Database
if command -v docker &> /dev/null; then
    echo "Starting database..."
    docker compose up -d db
else
    echo "Warning: docker not found. Skipping database startup."
    echo "If you need vector search features, please install Docker and run: docker compose up -d db"
fi

echo "Setup complete!"
echo ""
echo "Virtual environment created at: .venv"
echo "To activate it, run:"
echo "    source .venv/bin/activate"
echo ""
echo "Or run commands directly with uv:"
echo "    uv run various-llm-benchmark --help"
