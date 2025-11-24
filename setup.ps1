Write-Host "Setting up various-llm-benchmark..."

# Check for uv
if (-not (Get-Command "uv" -ErrorAction SilentlyContinue)) {
    Write-Host "uv not found. Installing uv..."
    powershell -ExecutionPolicy ByPass -c "irm https://astral.sh/uv/install.ps1 | iex"
    
    # Attempt to add uv to PATH for the current session
    $uvPath1 = Join-Path $env:USERPROFILE ".cargo\bin"
    $uvPath2 = Join-Path $env:USERPROFILE ".local\bin"
    
    if (Test-Path $uvPath1) { $env:Path = "$uvPath1;$env:Path" }
    if (Test-Path $uvPath2) { $env:Path = "$uvPath2;$env:Path" }
    
    # Also reload User PATH from registry to be sure
    $userPath = [Environment]::GetEnvironmentVariable("Path", [EnvironmentVariableTarget]::User)
    $env:Path = "$env:Path;$userPath"
    
    if (-not (Get-Command "uv" -ErrorAction SilentlyContinue)) {
        Write-Error "Error: Failed to locate uv even after installation."
        Write-Error "Please restart your terminal and try again."
        exit 1
    }
}

# Install dependencies
Write-Host "Creating virtual environment and installing dependencies..."
uv sync --extra dev

# Setup .env
if (-not (Test-Path ".env")) {
    Write-Host "Creating .env from .env.example..."
    Copy-Item ".env.example" ".env"
    Write-Host "Please edit .env to set your API keys."
} else {
    Write-Host ".env already exists. Skipping."
}

# Install Gemini CLI if npm is available
if (Get-Command "npm" -ErrorAction SilentlyContinue) {
    Write-Host "Installing @google/gemini-cli..."
    npm install -g @google/gemini-cli
} else {
    Write-Warning "npm not found. Skipping @google/gemini-cli installation."
    Write-Host "To use Gemini CLI features, please install Node.js and run: npm install -g @google/gemini-cli"
}

# Start Database
if (Get-Command "docker" -ErrorAction SilentlyContinue) {
    Write-Host "Starting database..."
    docker compose up -d db
} else {
    Write-Warning "docker not found. Skipping database startup."
    Write-Host "If you need vector search features, please install Docker and run: docker compose up -d db"
}

Write-Host "Setup complete!"
Write-Host ""
Write-Host "NOTE: If uv was just installed, you may need to RESTART YOUR TERMINAL for it to be available globally."
Write-Host ""
Write-Host "Virtual environment created at: .venv"
Write-Host "To activate it, run:"
Write-Host "    .\.venv\Scripts\activate"
Write-Host ""
Write-Host "Or run commands directly with uv:"
Write-Host "    uv run various-llm-benchmark --help"
