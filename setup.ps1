# Stop on error
$ErrorActionPreference = "Stop"

Write-Host "=== Speech-to-Text Setup ===" -ForegroundColor Cyan

# 1. Check for Python
$pythonCmd = $null
if (Get-Command python -ErrorAction SilentlyContinue) {
    $pythonCmd = "python"
} elseif (Get-Command python3 -ErrorAction SilentlyContinue) {
    $pythonCmd = "python3"
} elseif (Get-Command py -ErrorAction SilentlyContinue) {
    $pythonCmd = "py"
}

if (-not $pythonCmd) {
    Write-Host "Python not found. Please install Python from https://www.python.org/" -ForegroundColor Red
    exit 1
}

Write-Host "Using Python: $pythonCmd" -ForegroundColor Green

# 2. Create virtual environment
Write-Host "Creating virtual environment (venv)..." -ForegroundColor Yellow
& $pythonCmd -m venv venv

# 3. Activate environment and run setup
$activateScript = Join-Path $PSScriptRoot "venv\Scripts\Activate.ps1"
if (-not (Test-Path $activateScript)) {
    Write-Host "venv activation script not found. Trying alternative path..." -ForegroundColor Yellow
    $activateScript = Join-Path $PSScriptRoot "venv\Scripts\activate.ps1"
}

# Execute in project directory (dot-source activates venv in current scope)
Push-Location $PSScriptRoot
try {
    . $activateScript
    Write-Host "Updating pip..." -ForegroundColor Yellow
    pip install --upgrade pip
    Write-Host "Installing dependencies from requirements.txt..." -ForegroundColor Yellow
    pip install -r requirements.txt
    Write-Host "Downloading Vosk models..." -ForegroundColor Yellow
    & $pythonCmd setup_models.py
} finally {
    Pop-Location
}

Write-Host "=== Setup complete! ===" -ForegroundColor Green
Write-Host "To run the project:" -ForegroundColor Cyan
Write-Host "  .\venv\Scripts\Activate.ps1" -ForegroundColor White
Write-Host "  $pythonCmd main.py" -ForegroundColor White
