param(
    [string]$InputFile = "example.py"
)

Write-Host "🔧 Installing 'code_analyser' in editable mode..."
pip install -e . | Out-Null
if ($LASTEXITCODE -ne 0) {
    Write-Error "❌ pip install failed. Ensure you're in the project root."
    exit 1
}

# Dynamically get Python's Scripts directory
$scriptPath = python -c "import sysconfig; print(sysconfig.get_path('scripts'))"
$exePath = Join-Path $scriptPath "ast-metrics.exe"

Write-Host "`n📂 Input File:" $InputFile
Write-Host "📦 Scripts Directory:" $scriptPath
Write-Host "🚀 Executable Path:" $exePath

if (-not (Test-Path $exePath)) {
    Write-Warning "ast-metrics.exe not found. Falling back to 'python -m metrics.main'..."
    python -m metrics.main --file $InputFile --out metrics.json --verbose
} else {
    & $exePath --file $InputFile --out metrics.json --verbose
}

if (Test-Path "metrics.json") {
    Write-Host "`n📄 Opening metrics.json in Notepad..."
    notepad metrics.json
} else {
    Write-Error "❌ metrics.json not created."
    exit 1
}
