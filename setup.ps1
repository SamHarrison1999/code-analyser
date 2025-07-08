param(
    [string]$InputFile = "example.py"
)

Write-Host "Installing 'code-analyser' in editable mode..."
pip install -e .

$scriptPath = "$env:APPDATA\Python\Python313\Scripts"

if (-not ($env:Path -split ";" | Where-Object { $_ -eq $scriptPath })) {
    Write-Host "Adding ast-metrics CLI path to user PATH..."
    $oldPath = [Environment]::GetEnvironmentVariable("Path", "User")
    $newPath = "$oldPath;$scriptPath"
    [Environment]::SetEnvironmentVariable("Path", $newPath, "User")
    Write-Host "Please restart your terminal for PATH changes to take effect."
} else {
    Write-Host "CLI path already in PATH."
}

Write-Host "Running ast-metrics on $InputFile..."
& "$scriptPath\ast-metrics.exe" --file $InputFile --out metrics.json --verbose

Write-Host "Opening metrics.json in Notepad..."
notepad metrics.json
