param(
    [int]$Port = 8000
)

$projectRoot = Resolve-Path (Join-Path $PSScriptRoot "..\..")
Set-Location $projectRoot

python -m uvicorn backend.app.main:app --host 0.0.0.0 --port $Port --reload
