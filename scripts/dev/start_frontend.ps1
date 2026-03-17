param(
    [int]$Port = 5173
)

$projectRoot = Resolve-Path (Join-Path $PSScriptRoot "..\..")
Set-Location (Join-Path $projectRoot "frontend")

npm.cmd run dev -- --host 0.0.0.0 --port $Port
