param(
    [switch]$NoCache
)

$ErrorActionPreference = "Stop"

$repoRoot = Split-Path -Parent $PSScriptRoot
Set-Location $repoRoot

docker compose config | Out-Host

$buildArgs = @("compose", "build")
if ($NoCache) {
    $buildArgs += "--no-cache"
}

& docker @buildArgs
& docker compose run --rm coursework-check
