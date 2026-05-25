param(
    [string]$Configuration = "Release"
)

$ErrorActionPreference = "Stop"

$ScriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path
$RepoRoot = (Resolve-Path (Join-Path $ScriptDir "..")).Path
$PackageRoot = Join-Path $RepoRoot ".release\RayTracerRTX-portable"
$ReleaseDir = Join-Path $RepoRoot "x64\$Configuration"
$ExePath = Join-Path $ReleaseDir "RayTracerRTX.exe"

if (!(Test-Path -LiteralPath $ExePath -PathType Leaf)) {
    throw "Executable not found: $ExePath. Build x64 $Configuration first."
}

$ResolvedRepo = (Resolve-Path $RepoRoot).Path
$ReleaseParent = Join-Path $RepoRoot ".release"
New-Item -ItemType Directory -Force -Path $ReleaseParent | Out-Null

if (Test-Path -LiteralPath $PackageRoot) {
    $ResolvedPackage = (Resolve-Path $PackageRoot).Path
    if (!$ResolvedPackage.StartsWith($ResolvedRepo, [System.StringComparison]::OrdinalIgnoreCase)) {
        throw "Refusing to remove package outside repository: $ResolvedPackage"
    }

    Remove-Item -LiteralPath $PackageRoot -Recurse -Force
}

New-Item -ItemType Directory -Force -Path $PackageRoot | Out-Null
New-Item -ItemType Directory -Force -Path (Join-Path $PackageRoot "RayTracerRTX") | Out-Null
New-Item -ItemType Directory -Force -Path (Join-Path $PackageRoot "RayTracerRTX\src") | Out-Null

Copy-Item -LiteralPath $ExePath -Destination (Join-Path $PackageRoot "RayTracerRTX.exe") -Force
Copy-Item -LiteralPath (Join-Path $RepoRoot "RayTracerRTX\assets") -Destination (Join-Path $PackageRoot "RayTracerRTX\assets") -Recurse -Force
Copy-Item -LiteralPath (Join-Path $RepoRoot "RayTracerRTX\src\common") -Destination (Join-Path $PackageRoot "RayTracerRTX\src\common") -Recurse -Force
Copy-Item -LiteralPath (Join-Path $RepoRoot "scripts\check_windows_environment.ps1") -Destination (Join-Path $PackageRoot "check_windows_environment.ps1") -Force

$RunScript = @'
$ErrorActionPreference = "Stop"
$Root = Split-Path -Parent $MyInvocation.MyCommand.Path
$env:RAYTRACERRTX_SOURCE_DIR = Join-Path $Root "RayTracerRTX\src"

if ([string]::IsNullOrWhiteSpace($env:CUDA_PATH)) {
    $DefaultCuda = "C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v13.1"
    if (Test-Path -LiteralPath $DefaultCuda) {
        $env:CUDA_PATH = $DefaultCuda
    }
}

if (![string]::IsNullOrWhiteSpace($env:CUDA_PATH)) {
    $CudaBin = Join-Path $env:CUDA_PATH "bin"
    $CudaBinX64 = Join-Path $env:CUDA_PATH "bin\x64"
    foreach ($Dir in @($CudaBin, $CudaBinX64)) {
        if ((Test-Path -LiteralPath $Dir) -and ($env:PATH -notlike "*$Dir*")) {
            $env:PATH = "$Dir;$env:PATH"
        }
    }
}

if ([string]::IsNullOrWhiteSpace($env:OPTIX_SDK_DIR)) {
    $DefaultOptix = "C:\ProgramData\NVIDIA Corporation\OptiX SDK 9.1.0"
    if (Test-Path -LiteralPath $DefaultOptix) {
        $env:OPTIX_SDK_DIR = $DefaultOptix
    }
}

$Exe = Join-Path $Root "RayTracerRTX.exe"
if ($args.Count -gt 0) {
    & $Exe @args
}
else {
    & $Exe --scene "RayTracerRTX\assets\scenes\demo_scene.json"
}
'@

Set-Content -LiteralPath (Join-Path $PackageRoot "run_release.ps1") -Value $RunScript -Encoding UTF8

$Readme = @'
# RayTracerRTX portable package

Запуск одной кнопкой:

- `Run RayTracerRTX.bat` - запускает демо-сцену;
- `Check Environment.bat` - проверяет драйвер, CUDA, OptiX и assets.

Запуск из PowerShell:

```powershell
.\check_windows_environment.ps1
.\run_release.ps1
```

Запуск конкретной сцены:

```powershell
.\run_release.ps1 --scene RayTracerRTX\assets\scenes\multi_mesh_scene.json
```

На компьютере должны быть установлены:

- NVIDIA driver для RTX GPU;
- CUDA Toolkit 13.1;
- NVIDIA OptiX SDK 9.1.0.

Если CUDA или OptiX установлены не в стандартные папки, задайте переменные окружения `CUDA_PATH` и `OPTIX_SDK_DIR`.
'@

Set-Content -LiteralPath (Join-Path $PackageRoot "README_RUN.md") -Value $Readme -Encoding UTF8

$RunBat = @'
@echo off
setlocal
cd /d "%~dp0"
powershell -ExecutionPolicy Bypass -File "%~dp0run_release.ps1"
pause
'@

Set-Content -LiteralPath (Join-Path $PackageRoot "Run RayTracerRTX.bat") -Value $RunBat -Encoding ASCII

$CheckBat = @'
@echo off
setlocal
cd /d "%~dp0"
powershell -ExecutionPolicy Bypass -File "%~dp0check_windows_environment.ps1" -Root "%~dp0"
pause
'@

Set-Content -LiteralPath (Join-Path $PackageRoot "Check Environment.bat") -Value $CheckBat -Encoding ASCII

$ArchivePath = Join-Path $RepoRoot ".release\RayTracerRTX-portable.zip"
if (Test-Path -LiteralPath $ArchivePath) {
    Remove-Item -LiteralPath $ArchivePath -Force
}
Compress-Archive -LiteralPath (Join-Path $PackageRoot "*") -DestinationPath $ArchivePath -Force

Write-Host "Package: $PackageRoot"
Write-Host "Archive: $ArchivePath"
