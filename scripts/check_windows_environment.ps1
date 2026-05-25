param(
    [string]$Root = ""
)

$ErrorActionPreference = "Continue"

if ([string]::IsNullOrWhiteSpace($Root)) {
    $ScriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path
    $Candidate = Resolve-Path (Join-Path $ScriptDir "..") -ErrorAction SilentlyContinue
    if ($Candidate) {
        $Root = $Candidate.Path
    }
    else {
        $Root = (Get-Location).Path
    }
}

$Failures = 0
$Warnings = 0

function Write-CheckOk($Message) {
    Write-Host "[OK]   $Message" -ForegroundColor Green
}

function Write-CheckWarn($Message) {
    $script:Warnings += 1
    Write-Host "[WARN] $Message" -ForegroundColor Yellow
}

function Write-CheckFail($Message) {
    $script:Failures += 1
    Write-Host "[FAIL] $Message" -ForegroundColor Red
}

function Test-File($Path, $Name) {
    if (Test-Path -LiteralPath $Path -PathType Leaf) {
        Write-CheckOk "$Name найден: $Path"
        return $true
    }

    Write-CheckFail "$Name не найден: $Path"
    return $false
}

function Test-Directory($Path, $Name) {
    if (Test-Path -LiteralPath $Path -PathType Container) {
        Write-CheckOk "$Name найден: $Path"
        return $true
    }

    Write-CheckFail "$Name не найден: $Path"
    return $false
}

Write-Host "RayTracerRTX environment check"
Write-Host "Root: $Root"
Write-Host ""

$NvidiaSmi = Get-Command nvidia-smi -ErrorAction SilentlyContinue
if ($NvidiaSmi) {
    Write-CheckOk "nvidia-smi доступен"
    try {
        $GpuInfo = & nvidia-smi --query-gpu=name,driver_version --format=csv,noheader 2>$null
        if ($LASTEXITCODE -eq 0 -and $GpuInfo) {
            Write-Host "       GPU: $GpuInfo"
        }
        else {
            Write-CheckWarn "nvidia-smi найден, но не смог получить данные GPU"
        }
    }
    catch {
        Write-CheckWarn "nvidia-smi найден, но вызов завершился ошибкой"
    }
}
else {
    Write-CheckFail "nvidia-smi не найден. Проверьте драйвер NVIDIA RTX"
}

$CudaRoot = $env:CUDA_PATH
if ([string]::IsNullOrWhiteSpace($CudaRoot)) {
    $CudaRoot = "C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v13.1"
    Write-CheckWarn "CUDA_PATH не задан, пробую стандартный путь: $CudaRoot"
}
else {
    Write-CheckOk "CUDA_PATH задан: $CudaRoot"
}

Test-Directory $CudaRoot "CUDA Toolkit" | Out-Null
Test-File (Join-Path $CudaRoot "include\cuda_runtime.h") "CUDA headers" | Out-Null

$CudaBin = Join-Path $CudaRoot "bin"
$CudaBinX64 = Join-Path $CudaRoot "bin\x64"
$CudaDllDirs = @($CudaBin, $CudaBinX64) | Where-Object { Test-Path -LiteralPath $_ -PathType Container }

$NvrtcDlls = foreach ($Dir in $CudaDllDirs) {
    Get-ChildItem $Dir -Filter "nvrtc*.dll" -ErrorAction SilentlyContinue
}
if ($NvrtcDlls) {
    Write-CheckOk "NVRTC DLL найден: $($NvrtcDlls[0].FullName)"
}
else {
    Write-CheckFail "NVRTC DLL не найден в $CudaRoot\bin или $CudaRoot\bin\x64"
}

$CudaRuntimeDlls = foreach ($Dir in $CudaDllDirs) {
    Get-ChildItem $Dir -Filter "cudart*.dll" -ErrorAction SilentlyContinue
}
if ($CudaRuntimeDlls) {
    Write-CheckOk "CUDA Runtime DLL найден: $($CudaRuntimeDlls[0].FullName)"
}
else {
    Write-CheckFail "CUDA Runtime DLL не найден в $CudaRoot\bin или $CudaRoot\bin\x64"
}

$OptixRoot = $env:OPTIX_SDK_DIR
if ([string]::IsNullOrWhiteSpace($OptixRoot)) {
    $OptixRoot = "C:\ProgramData\NVIDIA Corporation\OptiX SDK 9.1.0"
    Write-CheckWarn "OPTIX_SDK_DIR не задан, пробую стандартный путь: $OptixRoot"
}
else {
    Write-CheckOk "OPTIX_SDK_DIR задан: $OptixRoot"
}

Test-Directory $OptixRoot "OptiX SDK" | Out-Null
Test-File (Join-Path $OptixRoot "include\optix.h") "OptiX headers" | Out-Null

$SourceInclude = Join-Path $Root "RayTracerRTX\src\common\rtx_shared.h"
Test-File $SourceInclude "Project runtime shader header" | Out-Null

$AssetsDir = Join-Path $Root "RayTracerRTX\assets"
Test-Directory $AssetsDir "Assets" | Out-Null
Test-File (Join-Path $AssetsDir "scenes\demo_scene.json") "Demo scene" | Out-Null

$DebugExe = Join-Path $Root "x64\Debug\RayTracerRTX.exe"
$ReleaseExe = Join-Path $Root "x64\Release\RayTracerRTX.exe"
$PortableExe = Join-Path $Root "RayTracerRTX.exe"
if ((Test-Path -LiteralPath $ReleaseExe -PathType Leaf) -or
    (Test-Path -LiteralPath $DebugExe -PathType Leaf) -or
    (Test-Path -LiteralPath $PortableExe -PathType Leaf)) {
    Write-CheckOk "RayTracerRTX.exe найден"
}
else {
    Write-CheckWarn "RayTracerRTX.exe не найден. Сначала соберите проект в Visual Studio или MSBuild"
}

Write-Host ""
Write-Host "Result: $Failures error(s), $Warnings warning(s)"
if ($Failures -gt 0) {
    exit 1
}

exit 0
