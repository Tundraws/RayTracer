# Static Analysis Report

Date: 2026-05-19

## Tool

- Primary tool: `clang-tidy`
- Version: LLVM 19.1.5
- Source: Visual Studio LLVM tools
- `cppcheck` status: not installed in the current environment

## Scope

The analysis covers CPU-side project logic that can be checked without opening
the GLFW window or launching the OptiX renderer:

- `RayTracerRTX/src/app/camera.cpp`
- `RayTracerRTX/src/app/material.cpp`
- `RayTracerRTX/src/app/scene.cpp`

GPU/OptiX code is still covered by regular build and GPU smoke testing because
full static analysis of NVRTC-compiled OptiX device programs requires a
dedicated CUDA-aware toolchain configuration.

## Command

```powershell
& "C:\Program Files\Microsoft Visual Studio\2022\Community\VC\Tools\Llvm\x64\bin\clang-tidy.exe" `
  RayTracerRTX\src\app\camera.cpp `
  RayTracerRTX\src\app\material.cpp `
  RayTracerRTX\src\app\scene.cpp `
  --config="{Checks: 'clang-analyzer-*,bugprone-*,performance-*,-bugprone-easily-swappable-parameters'}" `
  --extra-arg=-std=c++20 `
  --extra-arg=-IC:\Users\User\source\repos\RayTracerRTX\RayTracerRTX\tests\stubs `
  --extra-arg=-IC:\Users\User\source\repos\RayTracerRTX\RayTracerRTX\src
```

## Result

Exit code: `0`

Summary:

- No project-level errors were reported.
- No actionable project-level warnings were reported.
- `clang-tidy` reported suppressed warnings from non-user/system headers.
- `bugprone-easily-swappable-parameters` was disabled because vector math
  helpers naturally use adjacent parameters of the same type.

## Follow-up

For final CI, add either:

- `clang-tidy` with a generated `compile_commands.json`; or
- `cppcheck` in a Docker/CI image.
