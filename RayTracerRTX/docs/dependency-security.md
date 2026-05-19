# Dependency And Security Notes

This document supplements `static-analysis.md` with dependency information and
security limitations for the coursework project.

## Runtime And Build Dependencies

| Dependency | Role | How It Is Used |
|---|---|---|
| Visual Studio 2022 / MSVC v143 | Native Windows C++ build | Builds the GUI application and native tests |
| Windows SDK | Platform headers and libraries | Required by MSVC project files |
| CUDA Toolkit 13.1 | CUDA runtime, NVRTC, GPU memory and timing | Used by `OptixRenderer` and linked through Visual Studio properties |
| NVIDIA OptiX SDK 9.1 | Hardware ray tracing API | Used for RTX pipeline, acceleration structures and device programs |
| NVIDIA display driver | Runtime GPU driver | Required to launch OptiX on RTX hardware |
| GLFW | Window and input handling | Stored in the repository as a local dependency |
| OpenGL | Image presentation | Used to display the rendered framebuffer |
| Docker / Docker Compose | Reproducible checks | Runs documentation/structure checks and CPU-only tests |

## Dependency Check Approach

The project does not use a package manager lock file such as `package-lock.json`,
`requirements.txt`, `Cargo.lock`, or `go.sum`. Most dependencies are native SDKs
installed on the Windows host. Because of that, automated dependency scanning is
limited.

The practical dependency check is:

1. Record required SDK versions in `README.md`.
2. Keep CUDA and OptiX paths in Visual Studio project properties instead of C++
   source constants.
3. Avoid downloading dependencies during application startup.
4. Use Docker only for reproducible source, documentation and CPU-test checks.
5. Keep vendored GLFW files limited to the required include/library paths.

## Security Controls

| Control | Status | Evidence |
|---|---|---|
| Static analysis | Implemented | `docs/static-analysis.md` |
| Unit tests | Implemented | `tests/test_scene_camera.cpp` |
| GPU smoke test | Implemented | `tests/RayTracerRTX.Tests.vcxproj` |
| Docker reproducibility check | Implemented | `Dockerfile`, `docker-compose.yml`, `docs/docker-check.md` |
| Hard-coded local paths in C++ | Avoided | Paths are configured through project properties |
| Dependency scan | Documented limitation | This file |

## Remaining Manual Checks

- Confirm CUDA Toolkit and OptiX SDK versions before final defense.
- Confirm the installed NVIDIA driver supports the target OptiX version.
- Re-run native GPU smoke test after GPU driver or SDK updates.
- Re-run Docker CPU check before final GitHub submission.

