# Docker / Docker Compose Check

Date: 2026-05-19

## Purpose

The Docker setup provides a reproducible coursework verification environment for:

- repository structure checks;
- documentation marker checks;
- CPU-only unit tests for scene, camera, and material logic.

The RTX GUI application itself is not launched in Docker because it depends on a
Windows desktop session, GLFW window creation, NVIDIA OptiX, CUDA, and RTX GPU
driver integration. GPU execution is verified separately by the native
`RayTracerRTX.Tests.exe` GPU smoke test.

## Files

- `Dockerfile`
- `docker-compose.yml`
- `RayTracerRTX/tests/stubs/cuda_runtime.h`
- `RayTracerRTX/tests/stubs/optix.h`

## Commands

Validate Compose configuration:

```powershell
docker compose config
```

Build and run the reproducible check:

```powershell
docker compose build
docker compose run --rm coursework-check
```

If Docker reports `short read` or `unexpected EOF` while reading an image layer,
the base image/cache is corrupted or the image download was interrupted. Clean
the builder cache and rebuild:

```powershell
docker builder prune -f
docker compose build --no-cache
```

The Dockerfile uses `gcc:13-bookworm` and does not run `apt-get`. This avoids
failures when Debian package mirrors are unavailable from the Docker network.
The image is larger than a minimal runtime image, but it already contains `g++`,
so the build does not depend on package-manager access.
Structure and documentation checks are performed with POSIX shell commands, then
the CPU-only C++ tests are compiled with the compiler already present in the
base image.

## Local Verification Status

`docker compose config` completed successfully.

`docker compose build --no-cache` completed successfully and produced:

```text
raytracerrtx-coursework-check:latest
```

`docker compose run --rm coursework-check` completed successfully:

```text
All tests passed. Tests: 11, skipped: 1, checks: 31
```

The skipped test is the GPU smoke test. This is expected in Docker because the
container intentionally runs the CPU-only test path. RTX/OptiX execution is
verified by the native Windows test executable.
