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
- `scripts/verify_coursework.py`
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

## Local Verification Status

`docker compose config` completed successfully. Docker emitted warnings about
access to the local Docker client config file in the sandboxed session, but the
Compose configuration was still rendered correctly.

`docker compose build` could not be executed in the current session because
Docker Desktop's Linux engine was not running:

```text
failed to connect to the docker API at npipe:////./pipe/dockerDesktopLinuxEngine
```

After starting Docker Desktop, rerun the build/run commands above.
