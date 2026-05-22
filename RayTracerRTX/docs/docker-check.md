# Docker / Docker Compose Check

Date: 2026-05-19

## Purpose

The Docker setup provides a reproducible coursework verification environment for:

- repository structure checks;
- documentation marker checks;
- demo OBJ mesh and MTL asset checks;
- demo textured OBJ, `map_Kd` PPM, and normal-map PPM asset checks;
- demo PPM environment-map asset checks;
- demo material showcase OBJ/MTL asset checks;
- demo glTF `.gltf/.bin` asset checks;
- demo JSON scene config checks;
- OBJ and glTF loader source-file checks;
- CPU-only unit tests for scene, camera, material, extended MTL, OBJ loader, and glTF loader logic.

The RTX GUI application itself is not launched in Docker because it depends on a
Windows desktop session, GLFW window creation, NVIDIA OptiX, CUDA, and RTX GPU
driver integration. GPU execution is verified separately by the native
`RayTracerRTX.Tests.exe` GPU smoke test.

## Files

- `Dockerfile`
- `docker-compose.yml`
- `RayTracerRTX/assets/meshes/demo.obj`
- `RayTracerRTX/assets/meshes/demo.mtl`
- `RayTracerRTX/assets/meshes/textured_demo.obj`
- `RayTracerRTX/assets/meshes/textured_cube.obj`
- `RayTracerRTX/assets/meshes/textured_demo.mtl`
- `RayTracerRTX/assets/meshes/material_showcase.obj`
- `RayTracerRTX/assets/meshes/material_showcase.mtl`
- `RayTracerRTX/assets/meshes/demo_shapes.mtl`
- `RayTracerRTX/assets/meshes/low_poly_crystal.obj`
- `RayTracerRTX/assets/meshes/simple_column.obj`
- `RayTracerRTX/assets/meshes/block_arch.obj`
- `RayTracerRTX/assets/meshes/minimal_gltf.gltf`
- `RayTracerRTX/assets/meshes/minimal_gltf.bin`
- `RayTracerRTX/assets/meshes/gltf_diamond.gltf`
- `RayTracerRTX/assets/meshes/gltf_diamond.bin`
- `RayTracerRTX/assets/meshes/checker.ppm`
- `RayTracerRTX/assets/meshes/checker_normal.ppm`
- `RayTracerRTX/assets/meshes/studio_env.ppm`
- `RayTracerRTX/assets/scenes/demo_scene.json`
- `RayTracerRTX/assets/scenes/textured_scene.json`
- `RayTracerRTX/assets/scenes/textured_cube_scene.json`
- `RayTracerRTX/assets/scenes/multi_mesh_scene.json`
- `RayTracerRTX/assets/scenes/obj_shapes_scene.json`
- `RayTracerRTX/assets/scenes/material_showcase_scene.json`
- `RayTracerRTX/assets/scenes/path_tracing_demo_scene.json`
- `RayTracerRTX/assets/scenes/gltf_scene.json`
- `RayTracerRTX/assets/scenes/gltf_diamond_scene.json`
- `RayTracerRTX/src/app/obj_loader.*`
- `RayTracerRTX/src/app/gltf_loader.*`
- `RayTracerRTX/src/app/scene_config.*`
- `RayTracerRTX/src/app/mesh.*`
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
Structure, documentation, OBJ mesh, and glTF mesh asset checks are performed with POSIX
shell commands, then the CPU-only C++ tests are compiled with the compiler
already present in the base image.

## Local Verification Status

`docker compose config` completed successfully.

`docker compose build --no-cache` completed successfully and produced:

```text
raytracerrtx-coursework-check:latest
```

`docker compose run --rm coursework-check` should complete with all CPU tests
passing:

```text
All tests passed. Tests: 81, skipped: 4
```

The skipped tests are the GPU smoke tests. This is expected in Docker because
the container intentionally runs the CPU-only test path. RTX/OptiX execution is
verified by the native Windows test executable.
