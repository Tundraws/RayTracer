# Coursework Completion Checklist (Variant 42)

## What is mandatory

- GPU-based real-time ray tracing implemented (`C++ + CUDA + OptiX`)
- Functional scene: geometry, materials, light, shadows, camera controls
- Scene description with analytic primitives, OBJ polygon meshes, and a limited glTF mesh subset
- Git-based development history with clear commits
- Tests (unit + at least one GPU smoke/integration check)
- Static analysis and quality report
- Documentation (`README`, architecture notes, API notes, test evidence)
- Dependency/security notes
- AI-assisted development notes
- CI workflow for reproducible CPU/documentation checks

## Current "ideal" test package

- Unit tests for scene logic:
  - default scene values
  - material toggling
  - movement and clamping bounds
  - invalid index behavior
- Unit tests for camera logic:
  - normalized and orthogonal basis vectors
  - aspect ratio fallback
  - FOV influence on camera scale
- Unit tests for OBJ mesh logic:
  - loader success case
  - missing file
  - invalid face rejection
  - missing normals fallback
  - multiple material mapping through MTL
  - extended MTL `Kd`/`Ks`/`Ns`/`Ni`/`d` parsing
  - metal and dielectric material-name mapping
  - roughness, IOR and alpha clamping
  - OBJ `vt` texture coordinates
  - MTL `map_Kd` diffuse texture loading and missing-texture fallback
  - MTL `bump`/`map_Bump`/`norm` normal map loading and safe fallback
  - tangent computation for textured triangles
  - minimal glTF `.gltf/.bin` mesh loading
  - glTF material factors
  - default scene mesh presence
  - mesh material index validation
- Negative tests and diagnostics:
  - missing OBJ and missing glTF files fail with readable errors
  - invalid OBJ faces fail without crashing
  - missing MTL and invalid MTL values fall back safely
  - missing, invalid, and unsupported textures fall back to material values
  - missing glTF buffers fail cleanly
  - invalid JSON, invalid material config, and invalid scene transforms fail cleanly
  - loader warnings are written to console and `RayTracerRTX.log`
- GPU smoke test:
  - renderer initialization
  - one-frame render
  - progressive accumulation render
  - accumulation reset on camera change
  - optional denoiser enabled/disabled path
  - rendering quality mode helpers
  - non-empty pixel buffer
  - valid GPU frame time

## Evidence to include in report

- Test run log (console output with PASS/FAIL lines)
- Screenshot of successful test run
- Performance table (FPS / Frame ms / GPU ms) for real-time and quality-mode scenarios
- Screenshot(s) of running renderer with HUD
- Static analysis summary (tool + key warnings + fixes)
- Architecture diagrams from `docs/architecture.md`
- Internal API notes from `docs/api.md`
- Dependency/security notes from `docs/dependency-security.md`
- AI-assisted development notes from `docs/ai-assisted-development.md`
- GitHub Actions workflow from `.github/workflows/coursework-check.yml`

## Current artifact map

| Requirement | Artifact |
|---|---|
| Git / semantic commits | Git history on `main` |
| CI/CD | `.github/workflows/coursework-check.yml` |
| Unit tests | `tests/test_scene_camera.cpp` |
| GPU smoke test | `tests/RayTracerRTX.Tests.vcxproj` |
| Demo OBJ mesh | `assets/meshes/demo.obj`, `assets/meshes/demo.mtl` |
| Demo textured OBJ mesh | `assets/meshes/textured_demo.obj`, `assets/meshes/textured_cube.obj`, `assets/meshes/textured_demo.mtl`, `assets/meshes/checker.ppm`, `assets/meshes/checker_normal.ppm` |
| Demo material showcase mesh | `assets/meshes/material_showcase.obj`, `assets/meshes/material_showcase.mtl` |
| Demo glTF mesh | `assets/meshes/minimal_gltf.gltf`, `assets/meshes/minimal_gltf.bin` |
| Demo scene configs | `assets/scenes/demo_scene.json`, `assets/scenes/textured_scene.json`, `assets/scenes/textured_cube_scene.json`, `assets/scenes/multi_mesh_scene.json`, `assets/scenes/material_showcase_scene.json`, `assets/scenes/path_tracing_demo_scene.json`, `assets/scenes/gltf_scene.json` |
| Docker / Compose | `Dockerfile`, `docker-compose.yml`, `docs/docker-check.md` |
| README | `README.md`, `RayTracerRTX/README.md` |
| Architecture diagrams | `docs/architecture.md` |
| API documentation | `docs/api.md` |
| SAST | `docs/static-analysis.md` |
| Dependency/security notes | `docs/dependency-security.md` |
| Performance analysis | `docs/performance-results.md` |
| AI tools | `docs/ai-assisted-development.md` |

## Remaining before final submission

1. Capture final application screenshots for the report.
2. Re-run native tests and Docker check before final GitHub submission.
3. Push final `main` branch to GitHub.
