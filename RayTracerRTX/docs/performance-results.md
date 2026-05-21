# Performance Results

Date: 2026-05-21

## Hardware and Environment

- GPU: NVIDIA GeForce RTX 3050 Ti Laptop GPU
- GPU memory: 4096 MiB
- NVIDIA driver: 595.71
- CPU: Intel64 Family 6 Model 154 Stepping 3, 12 logical processors
- OS: Microsoft Windows NT 10.0.22621.0
- Build: `Debug|x64`
- Renderer: NVIDIA OptiX + CUDA

## Method

Measurements were collected with the test executable benchmark mode:

```powershell
RayTracerRTX\tests\x64\Debug\RayTracerRTX.Tests.exe --benchmark
```

The benchmark initializes the OptiX renderer, renders one warm-up frame, then
renders a fixed number of frames for each resolution. Reported frame time is
measured on the host around `OptixRenderer::renderFrame`; reported GPU time is
measured with CUDA events inside the renderer.

The benchmark scene includes analytic primitives and the demo OBJ mesh loaded
from `assets/meshes/demo.obj` with multiple MTL materials. The material path
includes diffuse, mirror, metal and dielectric shader branches, procedural
environment lighting, Reinhard tone mapping, gamma correction, and optional
OBJ `map_Kd` diffuse texture sampling. Basic tangent-space normal mapping is
available for OBJ materials that provide a valid normal map and UVs. Mesh
objects are stored separately and placed in the IAS with per-object transforms.
Direct lighting uses a physically motivated GGX microfacet BRDF for diffuse,
mirror, and metal materials.

The benchmark is headless and does not include GLFW window presentation, HUD
drawing, or user input processing. Interactive FPS in the desktop app may differ
because it includes display presentation and VSync settings.

## Results

| Scenario | Resolution | FPS | Avg frame ms | Avg GPU ms |
|---|---:|---:|---:|---:|
| Low | 640x360 | 630.36 | 1.59 | 1.57 |
| HD | 1280x720 | 225.60 | 4.43 | 4.41 |
| Full HD | 1920x1080 | 103.30 | 9.68 | 9.64 |

## Interpretation

The renderer stays within real-time frame budgets for all tested resolutions
with the OBJ mesh scene enabled. Full HD averages about 103 FPS, so the current
scene remains above the 60 FPS target. The extra material, diffuse texture,
normal-map shading state, and per-object Triangle GAS/IAS layout remain within
the real-time budget: average GPU time is below 10 ms at 1920x1080, under the
16.67 ms frame budget for 60 FPS.

The close match between host frame time and GPU time indicates that the benchmark
is dominated by GPU rendering and synchronization rather than CPU-side scene
logic.
