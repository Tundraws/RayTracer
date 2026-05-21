# Performance Results

Date: 2026-05-22

## Hardware and Environment

- GPU: NVIDIA GeForce RTX 3050 Ti Laptop GPU
- GPU memory: 4096 MiB
- NVIDIA driver: 595.71
- CPU: Intel64 Family 6 Model 154 Stepping 3, 12 logical processors
- OS: Microsoft Windows NT 10.0.22621.0
- Build: `Debug|x64`
- Renderer: NVIDIA OptiX + CUDA

## Method

Measurements were collected with the test executable benchmark mode. The
benchmark reports High quality across common resolutions and compares the
quality modes at 640x360:

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
Progressive path tracing accumulation is available through the PathTracing
quality mode. Optional OptiX denoising is enabled for that quality mode and is
much heavier than the direct real-time modes.
The benchmark scene still uses the OBJ demo mesh; glTF import is covered by
loader and scene-input tests rather than this FPS table.

The benchmark is headless and does not include GLFW window presentation, HUD
drawing, or user input processing. Interactive FPS in the desktop app may differ
because it includes display presentation and VSync settings.

## Results

| Scenario | Resolution | FPS | Avg frame ms | Avg GPU ms |
|---|---:|---:|---:|---:|
| Low resolution / High quality | 640x360 | 455.73 | 2.19 | 2.16 |
| HD resolution / High quality | 1280x720 | 230.50 | 4.34 | 4.31 |
| Full HD resolution / High quality | 1920x1080 | 113.82 | 8.79 | 8.73 |
| Low quality | 640x360 | 954.48 | 1.05 | 1.01 |
| Medium quality | 640x360 | 176.49 | 5.67 | 5.63 |
| High quality | 640x360 | 188.54 | 5.30 | 5.28 |
| PathTracing quality + denoiser | 640x360 | 13.44 | 74.42 | 74.33 |

## Interpretation

The renderer stays within real-time frame budgets for all tested resolutions
with the OBJ mesh scene enabled. Full HD High quality averages about 114 FPS, so the current
scene remains above the 60 FPS target. The extra material, diffuse texture,
normal-map shading state, and per-object Triangle GAS/IAS layout remain within
the real-time budget: average GPU time is below 10 ms at 1920x1080, under the
16.67 ms frame budget for 60 FPS.

Quality modes are intended for demonstration and performance comparison. Low
quality disables direct shadow rays and uses fewer samples/depth, so it is much
faster but visually flatter. Medium and High keep shadows enabled and increase
sample/depth budgets. PathTracing quality trades immediate stability for
progressive convergence and denoising; moving the camera, light, spheres,
materials, or mesh scene resets the accumulation buffer so stale samples are not
mixed with the new view.

The optional OptiX denoiser greatly reduces progressive noise but is not a
real-time default path in this Debug build. It is intentionally controlled by a
HUD flag and only applies to progressive accumulation, not to the direct
real-time mode.

The close match between host frame time and GPU time indicates that the benchmark
is dominated by GPU rendering and synchronization rather than CPU-side scene
logic.
