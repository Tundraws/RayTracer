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
The scene also uses configurable exposure, sky intensity, and light intensity
values so the default image is less overexposed.
Direct lighting uses a physically motivated GGX microfacet BRDF for diffuse,
mirror, and metal materials. Glass uses a simplified dielectric model with
Schlick Fresnel, IOR-based refraction, and total internal reflection handling.
Rough reflective materials broaden reflection directions as roughness increases;
this is an approximation designed for stable real-time rendering, not a full
offline material model.
Scene config can also enable area-light soft shadows through `light.size` or
`light.radius`, and an optional ASCII PPM environment map. If no environment map
is provided, the renderer keeps using the procedural gradient sky.
Progressive path tracing accumulation is available through the PathTracing
quality mode. The progressive sampler uses per-pixel/per-sample random numbers,
cosine-weighted hemisphere sampling for matte bounces, rough reflection sampling
for mirror/metal materials, a depth limit, and Russian roulette termination
after the first few bounces. Optional OptiX denoising can be enabled for that
quality mode and is much heavier than the direct real-time modes.
The benchmark scene still uses the OBJ demo mesh; glTF import is covered by
loader and scene-input tests rather than this FPS table.

The benchmark is headless and does not include GLFW window presentation, HUD
drawing, or user input processing. Interactive FPS in the desktop app may differ
because it includes display presentation and VSync settings.

## Results

| Scenario | Resolution | FPS | Avg frame ms | Avg GPU ms |
|---|---:|---:|---:|---:|
| Low resolution / High quality | 640x360 | 634.32 | 1.58 | 1.55 |
| HD resolution / High quality | 1280x720 | 215.16 | 4.65 | 4.61 |
| Full HD resolution / High quality | 1920x1080 | 110.91 | 9.02 | 8.97 |
| Low quality | 640x360 | 327.92 | 3.05 | 3.02 |
| Medium quality | 640x360 | 173.47 | 5.76 | 5.72 |
| High quality | 640x360 | 184.83 | 5.41 | 5.37 |
| PathTracing quality | 640x360 | 78.67 | 12.71 | 12.65 |
| PathTracing quality + denoiser | 640x360 | 13.28 | 75.33 | 75.28 |

## Interpretation

The renderer stays within real-time frame budgets for all tested resolutions
with the OBJ mesh scene enabled. Full HD High quality averages about 111 FPS, so
the current scene remains above the 60 FPS target in the Debug build. The
extra material, diffuse texture, normal-map shading state, dielectric
refraction, rough reflection approximation, and per-object Triangle GAS/IAS
layout remain within the real-time budget: average GPU time is about 8.8 ms at
1920x1080, under the 16.67 ms frame budget for 60 FPS.

Quality modes are intended for demonstration and performance comparison. Low
quality disables direct shadow rays and uses fewer samples/depth, so it is much
faster but visually flatter. Medium and High keep shadows enabled and increase
sample/depth budgets. PathTracing quality trades immediate stability for
progressive convergence and denoising; moving the camera, light, spheres,
materials, exposure, quality, denoiser state, or mesh scene resets the
accumulation buffer so stale samples are not mixed with the new view. Without
the denoiser, PathTracing quality remains interactive at 640x360 in the
benchmark but is visibly noisy until samples accumulate.

The optional OptiX denoiser greatly reduces progressive noise but is not a
real-time default path in this Debug build. It is intentionally controlled by a
HUD flag and only applies to progressive accumulation, not to the direct
real-time mode.

The close match between host frame time and GPU time indicates that the benchmark
is dominated by GPU rendering and synchronization rather than CPU-side scene
logic.
