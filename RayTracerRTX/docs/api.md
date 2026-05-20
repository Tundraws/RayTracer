# Internal API Notes

This document records the main internal interfaces used by the ray tracer.
It is intentionally focused on implementation contracts rather than coursework
prose.

## Application Entry

### `run_optix_app()`

Declared in `src/app/application.h`.

Starts the interactive RTX renderer. The application owns the window loop,
input processing, scene mutation, render calls, and image presentation.

## Scene API

### `SphereGeometry`

Declared in `src/app/scene.h`.

| Field | Type | Meaning |
|---|---|---|
| `center` | `float3` | Sphere center in world coordinates |
| `radius` | `float` | Sphere radius |

### `SceneState`

Declared in `src/app/scene.h`.

| Field | Type | Meaning |
|---|---|---|
| `spheres` | `std::vector<SphereGeometry>` | Scene geometry controlled by the CPU |
| `materials` | `std::vector<SphereMaterial>` | Per-sphere material data uploaded to GPU |
| `lightPosition` | `float3` | Point-light position used by hit programs |
| `lightType` | `int` | `LightPoint` or `LightArea` |
| `lightRadius` | `float` | Area-light radius used for soft shadows |
| `selectedSphere` | `int` | Index used by interactive controls |

### Scene Functions

| Function | Responsibility |
|---|---|
| `makeDefaultScene()` | Creates the initial scene with spheres, materials and light |
| `clampScene(SceneState&)` | Keeps objects inside supported bounds |
| `moveSelectedSphere(SceneState&, float3)` | Moves selected sphere and applies bounds |
| `toggleSelectedMaterial(SceneState&)` | Switches selected sphere between diffuse and mirror material |
| `moveLight(SceneState&, float3)` | Moves the light and applies bounds |
| `toggleLightType(SceneState&)` | Switches between point and area light |
| `changeLightRadius(SceneState&, float)` | Changes and clamps area-light radius |

## Camera API

### `CameraState`

Declared in `src/app/camera.h`.

| Field | Type | Meaning |
|---|---|---|
| `position` | `float3` | Camera origin in world coordinates |
| `yaw` | `float` | Horizontal rotation in degrees |
| `pitch` | `float` | Vertical rotation in degrees |
| `fov` | `float` | Vertical field of view in degrees |

### `updateCameraBasis(...)`

Computes the camera basis used by `__raygen__rg`:

- `forward`: normalized viewing direction;
- `right`: normalized horizontal basis vector;
- `up`: normalized vertical basis vector;
- `scale`: ray-plane scale derived from FOV;
- `aspect`: viewport aspect ratio with fallback protection.

## Shared Host/Device API

### `MaterialType`

Declared in `src/common/rtx_shared.h`.

| Value | Meaning |
|---|---|
| `MaterialDiffuse` | Local diffuse shading with shadow and specular component |
| `MaterialMirror` | Recursive mirror reflection with depth limit |

### `LightType`

| Value | Meaning |
|---|---|
| `LightPoint` | Single shadow ray to a point source, hard shadows |
| `LightArea` | Multiple shadow rays to a square area source, soft shadows |

### `SphereMaterial`

| Field | Type | Meaning |
|---|---|---|
| `color` | `float3` | Base color for diffuse material |
| `materialType` | `int` | Material type from `MaterialType` |

### `LaunchParams`

Uploaded to the GPU before each OptiX launch.

| Field | Responsibility |
|---|---|
| `image`, `imageWidth`, `imageHeight` | Output framebuffer |
| `handle` | Top-level OptiX traversable handle |
| `cameraPosition`, `cameraForward`, `cameraRight`, `cameraUp` | Camera ray generation data |
| `cameraScale`, `cameraAspect` | Projection parameters |
| `lightPosition`, `lightType`, `lightRadius` | Lighting input and point/area mode for hit programs |
| `materials`, `sphereCount` | Per-sphere material data |
| `maxDepth` | Recursive reflection depth limit |

## Renderer API

### `OptixRenderer`

Declared in `src/gpu/optix_renderer.h`.

| Method | Responsibility |
|---|---|
| `setRenderSize(int, int)` | Sets framebuffer dimensions |
| `initialize()` | Creates CUDA stream, OptiX context, modules, program groups, pipeline, SBT and acceleration structures |
| `renderFrame(const SceneState&, const CameraState&, std::vector<uchar4>&, float*)` | Uploads frame data, launches OptiX, copies pixels to host, optionally returns GPU time |
| `destroy()` | Releases CUDA and OptiX resources |

### Important Private Steps

| Method | Responsibility |
|---|---|
| `createContext()` | Initializes CUDA/OptiX device context |
| `createScene()` | Allocates geometry buffers |
| `createModule()` | Compiles device program source with NVRTC |
| `createProgramGroups()` | Creates raygen, miss and hitgroup programs |
| `createPipeline()` | Links OptiX pipeline and computes stack sizes |
| `createSbt()` | Builds shader binding table records |
| `rebuildAccelerationStructure()` | Builds GAS/IAS acceleration structures |

