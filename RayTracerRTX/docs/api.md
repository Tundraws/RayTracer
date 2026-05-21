# Internal API Notes

This document records the main internal interfaces used by the ray tracer.
It is intentionally focused on implementation contracts rather than coursework
prose.

## Application Entry

### `run_optix_app()`

Declared in `src/app/application.h`.

Starts the interactive RTX renderer. The application owns the window loop,
input processing, scene mutation, render calls, and image presentation.

The executable accepts optional scene input arguments:

```powershell
RayTracerRTX.exe --mesh RayTracerRTX/assets/meshes/demo.obj
RayTracerRTX.exe --scene RayTracerRTX/assets/scenes/demo_scene.json
```

`--mesh` loads one OBJ mesh into the default scene. `--scene` loads a JSON scene
config with `meshObjects`, `camera`, `light`, and mesh transform fields.
Invalid input prints a diagnostic message and falls back to the default scene.

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
| `mesh` | `MeshData` | Polygonal OBJ mesh loaded from `assets/meshes/demo.obj` or fallback mesh |
| `lightPosition` | `float3` | Point-light position used by hit programs |
| `selectedSphere` | `int` | Index used by interactive controls |

### Scene Functions

| Function | Responsibility |
|---|---|
| `makeDefaultScene()` | Creates the initial scene with spheres, materials and light |
| `clampScene(SceneState&)` | Keeps objects inside supported bounds |
| `moveSelectedSphere(SceneState&, float3)` | Moves selected sphere and applies bounds |
| `toggleSelectedMaterial(SceneState&)` | Switches selected sphere between diffuse and mirror material |
| `moveLight(SceneState&, float3)` | Moves the light and applies bounds |

## OBJ Mesh API

### Scene Config

Declared in `src/app/scene_config.h`.

`loadSceneConfigFile(...)` parses the documented JSON scene input. The config
supports:

- `mesh`: shorthand path for one OBJ mesh;
- `meshObjects`: array of mesh objects with `path`, `position`, `rotation`, and
  `scale`;
- `camera`: `position`, `yaw`, `pitch`, `fov`;
- `light`: `position`.

For this stage, configured mesh objects are transformed on the CPU and merged
into the single `SceneState::mesh` used by the current renderer.

### `loadObjMesh(...)`

Declared in `src/app/obj_loader.h`.

Loads a simple OBJ mesh description into `MeshData`. The loader supports:

- `v` positions;
- `vn` normals;
- triangular `f` faces;
- `mtllib` and `usemtl` material references;
- MTL `Kd`, `Ks`, `Ns`, `Ni`, and `d` material values;
- multiple named materials.

Material names containing `mirror` are mapped to `MaterialMirror`; all other
OBJ materials are mapped by name: `metal` to `MaterialMetal`, `glass` or
`dielectric` to `MaterialDielectric`, and the remaining names to
`MaterialDiffuse`. `Ns` is converted to a clamped roughness value, `Ni` stores
index of refraction, `d` stores alpha, and `Ks` stores specular color. The
loader is intentionally limited to the listed OBJ and MTL records.

### `MeshData`

Declared in `src/app/mesh.h`.

| Field | Type | Meaning |
|---|---|---|
| `vertices` | `std::vector<MeshVertex>` | Packed mesh vertices with position and normal |
| `triangles` | `std::vector<MeshTriangle>` | Triangle indices plus material index |
| `materials` | `std::vector<MeshMaterial>` | OBJ/MTL materials used by mesh triangles |

`hasValidMeshMaterialIndices(const MeshData&)` validates that every triangle
material index is inside the material array.

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
| `MaterialMetal` | Tinted reflective material with roughness-controlled reflection |
| `MaterialDielectric` | Simple glass/dielectric approximation with Fresnel and refraction |

### `SphereMaterial`

| Field | Type | Meaning |
|---|---|---|
| `color` | `float3` | Base color for diffuse material |
| `materialType` | `int` | Material type from `MaterialType` |
| `specularColor` | `float3` | Specular tint, parsed from OBJ MTL `Ks` for mesh materials |
| `roughness` | `float` | Reflection blur approximation, parsed from MTL `Ns` for mesh materials |
| `ior` | `float` | Index of refraction, parsed from MTL `Ni` for mesh materials |
| `alpha` | `float` | Opacity/transparency control, parsed from MTL `d` for mesh materials |

### `LaunchParams`

Uploaded to the GPU before each OptiX launch.

| Field | Responsibility |
|---|---|
| `image`, `imageWidth`, `imageHeight` | Output framebuffer |
| `handle` | Top-level OptiX traversable handle |
| `cameraPosition`, `cameraForward`, `cameraRight`, `cameraUp` | Camera ray generation data |
| `cameraScale`, `cameraAspect` | Projection parameters |
| `lightPosition` | Point-light input for hit programs |
| `materials`, `sphereCount` | Per-sphere material data |
| `meshVertices`, `meshVertexCount` | OBJ mesh vertex buffer |
| `meshTriangles`, `meshTriangleCount` | OBJ mesh triangle buffer with material indices |
| `meshMaterials`, `meshMaterialCount` | OBJ mesh material buffer |
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

## Output Processing

Final framebuffer colors are processed in the OptiX device program before
conversion to `uchar4`:

- HDR values are clamped to a bounded range;
- Reinhard tone mapping compresses bright values;
- gamma correction with gamma 2.2 converts linear color to display color.

The miss shader uses a procedural gradient environment with a horizon glow and
small sun highlight. Reflective materials sample this environment through their
existing reflection rays.

