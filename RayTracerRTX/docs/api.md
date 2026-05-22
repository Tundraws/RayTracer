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

`--mesh` loads one OBJ or glTF mesh into the default scene. `--scene` loads a
JSON scene config with `meshObjects`, `camera`, `light`, image tuning fields,
and mesh transform fields.
Invalid input prints a diagnostic message and falls back to the default scene.
The interactive app starts in `RenderModeRealtime`; pressing `P` toggles
progressive path tracing accumulation and resets samples when the camera, light,
scene, material, or mesh signature changes. Pressing `N` requests the optional
OptiX denoiser for progressive mode. If denoiser initialization or invocation
fails, the renderer keeps running without denoising.
The lightweight runtime controls also expose demo scene presets with `G`,
selected mesh object cycling with `B`, selected mesh material preset cycling
with `V`, and selected sphere material preset cycling with `M`.
Pressing `Q` cycles rendering quality modes: Low, Medium, High, and
PathTracing. Quality controls max reflection depth, direct shadow rays,
primary samples per pixel, progressive mode, and denoiser use.

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
| `meshObjects` | `std::vector<MeshObject>` | Polygonal OBJ mesh objects with asset reference, mesh data, and transform |
| `mesh` | `MeshData` | Compatibility combined mesh used by tests/docs and fallback paths |
| `lightPosition` | `float3` | Point-light position used by hit programs |
| `exposure` | `float` | Tone-mapping exposure multiplier, clamped to a safe range |
| `skyIntensity` | `float` | Environment-light intensity multiplier |
| `lightIntensity` | `float` | Direct light intensity multiplier |
| `selectedSphere` | `int` | Index used by interactive controls |
| `selectedMeshObject` | `int` | Mesh object index used by interactive controls |
| `selectedMeshMaterial` | `int` | Material index inside the selected mesh object |

### Scene Functions

| Function | Responsibility |
|---|---|
| `makeDefaultScene()` | Creates the initial scene with spheres, materials and light |
| `clampScene(SceneState&)` | Keeps objects inside supported bounds |
| `moveSelectedSphere(SceneState&, float3)` | Moves selected sphere and applies bounds |
| `toggleSelectedMaterial(SceneState&)` | Switches selected sphere between diffuse and mirror material |
| `cycleSelectedSphereMaterialPreset(SceneState&)` | Cycles selected sphere through diffuse, mirror, metal and dielectric presets |
| `selectNextMeshObject(SceneState&)` | Selects the next mesh object for HUD/material editing |
| `cycleSelectedMeshMaterialPreset(SceneState&)` | Cycles selected mesh material through diffuse, mirror, metal and dielectric presets |
| `moveLight(SceneState&, float3)` | Moves the light and applies bounds |

## OBJ Mesh API

### Scene Config

Declared in `src/app/scene_config.h`.

`loadSceneConfigFile(...)` parses the documented JSON scene input. The config
supports:

- `mesh`: shorthand path for one OBJ or glTF mesh;
- `meshObjects`: array of mesh objects with `path`, `position`, `rotation`, and
  `scale`;
- `camera`: `position`, `yaw`, `pitch`, `fov`;
- `light`: `position` and optional `intensity`;
- root-level or `render` object fields: `exposure`, `skyIntensity`,
  `lightIntensity`.

If image-tuning fields are missing, the default scene values are used. Numeric
values outside the supported range are clamped so old or experimental scene
files do not make the renderer unstable.

Configured mesh objects are stored in `SceneState::meshObjects`. Each object
keeps its source mesh data and transform, while `SceneState::mesh` remains as a
compatibility combined mesh for tests and legacy fallback code. The OptiX
renderer builds a Triangle GAS per mesh object and places each object in the
IAS with its transform matrix.

### `loadObjMesh(...)`

Declared in `src/app/obj_loader.h`.

Loads a simple OBJ mesh description into `MeshData`. The loader supports:

- `v` positions;
- `vn` normals;
- `vt` texture coordinates;
- triangular `f` faces;
- `mtllib` and `usemtl` material references;
- MTL `Kd`, `Ks`, `Ns`, `Ni`, and `d` material values;
- MTL `map_Kd` diffuse texture references for ASCII PPM (`P3`) images;
- MTL `bump`, `map_Bump`, and `norm` normal map references for ASCII PPM
  (`P3`) images;
- multiple named materials.

Material names containing `mirror` are mapped to `MaterialMirror`; all other
OBJ materials are mapped by name: `metal` to `MaterialMetal`, `glass` or
`dielectric` to `MaterialDielectric`, and the remaining names to
`MaterialDiffuse`. `Ns` is converted to a clamped roughness value, `Ni` stores
index of refraction, `d` stores alpha, and `Ks` stores specular color. The
loader is intentionally limited to the listed OBJ and MTL records.

If `map_Kd` points to a missing or unsupported texture file, the loader keeps
the texture path for diagnostics and falls back to the material `Kd` color.
If a normal map points to a missing or unsupported texture file, the loader
keeps the path for diagnostics and the shader falls back to the interpolated
geometric normal. Basic normal mapping requires OBJ `vt` coordinates and a
valid computed tangent basis; it perturbs shading normals only and does not
perform displacement mapping.

## glTF Mesh API

Declared in `src/app/gltf_loader.h`.

```cpp
GltfLoadResult loadGltfMesh(const std::filesystem::path& path);
```

The glTF loader is a small coursework importer for a limited glTF 2.0 subset.
It supports `.gltf` JSON files with external `.bin` buffers, `bufferViews`,
`accessors`, mesh primitives, triangle indices, `POSITION`, `NORMAL`,
`TEXCOORD_0`, and simple node `translation`/`scale`. Material import reads
`pbrMetallicRoughness.baseColorFactor`, `metallicFactor`, `roughnessFactor`,
and optionally `baseColorTexture` when the referenced image is a simple PPM
file compatible with the existing texture path.

Unsupported glTF features include `.glb`, embedded base64 buffers, animations,
skins, morph targets, cameras, lights, sparse accessors, sampler state,
normal/metallic/roughness texture maps, and full node rotation matrices. This
is an additional mesh input path; OBJ/MTL remains supported separately.

### `MeshData`

Declared in `src/app/mesh.h`.

| Field | Type | Meaning |
|---|---|---|
| `vertices` | `std::vector<MeshVertex>` | Packed mesh vertices with position, normal, UV, tangent and UV availability flag |
| `triangles` | `std::vector<MeshTriangle>` | Triangle indices plus material index |
| `materials` | `std::vector<MeshMaterial>` | OBJ/MTL materials used by mesh triangles |
| `textures` | `std::vector<MeshTexture>` | Loaded diffuse and normal texture pixels referenced by MTL texture maps |

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

### `RenderQuality`

| Value | Meaning |
|---|---|
| `RenderQualityLow` | One primary sample, one reflection level, direct shadow rays disabled |
| `RenderQualityMedium` | Two primary samples, medium reflection depth, direct shadow rays enabled |
| `RenderQualityHigh` | Four primary samples, deeper reflections, direct shadow rays enabled |
| `RenderQualityPathTracing` | Progressive mode with path-tracing accumulation and requested denoiser |

Direct lighting for diffuse, mirror, and metal materials uses a physically
motivated GGX/Trowbridge-Reitz microfacet BRDF. The shader evaluates the normal
distribution term `D`, Smith geometry term `G`, and Schlick Fresnel `F`, with
clamped roughness and dot products to avoid NaN/Inf values. `Kd` maps to
baseColor, `Ks` tints the specular approximation, `Ns` maps to roughness, and
material names containing `metal` use metallic shading. This is a physically
motivated material model with the limited material inputs listed above.

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
| `exposure`, `skyIntensity`, `lightIntensity` | Image and lighting tuning inputs |
| `materials`, `sphereCount` | Per-sphere material data |
| `meshVertices`, `meshVertexCount` | OBJ mesh vertex buffer |
| `meshTriangles`, `meshTriangleCount` | OBJ mesh triangle buffer with material indices |
| `meshMaterials`, `meshMaterialCount` | OBJ mesh material buffer |
| `meshTexturePixels`, `meshTexturePixelCount` | Packed diffuse and normal texture pixels for mesh materials |
| `accumulation` | Progressive float accumulation buffer |
| `renderMode` | `RenderModeRealtime` or `RenderModeProgressive` |
| `renderQuality` | Current `RenderQuality` mode |
| `shadowEnabled` | Enables or disables direct shadow rays |
| `samplesPerPixel` | Primary samples per pixel for the current quality mode |
| `accumulationSample` | Current progressive sample index |
| `maxDepth` | Recursive reflection depth limit |

### Render Modes

`RenderModeRealtime` is the default mode used by the benchmark and interactive
startup. It evaluates direct lighting, shadows, reflections and material
branches without accumulating previous frames.

`RenderModeProgressive` keeps a float accumulation buffer on the GPU. Each frame
adds jittered primary samples and stochastic diffuse/rough reflection bounces,
then tone maps the running average. The renderer resets accumulation when the
camera or scene signature changes, and the HUD displays the current sample
count. This is a coursework-scale progressive path tracing mode, not a full
offline path tracer.

When the denoiser flag is enabled, progressive mode sends the HDR accumulation
buffer through the OptiX denoiser and displays the denoised result. The current
integration uses the color buffer only; albedo and normal guide layers are not
generated yet. Real-time direct mode does not apply the denoiser.

## Renderer API

### `OptixRenderer`

Declared in `src/gpu/optix_renderer.h`.

| Method | Responsibility |
|---|---|
| `setRenderSize(int, int)` | Sets framebuffer dimensions |
| `setRenderMode(int)` | Selects real-time direct or progressive accumulation mode |
| `setDenoiserEnabled(bool)` | Requests optional OptiX denoising for progressive mode |
| `isDenoiserAvailable()` | Reports whether denoiser setup/invocation is currently available |
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

