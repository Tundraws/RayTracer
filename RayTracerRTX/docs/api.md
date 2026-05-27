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
RayTracerRTX.exe --scene RayTracerRTX/assets/scenes/clean_floor_scene.json
```

`--mesh` loads one OBJ or glTF mesh into the default scene. `--scene` loads a
JSON scene config with `spheres`, `meshObjects`, `camera`, `light`, image tuning
fields, and transform fields.
Invalid input prints a diagnostic message and falls back to the default scene.
Warnings and errors are also appended to `RayTracerRTX.log` in the current
working directory through the lightweight logger in `src/app/logger.*`.
The interactive app starts in `RenderModeRealtime`; pressing `P` toggles
progressive path tracing accumulation and resets samples when the camera, light,
scene, material, or mesh signature changes. Pressing `N` requests the optional
OptiX denoiser for progressive mode. If denoiser initialization or invocation
fails, the renderer keeps running without denoising.
The progressive path currently uses a small stochastic sampler: per-pixel RNG,
cosine-weighted matte bounces, rough reflection bounces for mirror/metal
materials, a max-depth limit, and Russian roulette termination. It is still a
coursework quality mode, not a full offline path tracer.
The lightweight runtime controls also expose a short public preset list with `G`
(clean floor scene and floating sphere scene),
selected mesh object cycling with `B`, selected mesh material preset cycling
with `V`, and selected sphere material preset cycling with `M`.
Pressing `C` restores the current preset camera and light without changing the
active preset.
Pressing `F5` reloads the current JSON-backed scene preset from disk. If reload
fails, the previous in-memory scene remains active and the panel shows the
error. Successful reloads reset progressive accumulation and request renderer
GAS/IAS rebuild.
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
| `lightPosition` | `float3` | Light position used by hit programs |
| `exposure` | `float` | Tone-mapping exposure multiplier, clamped to a safe range |
| `skyIntensity` | `float` | Environment-light intensity multiplier |
| `lightIntensity` | `float` | Direct light intensity multiplier |
| `areaLightRadius` | `float` | Area-light radius; zero keeps point-light behavior |
| `environmentIntensity` | `float` | Extra multiplier for gradient sky or environment map |
| `floorFadeDistance` | `float` | Distance from camera where editable floor panels start blending into the horizon color |
| `floorFadeSoftness` | `float` | Smooth transition length for distant floor blending |
| `environmentMap` | `MeshTexture` | Optional PPM lat-long environment map |
| `selectedSphere` | `int` | Index used by interactive controls |
| `selectedMeshObject` | `int` | Mesh object index used by interactive controls |
| `selectedMeshMaterial` | `int` | Material index inside the selected mesh object |

### Scene Functions

| Function | Responsibility |
|---|---|
| `makeDefaultScene()` | Creates the initial scene with spheres, materials and light |
| `clampScene(SceneState&)` | Keeps objects inside supported bounds |
| `addSphere(SceneState&)` | Appends a sphere, copies the selected sphere material and selects the new sphere |
| `removeSelectedSphere(SceneState&)` | Removes the selected sphere; empty sphere lists are valid |
| `setSelectedSphereRadius(SceneState&, float)` | Changes selected sphere radius with clamping |
| `setSelectedSphereColor(SceneState&, float3)` | Changes selected sphere color with clamping |
| `SceneEditor::setSelectedSphereMaterialProperties(...)` | Updates selected sphere material color, roughness, IOR and alpha through dirty flags |
| `moveSelectedSphere(SceneState&, float3)` | Moves selected sphere and applies bounds |
| `toggleSelectedMaterial(SceneState&)` | Switches selected sphere between diffuse and mirror material |
| `cycleSelectedSphereMaterialPreset(SceneState&)` | Cycles selected sphere through diffuse, mirror, metal and dielectric presets |
| `selectNextMeshObject(SceneState&)` | Selects the next mesh object for HUD/material editing |
| `cycleSelectedMeshMaterialPreset(SceneState&)` | Cycles selected mesh material through diffuse, mirror, metal and dielectric presets |
| `SceneEditor::setSelectedMeshMaterialProperties(...)` | Updates selected mesh material color, roughness, IOR, alpha and texture toggle through dirty flags |
| `SceneEditor::setSelectedMeshBaseColorTexture(...)` | Loads a PNG/JPG/PPM base-color texture and assigns it to the selected mesh object |
| `addBuiltInMeshPrimitive(SceneState&, int)` | Adds a built-in mesh primitive: cube, pyramid, or finite plane/panel |
| `removeSelectedMeshObject(SceneState&)` | Removes selected mesh object; empty mesh object lists are valid |
| `removeAllSpheres(SceneState&)` | Removes all analytic spheres and their materials |
| `removeAllMeshObjects(SceneState&)` | Removes all mesh objects and clears the compatibility mesh |
| `clearSceneObjects(SceneState&)` | Removes all visible objects while keeping camera/light settings |
| `restoreDefaultSceneObjects(SceneState&)` | Restores the default editable objects |
| `applySceneEnvironmentMode(SceneState&, int)` | Creates editable environment panels: open floor, room panels, or empty environment |
| `moveLight(SceneState&, float3)` | Moves the light and applies bounds |
| `setSceneExposure(SceneState&, float)` | Sets tone-mapping exposure with clamping |
| `setSceneSkyIntensity(SceneState&, float)` | Sets environment-light intensity with clamping |
| `setSceneLightIntensity(SceneState&, float)` | Sets direct-light intensity with clamping |
| `setSceneFloorFadeDistance(SceneState&, float)` | Sets distant floor fade start distance with clamping |
| `setSceneFloorFadeSoftness(SceneState&, float)` | Sets distant floor fade smoothness with clamping |
| `adjustSceneExposure(SceneState&, float)` | Changes exposure from keyboard/HUD controls |
| `adjustSceneSkyIntensity(SceneState&, float)` | Changes sky intensity from keyboard/HUD controls |
| `adjustSceneLightIntensity(SceneState&, float)` | Changes light intensity from keyboard/HUD controls |

### Built-In Mesh Primitives

Declared in `src/app/mesh.h`.

| Function | Responsibility |
|---|---|
| `createCubeMesh()` | Builds a cube as triangle `MeshData` with per-face normals |
| `createPyramidMesh()` | Builds a pyramid as triangle `MeshData` |
| `createPlaneMesh()` | Builds a finite rectangular panel from two triangles |
| `getMeshTextureMetadata(...)` | Returns safe display metadata for base color, normal, roughness, or metallic texture slots |

The built-in cube, pyramid and plane/panel are not separate GPU primitive
types. They use the same `MeshData` path as imported OBJ/glTF geometry. The
plane is intentionally finite, so it can represent floors, walls, ceilings and
other editable rectangular surfaces.

The visible floor in the default editor scene is also a finite mesh panel. The
shader can fade distant floor hits into the horizon color through
`floorFadeDistance` and `floorFadeSoftness`; this makes a large finite panel
look less abrupt without adding an infinite-plane primitive. Very distant floor
hits skip expensive secondary floor reflections and shadow visibility tracing.
The older renderer-side service plane is hidden by default and is kept only as a
compatibility fallback.

## Interactive Material Editor

The Dear ImGui material editor is organized into Russian UI sections:

- `Поверхность`: base color/tint for matte, metal and glass-like materials;
- `Отражение`: roughness, mirror blur and metal reflection controls;
- `Прозрачность/стекло`: transparency, IOR and glass haze controls;
- `Текстуры`: base color texture assignment plus diagnostics for normal map,
  roughness map and metallic map.

The panel can assign a PNG, JPG/JPEG, or PPM base-color texture to the selected
mesh object. Normal, roughness, and metallic texture paths still come from
OBJ/MTL, glTF/GLB, or JSON scene input; for these slots the panel reports
whether the texture is missing, unresolved, or loaded with dimensions/channel
count. Material edits go through `SceneEditor` methods so dirty flags can reset
accumulation and request renderer refreshes consistently.

## OBJ Mesh API

### Scene Config

Declared in `src/app/scene_config.h`.

`loadSceneConfigFile(...)` parses the documented JSON scene input. The config
supports:

- `mesh`: shorthand path for one OBJ or glTF mesh;
- `spheres`: explicit analytic sphere objects with optional `name`,
  `position`, `radius`, and optional inline/named `material`;
- `meshObjects`: array of mesh objects with either `path` for OBJ/glTF/GLB or
  `primitive` for built-in `cube`, `pyramid`, or `plane`, plus `position`,
  `rotation`, `scale`, optional `name`, and optional `material`;
- `camera`: `position`, `yaw`, `pitch`, `fov`;
- `light`: `position`, optional `intensity`, and optional `size`/`radius` for
  area-light soft shadows;
- `environment`: optional object with `type`, `path`, and `intensity`; `path`
  currently supports small ASCII PPM (`P3`) lat-long maps, with fallback to
  gradient sky when the file is missing or invalid; gradient sky also accepts
  `horizonColor`, `zenithColor`, and `gradientBlend`;
- root-level or `render` object fields: `exposure`, `skyIntensity`,
  `lightIntensity`, `skyHorizonColor`, `skyZenithColor`, and
  `skyGradientBlend`, `floorFadeDistance`, and `floorFadeSoftness`;
- `materials`: named material inputs with `name`, `type`, `baseColor`,
  `roughness`, `metallic`, `specularColor`, `ior`, `alpha`, `texture`/`map_Kd`,
  and `normalMap`;
- `sphereMaterials`: legacy material names or inline material objects assigned
  to spheres by index;
- mesh object `material` or `materialOverride`: material name or inline material
  object applied to all materials of that mesh object.

If image-tuning fields are missing, the default scene values are used. Numeric
values outside the supported range are clamped so old or experimental scene
files do not make the renderer unstable. `saveSceneToConfigFile(...)` writes the
current camera, light, spheres, built-in primitives, loaded mesh objects, and
basic material parameters back to JSON; the ImGui button opens a native save
dialog and suggests a timestamped name such as `Scene_270526_1755.json`.

JSON material `type` accepts `matte`, `mirror`, `metal`, and `glass`. The loader
also accepts `diffuse` as an alias for `matte` and `dielectric` as an alias for
`glass`. Unknown material types fall back to matte material and add a build
warning instead of failing the scene. The `metallic` value is clamped to `[0, 1]`;
values at or above `0.5` make a matte JSON material use the metal shader path.
Texture paths from JSON are stored on mesh materials for diagnostics and future
upload work; when they are not already loaded by OBJ/MTL, rendering falls back
to the material color.

The runtime HUD exposes the same image-tuning values through debounced hotkeys:
`4/5` for exposure, `6/7` for sky intensity, and `8/9` for direct light
intensity. These controls update `SceneState`, reset progressive accumulation,
and reuse the same clamping helpers as JSON parsing.

The interactive app also includes a lightweight Dear ImGui panel rendered on top
of the existing GLFW/OpenGL window. The panel is split into tabs: Scene,
Editor, Materials, Light, Quality, and Diagnostics. These tabs call the same
scene-state functions as the keyboard controls: scene preset selection,
camera/light reset, mesh selection, sphere selection, sphere add/remove, sphere
position/radius/color editing, built-in mesh primitive creation, selected mesh
removal, selected mesh position/rotation/scale editing,
explicit material type selection, texture enable/disable for textured mesh
materials, exposure, sky intensity, light intensity, area-light size,
environment intensity, material-specific roughness/transparency/IOR tuning,
quality mode, progressive mode and denoiser toggle. It also provides a Windows
file dialog button for adding an additional `.obj`, `.gltf`, or `.glb` model to
the current scene, a JSON scene load button that adds the selected `.json` file
as a runtime preset, plus a compact JSON reload button for JSON-backed presets.
Loaded models are placed on the available floor/support panel when possible and
shifted to a nearby free spot to avoid overlapping existing editable objects.
The compact left overlay exposes quick controls for the right panel visibility,
the short FPS/GPU summary, and saving the current framebuffer as a PNG, JPG, or
BMP snapshot through the native Windows save dialog. Snapshot file names are
timestamped by default, for example `RayTracerRTXPhoto_270526_1755.png`.
UI widgets are not unit-tested directly; the underlying state helpers are
covered by tests.

Scene config construction uses `AssetCache` as a small asset manager layer:
mesh assets are cached by normalized path, and standalone image texture loads
can be cached by path and texture type. The cache keeps repeated mesh references
from re-reading the same OBJ/glTF/GLB asset during scene construction.
For external OBJ/glTF/GLB mesh objects, the scene keeps a copy of the source
materials loaded from the asset. Saving a scene omits a mesh material override
while those source materials are unchanged, so reopening the JSON reloads the
original per-material colors/textures from the model file instead of replacing
them with a single fallback color.

The application keeps auxiliary JSON assets for tests and examples, but the
runtime preset selector intentionally exposes only a small demonstration set so
user-loaded OBJ/glTF/GLB models remain the main way to add extra content.

Preset helpers `applyScenePresetByIndex(...)` and
`resetSceneViewFromPreset(...)` keep runtime scene switching and reset behavior
testable without depending on GLFW input.

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
- `f` faces with 3+ vertices; faces with 4+ vertices are converted to triangles by simple fan triangulation;
- `mtllib` and `usemtl` material references;
- MTL `Kd`, `Ks`, `Ns`, `Ni`, and `d` material values;
- MTL `map_Kd` diffuse texture references for PPM (`P3`), PNG, and JPG/JPEG images;
- MTL `bump`, `map_Bump`, and `norm` normal map references for PPM (`P3`),
  PNG, and JPG/JPEG images;
- MTL `map_Pr`/`map_roughness` roughness maps and `map_Pm`/`map_metallic`
  metallic maps for the same image formats;
- multiple named materials.

Material names containing `mirror` are mapped to `MaterialMirror`; all other
OBJ materials are mapped by name: `metal` to `MaterialMetal`, `glass` or
`dielectric` to `MaterialDielectric`, and the remaining names to
`MaterialDiffuse`. `Ns` is converted to a clamped roughness value, `Ni` stores
index of refraction, `d` stores alpha, and `Ks` stores specular color. The
loader is intentionally limited to the listed OBJ and MTL records.

Fan triangulation is intentionally simple: it is suitable for common convex OBJ
polygons, but complex concave polygons can triangulate inaccurately. The
renderer still receives triangle `MeshData`; the GPU pipeline is unchanged.

If a texture map points to a missing or unsupported file, the loader keeps the
path for diagnostics and falls back to the numeric material values. Basic normal
mapping requires OBJ `vt` coordinates and a valid computed tangent basis; it
perturbs shading normals only and does not perform displacement mapping.

## Diagnostics And Fallbacks

Asset loading uses graceful fallbacks where that is safe:

- missing MTL files keep generated materials and log a warning;
- invalid MTL numeric values keep previous material defaults and log a warning;
- missing, invalid, or unsupported texture files fall back to material values;
- missing normal maps fall back to interpolated surface normals;
- missing or invalid environment maps fall back to the gradient sky;
- invalid OBJ faces, missing OBJ/glTF files, missing glTF buffers, invalid JSON,
  and invalid scene transforms fail cleanly with an error string instead of an
  application crash.

The interactive application writes the same diagnostics to the console and to
`RayTracerRTX.log`. Unit tests cover the negative paths at loader/state level;
GPU smoke tests still verify that valid scenes render a non-zero framebuffer.

## glTF Mesh API

Declared in `src/app/gltf_loader.h`.

```cpp
GltfLoadResult loadGltfMesh(const std::filesystem::path& path);
```

The glTF loader is a small coursework importer for a limited glTF 2.0 subset.
It supports `.gltf` JSON files with external `.bin` buffers and `.glb` files
with JSON/BIN chunks. Supported geometry fields include `bufferViews`,
`accessors`, mesh primitives, triangle indices, `POSITION`, `NORMAL`, and
`TEXCOORD_0`. Node import supports hierarchy traversal plus `translation`,
quaternion `rotation`, and `scale` composition.

Material import reads `pbrMetallicRoughness.baseColorFactor`,
`metallicFactor`, `roughnessFactor`, `baseColorTexture`, `normalTexture`, and
`metallicRoughnessTexture` when the referenced image is PPM, PNG, or JPG/JPEG.
The packed glTF metallic/roughness texture is stored as one image reference and
sampled as roughness from the green channel and metallic from the blue channel.

Unsupported glTF features include embedded base64 buffers, animations, skins,
morph targets, cameras, lights, sparse accessors, sampler state, material
extensions, and node `matrix` transforms. Unsupported or malformed files return
a clean load failure with an error string instead of replacing OBJ/MTL support.

### `MeshData`

Declared in `src/app/mesh.h`.

| Field | Type | Meaning |
|---|---|---|
| `vertices` | `std::vector<MeshVertex>` | Packed mesh vertices with position, normal, UV, tangent and UV availability flag |
| `triangles` | `std::vector<MeshTriangle>` | Triangle indices plus material index |
| `materials` | `std::vector<MeshMaterial>` | OBJ/MTL materials used by mesh triangles |
| `textures` | `std::vector<MeshTexture>` | Loaded color, normal, roughness, and metallic texture pixels referenced by material maps |

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
| `MaterialDielectric` | Simplified glass/dielectric model with Schlick Fresnel, IOR refraction and total internal reflection handling |

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
clamped roughness and dot products to avoid NaN/Inf values. Glass materials use
Schlick Fresnel from `ior`, trace refraction rays, and fall back to reflection
when total internal reflection occurs. Rough mirror and metal reflections use a
deterministic broadened reflection direction so roughness has a visible effect
in real-time mode. `Kd` maps to baseColor, `Ks` tints the specular
approximation, `Ns` maps to roughness, and material names containing `metal`
use metallic shading. This is a physically motivated material model with the
limited material inputs listed above, not a complete physically accurate
renderer.

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
| `lightPosition`, `areaLightRadius` | Direct-light input for hit programs |
| `exposure`, `skyIntensity`, `lightIntensity`, `environmentIntensity` | Image and lighting tuning inputs |
| `materials`, `sphereCount` | Per-sphere material data |
| `meshVertices`, `meshVertexCount` | OBJ mesh vertex buffer |
| `meshTriangles`, `meshTriangleCount` | OBJ mesh triangle buffer with material indices |
| `meshMaterials`, `meshMaterialCount` | OBJ mesh material buffer |
| `meshTexturePixels`, `meshTexturePixelCount` | Packed diffuse and normal texture pixels for mesh materials |
| `environmentPixels`, `environmentWidth`, `environmentHeight` | Optional PPM environment map buffer |
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

The miss shader uses either a procedural gradient environment with a horizon
glow and small sun highlight, or an optional PPM lat-long environment map loaded
from scene config. Reflective materials sample the same environment through
their existing reflection rays. Area-light mode uses multiple deterministic
shadow samples in Medium/High/PathTracing quality and falls back to one sample
in Low quality or when the radius is zero.

