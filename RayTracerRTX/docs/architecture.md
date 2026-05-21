# Architecture

This document describes the implementation architecture of the real-time RTX
ray tracer. It is a technical project artifact and can be used as source
material for the coursework design section.

## Module View

```mermaid
flowchart LR
    Main["maingpu.cpp"] --> App["Application loop"]
    Main --> Args["--mesh / --scene args"]
    Args --> SceneConfig["SceneConfig JSON"]
    App --> Input["Input handling"]
    App --> Camera["CameraState / updateCameraBasis"]
    SceneConfig --> MeshObject["MeshObject + transform"]
    MeshObject --> ObjLoader
    ObjLoader["ObjLoader"] --> MeshData["MeshData"]
    ObjLoader --> Textures["map_Kd / normal PPM textures"]
    ObjLoader --> Tangents["Tangent basis"]
    Textures --> MeshData
    Tangents --> MeshData
    MeshData --> MeshObject
    MeshObject --> Scene
    App --> Scene["SceneState"]
    Scene --> Materials["SphereMaterial"]
    Scene --> MeshMaterials["Mesh materials"]
    App --> Renderer["OptixRenderer"]
    App --> Mode["Render mode toggle"]
    Camera --> Renderer
    Scene --> Renderer
    Mode --> Renderer
    Renderer --> TriangleGAS["Triangle GAS per mesh object"]
    Renderer --> IAS["IAS with mesh transforms"]
    Renderer --> Shared["LaunchParams / rtx_shared.h"]
    Renderer --> Device["OptiX device programs"]
    TriangleGAS --> IAS
    IAS --> Device
    Device --> Accum["Progressive accumulation buffer"]
    Device --> Framebuffer["CUDA framebuffer"]
    Device --> Post["Tone mapping + gamma"]
    Accum --> Post
    Post --> Framebuffer
    Framebuffer --> App
```

## Runtime Pipeline

```mermaid
sequenceDiagram
    participant User
    participant Args as CLI / JSON scene config
    participant App as GLFW application
    participant ObjLoader as ObjLoader
    participant Scene as SceneState
    participant Renderer as OptixRenderer
    participant CUDA as CUDA buffers
    participant OptiX as OptiX pipeline
    participant GPU as RTX GPU

    User->>Args: Optional --mesh or --scene input
    Args->>ObjLoader: Resolve mesh path(s) and transforms
    Args->>Scene: Apply camera and light config
    User->>App: Keyboard and mouse input
    ObjLoader->>Scene: Load MeshObject list with OBJ data, tangents, materials, textures and transforms
    App->>Scene: Move sphere, light, or toggle material
    App->>Renderer: Optional P toggle for progressive accumulation
    App->>Renderer: Optional N toggle for OptiX denoiser
    App->>Renderer: renderFrame(scene, camera)
    Renderer->>CUDA: Upload sphere materials, mesh buffers, diffuse/normal texture pixels, and LaunchParams
    Renderer->>OptiX: Build sphere GAS + one triangle GAS per mesh object + IAS transforms
    Renderer->>OptiX: optixLaunch
    OptiX->>GPU: Ray generation, sphere hit, mesh closest-hit, miss, shadow, reflection programs
    GPU-->>CUDA: Write uchar4 framebuffer and progressive HDR accumulation
    Renderer->>OptiX: Optional denoiser invoke for progressive color buffer
    Renderer->>CUDA: Copy framebuffer or denoised pixels to host
    Renderer-->>App: hostPixels + gpuTimeMs
    App->>App: Present image in OpenGL window
```

## OptiX Shader Flow

```mermaid
flowchart TD
    RG["__raygen__rg"] --> Primary["traceRadiance primary ray"]
    Primary --> HitSphere{"Sphere hit?"}
    Primary --> HitMesh{"Triangle mesh hit?"}
    Primary --> HitPlane{"Plane hit?"}
    Primary --> Miss["__miss__radiance sky color"]
    Miss --> Environment["Gradient environment lighting"]
    HitSphere --> Shadow["Point-light shadow ray"]
    HitMesh --> MeshClosestHit["Mesh closest-hit shader"]
    MeshClosestHit --> NormalMap["Optional tangent-space normal map"]
    NormalMap --> MeshMaterial
    MeshClosestHit --> MeshMaterial{"Mesh material type"}
    MeshMaterial --> MeshDiffuse["GGX direct lighting + shadow"]
    MeshMaterial --> MeshMirror["Mirror reflection ray"]
    MeshMaterial --> MeshMetal["Metallic GGX + tinted reflection"]
    MeshMaterial --> MeshGlass["Dielectric Fresnel/refraction approximation"]
    MeshMirror --> Depth
    MeshMetal --> Depth
    MeshGlass --> Depth
    MeshDiffuse --> Output
    Shadow --> Material{"Material type"}
    Material --> Diffuse["Diffuse lighting + hard shadow"]
    Material --> Mirror["Reflection ray"]
    Mirror --> Depth{"depth < maxDepth"}
    Depth -->|yes| Primary
    Depth -->|no| Fallback["Neutral fallback color"]
    HitPlane --> PlaneShadow["Point-light shadow ray"]
    PlaneShadow --> PlaneLight["Plane lighting + hard shadow + fog"]
    Diffuse --> Output["setRadiancePayload"]
    Mirror --> Output
    PlaneLight --> Output
    Environment --> Output
```

## Data Boundaries

```mermaid
flowchart LR
    Host["CPU side C++"] -->|SceneState| Upload["CUDA upload"]
    Host -->|CameraState| Upload
    Upload --> Params["LaunchParams"]
    Params --> Device["GPU device code"]
    Device -->|uchar4 pixels| Buffer["dFrameBuffer"]
    Device -->|HDR average| Accum["float4 accumulation buffer"]
    Accum --> Denoiser["Optional OptiX denoiser"]
    Denoiser --> Denoised["float4 denoised output"]
    Buffer -->|cudaMemcpyAsync| HostPixels["std::vector<uchar4>"]
    Denoised -->|tone map + gamma| HostPixels
```

## Responsibility Split

| Area | Files | Responsibility |
|---|---|---|
| Application loop | `src/app/maingpu.cpp`, `src/app/application.*` | Window creation, input, scene updates, presentation |
| Scene model | `src/app/scene.*`, `src/app/material.*`, `src/app/mesh.*`, `src/app/obj_loader.*` | Spheres, OBJ mesh data, materials, selected object, light movement and clamping |
| Camera | `src/app/camera.*` | Camera state and basis vectors for ray generation |
| Shared GPU data | `src/common/rtx_shared.h` | Host/device structures used by CUDA and OptiX |
| Renderer host side | `src/gpu/optix_renderer.*` | CUDA resources, OptiX context, sphere GAS, triangle GAS, IAS, SBT, pipeline, launch, progressive accumulation, and optional denoiser |
| Renderer device side | `src/gpu/optix_device_programs.h` | Ray generation, sphere hit, mesh closest-hit shader, miss programs, shadow and reflection logic |
| Tests | `tests/*` | CPU unit tests and native GPU smoke test |

