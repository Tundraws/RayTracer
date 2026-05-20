# Architecture

This document describes the implementation architecture of the real-time RTX
ray tracer. It is a technical project artifact and can be used as source
material for the coursework design section.

## Module View

```mermaid
flowchart LR
    Main["maingpu.cpp"] --> App["Application loop"]
    App --> Input["Input handling"]
    App --> Camera["CameraState / updateCameraBasis"]
    App --> Scene["SceneState"]
    Scene --> Materials["SphereMaterial"]
    App --> Renderer["OptixRenderer"]
    Camera --> Renderer
    Scene --> Renderer
    Renderer --> Shared["LaunchParams / rtx_shared.h"]
    Renderer --> Device["OptiX device programs"]
    Device --> Framebuffer["CUDA framebuffer"]
    Framebuffer --> App
```

## Runtime Pipeline

```mermaid
sequenceDiagram
    participant User
    participant App as GLFW application
    participant Scene as SceneState
    participant Renderer as OptixRenderer
    participant CUDA as CUDA buffers
    participant OptiX as OptiX pipeline
    participant GPU as RTX GPU

    User->>App: Keyboard and mouse input
    App->>Scene: Move sphere, light, or toggle material
    App->>Renderer: renderFrame(scene, camera)
    Renderer->>CUDA: Upload materials and LaunchParams
    Renderer->>OptiX: optixLaunch
    OptiX->>GPU: Ray generation, hit, miss, shadow, reflection programs
    GPU-->>CUDA: Write uchar4 framebuffer
    Renderer->>CUDA: Copy framebuffer to host
    Renderer-->>App: hostPixels + gpuTimeMs
    App->>App: Present image in OpenGL window
```

## OptiX Shader Flow

```mermaid
flowchart TD
    RG["__raygen__rg"] --> Primary["traceRadiance primary ray"]
    Primary --> HitSphere{"Sphere hit?"}
    Primary --> HitPlane{"Plane hit?"}
    Primary --> Miss["__miss__radiance sky color"]
    HitSphere --> Visibility["Point or area-light visibility"]
    Visibility --> Material{"Material type"}
    Material --> Diffuse["Diffuse lighting + hard/soft shadow"]
    Material --> Mirror["Reflection ray"]
    Mirror --> Depth{"depth < maxDepth"}
    Depth -->|yes| Primary
    Depth -->|no| Fallback["Neutral fallback color"]
    HitPlane --> PlaneVisibility["Point or area-light visibility"]
    PlaneVisibility --> PlaneLight["Plane lighting + hard/soft shadow + fog"]
    Diffuse --> Output["setRadiancePayload"]
    Mirror --> Output
    PlaneLight --> Output
    Miss --> Output
```

## Data Boundaries

```mermaid
flowchart LR
    Host["CPU side C++"] -->|SceneState| Upload["CUDA upload"]
    Host -->|CameraState| Upload
    Upload --> Params["LaunchParams"]
    Params --> Device["GPU device code"]
    Device -->|uchar4 pixels| Buffer["dFrameBuffer"]
    Buffer -->|cudaMemcpyAsync| HostPixels["std::vector<uchar4>"]
```

## Responsibility Split

| Area | Files | Responsibility |
|---|---|---|
| Application loop | `src/app/maingpu.cpp`, `src/app/application.*` | Window creation, input, scene updates, presentation |
| Scene model | `src/app/scene.*`, `src/app/material.*` | Spheres, materials, selected object, light movement and clamping |
| Camera | `src/app/camera.*` | Camera state and basis vectors for ray generation |
| Shared GPU data | `src/common/rtx_shared.h` | Host/device structures used by CUDA and OptiX |
| Renderer host side | `src/gpu/optix_renderer.*` | CUDA resources, OptiX context, acceleration structures, SBT, pipeline and launch |
| Renderer device side | `src/gpu/optix_device_programs.h` | Ray generation, hit programs, miss programs, shadow and reflection logic |
| Tests | `tests/*` | CPU unit tests and native GPU smoke test |

