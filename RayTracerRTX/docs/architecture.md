# Архитектура

Документ описывает архитектуру интерактивного RTX-трассировщика лучей. Материал относится к технической документации проекта и может использоваться как основа для проектного раздела пояснительной записки.

## Модульное представление

```mermaid
flowchart LR
    Main["maingpu.cpp"] --> App["Главный цикл приложения"]
    Main --> Args["Аргументы --mesh / --scene"]
    Args --> SceneConfig["JSON-конфигурация сцены"]
    App --> Input["Обработка ввода"]
    App --> UI["Редактор сцены Dear ImGui"]
    App --> AppState["AppState"]
    UI --> SceneEditor["SceneEditor"]
    Input --> SceneEditor
    SceneEditor --> DirtyFlags["SceneDirtyFlags"]
    App --> Controller["RendererController"]
    App --> Camera["CameraState / updateCameraBasis"]
    SceneConfig --> JsonMaterials["Материалы JSON"]
    SceneConfig --> MeshObject["MeshObject + transform"]
    UI --> SceneGroups["Группы объектов"]
    SceneGroups --> SceneEditor
    SceneConfig --> AssetCache["AssetCache"]
    AssetCache --> MeshCache["Кэш моделей по пути"]
    AssetCache --> TextureCache["Кэш текстур по пути"]
    MeshObject --> AssetCache
    MeshCache --> ObjLoader["OBJ/glTF loaders"]
    ObjLoader --> Triangulation["Fan-триангуляция OBJ-граней"]
    Triangulation --> MeshData["MeshData"]
    ObjLoader --> Textures["map_Kd / normal textures"]
    ObjLoader --> Tangents["Касательная основа"]
    Textures --> MeshData
    TextureCache --> Textures
    Tangents --> MeshData
    MeshData --> MeshObject
    MeshObject --> Scene["SceneState"]
    App --> Scene
    UI --> Scene
    Scene --> Materials["SphereMaterial"]
    Scene --> MeshMaterials["Материалы моделей"]
    JsonMaterials --> Materials
    JsonMaterials --> MeshMaterials
    Controller --> Mode["Режим рендера"]
    Controller --> Rebuild["Запрос перестроения сцены"]
    Controller --> AccumReset["Сброс накопления"]
    DirtyFlags --> Controller
    AppState --> Scene
    App --> Renderer["OptixRenderer"]
    App --> Mode
    Camera --> Renderer
    Scene --> Renderer
    Mode --> Renderer
    Renderer --> TriangleGAS["Triangle GAS для mesh-объектов"]
    Renderer --> IAS["IAS с трансформациями"]
    Renderer --> Shared["LaunchParams / rtx_shared.h"]
    Renderer --> Device["GPU-программы OptiX"]
    TriangleGAS --> IAS
    IAS --> Device
    Device --> Accum["Буфер накопления"]
    Device --> Framebuffer["CUDA framebuffer"]
    Device --> Post["Tone mapping + gamma"]
    Accum --> Post
    Post --> Framebuffer
    Framebuffer --> App
```

## Конвейер выполнения

```mermaid
sequenceDiagram
    participant User as Пользователь
    participant Args as CLI / JSON
    participant App as GLFW-приложение
    participant Cache as AssetCache
    participant Loader as OBJ/glTF loaders
    participant Scene as SceneState
    participant Renderer as OptixRenderer
    participant CUDA as CUDA-буферы
    participant OptiX as OptiX pipeline
    participant GPU as RTX GPU

    User->>Args: Аргументы --mesh или --scene
    Args->>Cache: Разрешение путей моделей, текстур и трансформаций
    Args->>Scene: Применение камеры и света
    User->>App: Клавиатура и мышь
    Cache->>Loader: Загрузка отсутствующих ресурсов
    Loader-->>Cache: MeshData, материалы и текстуры
    Cache->>Scene: Формирование списка MeshObject
    App->>Scene: Изменение объектов, света и материалов
    App->>Renderer: Переключение режима, качества и шумоподавителя
    App->>Renderer: renderFrame(scene, camera)
    Renderer->>CUDA: Копирование геометрии, материалов, текстур и LaunchParams
    Renderer->>OptiX: Построение GAS/IAS
    Renderer->>OptiX: optixLaunch
    OptiX->>GPU: Raygen, hit, miss, shadow, reflection программы
    GPU-->>CUDA: Запись framebuffer и HDR-накопления
    Renderer->>OptiX: Дополнительный denoiser для прогрессивного режима
    Renderer->>CUDA: Копирование пикселей в память CPU
    Renderer-->>App: hostPixels + gpuTimeMs
    App->>App: Вывод изображения в OpenGL-окно
```

## Поток GPU-программ OptiX

```mermaid
flowchart TD
    RG["__raygen__rg"] --> Primary["Первичный traceRadiance"]
    Primary --> HitSphere{"Попадание в сферу?"}
    Primary --> HitMesh{"Попадание в mesh?"}
    Primary --> HitPlane{"Попадание в плоскость?"}
    Primary --> Miss["__miss__radiance"]
    Miss --> Environment["Цвет окружения"]
    HitSphere --> Shadow["Теневые лучи"]
    HitMesh --> MeshClosestHit["__closesthit__radiance_mesh"]
    MeshClosestHit --> NormalMap["Карта нормалей"]
    NormalMap --> MeshMaterial
    MeshClosestHit --> MeshMaterial{"Тип материала mesh"}
    MeshMaterial --> MeshDiffuse["Диффузный расчёт + GGX"]
    MeshMaterial --> MeshMirror["Зеркальное отражение"]
    MeshMaterial --> MeshMetal["Металлическое отражение"]
    MeshMaterial --> MeshGlass["Преломление и Fresnel"]
    MeshMirror --> Depth
    MeshMetal --> Depth
    MeshGlass --> Depth
    MeshDiffuse --> Output
    Shadow --> Material{"Тип материала сферы"}
    Material --> Diffuse["Диффузное освещение"]
    Material --> Mirror["Луч отражения"]
    Mirror --> Depth{"depth < maxDepth"}
    Depth -->|да| Primary
    Depth -->|нет| Fallback["Резервный цвет"]
    HitPlane --> PlaneShadow["Тени плоскости"]
    PlaneShadow --> PlaneLight["Освещение плоскости"]
    Diffuse --> Output["setRadiancePayload"]
    Mirror --> Output
    PlaneLight --> Output
    Environment --> Output
```

## Граница данных CPU/GPU

```mermaid
flowchart LR
    Host["CPU-код C++"] -->|SceneState| Upload["Копирование в CUDA"]
    Host -->|CameraState| Upload
    Upload --> Params["LaunchParams"]
    Params --> Device["GPU-код OptiX"]
    Device -->|uchar4 pixels| Buffer["dFrameBuffer"]
    Device -->|HDR average| Accum["float4 accumulation"]
    Accum --> Denoiser["OptiX denoiser"]
    Denoiser --> Denoised["float4 denoised output"]
    Buffer -->|cudaMemcpyAsync| HostPixels["std::vector<uchar4>"]
    Denoised -->|tone map + gamma| HostPixels
```

## Разделение ответственности

| Область | Файлы | Ответственность |
| --- | --- | --- |
| Главный цикл | `src/app/maingpu.cpp`, `src/app/application.*` | окно, ввод, обновление сцены и вывод изображения |
| Состояние приложения | `src/app/app_state.h` | камера, сцена, выбранные объекты, режим качества и запросы перестроения |
| Редактор сцены | `src/app/scene_editor.*` | изменение сцены и формирование `SceneDirtyFlags` |
| Управление рендерером | `src/app/renderer_controller.*` | сброс накопления, применение качества и пометка перестроения |
| Панель UI | `src/app/application.*`, Dear ImGui | выбор и редактирование объектов, материалов, камеры, света и качества |
| Группы объектов | `src/app/scene.*`, `src/app/scene_editor.*` | editor-level группы сфер и mesh-объектов без нового GPU-примитива |
| Встроенные mesh-примитивы | `src/app/mesh.*` | создание куба, пирамиды и прямоугольной панели как `MeshData` |
| Сцена и ресурсы | `src/app/scene.*`, `src/app/material.*`, `src/app/mesh.*`, `src/app/obj_loader.*`, `src/app/gltf_loader.*` | сферы, модели, материалы, загрузка ресурсов, свет и ограничения параметров |
| Камера | `src/app/camera.*` | состояние камеры и базисные векторы для генерации лучей |
| Общие GPU-данные | `src/common/rtx_shared.h` | структуры, общие для CPU- и GPU-кода |
| Host-часть рендерера | `src/gpu/optix_renderer.*` | CUDA-ресурсы, контекст OptiX, GAS/IAS, SBT, pipeline, запуск и denoiser |
| Device-часть рендерера | `src/gpu/optix_device_programs.h` | генерация лучей, hit/miss/shadow программы, отражения и преломления |
| Тесты | `tests/*` | CPU-проверки и GPU smoke-тесты |
