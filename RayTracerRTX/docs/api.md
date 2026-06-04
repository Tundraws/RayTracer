# Внутренние API проекта

Документ фиксирует основные внутренние интерфейсы трассировщика лучей. Описание сосредоточено на контрактах реализации, а не на тексте пояснительной записки.

## Вход приложения

### `run_optix_app()`

Функция объявлена в `src/app/application.h`.

`run_optix_app()` запускает интерактивный RTX-рендерер. Приложение управляет оконным циклом, обработкой ввода, изменением сцены, вызовами рендера и выводом изображения.

Поддерживаемые аргументы:

```powershell
RayTracerRTX.exe --mesh RayTracerRTX/assets/meshes/demo.obj
RayTracerRTX.exe --scene RayTracerRTX/assets/scenes/clean_floor_scene.json
```

`--mesh` загружает OBJ или glTF/GLB-модель в стандартную сцену. `--scene` загружает JSON-конфигурацию со сферами, mesh-объектами, камерой, светом, параметрами изображения и трансформациями. Некорректный ввод выводит диагностическое сообщение и использует стандартную сцену. Предупреждения и ошибки также записываются в `RayTracerRTX.log` через `src/app/logger.*`.

Приложение запускается в режиме `RenderModeRealtime`. Клавиша `P` переключает прогрессивное накопление, `N` включает или выключает шумоподавитель OptiX, `Q` переключает качество `Low`, `Medium`, `High`, `PathTracing`. При изменении камеры, света, сцены, материала или подписи геометрии накопление сбрасывается.

## API сцены

### `SphereGeometry`

Структура объявлена в `src/app/scene.h`.

| Поле | Тип | Назначение |
| --- | --- | --- |
| `center` | `float3` | центр сферы в мировых координатах |
| `radius` | `float` | радиус сферы |

### `SceneState`

Структура объявлена в `src/app/scene.h`.

| Поле | Тип | Назначение |
| --- | --- | --- |
| `spheres` | `std::vector<SphereGeometry>` | аналитические сферы сцены |
| `materials` | `std::vector<SphereMaterial>` | материалы сфер, передаваемые на GPU |
| `meshObjects` | `std::vector<MeshObject>` | полигональные объекты с ресурсом, mesh-данными и трансформацией |
| `mesh` | `MeshData` | совместимое объединённое представление для тестов и резервных путей |
| `lightPosition` | `float3` | положение источника света |
| `exposure` | `float` | множитель экспозиции для tone mapping |
| `skyIntensity` | `float` | интенсивность окружения |
| `lightIntensity` | `float` | интенсивность прямого света |
| `areaLightRadius` | `float` | радиус area light; ноль означает точечный источник |
| `environmentIntensity` | `float` | множитель карты окружения или градиентного неба |
| `environmentMap` | `MeshTexture` | optional PPM lat-long карта окружения |
| `selectedSphere` | `int` | индекс выбранной сферы |
| `selectedMeshObject` | `int` | индекс выбранного mesh-объекта |
| `selectedMeshMaterial` | `int` | индекс материала выбранного mesh-объекта |

### Основные функции сцены

| Функция | Назначение |
| --- | --- |
| `makeDefaultScene()` | создаёт начальную сцену |
| `clampScene(SceneState&)` | ограничивает параметры объектов допустимыми диапазонами |
| `addSphere(SceneState&)` | добавляет сферу и выбирает её |
| `removeSelectedSphere(SceneState&)` | удаляет выбранную сферу |
| `setSelectedSphereRadius(SceneState&, float)` | изменяет радиус выбранной сферы |
| `setSelectedSphereColor(SceneState&, float3)` | изменяет цвет выбранной сферы |
| `SceneEditor::setSelectedSphereMaterialProperties(...)` | изменяет цвет, roughness, IOR и alpha материала сферы |
| `moveSelectedSphere(SceneState&, float3)` | перемещает выбранную сферу |
| `toggleSelectedMaterial(SceneState&)` | переключает материал выбранной сферы |
| `cycleSelectedSphereMaterialPreset(SceneState&)` | перебирает preset-материалы сферы |
| `selectNextMeshObject(SceneState&)` | выбирает следующий mesh-объект |
| `cycleSelectedMeshMaterialPreset(SceneState&)` | перебирает preset-материалы mesh-объекта |
| `SceneEditor::setSelectedMeshMaterialProperties(...)` | изменяет материал выбранной модели |
| `SceneEditor::setSelectedMeshBaseColorTexture(...)` | назначает PNG/JPG/PPM base-color texture выбранной модели |
| `addBuiltInMeshPrimitive(SceneState&, int)` | добавляет куб, пирамиду или прямоугольную панель |
| `removeSelectedMeshObject(SceneState&)` | удаляет выбранный mesh-объект |
| `removeAllSpheres(SceneState&)` | удаляет все сферы |
| `removeAllMeshObjects(SceneState&)` | удаляет все mesh-объекты |
| `clearSceneObjects(SceneState&)` | очищает видимые объекты, сохраняя параметры камеры и света |
| `restoreDefaultSceneObjects(SceneState&)` | восстанавливает стандартные объекты |
| `applySceneEnvironmentMode(SceneState&, int)` | создаёт панели окружения: пол, комнату или пустую сцену |
| `moveLight(SceneState&, float3)` | перемещает источник света |
| `setSceneExposure(SceneState&, float)` | задаёт экспозицию |
| `setSceneSkyIntensity(SceneState&, float)` | задаёт интенсивность неба |
| `setSceneLightIntensity(SceneState&, float)` | задаёт интенсивность прямого света |

## Встроенные mesh-примитивы

Функции объявлены в `src/app/mesh.h`.

| Функция | Назначение |
| --- | --- |
| `createCubeMesh()` | создаёт куб как `MeshData` из треугольников |
| `createPyramidMesh()` | создаёт пирамиду как `MeshData` |
| `createPlaneMesh()` | создаёт конечную прямоугольную панель из двух треугольников |
| `getMeshTextureMetadata(...)` | возвращает безопасные сведения о texture slot материала |

Куб, пирамида и панель не являются отдельными типами GPU-примитивов. Они проходят через тот же путь `MeshData`, что и импортированные OBJ/glTF-модели.

## Редактор материалов

Панель Dear ImGui делит параметры материалов на разделы:

- `Поверхность` – базовый цвет или tint;
- `Отражение` – roughness, зеркальное размытие и металлическое отражение;
- `Прозрачность/стекло` – alpha, IOR и haze;
- `Текстуры` – base color texture и диагностика normal/roughness/metallic texture slots.

Панель назначает PNG, JPG/JPEG или PPM base-color texture выбранному mesh-объекту. Normal, roughness и metallic texture paths приходят из OBJ/MTL, glTF/GLB или JSON. Изменения материалов проходят через `SceneEditor`, поэтому dirty flags сбрасывают накопление и запрашивают обновление рендера.

## JSON-сцены

Функции объявлены в `src/app/scene_config.h`.

`loadSceneConfigFile(...)` разбирает JSON-сцену. Поддерживаемые поля:

- `mesh` – краткий путь к одной OBJ/glTF/GLB-модели;
- `spheres` – аналитические сферы с `name`, `position`, `radius`, `material`;
- `meshObjects` – массив объектов с `path` или `primitive`, `position`, `rotation`, `scale`, `name`, `material`;
- `camera` – `position`, `yaw`, `pitch`, `fov`;
- `light` – `position`, `intensity`, `size`/`radius`;
- `environment` – `type`, `path`, `intensity`, цвета градиента;
- `render` – `exposure`, `skyIntensity`, `lightIntensity`, `skyHorizonColor`, `skyZenithColor`, `skyGradientBlend`;
- `materials` – именованные материалы с `name`, `type`, `baseColor`, `roughness`, `metallic`, `specularColor`, `ior`, `alpha`, `texture`, `normalMap`;
- `sphereMaterials` – старый формат назначения материалов сферам;
- `material` или `materialOverride` у mesh-объекта – имя материала или inline-описание.

`saveSceneToConfigFile(...)` записывает текущую камеру, свет, сферы, встроенные примитивы, загруженные модели и базовые параметры материалов в JSON. Типы материалов принимают значения `matte`, `mirror`, `metal`, `glass`. Также принимаются alias-значения `diffuse` для `matte` и `dielectric` для `glass`.

`AssetCache` используется при построении сцены: модели кэшируются по нормализованному пути, текстуры кэшируются по пути и типу. Повторные ссылки на один ресурс не перечитывают файл.

## OBJ/MTL API

### `loadObjMesh(...)`

Функция объявлена в `src/app/obj_loader.h`.

Загрузчик поддерживает:

- `v` – позиции;
- `vn` – нормали;
- `vt` – текстурные координаты;
- `f` – грани с 3+ вершинами;
- `mtllib` и `usemtl`;
- MTL-поля `Kd`, `Ks`, `Ns`, `Ni`, `d`;
- `map_Kd` для base-color texture;
- `bump`, `map_Bump`, `norm` для normal map;
- `map_Pr`/`map_roughness` для roughness map;
- `map_Pm`/`map_metallic` для metallic map;
- несколько именованных материалов.

Грани с четырьмя и более вершинами разбиваются на треугольники простым fan-методом. Для сложных вогнутых многоугольников результат может быть неточным, поэтому такие модели лучше триангулировать заранее.

Имена материалов, содержащие `mirror`, переводятся в `MaterialMirror`; `metal` – в `MaterialMetal`; `glass` или `dielectric` – в `MaterialDielectric`; остальные – в `MaterialDiffuse`. `Ns` преобразуется в roughness, `Ni` – в IOR, `d` – в alpha, `Ks` – в specular color.

При отсутствии или повреждении texture map загрузчик сохраняет путь для диагностики и использует числовые параметры материала.

## glTF/GLB API

### `loadGltfMesh(...)`

Функция объявлена в `src/app/gltf_loader.h`.

```cpp
GltfLoadResult loadGltfMesh(const std::filesystem::path& path);
```

Загрузчик поддерживает ограниченный набор glTF 2.0:

- `.gltf` с внешним `.bin`;
- `.glb` с JSON/BIN-частями;
- `bufferViews`, `accessors`, mesh primitives;
- triangle indices;
- `POSITION`, `NORMAL`, `TEXCOORD_0`;
- иерархию узлов;
- `translation`, quaternion `rotation`, `scale`;
- `pbrMetallicRoughness.baseColorFactor`;
- `metallicFactor`, `roughnessFactor`;
- `baseColorTexture`, `normalTexture`, `metallicRoughnessTexture`.

Packed glTF metallic/roughness texture хранится как одна текстура: roughness берётся из green channel, metallic – из blue channel.

Не поддерживаются embedded base64 buffers, animations, skins, morph targets, cameras, lights, sparse accessors, sampler state, material extensions и node `matrix`.

## `MeshData`

Структура объявлена в `src/app/mesh.h`.

| Поле | Тип | Назначение |
| --- | --- | --- |
| `vertices` | `std::vector<MeshVertex>` | вершины с позицией, нормалью, UV, tangent и флагом наличия UV |
| `triangles` | `std::vector<MeshTriangle>` | индексы треугольников и индекс материала |
| `materials` | `std::vector<MeshMaterial>` | материалы модели |
| `textures` | `std::vector<MeshTexture>` | загруженные base-color, normal, roughness и metallic texture pixels |

`hasValidMeshMaterialIndices(const MeshData&)` проверяет, что каждый индекс материала треугольника находится внутри массива материалов.

## API камеры

### `CameraState`

Структура объявлена в `src/app/camera.h`.

| Поле | Тип | Назначение |
| --- | --- | --- |
| `position` | `float3` | положение камеры |
| `yaw` | `float` | горизонтальный поворот в градусах |
| `pitch` | `float` | вертикальный поворот в градусах |
| `fov` | `float` | вертикальный угол обзора |

### `updateCameraBasis(...)`

Функция рассчитывает базис камеры для `__raygen__rg`:

- `forward` – направление взгляда;
- `right` – горизонтальный базисный вектор;
- `up` – вертикальный базисный вектор;
- `scale` – масштаб плоскости лучей из FOV;
- `aspect` – соотношение сторон viewport.

## Общие структуры CPU/GPU

Общие структуры объявлены в `src/common/rtx_shared.h`.

### `MaterialType`

| Значение | Назначение |
| --- | --- |
| `MaterialDiffuse` | диффузное освещение с тенью и specular-компонентом |
| `MaterialMirror` | зеркальное отражение с ограничением глубины |
| `MaterialMetal` | окрашенное отражение с roughness |
| `MaterialDielectric` | упрощённое стекло с Fresnel, IOR и total internal reflection |

### `RenderQuality`

| Значение | Назначение |
| --- | --- |
| `RenderQualityLow` | один первичный сэмпл, малое число отражений, прямые тени отключены |
| `RenderQualityMedium` | два первичных сэмпла, средняя глубина отражений, прямые тени включены |
| `RenderQualityHigh` | четыре первичных сэмпла, увеличенная глубина отражений, прямые тени включены |
| `RenderQualityPathTracing` | прогрессивное накопление с path-tracing режимом и denoiser-запросом |

Диффузные, зеркальные и металлические материалы используют физически мотивированную GGX/Trowbridge-Reitz microfacet BRDF. Стеклянные материалы используют Schlick Fresnel, IOR refraction и fallback на отражение при total internal reflection. Модель материалов ограничена указанными входными параметрами и не является полной PBR-системой.

### `SphereMaterial`

| Поле | Тип | Назначение |
| --- | --- | --- |
| `color` | `float3` | базовый цвет |
| `materialType` | `int` | тип материала из `MaterialType` |
| `specularColor` | `float3` | specular tint |
| `roughness` | `float` | приближение размытия отражения |
| `ior` | `float` | показатель преломления |
| `alpha` | `float` | прозрачность/непрозрачность |

### `LaunchParams`

Структура копируется на GPU перед каждым запуском OptiX.

| Поле | Назначение |
| --- | --- |
| `image`, `imageWidth`, `imageHeight` | выходной framebuffer |
| `handle` | top-level OptiX traversable handle |
| `cameraPosition`, `cameraForward`, `cameraRight`, `cameraUp` | данные камеры для генерации лучей |
| `cameraScale`, `cameraAspect` | параметры проекции |
| `lightPosition`, `areaLightRadius` | параметры прямого света |
| `exposure`, `skyIntensity`, `lightIntensity`, `environmentIntensity` | параметры изображения и освещения |
| `materials`, `sphereCount` | материалы сфер |
| `meshVertices`, `meshVertexCount` | GPU-буфер вершин моделей |
| `meshTriangles`, `meshTriangleCount` | GPU-буфер треугольников |
| `meshMaterials`, `meshMaterialCount` | GPU-буфер материалов моделей |
| `meshTexturePixels`, `meshTexturePixelCount` | упакованные texture pixels |
| `environmentPixels`, `environmentWidth`, `environmentHeight` | карта окружения |
| `accumulation` | float-буфер прогрессивного накопления |
| `renderMode` | `RenderModeRealtime` или `RenderModeProgressive` |
| `renderQuality` | текущий режим качества |
| `shadowEnabled` | флаг прямых теневых лучей |
| `samplesPerPixel` | число первичных сэмплов на пиксель |
| `accumulationSample` | индекс текущего сэмпла накопления |
| `maxDepth` | ограничение глубины рекурсии |

## Режимы рендера

`RenderModeRealtime` является режимом по умолчанию. Он рассчитывает прямое освещение, тени, отражения и материалы без накопления предыдущих кадров.

`RenderModeProgressive` хранит float-буфер накопления на GPU. Каждый кадр добавляет jittered primary samples и стохастические отскоки, после чего отображается усреднённый результат. При изменении камеры или сцены накопление сбрасывается.

При включённом denoiser прогрессивный режим передаёт HDR-буфер накопления в OptiX denoiser. В текущей реализации используются только данные цвета; albedo и normal guide layers не создаются.

## API рендерера

### `OptixRenderer`

Класс объявлен в `src/gpu/optix_renderer.h`.

| Метод | Назначение |
| --- | --- |
| `setRenderSize(int, int)` | задаёт размер framebuffer |
| `setRenderMode(int)` | выбирает режим реального времени или накопления |
| `setDenoiserEnabled(bool)` | запрашивает OptiX denoiser для прогрессивного режима |
| `isDenoiserAvailable()` | сообщает, доступен ли denoiser |
| `initialize()` | создаёт CUDA stream, OptiX context, modules, program groups, pipeline, SBT и acceleration structures |
| `renderFrame(const SceneState&, const CameraState&, std::vector<uchar4>&, float*)` | загружает данные кадра, запускает OptiX и копирует пиксели в память CPU |
| `destroy()` | освобождает CUDA- и OptiX-ресурсы |

### Основные внутренние этапы

| Метод | Назначение |
| --- | --- |
| `createContext()` | инициализирует CUDA/OptiX device context |
| `createScene()` | выделяет буферы геометрии |
| `createModule()` | компилирует device program source через NVRTC |
| `createProgramGroups()` | создаёт raygen, miss и hitgroup программы |
| `createPipeline()` | связывает OptiX pipeline и рассчитывает stack sizes |
| `createSbt()` | создаёт shader binding table records |
| `rebuildAccelerationStructure()` | строит GAS/IAS структуры ускорения |

## Обработка результата

Перед записью в `uchar4` GPU-программа выполняет:

- ограничение HDR-значений;
- Reinhard tone mapping;
- gamma correction с gamma 2.2.

Miss shader использует процедурное градиентное окружение или PPM lat-long environment map из JSON-сцены. Area-light режим использует несколько deterministic shadow samples в `Medium`, `High` и `PathTracing`; при `Low` или нулевом радиусе применяется один сэмпл.
