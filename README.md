# RayTracerRTX

Интерактивный трассировщик лучей в реальном времени на `C++`, `CUDA` и `NVIDIA OptiX`.

Проект рендерит 3D-сцену со сферами, плоскостью и полигональными сетками. Программа принимает описание сцены из аргументов командной строки или JSON-конфига и может использовать OBJ/MTL-модели, а также ограниченный subset glTF 2.0.

## Возможности

- Аппаратная трассировка лучей через NVIDIA OptiX.
- Аналитические примитивы: сферы и плоскость.
- Полигональные OBJ-сетки с вершинами, нормалями, `vt`-координатами и треугольными гранями.
- Несколько OBJ-материалов через MTL.
- Поддержка MTL-параметров `Kd`, `Ks`, `Ns`, `Ni`, `d`.
- Упрощенные типы материалов: diffuse, mirror, metal, dielectric/glass.
- Диффузные PPM-текстуры через `map_Kd`.
- Basic normal mapping через `bump`, `map_Bump` или `norm`; normal map меняет только shading normal и не делает displacement геометрии.
- Несколько mesh objects в сцене с transform-параметрами: position, rotation, scale.
- Ограниченный импорт glTF 2.0 `.gltf` + внешний `.bin`: positions, normals, texcoords, indices, простые translation/scale transforms, `baseColorFactor`, `metallicFactor`, `roughnessFactor`, `baseColorTexture`.
- Gradient environment lighting, Reinhard tone mapping и gamma correction.
- Physically motivated GGX/Trowbridge-Reitz shading для direct lighting у diffuse, mirror и metal материалов.
- Real-time direct lighting mode и progressive path tracing mode с накоплением сэмплов.
- Optional OptiX denoiser для progressive mode; если denoiser недоступен, рендеринг продолжается без него.
- Переключаемые quality modes: Low, Medium, High, PathTracing.
- Интерактивное управление камерой, светом, сферами, mesh/material presets и режимами качества.
- CPU/GPU-тесты, benchmark и Docker coursework-check.

## Ограничения

Проект является учебным RTX renderer для курсовой работы с ограниченным набором возможностей. Поддержка OBJ, MTL, текстур, normal maps и glTF ограничена перечисленным subset. Не поддерживаются `.glb`, embedded base64 buffers, animation, skinning, morph targets, displacement, texture types кроме `map_Kd` для OBJ и базовой `baseColorTexture` для glTF.

## Технологии

- C++20, MSVC v143, Visual Studio 2022
- CUDA Toolkit 13.1
- NVIDIA OptiX SDK 9.1
- GLFW + OpenGL для вывода изображения
- Docker Compose для CPU-only coursework-check

## Требования

- Windows 10/11 x64
- NVIDIA GPU с поддержкой RTX
- Установленные драйверы NVIDIA
- Visual Studio 2022 с компонентом Desktop development with C++
- CUDA Toolkit 13.1
- NVIDIA OptiX SDK 9.1

## Сборка и запуск

1. Откройте `RayTracerRTX.sln` в Visual Studio 2022.
2. Выберите конфигурацию `Debug|x64` или `Release|x64`.
3. Убедитесь, что пути к CUDA и OptiX корректны.
4. Соберите проект и запустите `RayTracerRTX.exe`.

Примеры запуска из корня репозитория:

```powershell
.\x64\Debug\RayTracerRTX.exe
.\x64\Debug\RayTracerRTX.exe --mesh RayTracerRTX\assets\meshes\demo.obj
.\x64\Debug\RayTracerRTX.exe --scene RayTracerRTX\assets\scenes\demo_scene.json
.\x64\Debug\RayTracerRTX.exe --scene RayTracerRTX\assets\scenes\textured_cube_scene.json
.\x64\Debug\RayTracerRTX.exe --scene RayTracerRTX\assets\scenes\multi_mesh_scene.json
.\x64\Debug\RayTracerRTX.exe --scene RayTracerRTX\assets\scenes\gltf_scene.json
```

Если аргументы не переданы, запускается стандартная demo-сцена. Если JSON-конфиг отсутствует или некорректен, приложение выводит диагностическое сообщение в консоль и возвращается к fallback demo-сцене.

JSON scene config поддерживает:

- `mesh` или `meshObjects`;
- путь к OBJ или glTF mesh;
- параметры камеры: position, yaw, pitch, fov;
- позицию источника света;
- transform для mesh object: position, rotation, scale.

## Управление

- `W/A/S/D` - перемещение камеры
- `Space` - движение камеры вверх
- мышь - поворот камеры
- ЛКМ - захват или освобождение курсора
- `1/2/3` - выбор сферы
- `Left/Right/Up/Down` - движение выбранной сферы
- `R/F` - движение сферы по оси Y
- `J/L`, `I/K`, `U/O` - перемещение источника света
- `G` - переключение scene/mesh preset
- `M` - переключение material preset выбранной сферы
- `B` - выбор следующего mesh object
- `V` - переключение material preset выбранного mesh material
- `P` - переключение real-time direct lighting / progressive path tracing
- `Q` - переключение quality mode: Low / Medium / High / PathTracing
- `N` - включение или выключение optional OptiX denoiser для progressive mode
- `Esc` - выход

## Структура проекта

- `RayTracerRTX/src/app` - приложение, ввод, камера, сцена, материалы, загрузчики OBJ/glTF
- `RayTracerRTX/src/gpu` - OptiX renderer, pipeline, SBT, device-программы
- `RayTracerRTX/src/common` - общие host/device структуры
- `RayTracerRTX/tests` - unit-тесты, GPU smoke-tests и benchmark
- `RayTracerRTX/assets/meshes/demo.obj`, `demo.mtl` - demo OBJ mesh с несколькими материалами
- `RayTracerRTX/assets/meshes/textured_demo.obj`, `textured_cube.obj`, `textured_demo.mtl`, `checker.ppm`, `checker_normal.ppm` - demo OBJ assets с `map_Kd` и normal map
- `RayTracerRTX/assets/meshes/minimal_gltf.gltf`, `minimal_gltf.bin` - минимальный пример glTF mesh
- `RayTracerRTX/assets/scenes/*.json` - примеры JSON scene config
- `RayTracerRTX/docs` - проектная документация

## Проверки

Основные локальные проверки:

```powershell
& "C:\Program Files\Microsoft Visual Studio\2022\Community\MSBuild\Current\Bin\MSBuild.exe" RayTracerRTX.sln /p:Configuration=Debug /p:Platform=x64 /m
& "C:\Program Files\Microsoft Visual Studio\2022\Community\MSBuild\Current\Bin\MSBuild.exe" RayTracerRTX\tests\RayTracerRTX.Tests.vcxproj /p:Configuration=Debug /p:Platform=x64 /m
RayTracerRTX\tests\x64\Debug\RayTracerRTX.Tests.exe
RayTracerRTX\tests\x64\Debug\RayTracerRTX.Tests.exe --benchmark
docker compose build --no-cache
docker compose run --rm coursework-check
```

Docker coursework-check не запускает GPU/OptiX GUI. Он проверяет структуру проекта, наличие assets и CPU-only тесты загрузчиков/сцены.

## Статус

Проект находится на финальном техническом этапе подготовки курсовой работы по компьютерной графике. Реализованы RTX-ядро, интерактивная сцена, входные OBJ/glTF mesh данные, материалы, текстуры базового subset, progressive mode, optional denoiser, quality modes, тесты, benchmark, Docker-check и документация.
