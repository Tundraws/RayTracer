# RayTracerRTX

Интерактивное приложение трассировки лучей в реальном времени на `C++`, `CUDA` и `NVIDIA OptiX`.

## Описание проекта

Программа рендерит сцену со сферами, плоскостью и полигональными mesh-объектами. Входные данные задаются через аргументы командной строки или JSON scene config. По умолчанию используется встроенная demo-сцена.

Поддерживаемые возможности:

- аналитические примитивы: сферы и плоскость;
- OBJ/MTL mesh pipeline;
- несколько mesh objects в сцене;
- transform для mesh object: position, rotation, scale;
- материалы diffuse, mirror, metal и dielectric/glass;
- MTL-параметры `Kd`, `Ks`, `Ns`, `Ni`, `d`;
- diffuse texture `map_Kd` для ASCII PPM (`P3`);
- basic normal mapping через `bump`, `map_Bump` или `norm`;
- ограниченный glTF 2.0 `.gltf` + внешний `.bin` import;
- gradient environment lighting;
- Reinhard tone mapping и gamma correction;
- physically motivated GGX/Trowbridge-Reitz direct lighting для diffuse, mirror и metal;
- real-time direct lighting mode;
- progressive path tracing mode с accumulation buffer;
- optional OptiX denoiser для progressive mode;
- quality modes: Low, Medium, High, PathTracing;
- HUD с FPS, GPU time, mode, denoiser, quality и текущими mesh/material presets.

## Ограничения

Это учебный RTX renderer для курсовой работы с ограниченным набором возможностей.

Поддержка форматов ограничена:

- OBJ: `v`, `vn`, `vt`, triangular faces, `usemtl`, `mtllib`;
- MTL: `Kd`, `Ks`, `Ns`, `Ni`, `d`, `map_Kd`, `bump`, `map_Bump`, `norm`;
- texture files: ASCII PPM (`P3`) для demo diffuse/normal maps;
- glTF: `.gltf` JSON с внешним `.bin`, positions, normals, texcoords, indices, simple translation/scale transforms, `baseColorFactor`, `metallicFactor`, `roughnessFactor`, `baseColorTexture`.

Не поддерживаются `.glb`, embedded base64 buffers, animations, skinning, morph targets, displacement, full node rotation matrices, полный набор texture maps glTF и произвольные image formats.

## Технологии

- C++20
- Visual Studio 2022 / MSVC v143
- CUDA Toolkit 13.1
- NVIDIA OptiX SDK 9.1
- GLFW + OpenGL
- Docker Compose для CPU-only проверки

## Сборка

1. Откройте `RayTracerRTX.sln` в Visual Studio 2022.
2. Выберите `Debug|x64` или `Release|x64`.
3. Проверьте пути к CUDA и OptiX.
4. Соберите проект и запустите `RayTracerRTX.exe`.

Пути CUDA, OptiX и source include задаются через свойства Visual Studio project. Если OptiX SDK установлен не в стандартную директорию, переопределите `OptixSdkDir` через user property sheet или параметр MSBuild `/p:OptixSdkDir=...`.

## Запуск

Примеры из корня репозитория:

```powershell
.\x64\Debug\RayTracerRTX.exe
.\x64\Debug\RayTracerRTX.exe --mesh RayTracerRTX\assets\meshes\demo.obj
.\x64\Debug\RayTracerRTX.exe --scene RayTracerRTX\assets\scenes\demo_scene.json
.\x64\Debug\RayTracerRTX.exe --scene RayTracerRTX\assets\scenes\textured_scene.json
.\x64\Debug\RayTracerRTX.exe --scene RayTracerRTX\assets\scenes\textured_cube_scene.json
.\x64\Debug\RayTracerRTX.exe --scene RayTracerRTX\assets\scenes\multi_mesh_scene.json
.\x64\Debug\RayTracerRTX.exe --scene RayTracerRTX\assets\scenes\gltf_scene.json
```

`--mesh` загружает один OBJ или glTF mesh в стандартную сцену. `--scene` загружает JSON scene config с `meshObjects`, `camera`, `light` и transform-полями `position`, `rotation`, `scale`. При ошибке чтения входных данных приложение выводит сообщение в консоль и использует fallback demo-сцену.

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

## Структура

- `src/app` - окно, цикл приложения, ввод, камера, сцена, материалы, загрузчики OBJ/glTF
- `src/gpu` - OptiX renderer, pipeline, SBT, device-программы
- `src/common` - общие структуры host/device
- `tests` - unit-тесты, GPU smoke-tests и benchmark
- `assets/meshes/demo.obj`, `demo.mtl` - demo OBJ mesh
- `assets/meshes/textured_demo.obj`, `textured_cube.obj`, `textured_demo.mtl`, `checker.ppm`, `checker_normal.ppm` - demo assets с diffuse texture и normal map
- `assets/meshes/minimal_gltf.gltf`, `minimal_gltf.bin` - минимальный glTF example
- `assets/scenes/*.json` - примеры scene config
- `docs` - документация по API, архитектуре, Docker-check, performance и security

## Проверки

```powershell
& "C:\Program Files\Microsoft Visual Studio\2022\Community\MSBuild\Current\Bin\MSBuild.exe" ..\RayTracerRTX.sln /p:Configuration=Debug /p:Platform=x64 /m
& "C:\Program Files\Microsoft Visual Studio\2022\Community\MSBuild\Current\Bin\MSBuild.exe" tests\RayTracerRTX.Tests.vcxproj /p:Configuration=Debug /p:Platform=x64 /m
tests\x64\Debug\RayTracerRTX.Tests.exe
tests\x64\Debug\RayTracerRTX.Tests.exe --benchmark
```

Для проверки coursework-check из корня репозитория:

```powershell
docker compose build --no-cache
docker compose run --rm coursework-check
```

## Статус

Реализованы RTX rendering core, scene config input, OBJ/MTL mesh rendering, ограниченный glTF import, материалы, texture subset, normal mapping, progressive path tracing, optional denoiser, quality modes, тесты, benchmark, Docker-check и документация для курсовой работы.
