# RayTracerRTX

Интерактивный трассировщик лучей в реальном времени на `C++`, `CUDA` и `NVIDIA OptiX`.

## Описание проекта

Программа рендерит 3D-сцену (сферы, плоскость и OBJ mesh) с поддержкой:

- диффузных и зеркальных материалов;
- освещения и теней;
- отражений для зеркального материала;
- gradient environment lighting, tone mapping and gamma correction;
- интерактивного управления камерой и объектами сцены;
- отображения метрик производительности (FPS, время кадра, GPU time).

Программа принимает описание сцены, включающее аналитические примитивы и
полигональные сетки формата OBJ. Для OBJ поддерживаются вершины, нормали,
треугольные грани и несколько материалов через MTL. Значения `Kd` задают цвет
диффузного материала; `Ks`, `Ns`, `Ni` и `d` задают specular color,
roughness/glossiness, index of refraction и alpha. Если имя материала содержит
`mirror`, `metal`, `glass` или `dielectric`, выбирается соответствующая
упрощенная модель материала; остальные OBJ-материалы отображаются как
диффузные. Для demo PPM-текстур поддерживаются diffuse `map_Kd` и basic normal
mapping через `bump`, `map_Bump` или `norm`; normal map изменяет только shading
normal и не делает displacement геометрии. Direct lighting использует
physically motivated GGX/Trowbridge-Reitz microfacet BRDF для diffuse, mirror и
metal материалов; glass остается отдельной упрощенной dielectric-моделью.
Renderer поддерживает default real-time direct lighting mode и optional
progressive path tracing mode с GPU accumulation. Progressive mode can
optionally run the OptiX denoiser; if denoiser initialization or invocation
fails, rendering continues without it.
Basic glTF 2.0 `.gltf` import is also supported for external `.bin` buffers,
positions, normals, texcoords, indices, simple node translation/scale, and
`pbrMetallicRoughness` base color, metallic and roughness factors. It is an
additional mesh input path and does not replace OBJ/MTL.

## Технологии

- C++ (MSVC v143, Visual Studio 2022)
- CUDA 13.1
- NVIDIA OptiX 9.1
- GLFW + OpenGL

## Требования

- Windows 10/11 x64
- NVIDIA GPU с поддержкой RTX
- Установленные драйверы NVIDIA
- Visual Studio 2022 (Desktop development with C++)
- CUDA Toolkit 13.1
- NVIDIA OptiX SDK 9.1

## Сборка и запуск

1. Откройте `RayTracerRTX.vcxproj` в Visual Studio 2022.
2. Выберите конфигурацию `Debug|x64` или `Release|x64`.
3. Убедитесь, что пути к CUDA и OptiX корректны.
4. Соберите проект и запустите приложение.

Приложение принимает входные данные через аргументы командной строки:

```powershell
RayTracerRTX.exe --mesh RayTracerRTX/assets/meshes/demo.obj
RayTracerRTX.exe --scene RayTracerRTX/assets/scenes/demo_scene.json
RayTracerRTX.exe --scene RayTracerRTX/assets/scenes/textured_scene.json
RayTracerRTX.exe --scene RayTracerRTX/assets/scenes/textured_cube_scene.json
RayTracerRTX.exe --scene RayTracerRTX/assets/scenes/multi_mesh_scene.json
```

`--mesh` заменяет встроенный demo mesh на указанный OBJ. `--scene` загружает
JSON scene config с `meshObjects`, `camera`, `light` и transform-полями
`position`, `rotation`, `scale`. При ошибке чтения входных данных приложение
пишет диагностическое сообщение в консоль и запускает стандартную demo-сцену.

## Управление

- `W/A/S/D` — перемещение камеры
- `Space` — движение камеры вверх
- Мышь — поворот камеры
- ЛКМ — захват/освобождение курсора
- `1/2/3` — выбор сферы
- `Left/Right/Up/Down` — движение выбранной сферы
- `R/F` — движение сферы по оси Y
- `J/L`, `I/K`, `U/O` — перемещение источника света
- `M` — переключение материала выбранной сферы
- `P` — переключение real-time direct / progressive path tracing mode
- `N` — optional OptiX denoiser for progressive path tracing mode
- `Esc` — выход

## Структура проекта

- `src/app` — окно, цикл приложения, ввод, камера, сцена, материалы
- `src/gpu` — OptiX-рендерер, pipeline, SBT, device-программы
- `src/common` — общие структуры данных host/device
- `tests` — unit-тесты и тестовый проект
- `assets/meshes/demo.obj` and `assets/meshes/demo.mtl` - demo OBJ mesh with diffuse and mirror materials
- `assets/meshes/textured_demo.obj`, `textured_cube.obj`, `textured_demo.mtl`, `checker.ppm`, and `checker_normal.ppm` - demo OBJ meshes with `map_Kd` diffuse texture and basic normal map
- `assets/meshes/minimal_gltf.gltf` and `minimal_gltf.bin` - minimal glTF 2.0 mesh import example
- `assets/scenes/demo_scene.json`, `textured_scene.json`, `textured_cube_scene.json`, `multi_mesh_scene.json`, and `gltf_scene.json` - documented JSON scene input examples

## Статус

Проект разрабатывается как курсовая работа по компьютерной графике (вариант 42). Реализованы рабочее RTX-ядро, интерактивная сцена, CPU/GPU-тесты, Docker Compose-проверка, CI workflow и проектная документация; выполняется финальная подготовка к защите.

## Build path configuration

CUDA, OptiX, and source include paths are configured through Visual Studio
project properties instead of hard-coded C++ strings. Override `OptixSdkDir`
in a user property sheet or from MSBuild with `/p:OptixSdkDir=...` when the
OptiX SDK is installed outside the default directory.
