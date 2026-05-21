# RayTracerRTX

Интерактивный трассировщик лучей в реальном времени на `C++`, `CUDA` и `NVIDIA OptiX`.

Проект рендерит простую 3D-сцену (сферы, плоскость и OBJ mesh) с поддержкой:

- диффузных и зеркальных материалов;
- направленного освещения и теней;
- отражений (для зеркального материала);
- gradient environment lighting, tone mapping and gamma correction;
- интерактивного управления камерой и параметрами сцены;
- отображения FPS и времени кадра/GPU.

Program input is represented as a scene description with analytic primitives
and polygonal OBJ meshes. The OBJ path supports vertices, normals, triangular
faces, and multiple materials through MTL files. MTL `Kd` values define diffuse
colors; `Ks`, `Ns`, `Ni`, and `d` provide specular color, roughness/glossiness,
index of refraction, and alpha values. Material names containing `mirror`,
`metal`, `glass`, or `dielectric` select the corresponding simplified material
model; other OBJ materials use diffuse shading. Diffuse `map_Kd` and basic
normal maps through `bump`, `map_Bump`, or `norm` are supported for PPM demo
textures; normal mapping perturbs shading normals only and does not displace
geometry. Direct lighting uses a physically motivated GGX/Trowbridge-Reitz
microfacet BRDF for diffuse, mirror and metal materials; glass remains a
separate simplified dielectric approximation. The renderer has a default
real-time direct lighting mode and an optional progressive path tracing mode
with GPU accumulation. Progressive mode can optionally run the OptiX denoiser;
if denoiser initialization or invocation fails, rendering continues without it.
Basic glTF 2.0 `.gltf` import is also supported for external `.bin` buffers,
positions, normals, texcoords, indices, simple node translation/scale, and
`pbrMetallicRoughness` base color, metallic and roughness factors. It is an
additional mesh input path and does not replace OBJ/MTL.

## Технологии

- C++ (MSVC v143, Visual Studio 2022)
- CUDA 13.1
- NVIDIA OptiX 9.1
- GLFW + OpenGL (вывод изображения)

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
3. Убедитесь, что пути к CUDA/OptiX доступны в системе.
4. Соберите проект и запустите `RayTracerRTX`.

Программа может использовать встроенную demo-сцену или принимать входные
данные из командной строки:

```powershell
RayTracerRTX.exe --mesh RayTracerRTX/assets/meshes/demo.obj
RayTracerRTX.exe --scene RayTracerRTX/assets/scenes/demo_scene.json
RayTracerRTX.exe --scene RayTracerRTX/assets/scenes/textured_scene.json
RayTracerRTX.exe --scene RayTracerRTX/assets/scenes/textured_cube_scene.json
RayTracerRTX.exe --scene RayTracerRTX/assets/scenes/multi_mesh_scene.json
```

JSON scene config поддерживает `meshObjects`, `camera`, `light` и transform
для mesh object: `position`, `rotation`, `scale`. Если входной файл отсутствует
или некорректен, приложение выводит сообщение в консоль и возвращается к
стандартной demo-сцене.

## Управление

- `W/A/S/D` - перемещение камеры
- `Space` - движение камеры вверх
- Мышь - поворот камеры
- ЛКМ - захват/освобождение курсора
- `1/2/3` - выбор сферы
- `Left/Right/Up/Down` - движение выбранной сферы
- `R/F` - движение сферы по оси Y
- `J/L`, `I/K`, `U/O` - перемещение источника света
- `M` - переключение материала выбранной сферы
- `P` - переключение real-time direct / progressive path tracing mode
- `N` - optional OptiX denoiser for progressive path tracing mode
- `Esc` - выход

## Структура проекта

- `src/app` - приложение, ввод, камера, сцена, материалы
- `src/gpu` - OptiX renderer, pipeline, SBT, device-программы
- `src/common` - общие структуры данных host/device
- `assets/meshes/demo.obj` and `assets/meshes/demo.mtl` - demo OBJ mesh with diffuse and mirror materials
- `assets/meshes/textured_demo.obj`, `textured_cube.obj`, `textured_demo.mtl`, `checker.ppm`, and `checker_normal.ppm` - demo OBJ meshes with `map_Kd` diffuse texture and basic normal map
- `assets/meshes/minimal_gltf.gltf` and `minimal_gltf.bin` - minimal glTF 2.0 mesh import example
- `assets/scenes/demo_scene.json`, `textured_scene.json`, `textured_cube_scene.json`, `multi_mesh_scene.json`, and `gltf_scene.json` - documented JSON scene input examples

## Текущий статус

Проект находится на финальном техническом этапе подготовки курсовой работы по компьютерной графике. Реализованы RTX-ядро, интерактивная сцена, CPU/GPU-тесты, Docker Compose-проверка, CI workflow и комплект проектной документации.
