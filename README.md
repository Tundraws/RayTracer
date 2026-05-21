# RayTracerRTX

Интерактивный трассировщик лучей в реальном времени на `C++`, `CUDA` и `NVIDIA OptiX`.

Проект рендерит простую 3D-сцену (сферы, плоскость и OBJ mesh) с поддержкой:

- диффузных и зеркальных материалов;
- направленного освещения и теней;
- отражений (для зеркального материала);
- интерактивного управления камерой и параметрами сцены;
- отображения FPS и времени кадра/GPU.

Program input is represented as a scene description with analytic primitives
and polygonal OBJ meshes. The OBJ path supports vertices, normals, triangular
faces, and multiple materials through MTL files. MTL `Kd` values define diffuse
colors; `Ks`, `Ns`, `Ni`, and `d` provide specular color, roughness/glossiness,
index of refraction, and alpha values. Material names containing `mirror`,
`metal`, `glass`, or `dielectric` select the corresponding simplified material
model; other OBJ materials use diffuse shading.

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
- `Esc` - выход

## Структура проекта

- `src/app` - приложение, ввод, камера, сцена, материалы
- `src/gpu` - OptiX renderer, pipeline, SBT, device-программы
- `src/common` - общие структуры данных host/device
- `assets/meshes/demo.obj` and `assets/meshes/demo.mtl` - demo OBJ mesh with diffuse and mirror materials
- `assets/scenes/demo_scene.json` - documented JSON scene input example

## Текущий статус

Проект находится на финальном техническом этапе подготовки курсовой работы по компьютерной графике. Реализованы RTX-ядро, интерактивная сцена, CPU/GPU-тесты, Docker Compose-проверка, CI workflow и комплект проектной документации.
