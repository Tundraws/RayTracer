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
диффузные.

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
- `Esc` — выход

## Структура проекта

- `src/app` — окно, цикл приложения, ввод, камера, сцена, материалы
- `src/gpu` — OptiX-рендерер, pipeline, SBT, device-программы
- `src/common` — общие структуры данных host/device
- `tests` — unit-тесты и тестовый проект
- `assets/meshes/demo.obj` and `assets/meshes/demo.mtl` - demo OBJ mesh with diffuse and mirror materials
- `assets/scenes/demo_scene.json` - documented JSON scene input example

## Статус

Проект разрабатывается как курсовая работа по компьютерной графике (вариант 42). Реализованы рабочее RTX-ядро, интерактивная сцена, CPU/GPU-тесты, Docker Compose-проверка, CI workflow и проектная документация; выполняется финальная подготовка к защите.

## Build path configuration

CUDA, OptiX, and source include paths are configured through Visual Studio
project properties instead of hard-coded C++ strings. Override `OptixSdkDir`
in a user property sheet or from MSBuild with `/p:OptixSdkDir=...` when the
OptiX SDK is installed outside the default directory.
