# RayTracerRTX

RayTracerRTX - интерактивный RTX-трассировщик лучей для курсовой работы.

Программа рендерит 3D-сцену в реальном времени. В сцене есть сферы, плоскость и загружаемые сеточные модели. Можно запускать стандартную демонстрационную сцену или передавать свои входные данные через OBJ/glTF-файл или JSON-конфиг сцены.

## Быстрый запуск

Из корня репозитория:

```powershell
.\x64\Debug\RayTracerRTX.exe
```

Запуск с OBJ:

```powershell
.\x64\Debug\RayTracerRTX.exe --mesh RayTracerRTX\assets\meshes\demo.obj
```

Запуск с готовой JSON-сценой:

```powershell
.\x64\Debug\RayTracerRTX.exe --scene RayTracerRTX\assets\scenes\textured_cube_scene.json
```

Другие полезные сцены:

```powershell
.\x64\Debug\RayTracerRTX.exe --scene RayTracerRTX\assets\scenes\multi_mesh_scene.json
.\x64\Debug\RayTracerRTX.exe --scene RayTracerRTX\assets\scenes\material_showcase_scene.json
.\x64\Debug\RayTracerRTX.exe --scene RayTracerRTX\assets\scenes\path_tracing_demo_scene.json
.\x64\Debug\RayTracerRTX.exe --scene RayTracerRTX\assets\scenes\gltf_scene.json
```

## Что можно показать на защите

- RTX/OptiX рендеринг в реальном времени.
- OBJ mesh с несколькими материалами.
- Текстурированный куб.
- Несколько mesh-объектов в одной сцене.
- Отдельную сцену с матовым, металлическим, стеклянным и зеркальным материалом.
- Простую демонстрационную glTF-модель.
- Переключение материалов прямо в программе.
- Переключение качества рендера.
- Режим накопления сэмплов для более плавного изображения.
- Дополнительный шумоподавитель OptiX.
- Настройку экспозиции, яркости неба и яркости света из JSON-сцены.
- Benchmark и тесты.

## Поддерживаемые входные данные

### OBJ/MTL

Поддерживается:

- вершины `v`;
- нормали `vn`;
- текстурные координаты `vt`;
- треугольные грани;
- несколько материалов через MTL;
- `Kd`, `Ks`, `Ns`, `Ni`, `d`;
- `map_Kd` для матовой цветовой текстуры;
- `bump`, `map_Bump`, `norm` для базовой карты нормалей.

### JSON scene config

JSON-сцена может задавать:

- путь к модели;
- camera;
- light;
- exposure;
- skyIntensity;
- lightIntensity;
- материалы через `materials`;
- назначение материалов сферам через `sphereMaterials`;
- назначение материала всему mesh-объекту через поле `material`.
- список объектов-сеток;
- положение, поворот и масштаб для объекта-сетки.

### glTF

glTF поддерживается в простом варианте для демонстрационного импорта:

- `.gltf` + внешний `.bin`;
- positions;
- normals;
- texcoords;
- indices;
- base color;
- металлическость;
- roughness;
- simple translation/scale.

Сложные glTF-возможности вроде `.glb`, анимаций, skinning и morph targets не поддерживаются.

## Управление

- `W/A/S/D` - движение камеры
- `Space` - движение вверх
- мышь - поворот камеры
- ЛКМ - освободить курсор для панели
- ПКМ - вернуть управление камерой
- `1/2/3` - выбрать сферу
- стрелки - двигать выбранную сферу
- `R/F` - двигать сферу вверх/вниз
- `J/L`, `I/K`, `U/O` - двигать свет
- `G` - переключить демонстрационный набор сцены/модели
- `C` - сбросить камеру и свет для текущей сцены
- `M` - сменить материал сферы
- `B` - выбрать следующий объект-сетку
- `V` - сменить материал сетки
- `P` - режим реального времени / режим накопления
- `Q` - режим качества
- `N` - шумоподавитель
- `H` - показать или скрыть ImGui-панель настроек
- `4/5` - уменьшить/увеличить экспозицию
- `6/7` - уменьшить/увеличить яркость неба
- `8/9` - уменьшить/увеличить яркость света
- `Esc` - выход

HUD показывает текущую сцену, выбранный объект, материал, режим рендера,
качество, шумоподавитель, FPS/GPU time и компактную панель настройки
экспозиции, неба и света. Справа находится ImGui-панель для выбора сцены,
объекта, материала, качества и настройки экспозиции, неба, света,
roughness/metallic.

## Сборка

Требуется:

- Visual Studio 2022;
- CUDA Toolkit 13.1;
- NVIDIA OptiX SDK 9.1;
- NVIDIA RTX GPU.

Откройте `RayTracerRTX.sln`, выберите `Debug|x64` или `Release|x64` и соберите проект.

Если OptiX установлен не в стандартную папку, путь можно переопределить через `OptixSdkDir`.

## Проверки

Сборка тестов:

```powershell
& "C:\Program Files\Microsoft Visual Studio\2022\Community\MSBuild\Current\Bin\MSBuild.exe" tests\RayTracerRTX.Tests.vcxproj /p:Configuration=Debug /p:Platform=x64 /m
```

Запуск тестов:

```powershell
tests\x64\Debug\RayTracerRTX.Tests.exe
```

Замер производительности:

```powershell
tests\x64\Debug\RayTracerRTX.Tests.exe --benchmark
```

Docker-check из корня репозитория:

```powershell
docker compose build --no-cache
docker compose run --rm coursework-check
```

## Структура

- `src/app` - приложение, сцена, камера, материалы, загрузчики
- `src/gpu` - рендерер OptiX и GPU-программы
- `src/common` - общие структуры для CPU и GPU
- `assets/meshes` - демонстрационные модели
- `assets/scenes` - JSON-сцены
- `tests` - тесты и замер производительности
- `docs` - подробная документация
