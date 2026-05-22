# RayTracerRTX

RayTracerRTX - учебный трассировщик лучей в реальном времени на `C++`, `CUDA` и `NVIDIA OptiX`.

Проект показывает интерактивную 3D-сцену: сферы, плоскость и загружаемые 3D-модели. Камерой, светом, материалами и режимами качества можно управлять прямо во время работы программы.

## Что умеет программа

- Рендерит сцену через RTX/OptiX.
- Загружает OBJ-модели через `--mesh`.
- Загружает сцену из JSON через `--scene`.
- Поддерживает несколько объектов в сцене.
- Поддерживает материалы из MTL: матовый, зеркальный, металлический, стеклянный.
- Поддерживает простые матовые цветовые текстуры `map_Kd`.
- Поддерживает базовые карты нормалей для OBJ.
- Поддерживает простой импорт glTF для демонстрационных сцен.
- Имеет режим рендеринга в реальном времени и режим накопления сэмплов.
- Может использовать шумоподавитель OptiX в режиме накопления.
- Позволяет настраивать экспозицию, яркость неба и яркость света через JSON-сцену.
- Показывает FPS, время GPU и текущие режимы в HUD.

## Важно про ограничения

Это учебный рендерер для курсовой работы. Он поддерживает не все возможности OBJ, MTL и glTF, а только нужный для проекта ограниченный набор.

Например, glTF поддерживается только в простом варианте: `.gltf` + внешний `.bin`, геометрия, базовые материалы и простые преобразования объекта. `.glb`, анимации, скелетная анимация и сложные карты текстур не поддерживаются.

## Требования

- Windows 10/11 x64
- NVIDIA GPU с поддержкой RTX
- Visual Studio 2022
- CUDA Toolkit 13.1
- NVIDIA OptiX SDK 9.1

## Как собрать

1. Откройте `RayTracerRTX.sln` в Visual Studio 2022.
2. Выберите `Debug|x64` или `Release|x64`.
3. Проверьте, что CUDA и OptiX установлены.
4. Соберите проект.
5. Запустите `RayTracerRTX.exe`.

## Как запустить

Из корня репозитория:

```powershell
.\x64\Debug\RayTracerRTX.exe
```

Запуск с OBJ-моделью:

```powershell
.\x64\Debug\RayTracerRTX.exe --mesh RayTracerRTX\assets\meshes\demo.obj
```

Запуск с JSON-сценой:

```powershell
.\x64\Debug\RayTracerRTX.exe --scene RayTracerRTX\assets\scenes\textured_cube_scene.json
```

Другие демонстрационные сцены:

```powershell
.\x64\Debug\RayTracerRTX.exe --scene RayTracerRTX\assets\scenes\demo_scene.json
.\x64\Debug\RayTracerRTX.exe --scene RayTracerRTX\assets\scenes\multi_mesh_scene.json
.\x64\Debug\RayTracerRTX.exe --scene RayTracerRTX\assets\scenes\material_showcase_scene.json
.\x64\Debug\RayTracerRTX.exe --scene RayTracerRTX\assets\scenes\path_tracing_demo_scene.json
.\x64\Debug\RayTracerRTX.exe --scene RayTracerRTX\assets\scenes\gltf_scene.json
```

Если ничего не передавать, программа запустит стандартную демонстрационную сцену.

В JSON-сцене можно задавать камеру, свет, размер area light через `size`/`radius`,
environment map в PPM, список моделей, простые настройки изображения
(`exposure`, `skyIntensity`, `lightIntensity`, `environmentIntensity`) и материалы:
`matte`, `mirror`, `metal`, `glass`. Для материала поддерживаются
`baseColor`, `roughness`, `metallic`, `specularColor`, `ior`, `alpha`, а также
пути `texture`/`map_Kd` и `normalMap`.

## Управление

- `W/A/S/D` - движение камеры
- `Space` - движение камеры вверх
- мышь - поворот камеры
- ЛКМ - переключить курсор/управление камерой; если курсор уже свободен, ЛКМ по панели нажимает кнопки и двигает слайдеры
- `1/2/3` - выбрать сферу
- стрелки - двигать выбранную сферу
- `R/F` - двигать сферу вверх/вниз
- `J/L`, `I/K`, `U/O` - двигать источник света
- `G` - переключить демонстрационную сцену или модель
- `C` - сбросить камеру и свет для текущей демонстрационной сцены
- `M` - сменить материал выбранной сферы
- `B` - выбрать следующий объект-сетку
- `V` - сменить материал выбранной сетки
- `P` - переключить режим реального времени / режим накопления
- `Q` - переключить качество: низкое / среднее / высокое / режим накопления
- `N` - включить или выключить шумоподавитель
- `H` - показать или скрыть ImGui-панель настроек
- `4/5` - уменьшить/увеличить экспозицию
- `6/7` - уменьшить/увеличить яркость неба
- `8/9` - уменьшить/увеличить яркость света
- `Esc` - выход

HUD в окне показывает текущую сцену, выбранный объект, материал, режим рендера,
качество, шумоподавитель, FPS/GPU time и компактную панель настройки
экспозиции, неба и света. Справа есть ImGui-панель: в ней можно мышкой выбрать
сцену, модель, сферу, положение/поворот/масштаб модели, тип материала модели
или сферы, качество, добавить или удалить сферу, изменить ее размер и цвет, а
также подкрутить экспозицию, небо, свет, шероховатость, прозрачность и
показатель преломления. Для текстурированных моделей можно временно отключить
текстуру, чтобы цвет и тип материала были заметнее. В этой же панели есть
кнопка `Загрузить модель...`: она открывает стандартное окно выбора файла
Windows и позволяет добавить в демонстрацию `.obj` или `.gltf` модель без
перезапуска программы.

## Где лежат важные файлы

- `RayTracerRTX/src/app` - приложение, сцена, камера, загрузчики OBJ/glTF
- `RayTracerRTX/src/gpu` - рендерер OptiX/CUDA
- `RayTracerRTX/src/common` - общие структуры данных
- `RayTracerRTX/assets/meshes` - демонстрационные модели и текстуры
- `RayTracerRTX/assets/scenes` - демонстрационные JSON-сцены
- `RayTracerRTX/tests` - тесты и замер производительности
- `RayTracerRTX/docs` - подробная документация

## Проверки

Сборка:

```powershell
& "C:\Program Files\Microsoft Visual Studio\2022\Community\MSBuild\Current\Bin\MSBuild.exe" RayTracerRTX.sln /p:Configuration=Debug /p:Platform=x64 /m
```

Тесты:

```powershell
& "C:\Program Files\Microsoft Visual Studio\2022\Community\MSBuild\Current\Bin\MSBuild.exe" RayTracerRTX\tests\RayTracerRTX.Tests.vcxproj /p:Configuration=Debug /p:Platform=x64 /m
RayTracerRTX\tests\x64\Debug\RayTracerRTX.Tests.exe
```

Замер производительности:

```powershell
RayTracerRTX\tests\x64\Debug\RayTracerRTX.Tests.exe --benchmark
```

Docker-check:

```powershell
docker compose build --no-cache
docker compose run --rm coursework-check
```
