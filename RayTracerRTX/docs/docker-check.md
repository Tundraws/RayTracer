# Проверка Docker / Docker Compose

Дата: 2026-05-19.

## Назначение

Docker-конфигурация используется как воспроизводимая среда проверки курсового проекта. Проверка охватывает:

- структуру репозитория;
- наличие документации;
- демонстрационные OBJ/MTL-ресурсы;
- текстурированные OBJ-ресурсы, `map_Kd`, PPM-текстуры и PPM-карты нормалей;
- наличие `stb_image.h` для загрузки PNG/JPG;
- демонстрационную PPM-карту окружения;
- демонстрационные материалы;
- минимальные glTF-ресурсы `.gltf/.bin`;
- JSON-сцены;
- файлы загрузчиков OBJ и glTF;
- файлы логирования;
- CPU-only тесты для сцены, камеры, материалов, MTL, OBJ-загрузчика и glTF-загрузчика.

RTX GUI-приложение в Docker не запускается, поскольку зависит от Windows desktop session, GLFW-окна, NVIDIA OptiX, CUDA и драйвера RTX-видеокарты. GPU-выполнение проверяется отдельно нативным `RayTracerRTX.Tests.exe` через GPU smoke-тесты.

## Проверяемые файлы

- `Dockerfile`;
- `docker-compose.yml`;
- `RayTracerRTX/assets/meshes/demo.obj`;
- `RayTracerRTX/assets/meshes/demo.mtl`;
- `RayTracerRTX/assets/meshes/textured_demo.obj`;
- `RayTracerRTX/assets/meshes/textured_cube.obj`;
- `RayTracerRTX/assets/meshes/textured_demo.mtl`;
- `RayTracerRTX/assets/meshes/material_showcase.obj`;
- `RayTracerRTX/assets/meshes/material_showcase.mtl`;
- `RayTracerRTX/assets/meshes/minimal_gltf.gltf`;
- `RayTracerRTX/assets/meshes/minimal_gltf.bin`;
- `RayTracerRTX/assets/meshes/checker.ppm`;
- `RayTracerRTX/assets/meshes/checker_normal.ppm`;
- `RayTracerRTX/assets/meshes/studio_env.ppm`;
- `RayTracerRTX/assets/scenes/*.json`;
- `RayTracerRTX/src/app/obj_loader.*`;
- `RayTracerRTX/src/app/gltf_loader.*`;
- `RayTracerRTX/src/app/logger.*`;
- `RayTracerRTX/src/app/scene_config.*`;
- `RayTracerRTX/src/app/mesh.*`;
- `RayTracerRTX/tests/stubs/cuda_runtime.h`;
- `RayTracerRTX/tests/stubs/optix.h`.

## Команды

Проверка конфигурации Compose:

```powershell
docker compose config
```

Сборка и запуск проверки:

```powershell
docker compose build
docker compose run --rm coursework-check
```

Запуск через вспомогательный скрипт:

```powershell
powershell -ExecutionPolicy Bypass -File .\scripts\docker_coursework_build.ps1
```

Чистая пересборка Docker-образа:

```powershell
powershell -ExecutionPolicy Bypass -File .\scripts\docker_coursework_build.ps1 -NoCache
```

Если Docker сообщает `short read` или `unexpected EOF` при чтении слоя образа, причина обычно относится к повреждённому cache или прерванной загрузке. В этом случае можно очистить cache builder и повторить сборку:

```powershell
docker builder prune -f
docker compose build --no-cache
```

`Dockerfile` использует `gcc:13-bookworm` и не запускает `apt-get`. Базовый образ уже содержит `g++`, поэтому проверка не зависит от доступа к пакетным зеркалам Debian. Образ крупнее минимального runtime-образа, но подходит для воспроизводимой проверки исходников и CPU-тестов.

## Статус локальной проверки

`docker compose config` завершился успешно.

`docker compose build --no-cache` завершился успешно и создал образ:

```text
raytracerrtx-coursework-check:latest
```

Ожидаемый результат `docker compose run --rm coursework-check`:

```text
All tests passed. Tests: 106, skipped: 4
```

Пропущенные тесты относятся к GPU smoke-тестам. Это ожидаемо для Docker-среды, поскольку контейнер выполняет CPU-only путь. RTX/OptiX проверяется нативным Windows-приложением тестов.
