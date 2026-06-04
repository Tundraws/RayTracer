# Тестовое приложение

Тесты находятся в каталоге `tests` и собираются как отдельное консольное приложение `RayTracerRTX.Tests`.

## Состав проверок

Тестовый набор проверяет:

- значения сцены по умолчанию;
- переключение материалов;
- ограничения координат и параметров;
- базис камеры;
- загрузку OBJ/MTL;
- загрузку glTF/GLB;
- загрузку текстур;
- JSON-сцены;
- кэш ресурсов;
- редактор сцены;
- расчёты пересечений луча со сферой и треугольником;
- GPU smoke-тесты OptiX-рендерера.

## Сборка

Файл проекта:

```text
tests/RayTracerRTX.Tests.vcxproj
```

Сборка в Visual Studio:

1. Открыть `tests/RayTracerRTX.Tests.sln`.
2. Выбрать конфигурацию `Debug|x64`.
3. Собрать проект.
4. Запустить без отладки.

Сборка из PowerShell:

```powershell
& "C:\Program Files\Microsoft Visual Studio\2022\Community\MSBuild\Current\Bin\MSBuild.exe" RayTracerRTX\tests\RayTracerRTX.Tests.sln /m /p:Configuration=Debug /p:Platform=x64
```

## Запуск

```powershell
.\RayTracerRTX\tests\x64\Debug\RayTracerRTX.Tests.exe
```

Ожидаемый итог:

```text
All tests passed. Tests: 166, skipped: 0, checks: 810
```

Для каждой проверки выводится строка `[PASS]`. При ошибке выводится `[FAIL]` и сообщение из `TestContext::expect`.

## GPU smoke-тесты

GPU smoke-тесты включены в `RayTracerRTX.Tests.vcxproj` по умолчанию. Проект задаёт макрос `RAYTRACERRTX_ENABLE_GPU_TESTS`, компилирует `src/gpu/optix_renderer.cpp`, подключает CUDA runtime libraries и запускает проверки OptiX-рендерера.

GPU-проверки выполняют:

- инициализацию `OptixRenderer`;
- рендер одного кадра 64x64;
- проверку ненулевого буфера пикселей;
- проверку неотрицательного GPU time;
- прогрессивное накопление;
- динамическое изменение сферы и полигональной модели;
- включение шумоподавителя OptiX.

Для запуска GPU smoke-тестов требуется видеокарта NVIDIA с поддержкой используемых версий CUDA и OptiX.
