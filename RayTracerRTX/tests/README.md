# Тесты

Unit-тесты находятся в каталоге `tests` и собираются как отдельное консольное приложение.

Текущее покрытие:

- значения сцены по умолчанию и ограничение координат;
- переключение материалов;
- корректность базиса камеры (нормализация, ортогональность, aspect);
- опциональный GPU smoke-тест (рендер одного кадра через OptiX).

Готовый проект:

- `tests/RayTracerRTX.Tests.vcxproj`

Запуск в Visual Studio:

1. Откройте `tests/RayTracerRTX.Tests.vcxproj`.
2. Выберите конфигурацию `Debug|x64`.
3. Соберите проект (`Ctrl+Shift+B`).
4. Запустите без отладки (`Ctrl+F5`).

Ожидаемый вывод:

- строка `All tests passed...`;
- поимённые строки `[PASS]` для каждого теста;
- при необходимости строка `[SKIP] GPU smoke test ...`, если среда не готова.

## Как включить GPU smoke-тест

По умолчанию тестовый проект CPU-only и не линкует CUDA/OptiX.

Чтобы включить GPU smoke-тест:

1. В свойствах проекта добавьте define:
   - `RAYTRACERRTX_ENABLE_GPU_TESTS`
2. Добавьте исходник:
   - `..\src\gpu\optix_renderer.cpp`
3. Добавьте include-директории:
   - `$(ProjectDir)..\glfw\include`
   - путь к `CUDA\include`
   - путь к `OptiX SDK\include`
4. Добавьте зависимости линковки:
   - `cuda.lib;cudart.lib;nvrtc.lib;opengl32.lib;glfw3.lib`
5. Добавьте директорию библиотек:
   - `$(ProjectDir)..\glfw\lib`

## GPU smoke test status

The GPU smoke test is now enabled in `RayTracerRTX.Tests.vcxproj` by default.
It defines `RAYTRACERRTX_ENABLE_GPU_TESTS`, links CUDA runtime libraries,
compiles `src/gpu/optix_renderer.cpp`, initializes OptiX, renders one
64x64 frame, checks that the framebuffer is non-empty, and validates that
GPU frame time is non-negative.
