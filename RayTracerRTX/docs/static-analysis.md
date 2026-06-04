# Отчёт о статическом анализе

Дата: 2026-05-19.

## Инструмент

- Основной инструмент: `clang-tidy`.
- Версия: LLVM 19.1.5.
- Источник: LLVM tools for Visual Studio.
- Статус `cppcheck`: не установлен в текущем окружении.

## Область проверки

Анализ охватывает CPU-часть проекта, которую можно проверить без открытия окна GLFW и запуска OptiX-рендерера:

- `RayTracerRTX/src/app/camera.cpp`;
- `RayTracerRTX/src/app/material.cpp`;
- `RayTracerRTX/src/app/scene.cpp`.

GPU/OptiX-код проверяется регулярной сборкой и GPU smoke-тестами. Полноценный статический анализ NVRTC-компилируемых GPU-программ OptiX требует отдельной CUDA-aware конфигурации инструмента.

## Команда запуска

```powershell
& "C:\Program Files\Microsoft Visual Studio\2022\Community\VC\Tools\Llvm\x64\bin\clang-tidy.exe" `
  RayTracerRTX\src\app\camera.cpp `
  RayTracerRTX\src\app\material.cpp `
  RayTracerRTX\src\app\scene.cpp `
  --config="{Checks: 'clang-analyzer-*,bugprone-*,performance-*,-bugprone-easily-swappable-parameters'}" `
  --extra-arg=-std=c++20 `
  --extra-arg=-IC:\Users\User\source\repos\RayTracerRTX\RayTracerRTX\tests\stubs `
  --extra-arg=-IC:\Users\User\source\repos\RayTracerRTX\RayTracerRTX\src
```

## Результат

Код завершения: `0`.

Итог:

- ошибок уровня проекта не найдено;
- предупреждений, требующих исправления в коде проекта, не найдено;
- `clang-tidy` вывел подавленные предупреждения из системных и внешних заголовков;
- проверка `bugprone-easily-swappable-parameters` отключена, поскольку в математических функциях векторов естественно используются соседние параметры одного типа.

## Дальнейшие действия

Для финальной CI-проверки можно добавить один из вариантов:

- `clang-tidy` с созданным `compile_commands.json`;
- `cppcheck` в Docker/CI-образе.
