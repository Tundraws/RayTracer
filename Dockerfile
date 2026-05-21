FROM gcc:13-bookworm

WORKDIR /workspace

COPY . .

RUN test -f README.md \
    && test -f RayTracerRTX/README.md \
    && test -f RayTracerRTX/docs/coursework-checklist.md \
    && test -f RayTracerRTX/docs/architecture.md \
    && test -f RayTracerRTX/docs/api.md \
    && test -f RayTracerRTX/docs/dependency-security.md \
    && test -f RayTracerRTX/docs/ai-assisted-development.md \
    && test -f RayTracerRTX/src/gpu/optix_device_programs.h \
    && grep -q "OptiX" RayTracerRTX/README.md \
    && grep -q "__raygen__rg" RayTracerRTX/src/gpu/optix_device_programs.h

RUN g++ -std=c++20 -Wall -Wextra -pedantic \
    -IRayTracerRTX/tests/stubs \
    -IRayTracerRTX/src \
    RayTracerRTX/tests/test_scene_camera.cpp \
    RayTracerRTX/src/app/camera.cpp \
    RayTracerRTX/src/app/material.cpp \
    RayTracerRTX/src/app/mesh.cpp \
    RayTracerRTX/src/app/obj_loader.cpp \
    RayTracerRTX/src/app/scene.cpp \
    -o /usr/local/bin/raytracerrtx_cpu_tests

CMD ["/usr/local/bin/raytracerrtx_cpu_tests"]
