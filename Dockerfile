FROM gcc:13-bookworm

WORKDIR /workspace

COPY . .

RUN g++ -std=c++20 -Wall -Wextra -pedantic \
    -IRayTracerRTX/tests/stubs \
    -IRayTracerRTX/src \
    RayTracerRTX/tests/test_scene_camera.cpp \
    RayTracerRTX/src/app/camera.cpp \
    RayTracerRTX/src/app/material.cpp \
    RayTracerRTX/src/app/scene.cpp \
    -o /usr/local/bin/raytracerrtx_cpu_tests

CMD ["bash", "-lc", "python3 scripts/verify_coursework.py && /usr/local/bin/raytracerrtx_cpu_tests"]
