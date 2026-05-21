FROM gcc:13-bookworm

WORKDIR /workspace

COPY . .

RUN test -f README.md \
    && test -f RayTracerRTX/README.md \
    && test -f RayTracerRTX/docs/coursework-checklist.md \
    && test -f RayTracerRTX/docs/docker-check.md \
    && test -f RayTracerRTX/docs/architecture.md \
    && test -f RayTracerRTX/docs/api.md \
    && test -f RayTracerRTX/docs/dependency-security.md \
    && test -f RayTracerRTX/docs/ai-assisted-development.md \
    && test -f RayTracerRTX/assets/meshes/demo.obj \
    && test -f RayTracerRTX/assets/meshes/demo.mtl \
    && test -f RayTracerRTX/assets/meshes/textured_demo.obj \
    && test -f RayTracerRTX/assets/meshes/textured_cube.obj \
    && test -f RayTracerRTX/assets/meshes/textured_demo.mtl \
    && test -f RayTracerRTX/assets/meshes/minimal_gltf.gltf \
    && test -f RayTracerRTX/assets/meshes/minimal_gltf.bin \
    && test -f RayTracerRTX/assets/meshes/checker.ppm \
    && test -f RayTracerRTX/assets/meshes/checker_normal.ppm \
    && test -f RayTracerRTX/assets/scenes/demo_scene.json \
    && test -f RayTracerRTX/assets/scenes/textured_scene.json \
    && test -f RayTracerRTX/assets/scenes/textured_cube_scene.json \
    && test -f RayTracerRTX/assets/scenes/multi_mesh_scene.json \
    && test -f RayTracerRTX/assets/scenes/gltf_scene.json \
    && test -f RayTracerRTX/src/app/mesh.cpp \
    && test -f RayTracerRTX/src/app/mesh.h \
    && test -f RayTracerRTX/src/app/gltf_loader.cpp \
    && test -f RayTracerRTX/src/app/gltf_loader.h \
    && test -f RayTracerRTX/src/app/obj_loader.cpp \
    && test -f RayTracerRTX/src/app/obj_loader.h \
    && test -f RayTracerRTX/src/app/scene_config.cpp \
    && test -f RayTracerRTX/src/app/scene_config.h \
    && test -f RayTracerRTX/src/gpu/optix_device_programs.h \
    && grep -q "OBJ mesh" RayTracerRTX/README.md \
    && grep -q "OBJ mesh" RayTracerRTX/docs/docker-check.md \
    && grep -q "usemtl mat_mirror" RayTracerRTX/assets/meshes/demo.obj \
    && grep -q "newmtl mat_green_diffuse" RayTracerRTX/assets/meshes/demo.mtl \
    && grep -q "map_Kd checker.ppm" RayTracerRTX/assets/meshes/textured_demo.mtl \
    && grep -q "bump checker_normal.ppm" RayTracerRTX/assets/meshes/textured_demo.mtl \
    && grep -q "vt " RayTracerRTX/assets/meshes/textured_demo.obj \
    && grep -q "TexturedCourseworkCube" RayTracerRTX/assets/meshes/textured_cube.obj \
    && grep -q "MinimalGltfTriangle" RayTracerRTX/assets/meshes/minimal_gltf.gltf \
    && grep -q "OptiX" RayTracerRTX/README.md \
    && grep -q "__raygen__rg" RayTracerRTX/src/gpu/optix_device_programs.h

RUN g++ -std=c++20 -Wall -Wextra -pedantic \
    -IRayTracerRTX/tests/stubs \
    -IRayTracerRTX/src \
    RayTracerRTX/tests/test_scene_camera.cpp \
    RayTracerRTX/src/app/camera.cpp \
    RayTracerRTX/src/app/gltf_loader.cpp \
    RayTracerRTX/src/app/material.cpp \
    RayTracerRTX/src/app/mesh.cpp \
    RayTracerRTX/src/app/obj_loader.cpp \
    RayTracerRTX/src/app/scene.cpp \
    RayTracerRTX/src/app/scene_config.cpp \
    -o /usr/local/bin/raytracerrtx_cpu_tests

CMD ["/usr/local/bin/raytracerrtx_cpu_tests"]
