from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[1]

REQUIRED_PATHS = [
    "README.md",
    "RayTracerRTX/README.md",
    "RayTracerRTX/RayTracerRTX.vcxproj",
    "RayTracerRTX/src/app/application.cpp",
    "RayTracerRTX/src/app/camera.cpp",
    "RayTracerRTX/src/app/mesh.cpp",
    "RayTracerRTX/src/app/mesh.h",
    "RayTracerRTX/src/app/obj_loader.cpp",
    "RayTracerRTX/src/app/obj_loader.h",
    "RayTracerRTX/src/app/scene_config.cpp",
    "RayTracerRTX/src/app/scene_config.h",
    "RayTracerRTX/src/app/scene.cpp",
    "RayTracerRTX/src/gpu/optix_renderer.cpp",
    "RayTracerRTX/src/gpu/optix_device_programs.h",
    "RayTracerRTX/src/common/rtx_shared.h",
    "RayTracerRTX/assets/meshes/demo.obj",
    "RayTracerRTX/assets/meshes/demo.mtl",
    "RayTracerRTX/assets/scenes/demo_scene.json",
    "RayTracerRTX/tests/test_scene_camera.cpp",
    "RayTracerRTX/docs/coursework-checklist.md",
    "RayTracerRTX/docs/docker-check.md",
]

REQUIRED_DOC_MARKERS = [
    "RayTracerRTX",
    "CUDA",
    "OptiX",
    "GLFW",
    "OBJ mesh",
]


def fail(message: str) -> None:
    print(f"[FAIL] {message}")
    sys.exit(1)


def main() -> None:
    missing = [path for path in REQUIRED_PATHS if not (ROOT / path).exists()]
    if missing:
        fail("Missing required project files: " + ", ".join(missing))

    readme = (ROOT / "RayTracerRTX/README.md").read_text(encoding="utf-8", errors="ignore")
    absent_markers = [marker for marker in REQUIRED_DOC_MARKERS if marker not in readme]
    if absent_markers:
        fail("README misses required markers: " + ", ".join(absent_markers))

    device_program = (ROOT / "RayTracerRTX/src/gpu/optix_device_programs.h").read_text(encoding="utf-8", errors="ignore")
    for symbol in ["__raygen__rg", "__closesthit__radiance", "traceShadow", "MaterialMirror"]:
        if symbol not in device_program:
            fail(f"Device program misses symbol: {symbol}")

    demo_obj = (ROOT / "RayTracerRTX/assets/meshes/demo.obj").read_text(encoding="utf-8", errors="ignore")
    demo_mtl = (ROOT / "RayTracerRTX/assets/meshes/demo.mtl").read_text(encoding="utf-8", errors="ignore")
    for marker in ["mtllib demo.mtl", "usemtl mat_white_diffuse", "usemtl mat_green_diffuse", "usemtl mat_mirror"]:
        if marker not in demo_obj:
            fail(f"Demo OBJ misses marker: {marker}")
    for marker in ["newmtl mat_white_diffuse", "newmtl mat_green_diffuse", "newmtl mat_mirror", "Kd"]:
        if marker not in demo_mtl:
            fail(f"Demo MTL misses marker: {marker}")

    docker_doc = (ROOT / "RayTracerRTX/docs/docker-check.md").read_text(encoding="utf-8", errors="ignore")
    if "OBJ mesh" not in docker_doc:
        fail("Docker documentation must mention OBJ mesh checks.")

    print("[PASS] Coursework structure, OBJ assets, and documentation checks passed.")


if __name__ == "__main__":
    main()
