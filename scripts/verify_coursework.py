from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[1]

REQUIRED_PATHS = [
    "README.md",
    "RayTracerRTX/README.md",
    "RayTracerRTX/RayTracerRTX.vcxproj",
    "RayTracerRTX/src/app/application.cpp",
    "RayTracerRTX/src/app/camera.cpp",
    "RayTracerRTX/src/app/scene.cpp",
    "RayTracerRTX/src/gpu/optix_renderer.cpp",
    "RayTracerRTX/src/gpu/optix_device_programs.h",
    "RayTracerRTX/src/common/rtx_shared.h",
    "RayTracerRTX/tests/test_scene_camera.cpp",
    "RayTracerRTX/docs/coursework-checklist.md",
]

REQUIRED_DOC_MARKERS = [
    "RayTracerRTX",
    "CUDA",
    "OptiX",
    "GLFW",
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

    print("[PASS] Coursework structure and documentation checks passed.")


if __name__ == "__main__":
    main()
