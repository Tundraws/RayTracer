#include "../src/app/camera.h"
#include "../src/app/material.h"
#include "../src/app/scene.h"
#if defined(RAYTRACERRTX_ENABLE_GPU_TESTS)
#include "../src/gpu/optix_renderer.h"
#endif

#include "test_framework.h"

#include <chrono>
#include <cmath>
#include <iomanip>
#include <iostream>
#include <string>
#include <vector>

namespace
{
float length3(const float3& v)
{
    return std::sqrt(v.x * v.x + v.y * v.y + v.z * v.z);
}

float dot3(const float3& a, const float3& b)
{
    return a.x * b.x + a.y * b.y + a.z * b.z;
}

void testDefaultScene(TestContext& t)
{
    const SceneState scene = makeDefaultScene();
    t.expect(scene.spheres.size() == 3, "Default scene must contain 3 spheres.");
    t.expect(scene.materials.size() == 3, "Default scene must contain 3 materials.");
    t.expect(scene.selectedSphere == 0, "Default selected sphere should be index 0.");
    t.expect(scene.lightType == LightPoint, "Default light should be point light.");
    t.expect(scene.lightRadius > 0.0f, "Default area light radius should be positive.");
}

void testToggleMaterial(TestContext& t)
{
    SceneState scene = makeDefaultScene();
    scene.selectedSphere = 1;
    const int initial = scene.materials[1].materialType;
    toggleSelectedMaterial(scene);
    t.expect(scene.materials[1].materialType != initial, "Material toggle should change type.");
    toggleSelectedMaterial(scene);
    t.expect(scene.materials[1].materialType == initial, "Material toggle should return to initial type.");
}

void testMoveSphereClamp(TestContext& t)
{
    SceneState scene = makeDefaultScene();
    scene.selectedSphere = 0;
    const float radius = scene.spheres[0].radius;
    moveSelectedSphere(scene, make_float3(200.0f, -200.0f, 200.0f));
    const float3 center = scene.spheres[0].center;

    t.expect(center.x <= 24.0f, "Sphere X should be clamped by scene bounds.");
    t.expect(center.z <= 24.0f, "Sphere Z should be clamped by scene bounds.");
    t.expect(center.y >= radius, "Sphere Y should stay above floor with radius offset.");
}

void testMoveLightClamp(TestContext& t)
{
    SceneState scene = makeDefaultScene();
    moveLight(scene, make_float3(500.0f, -500.0f, -500.0f));
    t.expect(scene.lightPosition.x <= 40.0f, "Light X should be clamped.");
    t.expect(scene.lightPosition.y >= 6.0f, "Light Y should be clamped.");
    t.expect(scene.lightPosition.z >= -40.0f, "Light Z should be clamped.");
}

void testLightTypeAndRadius(TestContext& t)
{
    SceneState scene = makeDefaultScene();
    toggleLightType(scene);
    t.expect(scene.lightType == LightArea, "Light type should switch to area.");
    toggleLightType(scene);
    t.expect(scene.lightType == LightPoint, "Light type should switch back to point.");

    changeLightRadius(scene, 100.0f);
    t.expect(scene.lightRadius <= 12.0f, "Area light radius should clamp to upper bound.");
    changeLightRadius(scene, -100.0f);
    t.expect(scene.lightRadius >= 0.5f, "Area light radius should clamp to lower bound.");
}

void testClampSceneSelectedSphereBounds(TestContext& t)
{
    SceneState scene = makeDefaultScene();
    scene.selectedSphere = -100;
    clampScene(scene);
    t.expect(scene.selectedSphere == 0, "Selected sphere must clamp to lower bound.");

    scene.selectedSphere = 999;
    clampScene(scene);
    t.expect(scene.selectedSphere == static_cast<int>(scene.spheres.size()) - 1, "Selected sphere must clamp to upper bound.");
}

void testInvalidSelectedSphereOps(TestContext& t)
{
    SceneState scene = makeDefaultScene();
    const float3 centerBefore = scene.spheres[0].center;
    const int materialBefore = scene.materials[0].materialType;

    scene.selectedSphere = -1;
    moveSelectedSphere(scene, make_float3(1.0f, 1.0f, 1.0f));
    toggleSelectedMaterial(scene);
    t.expect(almostEqual(scene.spheres[0].center.x, centerBefore.x), "Invalid selected sphere must not move geometry.");
    t.expect(scene.materials[0].materialType == materialBefore, "Invalid selected sphere must not toggle material.");

    scene.selectedSphere = 100;
    moveSelectedSphere(scene, make_float3(1.0f, 1.0f, 1.0f));
    toggleSelectedMaterial(scene);
    t.expect(almostEqual(scene.spheres[0].center.x, centerBefore.x), "Out-of-range sphere index must not move geometry.");
    t.expect(scene.materials[0].materialType == materialBefore, "Out-of-range sphere index must not toggle material.");
}

void testAllSpheresStayAboveFloorAfterClamp(TestContext& t)
{
    SceneState scene = makeDefaultScene();
    for (size_t i = 0; i < scene.spheres.size(); ++i)
    {
        scene.spheres[i].center = make_float3(0.0f, -100.0f, 0.0f);
    }

    clampScene(scene);

    for (size_t i = 0; i < scene.spheres.size(); ++i)
    {
        const SphereGeometry& sphere = scene.spheres[i];
        t.expect(sphere.center.y >= sphere.radius, "Sphere center Y must stay above floor + radius.");
    }
}

void testCameraBasis(TestContext& t)
{
    CameraState camera;
    float3 forward{};
    float3 right{};
    float3 up{};
    float scale = 0.0f;
    float aspect = 0.0f;

    updateCameraBasis(camera, 1280, 720, forward, right, up, scale, aspect);

    t.expect(almostEqual(length3(forward), 1.0f, 1e-3f), "Forward vector must be normalized.");
    t.expect(almostEqual(length3(right), 1.0f, 1e-3f), "Right vector must be normalized.");
    t.expect(almostEqual(length3(up), 1.0f, 1e-3f), "Up vector must be normalized.");

    t.expect(std::fabs(dot3(forward, right)) < 1e-3f, "Forward and right must be orthogonal.");
    t.expect(std::fabs(dot3(forward, up)) < 1e-3f, "Forward and up must be orthogonal.");
    t.expect(std::fabs(dot3(right, up)) < 1e-3f, "Right and up must be orthogonal.");

    t.expect(almostEqual(aspect, 1280.0f / 720.0f, 1e-6f), "Aspect ratio should match viewport.");
    t.expect(scale > 0.0f, "Camera scale must be positive.");
}

void testCameraAspectFallback(TestContext& t)
{
    CameraState camera;
    float3 forward{};
    float3 right{};
    float3 up{};
    float scale = 0.0f;
    float aspect = 0.0f;

    updateCameraBasis(camera, 1920, 0, forward, right, up, scale, aspect);
    t.expect(almostEqual(aspect, 1.0f, 1e-6f), "Aspect ratio should fallback to 1.0 when height is zero.");
    t.expect(scale > 0.0f, "Camera scale must remain valid for zero-height fallback.");
}

void testCameraScaleIncreasesWithFov(TestContext& t)
{
    CameraState cameraNarrow;
    cameraNarrow.fov = 30.0f;
    CameraState cameraWide;
    cameraWide.fov = 90.0f;

    float3 forward{};
    float3 right{};
    float3 up{};
    float scaleNarrow = 0.0f;
    float aspect = 0.0f;
    updateCameraBasis(cameraNarrow, 1280, 720, forward, right, up, scaleNarrow, aspect);

    float scaleWide = 0.0f;
    updateCameraBasis(cameraWide, 1280, 720, forward, right, up, scaleWide, aspect);

    t.expect(scaleWide > scaleNarrow, "Camera scale should increase with larger FOV.");
}

bool runGpuSmokeTest(TestContext& t)
{
#if !defined(RAYTRACERRTX_ENABLE_GPU_TESTS)
    (void)t;
    std::cout << "[SKIP] GPU smoke test skipped: RAYTRACERRTX_ENABLE_GPU_TESTS is not enabled.\n";
    return false;
#else
    try
    {
        OptixRenderer renderer;
        renderer.setRenderSize(64, 64);
        renderer.initialize();

        SceneState scene = makeDefaultScene();
        CameraState camera;
        std::vector<uchar4> pixels(64u * 64u);
        float gpuTimeMs = -1.0f;

        renderer.renderFrame(scene, camera, pixels, &gpuTimeMs);
        renderer.destroy();

        bool hasNonZeroPixel = false;
        for (const uchar4 px : pixels)
        {
            if (px.x != 0u || px.y != 0u || px.z != 0u || px.w != 0u)
            {
                hasNonZeroPixel = true;
                break;
            }
        }

        t.expect(hasNonZeroPixel, "GPU smoke: rendered frame must contain non-zero pixels.");
        t.expect(gpuTimeMs >= 0.0f, "GPU smoke: GPU time must be non-negative.");
        return true;
    }
    catch (const std::exception& ex)
    {
        std::cout << "[SKIP] GPU smoke test skipped: " << ex.what() << '\n';
        return false;
    }
#endif
}

int runGpuBenchmark()
{
#if !defined(RAYTRACERRTX_ENABLE_GPU_TESTS)
    std::cout << "GPU benchmark is not available: RAYTRACERRTX_ENABLE_GPU_TESTS is not enabled.\n";
    return 1;
#else
    struct Scenario
    {
        const char* name;
        int width;
        int height;
        int frames;
    };

    const Scenario scenarios[] = {
        {"Low", 640, 360, 30},
        {"HD", 1280, 720, 30},
        {"Full HD", 1920, 1080, 20},
    };

    std::cout << "| Scenario | Resolution | FPS | Avg frame ms | Avg GPU ms |\n";
    std::cout << "|---|---:|---:|---:|---:|\n";

    for (const Scenario& scenario : scenarios)
    {
        OptixRenderer renderer;
        renderer.setRenderSize(scenario.width, scenario.height);
        renderer.initialize();

        SceneState scene = makeDefaultScene();
        CameraState camera;
        std::vector<uchar4> pixels(static_cast<size_t>(scenario.width) * static_cast<size_t>(scenario.height));

        float warmupGpuTimeMs = 0.0f;
        renderer.renderFrame(scene, camera, pixels, &warmupGpuTimeMs);

        double hostTotalMs = 0.0;
        double gpuTotalMs = 0.0;
        for (int frame = 0; frame < scenario.frames; ++frame)
        {
            float gpuTimeMs = 0.0f;
            const auto start = std::chrono::steady_clock::now();
            renderer.renderFrame(scene, camera, pixels, &gpuTimeMs);
            const auto stop = std::chrono::steady_clock::now();

            hostTotalMs += std::chrono::duration<double, std::milli>(stop - start).count();
            gpuTotalMs += static_cast<double>(gpuTimeMs);
        }

        renderer.destroy();

        const double avgHostMs = hostTotalMs / static_cast<double>(scenario.frames);
        const double avgGpuMs = gpuTotalMs / static_cast<double>(scenario.frames);
        const double fps = avgHostMs > 0.0 ? 1000.0 / avgHostMs : 0.0;

        std::cout << "| " << scenario.name
                  << " | " << scenario.width << "x" << scenario.height
                  << " | " << std::fixed << std::setprecision(2) << fps
                  << " | " << avgHostMs
                  << " | " << avgGpuMs
                  << " |\n";
    }

    return 0;
#endif
}
} // namespace

int main(int argc, char** argv)
{
    if (argc > 1 && std::string(argv[1]) == "--benchmark")
    {
        return runGpuBenchmark();
    }

    TestContext t;
    int testsRun = 0;
    int testsFailed = 0;
    int testsSkipped = 0;

    const auto runTest = [&](const char* name, void (*fn)(TestContext&))
    {
        const int failuresBefore = t.failures;
        const int checksBefore = t.checks;
        ++testsRun;
        fn(t);
        const int checksDelta = t.checks - checksBefore;
        if (t.failures == failuresBefore)
        {
            std::cout << "[PASS] " << name << " (checks: " << checksDelta << ")\n";
        }
        else
        {
            ++testsFailed;
            std::cout << "[FAIL] " << name << " (new failures: " << (t.failures - failuresBefore) << ")\n";
        }
    };

    runTest("Default scene", testDefaultScene);
    runTest("Toggle material", testToggleMaterial);
    runTest("Move sphere clamp", testMoveSphereClamp);
    runTest("Move light clamp", testMoveLightClamp);
    runTest("Light type and radius", testLightTypeAndRadius);
    runTest("Clamp selected sphere index", testClampSceneSelectedSphereBounds);
    runTest("Invalid selected sphere operations", testInvalidSelectedSphereOps);
    runTest("All spheres above floor after clamp", testAllSpheresStayAboveFloorAfterClamp);
    runTest("Camera basis", testCameraBasis);
    runTest("Camera aspect fallback", testCameraAspectFallback);
    runTest("Camera scale vs FOV", testCameraScaleIncreasesWithFov);

    ++testsRun;
    if (runGpuSmokeTest(t))
    {
        std::cout << "[PASS] GPU smoke test (checks: 2)\n";
    }
    else
    {
        ++testsSkipped;
    }

    if (t.failures == 0)
    {
        std::cout << "\nAll tests passed. Tests: " << testsRun
                  << ", skipped: " << testsSkipped
                  << ", checks: " << t.checks << '\n';
        return 0;
    }

    std::cerr << "\nTests failed. Failed tests: " << testsFailed
              << ", failed checks: " << t.failures
              << " / total checks: " << t.checks << '\n';
    return 1;
}
