#include "../src/app/camera.h"
#include "../src/app/material.h"
#include "../src/app/obj_loader.h"
#include "../src/app/scene.h"
#include "../src/app/scene_config.h"
#if defined(RAYTRACERRTX_ENABLE_GPU_TESTS)
#include "../src/gpu/optix_renderer.h"
#endif

#include "test_framework.h"

#include <chrono>
#include <cmath>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <string>
#include <vector>

#ifndef RAYTRACERRTX_SOURCE_DIR
#define RAYTRACERRTX_SOURCE_DIR ""
#endif

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

std::filesystem::path writeFixtureFile(const std::string& name, const std::string& content)
{
    const std::filesystem::path directory = std::filesystem::temp_directory_path() / "raytracerrtx_obj_loader_tests";
    std::filesystem::create_directories(directory);

    const std::filesystem::path path = directory / name;
    std::ofstream file(path, std::ios::binary);
    file << content;
    return path;
}

std::filesystem::path findDemoObjAsset()
{
    const std::filesystem::path sourceDir = RAYTRACERRTX_SOURCE_DIR;
    const std::filesystem::path fromSource = sourceDir.empty()
        ? std::filesystem::path{}
        : sourceDir.parent_path() / "assets" / "meshes" / "demo.obj";

    const std::filesystem::path candidates[] = {
        fromSource,
        std::filesystem::path("RayTracerRTX") / "assets" / "meshes" / "demo.obj",
        std::filesystem::path("assets") / "meshes" / "demo.obj",
        std::filesystem::path("..") / "assets" / "meshes" / "demo.obj"
    };

    for (const std::filesystem::path& candidate : candidates)
    {
        if (!candidate.empty() && std::filesystem::exists(candidate))
        {
            return candidate;
        }
    }

    return {};
}

void testObjLoaderTriangleWithNormals(TestContext& t)
{
    const std::filesystem::path objPath = writeFixtureFile(
        "triangle_normals.obj",
        "mtllib triangle_normals.mtl\n"
        "v 0 0 0\n"
        "v 1 0 0\n"
        "v 0 1 0\n"
        "vn 0 0 1\n"
        "usemtl white\n"
        "f 1//1 2//1 3//1\n");

    writeFixtureFile(
        "triangle_normals.mtl",
        "newmtl white\n"
        "Kd 0.9 0.8 0.7\n");

    const ObjLoadResult result = loadObjMesh(objPath);
    t.expect(result.ok, "OBJ with vertices, normals, and one material should load.");
    t.expect(result.mesh.vertices.size() == 3, "Loaded triangle should create 3 packed vertices.");
    t.expect(result.mesh.triangles.size() == 1, "Loaded OBJ should contain one triangle.");
    t.expect(result.mesh.materials.size() == 2, "Default and MTL material should be available.");
    t.expect(almostEqual(result.mesh.vertices[0].normal.z, 1.0f), "OBJ normal should be assigned to vertices.");
    t.expect(almostEqual(result.mesh.materials[1].color.x, 0.9f), "MTL Kd color should be loaded.");
}

void testObjLoaderMultipleMaterials(TestContext& t)
{
    const std::filesystem::path objPath = writeFixtureFile(
        "multi_material.obj",
        "mtllib multi_material.mtl\n"
        "v 0 0 0\n"
        "v 1 0 0\n"
        "v 0 1 0\n"
        "v 1 1 0\n"
        "vn 0 0 1\n"
        "usemtl red\n"
        "f 1//1 2//1 3//1\n"
        "usemtl green\n"
        "f 2//1 4//1 3//1\n");

    writeFixtureFile(
        "multi_material.mtl",
        "newmtl red\n"
        "Kd 1 0 0\n"
        "newmtl green\n"
        "Kd 0 1 0\n");

    const ObjLoadResult result = loadObjMesh(objPath);
    t.expect(result.ok, "OBJ with multiple usemtl commands should load.");
    t.expect(result.mesh.triangles.size() == 2, "Multiple material OBJ should contain two triangles.");
    t.expect(result.mesh.materials.size() == 3, "Default plus two named materials should be loaded.");
    t.expect(result.mesh.triangles[0].materialIndex != result.mesh.triangles[1].materialIndex, "Triangles should reference different materials.");
    t.expect(hasValidMeshMaterialIndices(result.mesh), "Material indices should be valid.");
}

void testObjLoaderMirrorMaterialNameMapping(TestContext& t)
{
    const std::filesystem::path objPath = writeFixtureFile(
        "mirror_material.obj",
        "v 0 0 0\n"
        "v 1 0 0\n"
        "v 0 1 0\n"
        "usemtl mat_mirror\n"
        "f 1 2 3\n");

    const ObjLoadResult result = loadObjMesh(objPath);
    t.expect(result.ok, "OBJ with mirror material name should load.");
    t.expect(result.mesh.materials.size() == 2, "Mirror usemtl should create a named material.");
    t.expect(result.mesh.materials[1].materialType == MaterialMirror, "Material name containing mirror should map to MaterialMirror.");
}

void testObjLoaderMissingNormalsFallback(TestContext& t)
{
    const std::filesystem::path objPath = writeFixtureFile(
        "missing_normals.obj",
        "v 0 0 0\n"
        "v 1 0 0\n"
        "v 0 1 0\n"
        "f 1 2 3\n");

    const ObjLoadResult result = loadObjMesh(objPath);
    t.expect(result.ok, "OBJ without normals should load with generated normals.");
    t.expect(result.mesh.vertices.size() == 3, "Fallback-normal OBJ should create vertices.");
    t.expect(almostEqual(result.mesh.vertices[0].normal.z, 1.0f), "Fallback normal should be computed from triangle winding.");
}

void testObjLoaderEmptyFile(TestContext& t)
{
    const std::filesystem::path objPath = writeFixtureFile("empty.obj", "");
    const ObjLoadResult result = loadObjMesh(objPath);
    t.expect(!result.ok, "Empty OBJ should fail validation.");
    t.expect(!result.error.empty(), "Empty OBJ failure should include an error message.");
}

void testObjLoaderInvalidFace(TestContext& t)
{
    const std::filesystem::path objPath = writeFixtureFile(
        "invalid_face.obj",
        "v 0 0 0\n"
        "v 1 0 0\n"
        "v 0 1 0\n"
        "v 1 1 0\n"
        "f 1 2 3 4\n");

    const ObjLoadResult result = loadObjMesh(objPath);
    t.expect(!result.ok, "Non-triangulated face should fail.");
    t.expect(result.error.find("triangulated") != std::string::npos, "Invalid face error should explain triangulated requirement.");
}

void testObjLoaderMissingFile(TestContext& t)
{
    const std::filesystem::path objPath = std::filesystem::temp_directory_path() / "raytracerrtx_missing_mesh.obj";
    const ObjLoadResult result = loadObjMesh(objPath);
    t.expect(!result.ok, "Missing OBJ file should fail.");
    t.expect(result.error.find("could not be opened") != std::string::npos, "Missing file error should explain open failure.");
}

void testObjLoaderUnknownLinesIgnored(TestContext& t)
{
    const std::filesystem::path objPath = writeFixtureFile(
        "unknown_lines.obj",
        "o DemoObject\n"
        "s off\n"
        "v 0 0 0\n"
        "v 1 0 0\n"
        "v 0 1 0\n"
        "g ignored_group\n"
        "f 1 2 3\n");

    const ObjLoadResult result = loadObjMesh(objPath);
    t.expect(result.ok, "Unknown OBJ lines should be ignored.");
    t.expect(result.mesh.triangles.size() == 1, "Unknown lines should not prevent triangle loading.");
}

void testObjLoaderMaterialFallbackWhenMtlMissing(TestContext& t)
{
    const std::filesystem::path objPath = writeFixtureFile(
        "missing_mtl.obj",
        "mtllib does_not_exist.mtl\n"
        "v 0 0 0\n"
        "v 1 0 0\n"
        "v 0 1 0\n"
        "usemtl fallback_name\n"
        "f 1 2 3\n");

    const ObjLoadResult result = loadObjMesh(objPath);
    t.expect(result.ok, "Missing MTL should fall back to generated material.");
    t.expect(result.mesh.materials.size() == 2, "Fallback named material should be created.");
    t.expect(result.mesh.materials[1].name == "fallback_name", "Fallback material should keep usemtl name.");
    t.expect(hasValidMeshMaterialIndices(result.mesh), "Fallback material index should be valid.");
}

void testObjLoaderSingleTriangleBoundary(TestContext& t)
{
    const std::filesystem::path objPath = writeFixtureFile(
        "single_triangle.obj",
        "v 0 0 0\n"
        "v 0 0 1\n"
        "v 0 1 0\n"
        "f 1 2 3\n");

    const ObjLoadResult result = loadObjMesh(objPath);
    t.expect(result.ok, "Single triangle OBJ should load.");
    t.expect(!isEmptyMesh(result.mesh), "Single triangle mesh should not be empty.");
    t.expect(result.mesh.triangles[0].i0 == 0, "Single triangle first index should be 0.");
    t.expect(result.mesh.triangles[0].i2 == 2, "Single triangle third index should be 2.");
}

void testDemoObjAssetLoads(TestContext& t)
{
    const std::filesystem::path objPath = findDemoObjAsset();
    t.expect(!objPath.empty(), "Demo OBJ asset must exist.");
    if (objPath.empty())
    {
        return;
    }

    const ObjLoadResult result = loadObjMesh(objPath);
    t.expect(result.ok, "Demo OBJ asset should load.");
    t.expect(result.mesh.vertices.size() > 0, "Demo OBJ should contain vertices.");
    t.expect(result.mesh.triangles.size() > 0, "Demo OBJ should contain triangles.");
    t.expect(result.mesh.materials.size() >= 4, "Demo OBJ should load default plus named MTL materials.");
    t.expect(hasValidMeshMaterialIndices(result.mesh), "Demo OBJ material indices should be valid.");
}

void testDefaultScene(TestContext& t)
{
    const SceneState scene = makeDefaultScene();
    t.expect(scene.spheres.size() == 3, "Default scene must contain 3 spheres.");
    t.expect(scene.materials.size() == 3, "Default scene must contain 3 materials.");
    t.expect(!isEmptyMesh(scene.mesh), "Default scene must contain a triangle mesh.");
    t.expect(scene.mesh.materials.size() >= 3, "Default mesh should contain several materials.");
    t.expect(hasValidMeshMaterialIndices(scene.mesh), "Default mesh material indices should be valid.");
    bool hasMirrorMeshMaterial = false;
    for (const MeshMaterial& material : scene.mesh.materials)
    {
        hasMirrorMeshMaterial = hasMirrorMeshMaterial || material.materialType == MaterialMirror;
    }
    t.expect(hasMirrorMeshMaterial, "Default mesh should include a mirror material.");
    for (const SphereMaterial& material : scene.materials)
    {
        t.expect(material.materialType == MaterialDiffuse, "All default spheres should start as diffuse.");
    }
    t.expect(scene.selectedSphere == 0, "Default selected sphere should be index 0.");
}

void testDefaultSceneMeshGeometry(TestContext& t)
{
    const SceneState scene = makeDefaultScene();
    t.expect(scene.mesh.vertices.size() > 0, "Default mesh vertex count should be positive.");
    t.expect(scene.mesh.triangles.size() > 0, "Default mesh triangle count should be positive.");
    t.expect(!scene.mesh.materials.empty(), "Default mesh material count should be positive.");
    t.expect(hasValidMeshMaterialIndices(scene.mesh), "Default mesh material indices should stay in range.");
}

void testSceneConfigLoadsValidScene(TestContext& t)
{
    const std::filesystem::path objPath = writeFixtureFile(
        "config_mesh.obj",
        "v 0 0 0\n"
        "v 1 0 0\n"
        "v 0 1 0\n"
        "f 1 2 3\n");

    const std::filesystem::path configPath = writeFixtureFile(
        "valid_scene.json",
        "{\n"
        "  \"camera\": {\"position\": [1, 2, 3], \"yaw\": 15, \"pitch\": -10, \"fov\": 55},\n"
        "  \"light\": {\"position\": [4, 5, 6]},\n"
        "  \"meshObjects\": [\n"
        "    {\"path\": \"config_mesh.obj\", \"position\": [2, 0, 0], \"rotation\": [0, 0, 0], \"scale\": [2, 2, 2]}\n"
        "  ]\n"
        "}\n");

    (void)objPath;
    const SceneConfigResult config = loadSceneConfigFile(configPath);
    t.expect(config.ok, "Valid scene config should load: " + config.error);
    const SceneBuildResult scene = buildSceneFromConfig(config.config, configPath.parent_path());
    t.expect(scene.ok, "Valid scene config should build a scene.");
    t.expect(almostEqual(scene.camera.position.x, 1.0f), "Scene config should apply camera position.");
    t.expect(almostEqual(scene.scene.lightPosition.y, 5.0f), "Scene config should apply light position.");
    t.expect(scene.scene.mesh.vertices.size() == 3, "Scene config mesh path should load OBJ vertices.");
    t.expect(almostEqual(scene.scene.mesh.vertices[1].position.x, 4.0f), "Scene config transform should affect mesh vertices.");
}

void testSceneConfigMissingFile(TestContext& t)
{
    const std::filesystem::path configPath = std::filesystem::temp_directory_path() / "raytracerrtx_missing_scene_config.json";
    const SceneConfigResult config = loadSceneConfigFile(configPath);
    t.expect(!config.ok, "Missing scene config should fail cleanly.");
    t.expect(config.error.find("could not be opened") != std::string::npos, "Missing scene config should explain open failure.");
}

void testSceneConfigMeshPathApplied(TestContext& t)
{
    const std::filesystem::path objPath = writeFixtureFile(
        "direct_mesh.obj",
        "v 0 0 0\n"
        "v 0 2 0\n"
        "v 0 0 2\n"
        "f 1 2 3\n");

    const SceneBuildResult scene = buildSceneFromMeshPath(objPath);
    t.expect(scene.ok, "Direct mesh path should build a scene.");
    t.expect(scene.scene.mesh.vertices.size() == 3, "Direct mesh path should replace default mesh vertices.");
    t.expect(almostEqual(scene.scene.mesh.vertices[1].position.y, 2.0f), "Direct mesh path should use requested OBJ data.");
}

void testSceneConfigFallbackDemoScene(TestContext& t)
{
    const SceneBuildResult scene = buildDefaultSceneInput();
    t.expect(scene.ok, "Default scene input should build.");
    t.expect(!isEmptyMesh(scene.scene.mesh), "Default scene input should keep fallback/demo mesh.");
    t.expect(hasValidMeshMaterialIndices(scene.scene.mesh), "Default scene input mesh material indices should be valid.");
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
    runTest("Clamp selected sphere index", testClampSceneSelectedSphereBounds);
    runTest("Invalid selected sphere operations", testInvalidSelectedSphereOps);
    runTest("All spheres above floor after clamp", testAllSpheresStayAboveFloorAfterClamp);
    runTest("Camera basis", testCameraBasis);
    runTest("Camera aspect fallback", testCameraAspectFallback);
    runTest("Camera scale vs FOV", testCameraScaleIncreasesWithFov);
    runTest("OBJ loader triangle with normals", testObjLoaderTriangleWithNormals);
    runTest("OBJ loader multiple materials", testObjLoaderMultipleMaterials);
    runTest("OBJ loader mirror material name mapping", testObjLoaderMirrorMaterialNameMapping);
    runTest("OBJ loader missing normals fallback", testObjLoaderMissingNormalsFallback);
    runTest("OBJ loader empty file", testObjLoaderEmptyFile);
    runTest("OBJ loader invalid face", testObjLoaderInvalidFace);
    runTest("OBJ loader missing file", testObjLoaderMissingFile);
    runTest("OBJ loader unknown lines ignored", testObjLoaderUnknownLinesIgnored);
    runTest("OBJ loader material fallback when MTL missing", testObjLoaderMaterialFallbackWhenMtlMissing);
    runTest("OBJ loader single triangle boundary", testObjLoaderSingleTriangleBoundary);
    runTest("Demo OBJ asset loads", testDemoObjAssetLoads);
    runTest("Default scene mesh geometry", testDefaultSceneMeshGeometry);
    runTest("Scene config loads valid scene", testSceneConfigLoadsValidScene);
    runTest("Scene config missing file", testSceneConfigMissingFile);
    runTest("Scene config mesh path applied", testSceneConfigMeshPathApplied);
    runTest("Scene config fallback demo scene", testSceneConfigFallbackDemoScene);

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
