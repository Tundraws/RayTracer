#include "scene.h"

#include "obj_loader.h"

#include <algorithm>
#include <filesystem>

#ifndef RAYTRACERRTX_SOURCE_DIR
#define RAYTRACERRTX_SOURCE_DIR ""
#endif

namespace
{
float3 add3(const float3 a, const float3 b)
{
    return make_float3(a.x + b.x, a.y + b.y, a.z + b.z);
}

float3 clamp3(const float3 value, const float3 minValue, const float3 maxValue)
{
    const auto clampScalar = [](const float v, const float minV, const float maxV)
    {
        return v < minV ? minV : (v > maxV ? maxV : v);
    };

    return make_float3(
        clampScalar(value.x, minValue.x, maxValue.x),
        clampScalar(value.y, minValue.y, maxValue.y),
        clampScalar(value.z, minValue.z, maxValue.z));
}

MeshData makeFallbackMesh()
{
    MeshData mesh;
    mesh.materials = {
        {make_float3(0.72f, 0.86f, 0.95f), MaterialDiffuse, "fallback_blue", make_float3(1.0f, 1.0f, 1.0f), 0.35f, 1.5f, 1.0f, "", -1},
        {make_float3(0.95f, 0.78f, 0.55f), MaterialDiffuse, "fallback_warm", make_float3(1.0f, 1.0f, 1.0f), 0.35f, 1.5f, 1.0f, "", -1},
        {make_float3(0.92f, 0.92f, 0.92f), MaterialMirror, "fallback_mirror", make_float3(1.0f, 1.0f, 1.0f), 0.02f, 1.5f, 1.0f, "", -1}
    };

    mesh.vertices = {
        {make_float3(-1.8f, 0.0f, -7.6f), make_float3(0.0f, -1.0f, 0.0f), make_float2(0.0f, 0.0f)},
        {make_float3(1.8f, 0.0f, -7.6f), make_float3(0.0f, -1.0f, 0.0f), make_float2(1.0f, 0.0f)},
        {make_float3(1.8f, 0.0f, -4.0f), make_float3(0.0f, -1.0f, 0.0f), make_float2(1.0f, 1.0f)},
        {make_float3(-1.8f, 0.0f, -4.0f), make_float3(0.0f, -1.0f, 0.0f), make_float2(0.0f, 1.0f)},
        {make_float3(0.0f, 3.2f, -5.8f), make_float3(0.0f, 1.0f, 0.0f), make_float2(0.5f, 0.5f)}
    };

    mesh.triangles = {
        {0u, 1u, 4u, 0u},
        {1u, 2u, 4u, 1u},
        {2u, 3u, 4u, 2u},
        {3u, 0u, 4u, 0u}
    };

    return mesh;
}

MeshData loadDefaultMesh()
{
    const std::filesystem::path sourceDir = RAYTRACERRTX_SOURCE_DIR;
    const std::filesystem::path assetFromSource = sourceDir.empty()
        ? std::filesystem::path{}
        : sourceDir.parent_path() / "assets" / "meshes" / "demo.obj";

    const std::filesystem::path candidates[] = {
        assetFromSource,
        std::filesystem::path("RayTracerRTX") / "assets" / "meshes" / "demo.obj",
        std::filesystem::path("assets") / "meshes" / "demo.obj"
    };

    for (const std::filesystem::path& candidate : candidates)
    {
        if (!candidate.empty())
        {
            const ObjLoadResult result = loadObjMesh(candidate);
            if (result.ok)
            {
                return result.mesh;
            }
        }
    }

    return makeFallbackMesh();
}
} // namespace

SceneState makeDefaultScene()
{
    SceneState scene;
    scene.spheres = {
        {make_float3(0.0f, 2.55f, 0.0f), 2.55f},
        {make_float3(-5.9f, 2.05f, 3.4f), 2.05f},
        {make_float3(5.9f, 1.55f, 3.4f), 1.55f}
    };

    scene.materials = {
        {make_float3(0.98f, 0.98f, 0.95f), MaterialDiffuse},
        {make_float3(0.92f, 0.78f, 0.66f), MaterialDiffuse},
        {make_float3(0.72f, 0.92f, 0.84f), MaterialDiffuse}
    };

    scene.mesh = loadDefaultMesh();
    scene.lightPosition = make_float3(10.0f, 14.0f, -10.0f);
    scene.selectedSphere = 0;
    return scene;
}

void clampScene(SceneState& scene)
{
    const float3 sphereMin = make_float3(-24.0f, 0.0f, -24.0f);
    const float3 sphereMax = make_float3(24.0f, 14.0f, 24.0f);
    const float floorY = 0.0f;

    for (size_t i = 0; i < scene.spheres.size(); ++i)
    {
        SphereGeometry& sphere = scene.spheres[i];
        const float3 minBounds = make_float3(sphereMin.x, floorY + sphere.radius, sphereMin.z);
        const float3 maxBounds = make_float3(sphereMax.x, sphereMax.y, sphereMax.z);
        sphere.center = clamp3(sphere.center, minBounds, maxBounds);
    }

    scene.lightPosition = clamp3(
        scene.lightPosition,
        make_float3(-40.0f, 6.0f, -40.0f),
        make_float3(40.0f, 40.0f, 40.0f));

    if (scene.selectedSphere < 0)
    {
        scene.selectedSphere = 0;
    }
    if (scene.selectedSphere >= static_cast<int>(scene.spheres.size()))
    {
        scene.selectedSphere = static_cast<int>(scene.spheres.size()) - 1;
    }
}

void moveSelectedSphere(SceneState& scene, const float3 delta)
{
    const int index = scene.selectedSphere;
    if (index < 0 || index >= static_cast<int>(scene.spheres.size()))
    {
        return;
    }

    scene.spheres[index].center = add3(scene.spheres[index].center, delta);
    clampScene(scene);
}

void toggleSelectedMaterial(SceneState& scene)
{
    const int index = scene.selectedSphere;
    if (index < 0 || index >= static_cast<int>(scene.materials.size()))
    {
        return;
    }

    SphereMaterial& material = scene.materials[index];
    material.materialType = (material.materialType == MaterialDiffuse) ? MaterialMirror : MaterialDiffuse;
}

void moveLight(SceneState& scene, const float3 delta)
{
    scene.lightPosition = add3(scene.lightPosition, delta);
    clampScene(scene);
}
