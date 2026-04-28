#include "scene.h"

#include <algorithm>

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
} // namespace

SceneState makeDefaultScene()
{
    SceneState scene;
    scene.spheres = {
        {make_float3(0.0f, 3.0f, 0.0f), 3.0f},
        {make_float3(6.5f, 2.4f, 3.5f), 2.4f},
        {make_float3(-6.5f, 2.4f, 3.5f), 2.4f}
    };

    scene.materials = {
        {make_float3(1.0f, 1.0f, 1.0f), MaterialMirror},
        {make_float3(0.82f, 0.70f, 0.60f), MaterialDiffuse},
        {make_float3(0.65f, 0.80f, 0.75f), MaterialDiffuse}
    };

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
