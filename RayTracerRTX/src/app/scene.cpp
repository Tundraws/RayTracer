#include "scene.h"

#include "obj_loader.h"

#include <algorithm>
#include <filesystem>
#include <utility>

#ifndef RAYTRACERRTX_SOURCE_DIR
#define RAYTRACERRTX_SOURCE_DIR ""
#endif

namespace
{
constexpr float kMinSphereRadius = 0.25f;
constexpr float kMaxSphereRadius = 5.0f;

float clampScalar(const float v, const float minV, const float maxV)
{
    return v < minV ? minV : (v > maxV ? maxV : v);
}

float3 add3(const float3 a, const float3 b)
{
    return make_float3(a.x + b.x, a.y + b.y, a.z + b.z);
}

float3 clamp3(const float3 value, const float3 minValue, const float3 maxValue)
{
    return make_float3(
        clampScalar(value.x, minValue.x, maxValue.x),
        clampScalar(value.y, minValue.y, maxValue.y),
        clampScalar(value.z, minValue.z, maxValue.z));
}

SphereMaterial makeDefaultSphereMaterial()
{
    return {make_float3(0.72f, 0.76f, 0.72f), MaterialDiffuse, make_float3(0.72f, 0.72f, 0.72f), 0.52f, 1.5f, 1.0f};
}

MeshData makeFallbackMesh()
{
    MeshData mesh;
    mesh.materials = {
        {make_float3(0.72f, 0.86f, 0.95f), MaterialDiffuse, "fallback_blue", make_float3(1.0f, 1.0f, 1.0f), 0.35f, 1.5f, 1.0f, "", -1, "", -1},
        {make_float3(0.95f, 0.78f, 0.55f), MaterialDiffuse, "fallback_warm", make_float3(1.0f, 1.0f, 1.0f), 0.35f, 1.5f, 1.0f, "", -1, "", -1},
        {make_float3(0.92f, 0.92f, 0.92f), MaterialMirror, "fallback_mirror", make_float3(1.0f, 1.0f, 1.0f), 0.02f, 1.5f, 1.0f, "", -1, "", -1}
    };

    mesh.vertices = {
        {make_float3(-1.8f, 0.0f, -7.6f), make_float3(0.0f, -1.0f, 0.0f), make_float2(0.0f, 0.0f), make_float3(1.0f, 0.0f, 0.0f), 1},
        {make_float3(1.8f, 0.0f, -7.6f), make_float3(0.0f, -1.0f, 0.0f), make_float2(1.0f, 0.0f), make_float3(1.0f, 0.0f, 0.0f), 1},
        {make_float3(1.8f, 0.0f, -4.0f), make_float3(0.0f, -1.0f, 0.0f), make_float2(1.0f, 1.0f), make_float3(1.0f, 0.0f, 0.0f), 1},
        {make_float3(-1.8f, 0.0f, -4.0f), make_float3(0.0f, -1.0f, 0.0f), make_float2(0.0f, 1.0f), make_float3(1.0f, 0.0f, 0.0f), 1},
        {make_float3(0.0f, 3.2f, -5.8f), make_float3(0.0f, 1.0f, 0.0f), make_float2(0.5f, 0.5f), make_float3(1.0f, 0.0f, 0.0f), 1}
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
        {make_float3(0.72f, 0.76f, 0.72f), MaterialDiffuse, make_float3(0.72f, 0.72f, 0.72f), 0.52f, 1.5f, 1.0f},
        {make_float3(0.78f, 0.68f, 0.58f), MaterialDiffuse, make_float3(0.70f, 0.68f, 0.62f), 0.58f, 1.5f, 1.0f},
        {make_float3(0.58f, 0.70f, 0.82f), MaterialDiffuse, make_float3(0.66f, 0.68f, 0.72f), 0.55f, 1.5f, 1.0f}
    };

    scene.mesh = loadDefaultMesh();
    MeshObject defaultMeshObject;
    defaultMeshObject.assetReference = "assets/meshes/demo.obj";
    defaultMeshObject.mesh = scene.mesh;
    scene.meshObjects = {std::move(defaultMeshObject)};
    scene.lightPosition = make_float3(10.0f, 14.0f, -10.0f);
    scene.exposure = 0.82f;
    scene.skyIntensity = 0.78f;
    scene.lightIntensity = 0.95f;
    scene.areaLightRadius = 0.0f;
    scene.environmentIntensity = 1.0f;
    scene.environmentType = "gradient";
    scene.selectedSphere = 0;
    scene.selectedMeshObject = 0;
    scene.selectedMeshMaterial = 0;
    return scene;
}

void clampScene(SceneState& scene)
{
    const float3 sphereMin = make_float3(-24.0f, 0.0f, -24.0f);
    const float3 sphereMax = make_float3(24.0f, 14.0f, 24.0f);
    const float floorY = 0.0f;

    if (scene.materials.size() < scene.spheres.size())
    {
        scene.materials.resize(scene.spheres.size(), makeDefaultSphereMaterial());
    }
    if (scene.materials.size() > scene.spheres.size())
    {
        scene.materials.resize(scene.spheres.size());
    }

    for (size_t i = 0; i < scene.spheres.size(); ++i)
    {
        SphereGeometry& sphere = scene.spheres[i];
        sphere.radius = clampScalar(sphere.radius, kMinSphereRadius, kMaxSphereRadius);
        const float3 minBounds = make_float3(sphereMin.x, floorY + sphere.radius, sphereMin.z);
        const float3 maxBounds = make_float3(sphereMax.x, sphereMax.y, sphereMax.z);
        sphere.center = clamp3(sphere.center, minBounds, maxBounds);
    }

    scene.lightPosition = clamp3(
        scene.lightPosition,
        make_float3(-40.0f, 6.0f, -40.0f),
        make_float3(40.0f, 40.0f, 40.0f));
    scene.exposure = clampSceneExposure(scene.exposure);
    scene.skyIntensity = clampSceneSkyIntensity(scene.skyIntensity);
    scene.lightIntensity = clampSceneLightIntensity(scene.lightIntensity);
    scene.areaLightRadius = clampSceneAreaLightRadius(scene.areaLightRadius);
    scene.environmentIntensity = clampSceneEnvironmentIntensity(scene.environmentIntensity);

    if (scene.selectedSphere < 0)
    {
        scene.selectedSphere = 0;
    }
    if (scene.selectedSphere >= static_cast<int>(scene.spheres.size()))
    {
        scene.selectedSphere = scene.spheres.empty() ? 0 : static_cast<int>(scene.spheres.size()) - 1;
    }

    if (scene.selectedMeshObject < 0)
    {
        scene.selectedMeshObject = 0;
    }
    if (scene.selectedMeshObject >= static_cast<int>(scene.meshObjects.size()))
    {
        scene.selectedMeshObject = scene.meshObjects.empty() ? 0 : static_cast<int>(scene.meshObjects.size()) - 1;
    }

    const bool hasSelectedMesh = !scene.meshObjects.empty() &&
        scene.selectedMeshObject >= 0 &&
        scene.selectedMeshObject < static_cast<int>(scene.meshObjects.size());
    const int materialCount = hasSelectedMesh
        ? static_cast<int>(scene.meshObjects[static_cast<size_t>(scene.selectedMeshObject)].mesh.materials.size())
        : 0;
    if (scene.selectedMeshMaterial < 0)
    {
        scene.selectedMeshMaterial = 0;
    }
    if (scene.selectedMeshMaterial >= materialCount)
    {
        scene.selectedMeshMaterial = materialCount > 0 ? materialCount - 1 : 0;
    }
}

bool addSphere(SceneState& scene)
{
    clampScene(scene);

    SphereGeometry sphere{make_float3(0.0f, 1.25f, -2.5f), 1.25f};
    SphereMaterial material = makeDefaultSphereMaterial();

    if (!scene.spheres.empty() &&
        scene.selectedSphere >= 0 &&
        scene.selectedSphere < static_cast<int>(scene.spheres.size()))
    {
        const SphereGeometry& selected = scene.spheres[static_cast<size_t>(scene.selectedSphere)];
        sphere.radius = selected.radius;
        sphere.center = add3(selected.center, make_float3(selected.radius * 2.2f + 0.5f, 0.0f, 0.0f));

        if (scene.selectedSphere < static_cast<int>(scene.materials.size()))
        {
            material = scene.materials[static_cast<size_t>(scene.selectedSphere)];
        }
    }

    scene.spheres.push_back(sphere);
    scene.materials.push_back(material);
    scene.selectedSphere = static_cast<int>(scene.spheres.size()) - 1;
    clampScene(scene);
    return true;
}

bool removeSelectedSphere(SceneState& scene)
{
    clampScene(scene);
    if (scene.spheres.size() <= 1)
    {
        return false;
    }

    const int index = scene.selectedSphere;
    if (index < 0 || index >= static_cast<int>(scene.spheres.size()))
    {
        return false;
    }

    scene.spheres.erase(scene.spheres.begin() + index);
    if (index < static_cast<int>(scene.materials.size()))
    {
        scene.materials.erase(scene.materials.begin() + index);
    }
    scene.selectedSphere = std::min(index, static_cast<int>(scene.spheres.size()) - 1);
    clampScene(scene);
    return true;
}

void setSelectedSphereRadius(SceneState& scene, const float radius)
{
    const int index = scene.selectedSphere;
    if (index < 0 || index >= static_cast<int>(scene.spheres.size()))
    {
        return;
    }

    scene.spheres[static_cast<size_t>(index)].radius = clampScalar(radius, kMinSphereRadius, kMaxSphereRadius);
    clampScene(scene);
}

void setSelectedSphereColor(SceneState& scene, const float3 color)
{
    const int index = scene.selectedSphere;
    if (index < 0 || index >= static_cast<int>(scene.materials.size()))
    {
        return;
    }

    SphereMaterial& material = scene.materials[static_cast<size_t>(index)];
    material.color = clamp3(color, make_float3(0.0f, 0.0f, 0.0f), make_float3(1.0f, 1.0f, 1.0f));
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

int nextMaterialPreset(const int materialType)
{
    switch (materialType)
    {
    case MaterialDiffuse:
        return MaterialMirror;
    case MaterialMirror:
        return MaterialMetal;
    case MaterialMetal:
        return MaterialDielectric;
    default:
        return MaterialDiffuse;
    }
}

void applyMaterialDefaults(SphereMaterial& material)
{
    switch (material.materialType)
    {
    case MaterialMirror:
        material.specularColor = make_float3(1.0f, 1.0f, 1.0f);
        material.roughness = 0.02f;
        material.ior = 1.5f;
        material.alpha = 1.0f;
        break;
    case MaterialMetal:
        material.specularColor = make_float3(0.95f, 0.9f, 0.82f);
        material.roughness = 0.18f;
        material.ior = 1.5f;
        material.alpha = 1.0f;
        break;
    case MaterialDielectric:
        material.specularColor = make_float3(1.0f, 1.0f, 1.0f);
        material.roughness = 0.02f;
        material.ior = 1.45f;
        material.alpha = 0.45f;
        break;
    default:
        material.specularColor = make_float3(1.0f, 1.0f, 1.0f);
        material.roughness = 0.4f;
        material.ior = 1.5f;
        material.alpha = 1.0f;
        break;
    }
}

void applyMaterialDefaults(MeshMaterial& material)
{
    switch (material.materialType)
    {
    case MaterialMirror:
        material.specularColor = make_float3(1.0f, 1.0f, 1.0f);
        material.roughness = 0.02f;
        material.ior = 1.5f;
        material.alpha = 1.0f;
        break;
    case MaterialMetal:
        material.specularColor = make_float3(0.95f, 0.9f, 0.82f);
        material.roughness = 0.18f;
        material.ior = 1.5f;
        material.alpha = 1.0f;
        break;
    case MaterialDielectric:
        material.specularColor = make_float3(1.0f, 1.0f, 1.0f);
        material.roughness = 0.02f;
        material.ior = 1.45f;
        material.alpha = 0.45f;
        break;
    default:
        material.specularColor = make_float3(1.0f, 1.0f, 1.0f);
        material.roughness = 0.4f;
        material.ior = 1.5f;
        material.alpha = 1.0f;
        break;
    }
}

void cycleSelectedSphereMaterialPreset(SceneState& scene)
{
    const int index = scene.selectedSphere;
    if (index < 0 || index >= static_cast<int>(scene.materials.size()))
    {
        return;
    }

    SphereMaterial& material = scene.materials[static_cast<size_t>(index)];
    material.materialType = nextMaterialPreset(material.materialType);
    applyMaterialDefaults(material);
}

void selectNextMeshObject(SceneState& scene)
{
    if (scene.meshObjects.empty())
    {
        scene.selectedMeshObject = 0;
        scene.selectedMeshMaterial = 0;
        return;
    }

    scene.selectedMeshObject = (scene.selectedMeshObject + 1) % static_cast<int>(scene.meshObjects.size());
    scene.selectedMeshMaterial = 0;
    clampScene(scene);
}

void cycleSelectedMeshMaterialPreset(SceneState& scene)
{
    if (scene.meshObjects.empty())
    {
        return;
    }
    clampScene(scene);
    MeshObject& object = scene.meshObjects[static_cast<size_t>(scene.selectedMeshObject)];
    if (object.mesh.materials.empty())
    {
        return;
    }

    MeshMaterial& objectMaterial = object.mesh.materials[static_cast<size_t>(scene.selectedMeshMaterial)];
    objectMaterial.materialType = nextMaterialPreset(objectMaterial.materialType);
    applyMaterialDefaults(objectMaterial);

    if (!scene.mesh.materials.empty())
    {
        size_t combinedMaterialIndex = 0;
        for (int i = 0; i < scene.selectedMeshObject && i < static_cast<int>(scene.meshObjects.size()); ++i)
        {
            combinedMaterialIndex += scene.meshObjects[static_cast<size_t>(i)].mesh.materials.size();
        }
        combinedMaterialIndex += static_cast<size_t>(scene.selectedMeshMaterial);
        if (combinedMaterialIndex < scene.mesh.materials.size())
        {
            scene.mesh.materials[combinedMaterialIndex].materialType = objectMaterial.materialType;
            scene.mesh.materials[combinedMaterialIndex].specularColor = objectMaterial.specularColor;
            scene.mesh.materials[combinedMaterialIndex].roughness = objectMaterial.roughness;
            scene.mesh.materials[combinedMaterialIndex].ior = objectMaterial.ior;
            scene.mesh.materials[combinedMaterialIndex].alpha = objectMaterial.alpha;
        }
    }
}

void moveLight(SceneState& scene, const float3 delta)
{
    scene.lightPosition = add3(scene.lightPosition, delta);
    clampScene(scene);
}

void setSceneExposure(SceneState& scene, const float value)
{
    scene.exposure = clampSceneExposure(value);
}

void setSceneSkyIntensity(SceneState& scene, const float value)
{
    scene.skyIntensity = clampSceneSkyIntensity(value);
}

void setSceneLightIntensity(SceneState& scene, const float value)
{
    scene.lightIntensity = clampSceneLightIntensity(value);
}

void adjustSceneExposure(SceneState& scene, const float delta)
{
    setSceneExposure(scene, scene.exposure + delta);
}

void adjustSceneSkyIntensity(SceneState& scene, const float delta)
{
    setSceneSkyIntensity(scene, scene.skyIntensity + delta);
}

void adjustSceneLightIntensity(SceneState& scene, const float delta)
{
    setSceneLightIntensity(scene, scene.lightIntensity + delta);
}
