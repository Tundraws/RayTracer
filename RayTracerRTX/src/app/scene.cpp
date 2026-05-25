#include "scene.h"

#include "obj_loader.h"

#include <algorithm>
#include <cmath>
#include <filesystem>
#include <utility>

#ifndef RAYTRACERRTX_SOURCE_DIR
#define RAYTRACERRTX_SOURCE_DIR ""
#endif

namespace
{
constexpr float kMinSphereRadius = 0.25f;
constexpr float kMaxSphereRadius = 5.0f;
constexpr float kPi = 3.14159265358979323846f;

float clampScalar(const float v, const float minV, const float maxV)
{
    return v < minV ? minV : (v > maxV ? maxV : v);
}

float3 add3(const float3 a, const float3 b)
{
    return make_float3(a.x + b.x, a.y + b.y, a.z + b.z);
}

float radians(const float degrees)
{
    return degrees * kPi / 180.0f;
}

float3 rotateX(const float3 value, const float angle)
{
    const float c = std::cos(angle);
    const float s = std::sin(angle);
    return make_float3(value.x, value.y * c - value.z * s, value.y * s + value.z * c);
}

float3 rotateY(const float3 value, const float angle)
{
    const float c = std::cos(angle);
    const float s = std::sin(angle);
    return make_float3(value.x * c + value.z * s, value.y, -value.x * s + value.z * c);
}

float3 rotateZ(const float3 value, const float angle)
{
    const float c = std::cos(angle);
    const float s = std::sin(angle);
    return make_float3(value.x * c - value.y * s, value.x * s + value.y * c, value.z);
}

float3 rotateEulerXyz(const float3 value, const float3 degrees)
{
    float3 result = rotateX(value, radians(degrees.x));
    result = rotateY(result, radians(degrees.y));
    result = rotateZ(result, radians(degrees.z));
    return result;
}

std::array<float, 12> makeMeshTransformMatrix(const float3 position, const float3 rotation, const float3 scale)
{
    const float3 xAxis = rotateEulerXyz(make_float3(scale.x, 0.0f, 0.0f), rotation);
    const float3 yAxis = rotateEulerXyz(make_float3(0.0f, scale.y, 0.0f), rotation);
    const float3 zAxis = rotateEulerXyz(make_float3(0.0f, 0.0f, scale.z), rotation);
    return {
        xAxis.x, yAxis.x, zAxis.x, position.x,
        xAxis.y, yAxis.y, zAxis.y, position.y,
        xAxis.z, yAxis.z, zAxis.z, position.z};
}

void updateMeshObjectTransform(MeshObject& object)
{
    object.transform = makeMeshTransformMatrix(object.position, object.rotation, object.scale);
}

MeshObject* selectedMeshObject(SceneState& scene)
{
    if (scene.meshObjects.empty())
    {
        return nullptr;
    }
    clampScene(scene);
    if (scene.selectedMeshObject < 0 || scene.selectedMeshObject >= static_cast<int>(scene.meshObjects.size()))
    {
        return nullptr;
    }
    return &scene.meshObjects[static_cast<size_t>(scene.selectedMeshObject)];
}

void syncCompatibilityMesh(SceneState& scene)
{
    if (!scene.meshObjects.empty())
    {
        scene.mesh = scene.meshObjects[0].mesh;
    }
}

MeshObject makeBuiltInMeshObject(MeshData mesh, std::string name, const float3 position, const float3 rotation, const float3 scale)
{
    MeshObject object;
    object.assetReference = name;
    object.displayName = std::move(name);
    object.mesh = std::move(mesh);
    object.position = position;
    object.rotation = rotation;
    object.scale = scale;
    updateMeshObjectTransform(object);
    return object;
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
        {make_float3(0.72f, 0.86f, 0.95f), MaterialDiffuse, "fallback_blue", make_float3(1.0f, 1.0f, 1.0f), 0.35f, 1.5f, 1.0f, "", -1, 1, "", -1, "", -1, "", -1},
        {make_float3(0.95f, 0.78f, 0.55f), MaterialDiffuse, "fallback_warm", make_float3(1.0f, 1.0f, 1.0f), 0.35f, 1.5f, 1.0f, "", -1, 1, "", -1, "", -1, "", -1},
        {make_float3(0.92f, 0.92f, 0.92f), MaterialMirror, "fallback_mirror", make_float3(1.0f, 1.0f, 1.0f), 0.02f, 1.5f, 1.0f, "", -1, 1, "", -1, "", -1, "", -1}
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
    defaultMeshObject.displayName = "Demo OBJ";
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

    SphereGeometry& sphere = scene.spheres[static_cast<size_t>(index)];
    const float bottomY = sphere.center.y - sphere.radius;
    sphere.radius = clampScalar(radius, kMinSphereRadius, kMaxSphereRadius);
    sphere.center.y = bottomY + sphere.radius;
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

int sanitizeMaterialType(const int materialType)
{
    switch (materialType)
    {
    case MaterialDiffuse:
    case MaterialMirror:
    case MaterialMetal:
    case MaterialDielectric:
        return materialType;
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

void syncSelectedMeshMaterialToCombined(SceneState& scene)
{
    if (scene.meshObjects.empty() || scene.mesh.materials.empty())
    {
        return;
    }
    if (scene.selectedMeshObject < 0 ||
        scene.selectedMeshObject >= static_cast<int>(scene.meshObjects.size()))
    {
        return;
    }

    const MeshObject& object = scene.meshObjects[static_cast<size_t>(scene.selectedMeshObject)];
    if (object.mesh.materials.empty())
    {
        return;
    }

    size_t combinedMaterialIndex = 0;
    for (int i = 0; i < scene.selectedMeshObject && i < static_cast<int>(scene.meshObjects.size()); ++i)
    {
        combinedMaterialIndex += scene.meshObjects[static_cast<size_t>(i)].mesh.materials.size();
    }
    if (combinedMaterialIndex >= scene.mesh.materials.size())
    {
        return;
    }

    const size_t materialCount = std::min(
        object.mesh.materials.size(),
        scene.mesh.materials.size() - combinedMaterialIndex);
    for (size_t i = 0; i < materialCount; ++i)
    {
        scene.mesh.materials[combinedMaterialIndex + i] = object.mesh.materials[i];
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

void setSelectedSphereMaterialType(SceneState& scene, const int materialType)
{
    const int index = scene.selectedSphere;
    if (index < 0 || index >= static_cast<int>(scene.materials.size()))
    {
        return;
    }

    SphereMaterial& material = scene.materials[static_cast<size_t>(index)];
    material.materialType = sanitizeMaterialType(materialType);
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

    const int nextType = nextMaterialPreset(object.mesh.materials[static_cast<size_t>(scene.selectedMeshMaterial)].materialType);
    for (MeshMaterial& objectMaterial : object.mesh.materials)
    {
        objectMaterial.materialType = nextType;
        applyMaterialDefaults(objectMaterial);
    }
    syncSelectedMeshMaterialToCombined(scene);
}

void setSelectedMeshMaterialType(SceneState& scene, const int materialType)
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

    const int safeType = sanitizeMaterialType(materialType);
    for (MeshMaterial& objectMaterial : object.mesh.materials)
    {
        objectMaterial.materialType = safeType;
        applyMaterialDefaults(objectMaterial);
    }
    syncSelectedMeshMaterialToCombined(scene);
}

bool addBuiltInMeshPrimitive(SceneState& scene, const int primitiveType)
{
    const float xOffset = static_cast<float>(scene.meshObjects.size()) * 1.4f - 1.4f;
    if (primitiveType == BuiltInMeshCube)
    {
        scene.meshObjects.push_back(makeBuiltInMeshObject(
            createCubeMesh(),
            "built-in cube",
            make_float3(xOffset, 0.5f, -2.6f),
            make_float3(0.0f, 25.0f, 0.0f),
            make_float3(1.4f, 1.4f, 1.4f)));
    }
    else if (primitiveType == BuiltInMeshPyramid)
    {
        scene.meshObjects.push_back(makeBuiltInMeshObject(
            createPyramidMesh(),
            "built-in pyramid",
            make_float3(xOffset, 0.0f, -2.6f),
            make_float3(0.0f, -18.0f, 0.0f),
            make_float3(1.4f, 1.4f, 1.4f)));
    }
    else if (primitiveType == BuiltInMeshPlane)
    {
        scene.meshObjects.push_back(makeBuiltInMeshObject(
            createPlaneMesh(),
            "built-in panel",
            make_float3(xOffset, 0.02f, -2.6f),
            make_float3(0.0f, 0.0f, 0.0f),
            make_float3(3.0f, 1.0f, 3.0f)));
    }
    else
    {
        return false;
    }

    scene.selectedMeshObject = static_cast<int>(scene.meshObjects.size()) - 1;
    scene.selectedMeshMaterial = 0;
    syncCompatibilityMesh(scene);
    clampScene(scene);
    return true;
}

bool removeSelectedMeshObject(SceneState& scene)
{
    if (scene.meshObjects.size() <= 1)
    {
        clampScene(scene);
        return false;
    }
    if (scene.selectedMeshObject < 0 || scene.selectedMeshObject >= static_cast<int>(scene.meshObjects.size()))
    {
        clampScene(scene);
        return false;
    }

    scene.meshObjects.erase(scene.meshObjects.begin() + scene.selectedMeshObject);
    if (scene.selectedMeshObject >= static_cast<int>(scene.meshObjects.size()))
    {
        scene.selectedMeshObject = static_cast<int>(scene.meshObjects.size()) - 1;
    }
    scene.selectedMeshMaterial = 0;
    syncCompatibilityMesh(scene);
    clampScene(scene);
    return true;
}

bool setSelectedMeshPosition(SceneState& scene, const float3 position)
{
    MeshObject* object = selectedMeshObject(scene);
    if (object == nullptr)
    {
        return false;
    }

    object->position = clamp3(position, make_float3(-50.0f, -10.0f, -50.0f), make_float3(50.0f, 50.0f, 50.0f));
    updateMeshObjectTransform(*object);
    return true;
}

bool setSelectedMeshRotation(SceneState& scene, const float3 rotation)
{
    MeshObject* object = selectedMeshObject(scene);
    if (object == nullptr)
    {
        return false;
    }

    object->rotation = clamp3(rotation, make_float3(-360.0f, -360.0f, -360.0f), make_float3(360.0f, 360.0f, 360.0f));
    updateMeshObjectTransform(*object);
    return true;
}

bool setSelectedMeshScale(SceneState& scene, const float3 scale)
{
    MeshObject* object = selectedMeshObject(scene);
    if (object == nullptr)
    {
        return false;
    }

    object->scale = clamp3(scale, make_float3(0.05f, 0.05f, 0.05f), make_float3(20.0f, 20.0f, 20.0f));
    updateMeshObjectTransform(*object);
    return true;
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
