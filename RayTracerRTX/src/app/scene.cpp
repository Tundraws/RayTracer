#include "scene.h"

#include "obj_loader.h"

#include <algorithm>
#include <cmath>
#include <filesystem>
#include <functional>
#include <limits>
#include <utility>

#ifndef RAYTRACERRTX_SOURCE_DIR
#define RAYTRACERRTX_SOURCE_DIR ""
#endif

namespace
{
constexpr float kMinSphereRadius = 0.25f;
constexpr float kMaxSphereRadius = 5.0f;
constexpr float kPi = 3.14159265358979323846f;
constexpr int kSceneObjectSphere = 3;
constexpr int kSceneObjectMesh = 4;

struct SceneFootprint
{
    float x = 0.0f;
    float z = 0.0f;
    float radius = 1.0f;
};

float clampScalar(const float v, const float minV, const float maxV)
{
    return v < minV ? minV : (v > maxV ? maxV : v);
}

float3 add3(const float3 a, const float3 b)
{
    return make_float3(a.x + b.x, a.y + b.y, a.z + b.z);
}

float3 sub3(const float3 a, const float3 b)
{
    return make_float3(a.x - b.x, a.y - b.y, a.z - b.z);
}

float3 mul3(const float3 a, const float3 b)
{
    return make_float3(a.x * b.x, a.y * b.y, a.z * b.z);
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
        return;
    }

    scene.mesh = {};
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

bool isEnvironmentObject(const MeshObject& object)
{
    return object.assetReference.rfind("environment:", 0) == 0;
}

float normalizedAbsAngle(const float degrees)
{
    float angle = std::fmod(std::fabs(degrees), 360.0f);
    if (angle > 180.0f)
    {
        angle = 360.0f - angle;
    }
    return angle;
}

bool isHorizontalSupportPanel(const MeshObject& object)
{
    const bool isPanel = object.assetReference.find("panel") != std::string::npos ||
        object.assetReference.find("plane") != std::string::npos ||
        isEnvironmentObject(object);
    if (!isPanel)
    {
        return false;
    }

    return normalizedAbsAngle(object.rotation.x) < 8.0f &&
        normalizedAbsAngle(object.rotation.z) < 8.0f;
}

bool supportPanelContains(const MeshObject& object, const float x, const float z)
{
    const float halfX = std::max(0.25f, std::fabs(object.scale.x) * 0.5f);
    const float halfZ = std::max(0.25f, std::fabs(object.scale.z) * 0.5f);
    return x >= object.position.x - halfX &&
        x <= object.position.x + halfX &&
        z >= object.position.z - halfZ &&
        z <= object.position.z + halfZ;
}

float supportFloorYAt(const SceneState& scene, const float x, const float z)
{
    float floorY = 0.0f;
    bool found = false;
    for (const MeshObject& object : scene.meshObjects)
    {
        if (!isHorizontalSupportPanel(object) || !supportPanelContains(object, x, z))
        {
            continue;
        }
        if (!found || object.position.y > floorY)
        {
            floorY = object.position.y;
            found = true;
        }
    }
    return found ? floorY : 0.0f;
}

bool sceneHasSupportPanel(const SceneState& scene)
{
    for (const MeshObject& object : scene.meshObjects)
    {
        if (isHorizontalSupportPanel(object))
        {
            return true;
        }
    }
    return false;
}

bool hasSupportAt(const SceneState& scene, const float x, const float z)
{
    for (const MeshObject& object : scene.meshObjects)
    {
        if (isHorizontalSupportPanel(object) && supportPanelContains(object, x, z))
        {
            return true;
        }
    }
    return false;
}

bool placementIsOnAvailableSupport(const SceneState& scene, const float x, const float z)
{
    return !sceneHasSupportPanel(scene) || hasSupportAt(scene, x, z);
}

float meshBottomOffset(const MeshObject& object)
{
    if (object.assetReference.find("panel") != std::string::npos ||
        object.assetReference.find("plane") != std::string::npos)
    {
        return 0.02f;
    }

    if (object.mesh.vertices.empty())
    {
        return 0.0f;
    }

    float minY = std::numeric_limits<float>::max();
    for (const MeshVertex& vertex : object.mesh.vertices)
    {
        const float3 scaled = make_float3(
            vertex.position.x * object.scale.x,
            vertex.position.y * object.scale.y,
            vertex.position.z * object.scale.z);
        const float3 local = rotateEulerXyz(scaled, object.rotation);
        minY = std::min(minY, local.y);
    }
    return std::max(0.0f, -minY);
}

float3 placeSphereOnSupport(const SceneState& scene, const float3 position, const float radius)
{
    const float floorY = supportFloorYAt(scene, position.x, position.z);
    return make_float3(position.x, floorY + radius, position.z);
}

float3 placeMeshOnSupport(const SceneState& scene, const MeshObject& object, const float3 position)
{
    const float floorY = supportFloorYAt(scene, position.x, position.z);
    return make_float3(position.x, floorY + meshBottomOffset(object), position.z);
}

float footprintDistance2(const float x0, const float z0, const float x1, const float z1)
{
    const float dx = x0 - x1;
    const float dz = z0 - z1;
    return dx * dx + dz * dz;
}

float meshFootprintRadius(const MeshObject& object)
{
    const float maxHorizontalScale = std::max(std::fabs(object.scale.x), std::fabs(object.scale.z));
    if (object.assetReference.find("panel") != std::string::npos ||
        object.assetReference.find("plane") != std::string::npos)
    {
        return std::max(0.8f, maxHorizontalScale * 0.72f);
    }
    float radius = maxHorizontalScale * 0.82f;
    for (const MeshVertex& vertex : object.mesh.vertices)
    {
        const float3 scaled = make_float3(
            vertex.position.x * object.scale.x,
            vertex.position.y * object.scale.y,
            vertex.position.z * object.scale.z);
        const float3 local = rotateEulerXyz(scaled, object.rotation);
        radius = std::max(radius, std::sqrt(local.x * local.x + local.z * local.z));
    }
    return std::max(0.9f, radius);
}

std::vector<SceneFootprint> collectEditableFootprints(const SceneState& scene)
{
    std::vector<SceneFootprint> footprints;
    footprints.reserve(scene.spheres.size() + scene.meshObjects.size());

    for (const SphereGeometry& sphere : scene.spheres)
    {
        footprints.push_back({sphere.center.x, sphere.center.z, std::max(0.35f, sphere.radius)});
    }
    for (const MeshObject& object : scene.meshObjects)
    {
        if (isEnvironmentObject(object) || isHorizontalSupportPanel(object))
        {
            continue;
        }
        footprints.push_back({object.position.x, object.position.z, meshFootprintRadius(object)});
    }
    return footprints;
}

bool footprintIsFree(const std::vector<SceneFootprint>& footprints, const float x, const float z, const float radius)
{
    constexpr float padding = 0.04f;
    for (const SceneFootprint& footprint : footprints)
    {
        const float minDistance = radius + footprint.radius + padding;
        if (footprintDistance2(x, z, footprint.x, footprint.z) < minDistance * minDistance)
        {
            return false;
        }
    }
    return true;
}

float3 findAdjacentPlacementOnFloor(const SceneState& scene, const SceneFootprint& anchor, const float3 preferred, const float radius)
{
    const std::vector<SceneFootprint> footprints = collectEditableFootprints(scene);
    const float safeRadius = std::max(0.35f, radius);
    const float distance = anchor.radius + safeRadius + 0.06f;
    const float2 directions[] = {
        make_float2(1.0f, 0.0f),
        make_float2(-1.0f, 0.0f),
        make_float2(0.0f, 1.0f),
        make_float2(0.0f, -1.0f)
    };

    for (const float2 direction : directions)
    {
        const float x = anchor.x + direction.x * distance;
        const float z = anchor.z + direction.y * distance;
        if (placementIsOnAvailableSupport(scene, x, z) && footprintIsFree(footprints, x, z, safeRadius))
        {
            return make_float3(
                clampScalar(x, -35.0f, 35.0f),
                preferred.y,
                clampScalar(z, -35.0f, 35.0f));
        }
    }

    return preferred;
}

float3 findFreePlacementOnFloor(const SceneState& scene, const float3 preferred, const float radius)
{
    const std::vector<SceneFootprint> footprints = collectEditableFootprints(scene);
    const float safeRadius = std::max(0.35f, radius);
    if (placementIsOnAvailableSupport(scene, preferred.x, preferred.z) &&
        footprintIsFree(footprints, preferred.x, preferred.z, safeRadius))
    {
        return preferred;
    }

    const float spacing = safeRadius * 2.0f + 0.08f;
    const float2 directions[] = {
        make_float2(1.0f, 0.0f),
        make_float2(-1.0f, 0.0f),
        make_float2(0.0f, 1.0f),
        make_float2(0.0f, -1.0f),
        make_float2(1.0f, 1.0f),
        make_float2(-1.0f, 1.0f),
        make_float2(1.0f, -1.0f),
        make_float2(-1.0f, -1.0f)
    };

    for (int ring = 1; ring <= 12; ++ring)
    {
        for (const float2 direction : directions)
        {
            const float length = std::sqrt(direction.x * direction.x + direction.y * direction.y);
            const float x = preferred.x + direction.x / length * spacing * static_cast<float>(ring);
            const float z = preferred.z + direction.y / length * spacing * static_cast<float>(ring);
            if (placementIsOnAvailableSupport(scene, x, z) && footprintIsFree(footprints, x, z, safeRadius))
            {
                return make_float3(
                    clampScalar(x, -35.0f, 35.0f),
                    preferred.y,
                    clampScalar(z, -35.0f, 35.0f));
            }
        }
    }

    return make_float3(
        clampScalar(preferred.x + spacing, -35.0f, 35.0f),
        preferred.y,
        clampScalar(preferred.z, -35.0f, 35.0f));
}

bool selectedMeshMatchesPrimitive(const MeshObject& object, const int primitiveType)
{
    if (primitiveType == BuiltInMeshCube)
    {
        return object.assetReference.find("cube") != std::string::npos;
    }
    if (primitiveType == BuiltInMeshPyramid)
    {
        return object.assetReference.find("pyramid") != std::string::npos;
    }
    if (primitiveType == BuiltInMeshPlane)
    {
        return object.assetReference.find("panel") != std::string::npos ||
            object.assetReference.find("plane") != std::string::npos;
    }
    return false;
}

bool sceneObjectRefEquals(const SceneObjectRef a, const SceneObjectRef b)
{
    return a.kind == b.kind && a.index == b.index;
}

bool isValidSceneObjectRef(const SceneState& scene, const SceneObjectRef ref)
{
    if (ref.kind == kSceneObjectSphere)
    {
        return ref.index >= 0 && ref.index < static_cast<int>(scene.spheres.size());
    }
    if (ref.kind == kSceneObjectMesh)
    {
        return ref.index >= 0 && ref.index < static_cast<int>(scene.meshObjects.size());
    }
    return false;
}

float3 objectPosition(const SceneState& scene, const SceneObjectRef ref)
{
    if (ref.kind == kSceneObjectSphere)
    {
        return scene.spheres[static_cast<size_t>(ref.index)].center;
    }
    return scene.meshObjects[static_cast<size_t>(ref.index)].position;
}

float3 averageObjectPosition(const SceneState& scene, const std::vector<SceneObjectRef>& refs)
{
    if (refs.empty())
    {
        return make_float3(0.0f, 0.0f, 0.0f);
    }

    float3 total = make_float3(0.0f, 0.0f, 0.0f);
    int count = 0;
    for (const SceneObjectRef ref : refs)
    {
        if (!isValidSceneObjectRef(scene, ref))
        {
            continue;
        }
        total = add3(total, objectPosition(scene, ref));
        ++count;
    }
    if (count == 0)
    {
        return make_float3(0.0f, 0.0f, 0.0f);
    }
    const float invCount = 1.0f / static_cast<float>(count);
    return make_float3(total.x * invCount, total.y * invCount, total.z * invCount);
}

void pruneSceneGroups(SceneState& scene)
{
    for (SceneGroup& group : scene.groups)
    {
        group.objects.erase(
            std::remove_if(group.objects.begin(), group.objects.end(), [&](const SceneObjectRef ref)
            {
                return !isValidSceneObjectRef(scene, ref);
            }),
            group.objects.end());
    }
    scene.groups.erase(
        std::remove_if(scene.groups.begin(), scene.groups.end(), [](const SceneGroup& group)
        {
            return group.objects.size() < 2;
        }),
        scene.groups.end());
}

void removeObjectFromGroupsAfterErase(SceneState& scene, const int kind, const int erasedIndex)
{
    for (SceneGroup& group : scene.groups)
    {
        group.objects.erase(
            std::remove_if(group.objects.begin(), group.objects.end(), [&](SceneObjectRef& ref)
            {
                if (ref.kind != kind)
                {
                    return false;
                }
                if (ref.index == erasedIndex)
                {
                    return true;
                }
                if (ref.index > erasedIndex)
                {
                    --ref.index;
                }
                return false;
            }),
            group.objects.end());
    }
    pruneSceneGroups(scene);
}

void moveSceneObject(SceneState& scene, const SceneObjectRef ref, const float3 delta)
{
    if (ref.kind == kSceneObjectSphere)
    {
        SphereGeometry& sphere = scene.spheres[static_cast<size_t>(ref.index)];
        sphere.center = add3(sphere.center, delta);
    }
    else if (ref.kind == kSceneObjectMesh)
    {
        MeshObject& object = scene.meshObjects[static_cast<size_t>(ref.index)];
        object.position = add3(object.position, delta);
        updateMeshObjectTransform(object);
    }
}

void transformSceneObjectAroundPivot(
    SceneState& scene,
    const SceneObjectRef ref,
    const float3 pivot,
    const float3 rotationDelta,
    const float3 scaleRatio)
{
    float3 relative = sub3(objectPosition(scene, ref), pivot);
    relative = mul3(relative, scaleRatio);
    relative = rotateEulerXyz(relative, rotationDelta);
    const float3 nextPosition = add3(pivot, relative);

    if (ref.kind == kSceneObjectSphere)
    {
        SphereGeometry& sphere = scene.spheres[static_cast<size_t>(ref.index)];
        sphere.center = nextPosition;
        const float radiusScale = std::max(0.1f, (std::fabs(scaleRatio.x) + std::fabs(scaleRatio.y) + std::fabs(scaleRatio.z)) / 3.0f);
        sphere.radius *= radiusScale;
    }
    else if (ref.kind == kSceneObjectMesh)
    {
        MeshObject& object = scene.meshObjects[static_cast<size_t>(ref.index)];
        object.position = nextPosition;
        object.rotation = add3(object.rotation, rotationDelta);
        object.scale = mul3(object.scale, scaleRatio);
        updateMeshObjectTransform(object);
    }
}

MeshObject makeEnvironmentPanel(
    std::string displayName,
    const float3 position,
    const float3 rotation,
    const float3 scale,
    const float3 color)
{
    MeshData mesh = createPlaneMesh();
    for (MeshMaterial& material : mesh.materials)
    {
        material.name = displayName;
        material.color = color;
        material.materialType = MaterialDiffuse;
        material.roughness = 0.64f;
        material.alpha = 1.0f;
    }

    MeshObject panel = makeBuiltInMeshObject(
        std::move(mesh),
        "environment:" + displayName,
        position,
        rotation,
        scale);
    panel.displayName = std::move(displayName);
    return panel;
}

void removeEnvironmentObjects(SceneState& scene)
{
    scene.meshObjects.erase(
        std::remove_if(scene.meshObjects.begin(), scene.meshObjects.end(), isEnvironmentObject),
        scene.meshObjects.end());
    syncCompatibilityMesh(scene);
    clampScene(scene);
}

void addEditableFloor(SceneState& scene)
{
    scene.meshObjects.insert(
        scene.meshObjects.begin(),
        makeEnvironmentPanel(
            "Пол",
            make_float3(0.0f, 0.0f, 0.0f),
            make_float3(0.0f, 0.0f, 0.0f),
            make_float3(18.0f, 1.0f, 18.0f),
            make_float3(0.48f, 0.50f, 0.48f)));
    scene.selectedMeshObject = 0;
    scene.selectedMeshMaterial = 0;
    syncCompatibilityMesh(scene);
    clampScene(scene);
}

void addRoomPanels(SceneState& scene, const float width, const float depth, const float height)
{
    const float safeWidth = clampScalar(width, 4.0f, 80.0f);
    const float safeDepth = clampScalar(depth, 4.0f, 80.0f);
    const float safeHeight = clampScalar(height, 2.0f, 40.0f);
    const float halfWidth = safeWidth * 0.5f;
    const float halfDepth = safeDepth * 0.5f;
    const float halfHeight = safeHeight * 0.5f;

    scene.meshObjects.insert(scene.meshObjects.begin(), {
        makeEnvironmentPanel("Пол", make_float3(0.0f, 0.0f, 0.0f), make_float3(0.0f, 0.0f, 0.0f), make_float3(safeWidth, 1.0f, safeDepth), make_float3(0.46f, 0.48f, 0.46f)),
        makeEnvironmentPanel("Задняя стена", make_float3(0.0f, halfHeight, -halfDepth), make_float3(90.0f, 0.0f, 0.0f), make_float3(safeWidth, 1.0f, safeHeight), make_float3(0.54f, 0.56f, 0.58f)),
        makeEnvironmentPanel("Левая стена", make_float3(-halfWidth, halfHeight, 0.0f), make_float3(0.0f, 0.0f, -90.0f), make_float3(safeHeight, 1.0f, safeDepth), make_float3(0.52f, 0.50f, 0.48f)),
        makeEnvironmentPanel("Правая стена", make_float3(halfWidth, halfHeight, 0.0f), make_float3(0.0f, 0.0f, 90.0f), make_float3(safeHeight, 1.0f, safeDepth), make_float3(0.48f, 0.50f, 0.54f)),
        makeEnvironmentPanel("Потолок", make_float3(0.0f, safeHeight, 0.0f), make_float3(180.0f, 0.0f, 0.0f), make_float3(safeWidth, 1.0f, safeDepth), make_float3(0.50f, 0.50f, 0.49f))
    });
    scene.selectedMeshObject = 0;
    scene.selectedMeshMaterial = 0;
    syncCompatibilityMesh(scene);
    clampScene(scene);
}

bool updateRoomPanels(SceneState& scene, const float width, const float depth, const float height)
{
    std::vector<MeshObject*> panels;
    panels.reserve(5);
    for (MeshObject& object : scene.meshObjects)
    {
        if (isEnvironmentObject(object))
        {
            panels.push_back(&object);
        }
    }
    if (panels.size() < 5)
    {
        return false;
    }

    const float safeWidth = clampScalar(width, 4.0f, 80.0f);
    const float safeDepth = clampScalar(depth, 4.0f, 80.0f);
    const float safeHeight = clampScalar(height, 2.0f, 40.0f);
    const float halfWidth = safeWidth * 0.5f;
    const float halfDepth = safeDepth * 0.5f;
    const float halfHeight = safeHeight * 0.5f;

    panels[0]->position = make_float3(0.0f, 0.0f, 0.0f);
    panels[0]->rotation = make_float3(0.0f, 0.0f, 0.0f);
    panels[0]->scale = make_float3(safeWidth, 1.0f, safeDepth);

    panels[1]->position = make_float3(0.0f, halfHeight, -halfDepth);
    panels[1]->rotation = make_float3(90.0f, 0.0f, 0.0f);
    panels[1]->scale = make_float3(safeWidth, 1.0f, safeHeight);

    panels[2]->position = make_float3(-halfWidth, halfHeight, 0.0f);
    panels[2]->rotation = make_float3(0.0f, 0.0f, -90.0f);
    panels[2]->scale = make_float3(safeHeight, 1.0f, safeDepth);

    panels[3]->position = make_float3(halfWidth, halfHeight, 0.0f);
    panels[3]->rotation = make_float3(0.0f, 0.0f, 90.0f);
    panels[3]->scale = make_float3(safeHeight, 1.0f, safeDepth);

    panels[4]->position = make_float3(0.0f, safeHeight, 0.0f);
    panels[4]->rotation = make_float3(180.0f, 0.0f, 0.0f);
    panels[4]->scale = make_float3(safeWidth, 1.0f, safeDepth);

    for (MeshObject* panel : panels)
    {
        updateMeshObjectTransform(*panel);
    }
    syncCompatibilityMesh(scene);
    clampScene(scene);
    return true;
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
    addEditableFloor(scene);
    scene.mesh = scene.meshObjects.size() > 1 ? scene.meshObjects[1].mesh : scene.mesh;
    scene.lightPosition = make_float3(10.0f, 14.0f, -10.0f);
    scene.exposure = 0.82f;
    scene.skyIntensity = 0.78f;
    scene.lightIntensity = 0.95f;
    scene.areaLightRadius = 0.0f;
    scene.environmentIntensity = 1.0f;
    scene.environmentType = "gradient";
    scene.showGroundPlane = false;
    scene.selectedSphere = 0;
    scene.selectedMeshObject = scene.meshObjects.size() > 1 ? 1 : 0;
    scene.selectedMeshMaterial = 0;
    clampScene(scene);
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

    pruneSceneGroups(scene);
    if (scene.selectedGroup < 0)
    {
        scene.selectedGroup = 0;
    }
    if (scene.selectedGroup >= static_cast<int>(scene.groups.size()))
    {
        scene.selectedGroup = scene.groups.empty() ? 0 : static_cast<int>(scene.groups.size()) - 1;
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
        sphere.center = findAdjacentPlacementOnFloor(
            scene,
            SceneFootprint{selected.center.x, selected.center.z, selected.radius},
            selected.center,
            sphere.radius);

        if (scene.selectedSphere < static_cast<int>(scene.materials.size()))
        {
            material = scene.materials[static_cast<size_t>(scene.selectedSphere)];
        }
    }
    sphere.center = findFreePlacementOnFloor(scene, sphere.center, sphere.radius);
    sphere.center = placeSphereOnSupport(scene, sphere.center, sphere.radius);

    scene.spheres.push_back(sphere);
    scene.materials.push_back(material);
    scene.selectedSphere = static_cast<int>(scene.spheres.size()) - 1;
    clampScene(scene);
    return true;
}

bool removeSelectedSphere(SceneState& scene)
{
    clampScene(scene);
    if (scene.spheres.empty())
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
    removeObjectFromGroupsAfterErase(scene, kSceneObjectSphere, index);
    scene.selectedSphere = scene.spheres.empty() ? 0 : std::min(index, static_cast<int>(scene.spheres.size()) - 1);
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

bool resetSelectedSphereMaterial(SceneState& scene)
{
    const int index = scene.selectedSphere;
    if (index < 0 || index >= static_cast<int>(scene.materials.size()))
    {
        return false;
    }

    scene.materials[static_cast<size_t>(index)] = makeDefaultSphereMaterial();
    return true;
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

void selectPreviousMeshObject(SceneState& scene)
{
    if (scene.meshObjects.empty())
    {
        scene.selectedMeshObject = 0;
        scene.selectedMeshMaterial = 0;
        return;
    }

    const int meshCount = static_cast<int>(scene.meshObjects.size());
    scene.selectedMeshObject = (scene.selectedMeshObject - 1 + meshCount) % meshCount;
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

bool resetSelectedMeshMaterial(SceneState& scene)
{
    if (scene.meshObjects.empty())
    {
        return false;
    }
    clampScene(scene);
    MeshObject& object = scene.meshObjects[static_cast<size_t>(scene.selectedMeshObject)];
    if (object.mesh.materials.empty())
    {
        return false;
    }

    for (MeshMaterial& material : object.mesh.materials)
    {
        const std::string name = material.name;
        const std::string texturePath = material.texturePath;
        const int textureIndex = material.textureIndex;
        const int textureEnabled = material.textureEnabled;
        const std::string normalTexturePath = material.normalTexturePath;
        const int normalTextureIndex = material.normalTextureIndex;
        const std::string metallicTexturePath = material.metallicTexturePath;
        const int metallicTextureIndex = material.metallicTextureIndex;
        const std::string roughnessTexturePath = material.roughnessTexturePath;
        const int roughnessTextureIndex = material.roughnessTextureIndex;

        material = MeshMaterial{};
        material.name = name;
        material.color = make_float3(0.72f, 0.76f, 0.72f);
        material.materialType = MaterialDiffuse;
        material.specularColor = make_float3(0.72f, 0.72f, 0.72f);
        material.roughness = 0.52f;
        material.ior = 1.5f;
        material.alpha = 1.0f;
        material.texturePath = texturePath;
        material.textureIndex = textureIndex;
        material.textureEnabled = textureEnabled;
        material.normalTexturePath = normalTexturePath;
        material.normalTextureIndex = normalTextureIndex;
        material.metallicTexturePath = metallicTexturePath;
        material.metallicTextureIndex = metallicTextureIndex;
        material.roughnessTexturePath = roughnessTexturePath;
        material.roughnessTextureIndex = roughnessTextureIndex;
    }
    syncSelectedMeshMaterialToCombined(scene);
    return true;
}

void applySelectedMeshMaterialToWholeObject(SceneState& scene)
{
    if (scene.meshObjects.empty())
    {
        return;
    }
    clampScene(scene);
    MeshObject& object = scene.meshObjects[static_cast<size_t>(scene.selectedMeshObject)];
    if (object.mesh.materials.empty() ||
        scene.selectedMeshMaterial < 0 ||
        scene.selectedMeshMaterial >= static_cast<int>(object.mesh.materials.size()))
    {
        return;
    }

    const MeshMaterial material = object.mesh.materials[static_cast<size_t>(scene.selectedMeshMaterial)];
    for (MeshMaterial& objectMaterial : object.mesh.materials)
    {
        objectMaterial = material;
    }
    syncSelectedMeshMaterialToCombined(scene);
}

bool addBuiltInMeshPrimitive(SceneState& scene, const int primitiveType)
{
    clampScene(scene);
    MeshObject object;
    const MeshObject* selected = selectedMeshObject(scene);
    if (selected != nullptr &&
        !isEnvironmentObject(*selected) &&
        selectedMeshMatchesPrimitive(*selected, primitiveType))
    {
        object = *selected;
    }
    else if (primitiveType == BuiltInMeshCube)
    {
        object = makeBuiltInMeshObject(
            createCubeMesh(),
            "built-in cube",
            make_float3(0.0f, 0.7f, -2.6f),
            make_float3(0.0f, 25.0f, 0.0f),
            make_float3(1.4f, 1.4f, 1.4f));
    }
    else if (primitiveType == BuiltInMeshPyramid)
    {
        object = makeBuiltInMeshObject(
            createPyramidMesh(),
            "built-in pyramid",
            make_float3(0.0f, 0.0f, -2.6f),
            make_float3(0.0f, -18.0f, 0.0f),
            make_float3(1.4f, 1.4f, 1.4f));
    }
    else if (primitiveType == BuiltInMeshPlane)
    {
        object = makeBuiltInMeshObject(
            createPlaneMesh(),
            "built-in panel",
            make_float3(0.0f, 0.02f, -2.6f),
            make_float3(0.0f, 0.0f, 0.0f),
            make_float3(3.0f, 1.0f, 3.0f));
    }
    else
    {
        return false;
    }

    const float placementRadius = meshFootprintRadius(object);
    const bool duplicateSelected = selected != nullptr && selectedMeshMatchesPrimitive(*selected, primitiveType);
    const float3 preferred = duplicateSelected
        ? findAdjacentPlacementOnFloor(
            scene,
            SceneFootprint{selected->position.x, selected->position.z, meshFootprintRadius(*selected)},
            selected->position,
            placementRadius)
        : object.position;
    object.position = findFreePlacementOnFloor(scene, preferred, placementRadius);
    object.position = placeMeshOnSupport(scene, object, object.position);
    updateMeshObjectTransform(object);

    if (primitiveType == BuiltInMeshCube)
    {
        object.displayName = object.displayName.empty() ? "built-in cube" : object.displayName;
    }
    else if (primitiveType == BuiltInMeshPyramid)
    {
        object.displayName = object.displayName.empty() ? "built-in pyramid" : object.displayName;
    }
    else if (primitiveType == BuiltInMeshPlane)
    {
        object.displayName = object.displayName.empty() ? "built-in panel" : object.displayName;
    }

    scene.meshObjects.push_back(std::move(object));
    scene.selectedMeshObject = static_cast<int>(scene.meshObjects.size()) - 1;
    scene.selectedMeshMaterial = 0;
    syncCompatibilityMesh(scene);
    clampScene(scene);
    return true;
}

bool addMeshObjectToScene(SceneState& scene, MeshObject object)
{
    clampScene(scene);
    if (isEmptyMesh(object.mesh) || !hasValidMeshMaterialIndices(object.mesh))
    {
        return false;
    }

    const float placementRadius = meshFootprintRadius(object);
    object.position = findFreePlacementOnFloor(scene, object.position, placementRadius);
    object.position = placeMeshOnSupport(scene, object, object.position);
    updateMeshObjectTransform(object);
    scene.meshObjects.push_back(std::move(object));
    scene.selectedMeshObject = static_cast<int>(scene.meshObjects.size()) - 1;
    scene.selectedMeshMaterial = 0;
    syncCompatibilityMesh(scene);
    clampScene(scene);
    return true;
}

bool removeSelectedMeshObject(SceneState& scene)
{
    if (scene.meshObjects.empty())
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
    removeObjectFromGroupsAfterErase(scene, kSceneObjectMesh, scene.selectedMeshObject);
    if (scene.selectedMeshObject >= static_cast<int>(scene.meshObjects.size()))
    {
        scene.selectedMeshObject = scene.meshObjects.empty() ? 0 : static_cast<int>(scene.meshObjects.size()) - 1;
    }
    scene.selectedMeshMaterial = 0;
    syncCompatibilityMesh(scene);
    clampScene(scene);
    return true;
}

bool removeAllSpheres(SceneState& scene)
{
    const bool changed = !scene.spheres.empty() || !scene.materials.empty();
    scene.spheres.clear();
    scene.materials.clear();
    scene.selectedSphere = 0;
    for (SceneGroup& group : scene.groups)
    {
        group.objects.erase(
            std::remove_if(group.objects.begin(), group.objects.end(), [](const SceneObjectRef ref)
            {
                return ref.kind == kSceneObjectSphere;
            }),
            group.objects.end());
    }
    clampScene(scene);
    return changed;
}

bool removeAllMeshObjects(SceneState& scene)
{
    const bool changed = !scene.meshObjects.empty() || !isEmptyMesh(scene.mesh);
    scene.meshObjects.clear();
    scene.mesh = {};
    scene.selectedMeshObject = 0;
    scene.selectedMeshMaterial = 0;
    for (SceneGroup& group : scene.groups)
    {
        group.objects.erase(
            std::remove_if(group.objects.begin(), group.objects.end(), [](const SceneObjectRef ref)
            {
                return ref.kind == kSceneObjectMesh;
            }),
            group.objects.end());
    }
    clampScene(scene);
    return changed;
}

bool clearSceneObjects(SceneState& scene)
{
    const bool spheresChanged = removeAllSpheres(scene);
    const bool meshesChanged = removeAllMeshObjects(scene);
    const bool changed = spheresChanged || meshesChanged;
    scene.showGroundPlane = false;
    clampScene(scene);
    return changed;
}

bool restoreDefaultSceneObjects(SceneState& scene)
{
    const SceneState defaults = makeDefaultScene();
    scene.spheres = defaults.spheres;
    scene.materials = defaults.materials;
    scene.meshObjects = defaults.meshObjects;
    scene.groups.clear();
    scene.mesh = defaults.mesh;
    scene.showGroundPlane = defaults.showGroundPlane;
    scene.selectedSphere = defaults.selectedSphere;
    scene.selectedMeshObject = defaults.selectedMeshObject;
    scene.selectedMeshMaterial = defaults.selectedMeshMaterial;
    clampScene(scene);
    return true;
}

bool applySceneEnvironmentMode(SceneState& scene, const int environmentMode)
{
    removeEnvironmentObjects(scene);
    scene.showGroundPlane = false;

    if (environmentMode == SceneEnvironmentOpen)
    {
        addEditableFloor(scene);
    }
    else if (environmentMode == SceneEnvironmentRoom)
    {
        addRoomPanels(scene, 18.0f, 18.0f, 9.0f);
    }
    else if (environmentMode != SceneEnvironmentEmpty)
    {
        return false;
    }

    syncCompatibilityMesh(scene);
    clampScene(scene);
    return true;
}

bool applySceneRoomDimensions(SceneState& scene, const float width, const float depth, const float height)
{
    if (!updateRoomPanels(scene, width, depth, height))
    {
        removeEnvironmentObjects(scene);
        scene.showGroundPlane = false;
        addRoomPanels(scene, width, depth, height);
    }
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

    object->scale = clamp3(scale, make_float3(0.05f, 0.05f, 0.05f), make_float3(100.0f, 100.0f, 100.0f));
    updateMeshObjectTransform(*object);
    return true;
}

int findObjectGroupIndex(const SceneState& scene, const SceneObjectRef ref)
{
    for (int i = 0; i < static_cast<int>(scene.groups.size()); ++i)
    {
        const SceneGroup& group = scene.groups[static_cast<size_t>(i)];
        const auto found = std::find_if(group.objects.begin(), group.objects.end(), [&](const SceneObjectRef groupRef)
        {
            return sceneObjectRefEquals(groupRef, ref);
        });
        if (found != group.objects.end())
        {
            return i;
        }
    }
    return -1;
}

bool createSceneGroup(SceneState& scene, const std::vector<SceneObjectRef>& refs)
{
    clampScene(scene);

    std::vector<SceneObjectRef> filtered;
    filtered.reserve(refs.size());
    for (const SceneObjectRef ref : refs)
    {
        if (!isValidSceneObjectRef(scene, ref) || findObjectGroupIndex(scene, ref) >= 0)
        {
            continue;
        }
        const auto duplicate = std::find_if(filtered.begin(), filtered.end(), [&](const SceneObjectRef existing)
        {
            return sceneObjectRefEquals(existing, ref);
        });
        if (duplicate == filtered.end())
        {
            filtered.push_back(ref);
        }
    }

    if (filtered.size() < 2)
    {
        return false;
    }

    SceneGroup group;
    group.name = std::string("\xD0\x93\xD1\x80\xD1\x83\xD0\xBF\xD0\xBF\xD0\xB0 ") + std::to_string(scene.groups.size() + 1);
    group.objects = std::move(filtered);
    group.position = averageObjectPosition(scene, group.objects);
    group.rotation = make_float3(0.0f, 0.0f, 0.0f);
    group.scale = make_float3(1.0f, 1.0f, 1.0f);

    scene.groups.push_back(std::move(group));
    scene.selectedGroup = static_cast<int>(scene.groups.size()) - 1;
    clampScene(scene);
    return true;
}

bool ungroupSelectedSceneGroup(SceneState& scene)
{
    clampScene(scene);
    if (scene.groups.empty() ||
        scene.selectedGroup < 0 ||
        scene.selectedGroup >= static_cast<int>(scene.groups.size()))
    {
        return false;
    }

    scene.groups.erase(scene.groups.begin() + scene.selectedGroup);
    scene.selectedGroup = scene.groups.empty() ? 0 : std::min(scene.selectedGroup, static_cast<int>(scene.groups.size()) - 1);
    clampScene(scene);
    return true;
}

bool removeSelectedSceneGroup(SceneState& scene)
{
    clampScene(scene);
    if (scene.groups.empty() ||
        scene.selectedGroup < 0 ||
        scene.selectedGroup >= static_cast<int>(scene.groups.size()))
    {
        return false;
    }

    const std::vector<SceneObjectRef> refs = scene.groups[static_cast<size_t>(scene.selectedGroup)].objects;
    scene.groups.erase(scene.groups.begin() + scene.selectedGroup);

    std::vector<int> sphereIndices;
    std::vector<int> meshIndices;
    for (const SceneObjectRef ref : refs)
    {
        if (ref.kind == kSceneObjectSphere && ref.index >= 0 && ref.index < static_cast<int>(scene.spheres.size()))
        {
            sphereIndices.push_back(ref.index);
        }
        else if (ref.kind == kSceneObjectMesh && ref.index >= 0 && ref.index < static_cast<int>(scene.meshObjects.size()))
        {
            meshIndices.push_back(ref.index);
        }
    }

    std::sort(sphereIndices.begin(), sphereIndices.end(), std::greater<int>());
    sphereIndices.erase(std::unique(sphereIndices.begin(), sphereIndices.end()), sphereIndices.end());
    std::sort(meshIndices.begin(), meshIndices.end(), std::greater<int>());
    meshIndices.erase(std::unique(meshIndices.begin(), meshIndices.end()), meshIndices.end());

    for (const int meshIndex : meshIndices)
    {
        if (meshIndex < 0 || meshIndex >= static_cast<int>(scene.meshObjects.size()))
        {
            continue;
        }
        scene.meshObjects.erase(scene.meshObjects.begin() + meshIndex);
        removeObjectFromGroupsAfterErase(scene, kSceneObjectMesh, meshIndex);
    }

    for (const int sphereIndex : sphereIndices)
    {
        if (sphereIndex < 0 || sphereIndex >= static_cast<int>(scene.spheres.size()))
        {
            continue;
        }
        scene.spheres.erase(scene.spheres.begin() + sphereIndex);
        if (sphereIndex < static_cast<int>(scene.materials.size()))
        {
            scene.materials.erase(scene.materials.begin() + sphereIndex);
        }
        removeObjectFromGroupsAfterErase(scene, kSceneObjectSphere, sphereIndex);
    }

    scene.selectedSphere = scene.spheres.empty() ? 0 : std::min(scene.selectedSphere, static_cast<int>(scene.spheres.size()) - 1);
    scene.selectedMeshObject = scene.meshObjects.empty() ? 0 : std::min(scene.selectedMeshObject, static_cast<int>(scene.meshObjects.size()) - 1);
    scene.selectedMeshMaterial = 0;
    scene.selectedGroup = scene.groups.empty() ? 0 : std::min(scene.selectedGroup, static_cast<int>(scene.groups.size()) - 1);
    syncCompatibilityMesh(scene);
    clampScene(scene);
    return !refs.empty();
}

bool setSelectedGroupPosition(SceneState& scene, const float3 position)
{
    clampScene(scene);
    if (scene.groups.empty() ||
        scene.selectedGroup < 0 ||
        scene.selectedGroup >= static_cast<int>(scene.groups.size()))
    {
        return false;
    }

    SceneGroup& group = scene.groups[static_cast<size_t>(scene.selectedGroup)];
    const float3 clampedPosition = clamp3(position, make_float3(-50.0f, -10.0f, -50.0f), make_float3(50.0f, 50.0f, 50.0f));
    const float3 delta = sub3(clampedPosition, group.position);
    if (std::fabs(delta.x) < 0.0001f && std::fabs(delta.y) < 0.0001f && std::fabs(delta.z) < 0.0001f)
    {
        return false;
    }

    for (const SceneObjectRef ref : group.objects)
    {
        if (isValidSceneObjectRef(scene, ref))
        {
            moveSceneObject(scene, ref, delta);
        }
    }
    group.position = clampedPosition;
    syncCompatibilityMesh(scene);
    clampScene(scene);
    return true;
}

bool setSelectedGroupRotation(SceneState& scene, const float3 rotation)
{
    clampScene(scene);
    if (scene.groups.empty() ||
        scene.selectedGroup < 0 ||
        scene.selectedGroup >= static_cast<int>(scene.groups.size()))
    {
        return false;
    }

    SceneGroup& group = scene.groups[static_cast<size_t>(scene.selectedGroup)];
    const float3 clampedRotation = clamp3(rotation, make_float3(-360.0f, -360.0f, -360.0f), make_float3(360.0f, 360.0f, 360.0f));
    const float3 delta = sub3(clampedRotation, group.rotation);
    if (std::fabs(delta.x) < 0.0001f && std::fabs(delta.y) < 0.0001f && std::fabs(delta.z) < 0.0001f)
    {
        return false;
    }

    for (const SceneObjectRef ref : group.objects)
    {
        if (isValidSceneObjectRef(scene, ref))
        {
            transformSceneObjectAroundPivot(scene, ref, group.position, delta, make_float3(1.0f, 1.0f, 1.0f));
        }
    }
    group.rotation = clampedRotation;
    syncCompatibilityMesh(scene);
    clampScene(scene);
    return true;
}

bool setSelectedGroupScale(SceneState& scene, const float3 scale)
{
    clampScene(scene);
    if (scene.groups.empty() ||
        scene.selectedGroup < 0 ||
        scene.selectedGroup >= static_cast<int>(scene.groups.size()))
    {
        return false;
    }

    SceneGroup& group = scene.groups[static_cast<size_t>(scene.selectedGroup)];
    const float3 clampedScale = clamp3(scale, make_float3(0.05f, 0.05f, 0.05f), make_float3(100.0f, 100.0f, 100.0f));
    const float3 ratio = make_float3(
        clampedScale.x / std::max(0.05f, group.scale.x),
        clampedScale.y / std::max(0.05f, group.scale.y),
        clampedScale.z / std::max(0.05f, group.scale.z));
    if (std::fabs(ratio.x - 1.0f) < 0.0001f &&
        std::fabs(ratio.y - 1.0f) < 0.0001f &&
        std::fabs(ratio.z - 1.0f) < 0.0001f)
    {
        return false;
    }

    for (const SceneObjectRef ref : group.objects)
    {
        if (isValidSceneObjectRef(scene, ref))
        {
            transformSceneObjectAroundPivot(scene, ref, group.position, make_float3(0.0f, 0.0f, 0.0f), ratio);
        }
    }
    group.scale = clampedScale;
    syncCompatibilityMesh(scene);
    clampScene(scene);
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
