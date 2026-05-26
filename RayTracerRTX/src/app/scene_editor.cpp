#include "scene_editor.h"

#include <cmath>

namespace
{
bool equalFloat(const float a, const float b)
{
    return std::fabs(a - b) < 0.0001f;
}

bool equal3(const float3 a, const float3 b)
{
    return equalFloat(a.x, b.x) && equalFloat(a.y, b.y) && equalFloat(a.z, b.z);
}

bool hasSelectedSphere(const SceneState& scene)
{
    return scene.selectedSphere >= 0 && scene.selectedSphere < static_cast<int>(scene.spheres.size());
}

bool hasSelectedMesh(const SceneState& scene)
{
    return scene.selectedMeshObject >= 0 && scene.selectedMeshObject < static_cast<int>(scene.meshObjects.size());
}
}

bool hasDirtyFlags(const SceneDirtyFlags& dirty)
{
    return dirty.camera ||
        dirty.material ||
        dirty.transform ||
        dirty.geometry ||
        dirty.lighting ||
        dirty.renderSettings ||
        dirty.fullRebuild;
}

bool needsRendererSceneRebuild(const SceneDirtyFlags& dirty)
{
    return dirty.geometry || dirty.fullRebuild;
}

bool selectHierarchyObject(SceneState& scene, const int selectionKind, const int index)
{
    if (selectionKind == SceneHierarchySelectionSphere)
    {
        if (index < 0 || index >= static_cast<int>(scene.spheres.size()))
        {
            clampScene(scene);
            return false;
        }
        scene.selectedSphere = index;
        clampScene(scene);
        return true;
    }

    if (selectionKind == SceneHierarchySelectionMesh)
    {
        if (index < 0 || index >= static_cast<int>(scene.meshObjects.size()))
        {
            clampScene(scene);
            return false;
        }
        scene.selectedMeshObject = index;
        scene.selectedMeshMaterial = 0;
        clampScene(scene);
        return true;
    }

    return selectionKind == SceneHierarchySelectionScene ||
        selectionKind == SceneHierarchySelectionCamera ||
        selectionKind == SceneHierarchySelectionLight;
}

SceneEditResult makeSceneEditResult(const bool changed, const SceneDirtyFlags dirty)
{
    return {changed && hasDirtyFlags(dirty), dirty};
}

SceneEditResult makeCameraDirty(const bool changed)
{
    SceneDirtyFlags dirty;
    dirty.camera = true;
    return makeSceneEditResult(changed, dirty);
}

SceneEditResult makeMaterialDirty(const bool changed)
{
    SceneDirtyFlags dirty;
    dirty.material = true;
    return makeSceneEditResult(changed, dirty);
}

SceneEditResult makeTransformDirty(const bool changed)
{
    SceneDirtyFlags dirty;
    dirty.transform = true;
    return makeSceneEditResult(changed, dirty);
}

SceneEditResult makeGeometryDirty(const bool changed)
{
    SceneDirtyFlags dirty;
    dirty.geometry = true;
    return makeSceneEditResult(changed, dirty);
}

SceneEditResult makeLightingDirty(const bool changed)
{
    SceneDirtyFlags dirty;
    dirty.lighting = true;
    return makeSceneEditResult(changed, dirty);
}

SceneEditResult makeRenderSettingsDirty(const bool changed)
{
    SceneDirtyFlags dirty;
    dirty.renderSettings = true;
    return makeSceneEditResult(changed, dirty);
}

SceneEditResult makeMeshMaterialDirty(const bool changed)
{
    SceneDirtyFlags dirty;
    dirty.material = true;
    dirty.fullRebuild = true;
    return makeSceneEditResult(changed, dirty);
}

SceneEditor::SceneEditor(SceneState& scene)
    : scene_(scene)
{
}

SceneEditResult SceneEditor::addSphere()
{
    return makeGeometryDirty(::addSphere(scene_));
}

SceneEditResult SceneEditor::addMeshPrimitive(const int primitiveType)
{
    return makeGeometryDirty(addBuiltInMeshPrimitive(scene_, primitiveType));
}

SceneEditResult SceneEditor::deleteSelectedSphere()
{
    return makeGeometryDirty(removeSelectedSphere(scene_));
}

SceneEditResult SceneEditor::deleteSelectedMeshObject()
{
    return makeGeometryDirty(removeSelectedMeshObject(scene_));
}

SceneEditResult SceneEditor::clearScene()
{
    return makeGeometryDirty(clearSceneObjects(scene_));
}

SceneEditResult SceneEditor::removeAllSpheres()
{
    return makeGeometryDirty(::removeAllSpheres(scene_));
}

SceneEditResult SceneEditor::removeAllMeshObjects()
{
    return makeGeometryDirty(::removeAllMeshObjects(scene_));
}

SceneEditResult SceneEditor::restoreDefaultObjects()
{
    return makeGeometryDirty(restoreDefaultSceneObjects(scene_));
}

SceneEditResult SceneEditor::applyEnvironmentMode(const int environmentMode)
{
    return makeGeometryDirty(applySceneEnvironmentMode(scene_, environmentMode));
}

SceneEditResult SceneEditor::applyRoomDimensions(const float width, const float depth, const float height)
{
    size_t environmentPanelCount = 0;
    for (const MeshObject& object : scene_.meshObjects)
    {
        if (object.assetReference.rfind("environment:", 0) == 0)
        {
            ++environmentPanelCount;
        }
    }
    const bool changed = applySceneRoomDimensions(scene_, width, depth, height);
    return environmentPanelCount >= 5 ? makeTransformDirty(changed) : makeGeometryDirty(changed);
}

SceneEditResult SceneEditor::setSelectedSphereRadius(const float radius)
{
    if (!hasSelectedSphere(scene_))
    {
        return makeGeometryDirty(false);
    }

    const SphereGeometry before = scene_.spheres[static_cast<size_t>(scene_.selectedSphere)];
    ::setSelectedSphereRadius(scene_, radius);
    const SphereGeometry after = scene_.spheres[static_cast<size_t>(scene_.selectedSphere)];
    return makeGeometryDirty(!equalFloat(before.radius, after.radius) || !equal3(before.center, after.center));
}

SceneEditResult SceneEditor::setSelectedSphereColor(const float3 color)
{
    if (scene_.materials.empty() || !hasSelectedSphere(scene_))
    {
        return makeMaterialDirty(false);
    }

    const float3 before = scene_.materials[static_cast<size_t>(scene_.selectedSphere)].color;
    ::setSelectedSphereColor(scene_, color);
    const float3 after = scene_.materials[static_cast<size_t>(scene_.selectedSphere)].color;
    return makeMaterialDirty(!equal3(before, after));
}

SceneEditResult SceneEditor::moveSelectedSphere(const float3 delta)
{
    if (!hasSelectedSphere(scene_))
    {
        return makeTransformDirty(false);
    }

    const float3 before = scene_.spheres[static_cast<size_t>(scene_.selectedSphere)].center;
    ::moveSelectedSphere(scene_, delta);
    const float3 after = scene_.spheres[static_cast<size_t>(scene_.selectedSphere)].center;
    return makeTransformDirty(!equal3(before, after));
}

SceneEditResult SceneEditor::setSelectedSphereMaterialType(const int materialType)
{
    if (scene_.materials.empty() || !hasSelectedSphere(scene_))
    {
        return makeMaterialDirty(false);
    }

    const SphereMaterial before = scene_.materials[static_cast<size_t>(scene_.selectedSphere)];
    ::setSelectedSphereMaterialType(scene_, materialType);
    const SphereMaterial after = scene_.materials[static_cast<size_t>(scene_.selectedSphere)];
    return makeMaterialDirty(before.materialType != after.materialType ||
        !equal3(before.color, after.color) ||
        !equalFloat(before.roughness, after.roughness) ||
        !equalFloat(before.ior, after.ior) ||
        !equalFloat(before.alpha, after.alpha));
}

SceneEditResult SceneEditor::cycleSelectedSphereMaterialPreset()
{
    if (scene_.materials.empty() || !hasSelectedSphere(scene_))
    {
        return makeMaterialDirty(false);
    }

    const SphereMaterial before = scene_.materials[static_cast<size_t>(scene_.selectedSphere)];
    ::cycleSelectedSphereMaterialPreset(scene_);
    const SphereMaterial after = scene_.materials[static_cast<size_t>(scene_.selectedSphere)];
    return makeMaterialDirty(before.materialType != after.materialType ||
        !equal3(before.color, after.color) ||
        !equalFloat(before.roughness, after.roughness));
}

SceneEditResult SceneEditor::setSelectedMeshPosition(const float3 position)
{
    return makeTransformDirty(::setSelectedMeshPosition(scene_, position));
}

SceneEditResult SceneEditor::setSelectedMeshRotation(const float3 rotation)
{
    return makeTransformDirty(::setSelectedMeshRotation(scene_, rotation));
}

SceneEditResult SceneEditor::setSelectedMeshScale(const float3 scale)
{
    return makeTransformDirty(::setSelectedMeshScale(scene_, scale));
}

SceneEditResult SceneEditor::setSelectedMeshMaterialType(const int materialType)
{
    if (!hasSelectedMesh(scene_))
    {
        return makeMeshMaterialDirty(false);
    }

    const int materialIndex = scene_.selectedMeshMaterial;
    MeshObject& object = scene_.meshObjects[static_cast<size_t>(scene_.selectedMeshObject)];
    if (materialIndex < 0 || materialIndex >= static_cast<int>(object.mesh.materials.size()))
    {
        return makeMeshMaterialDirty(false);
    }

    const MeshMaterial before = object.mesh.materials[static_cast<size_t>(materialIndex)];
    ::setSelectedMeshMaterialType(scene_, materialType);
    const MeshMaterial after = object.mesh.materials[static_cast<size_t>(materialIndex)];
    return makeMeshMaterialDirty(before.materialType != after.materialType ||
        !equal3(before.color, after.color) ||
        !equalFloat(before.roughness, after.roughness) ||
        !equalFloat(before.ior, after.ior) ||
        !equalFloat(before.alpha, after.alpha));
}

SceneEditResult SceneEditor::cycleSelectedMeshMaterialPreset()
{
    if (!hasSelectedMesh(scene_))
    {
        return makeMeshMaterialDirty(false);
    }

    ::cycleSelectedMeshMaterialPreset(scene_);
    return makeMeshMaterialDirty(true);
}

SceneEditResult SceneEditor::setLightPosition(const float3 position)
{
    const float3 before = scene_.lightPosition;
    scene_.lightPosition = position;
    clampScene(scene_);
    return makeLightingDirty(!equal3(before, scene_.lightPosition));
}

SceneEditResult SceneEditor::moveLight(const float3 delta)
{
    const float3 before = scene_.lightPosition;
    ::moveLight(scene_, delta);
    return makeLightingDirty(!equal3(before, scene_.lightPosition));
}

SceneEditResult SceneEditor::setExposure(const float value)
{
    const float before = scene_.exposure;
    setSceneExposure(scene_, value);
    return makeLightingDirty(!equalFloat(before, scene_.exposure));
}

SceneEditResult SceneEditor::adjustExposure(const float delta)
{
    const float before = scene_.exposure;
    adjustSceneExposure(scene_, delta);
    return makeLightingDirty(!equalFloat(before, scene_.exposure));
}

SceneEditResult SceneEditor::setSkyIntensity(const float value)
{
    const float before = scene_.skyIntensity;
    setSceneSkyIntensity(scene_, value);
    return makeLightingDirty(!equalFloat(before, scene_.skyIntensity));
}

SceneEditResult SceneEditor::adjustSkyIntensity(const float delta)
{
    const float before = scene_.skyIntensity;
    adjustSceneSkyIntensity(scene_, delta);
    return makeLightingDirty(!equalFloat(before, scene_.skyIntensity));
}

SceneEditResult SceneEditor::setLightIntensity(const float value)
{
    const float before = scene_.lightIntensity;
    setSceneLightIntensity(scene_, value);
    return makeLightingDirty(!equalFloat(before, scene_.lightIntensity));
}

SceneEditResult SceneEditor::adjustLightIntensity(const float delta)
{
    const float before = scene_.lightIntensity;
    adjustSceneLightIntensity(scene_, delta);
    return makeLightingDirty(!equalFloat(before, scene_.lightIntensity));
}
