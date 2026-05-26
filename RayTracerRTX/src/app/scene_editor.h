#pragma once

#include "scene.h"

struct SceneDirtyFlags
{
    bool camera = false;
    bool material = false;
    bool transform = false;
    bool geometry = false;
    bool lighting = false;
    bool renderSettings = false;
    bool fullRebuild = false;
};

struct SceneEditResult
{
    bool changed = false;
    SceneDirtyFlags dirty;
};

enum SceneHierarchySelectionKind
{
    SceneHierarchySelectionScene = 0,
    SceneHierarchySelectionCamera = 1,
    SceneHierarchySelectionLight = 2,
    SceneHierarchySelectionSphere = 3,
    SceneHierarchySelectionMesh = 4
};

bool hasDirtyFlags(const SceneDirtyFlags& dirty);
bool needsRendererSceneRebuild(const SceneDirtyFlags& dirty);
bool selectHierarchyObject(SceneState& scene, int selectionKind, int index);

SceneEditResult makeSceneEditResult(bool changed, SceneDirtyFlags dirty);
SceneEditResult makeCameraDirty(bool changed = true);
SceneEditResult makeMaterialDirty(bool changed = true);
SceneEditResult makeTransformDirty(bool changed = true);
SceneEditResult makeGeometryDirty(bool changed = true);
SceneEditResult makeLightingDirty(bool changed = true);
SceneEditResult makeRenderSettingsDirty(bool changed = true);
SceneEditResult makeMeshMaterialDirty(bool changed = true);

class SceneEditor
{
public:
    explicit SceneEditor(SceneState& scene);

    SceneEditResult addSphere();
    SceneEditResult addMeshPrimitive(int primitiveType);
    SceneEditResult deleteSelectedSphere();
    SceneEditResult deleteSelectedMeshObject();
    SceneEditResult clearScene();
    SceneEditResult removeAllSpheres();
    SceneEditResult removeAllMeshObjects();
    SceneEditResult restoreDefaultObjects();

    SceneEditResult applyEnvironmentMode(int environmentMode);
    SceneEditResult applyRoomDimensions(float width, float depth, float height);

    SceneEditResult setSelectedSphereRadius(float radius);
    SceneEditResult setSelectedSphereColor(float3 color);
    SceneEditResult moveSelectedSphere(float3 delta);
    SceneEditResult setSelectedSphereMaterialType(int materialType);
    SceneEditResult cycleSelectedSphereMaterialPreset();

    SceneEditResult setSelectedMeshPosition(float3 position);
    SceneEditResult setSelectedMeshRotation(float3 rotation);
    SceneEditResult setSelectedMeshScale(float3 scale);
    SceneEditResult setSelectedMeshMaterialType(int materialType);
    SceneEditResult cycleSelectedMeshMaterialPreset();

    SceneEditResult setLightPosition(float3 position);
    SceneEditResult moveLight(float3 delta);
    SceneEditResult setExposure(float value);
    SceneEditResult adjustExposure(float delta);
    SceneEditResult setSkyIntensity(float value);
    SceneEditResult adjustSkyIntensity(float delta);
    SceneEditResult setLightIntensity(float value);
    SceneEditResult adjustLightIntensity(float delta);

private:
    SceneState& scene_;
};
