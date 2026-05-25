#pragma once

#include "../common/rtx_shared.h"
#include "mesh.h"

#include <array>
#include <string>
#include <vector>

struct SphereGeometry
{
    float3 center;
    float radius;
};

struct MeshObject
{
    std::string assetReference;
    std::string displayName;
    MeshData mesh;
    float3 position = make_float3(0.0f, 0.0f, 0.0f);
    float3 rotation = make_float3(0.0f, 0.0f, 0.0f);
    float3 scale = make_float3(1.0f, 1.0f, 1.0f);
    std::array<float, 12> transform{
        1.0f, 0.0f, 0.0f, 0.0f,
        0.0f, 1.0f, 0.0f, 0.0f,
        0.0f, 0.0f, 1.0f, 0.0f};
};

enum BuiltInMeshPrimitive
{
    BuiltInMeshCube = 0,
    BuiltInMeshPyramid = 1,
    BuiltInMeshPlane = 2
};

enum SceneEnvironmentMode
{
    SceneEnvironmentOpen = 0,
    SceneEnvironmentRoom = 1,
    SceneEnvironmentEmpty = 2
};

struct SceneState
{
    std::vector<SphereGeometry> spheres;
    std::vector<SphereMaterial> materials;
    std::vector<MeshObject> meshObjects;
    MeshData mesh;
    float3 lightPosition;
    float exposure = 0.82f;
    float skyIntensity = 0.78f;
    float lightIntensity = 0.95f;
    float areaLightRadius = 0.0f;
    float environmentIntensity = 1.0f;
    std::string environmentType = "gradient";
    std::string environmentPath;
    MeshTexture environmentMap;
    bool showGroundPlane = false;
    int selectedSphere = 0;
    int selectedMeshObject = 0;
    int selectedMeshMaterial = 0;
};

SceneState makeDefaultScene();
void clampScene(SceneState& scene);
bool addSphere(SceneState& scene);
bool removeSelectedSphere(SceneState& scene);
void setSelectedSphereRadius(SceneState& scene, float radius);
void setSelectedSphereColor(SceneState& scene, float3 color);
void moveSelectedSphere(SceneState& scene, const float3 delta);
void toggleSelectedMaterial(SceneState& scene);
void cycleSelectedSphereMaterialPreset(SceneState& scene);
void setSelectedSphereMaterialType(SceneState& scene, int materialType);
void selectNextMeshObject(SceneState& scene);
void cycleSelectedMeshMaterialPreset(SceneState& scene);
void setSelectedMeshMaterialType(SceneState& scene, int materialType);
bool addBuiltInMeshPrimitive(SceneState& scene, int primitiveType);
bool removeSelectedMeshObject(SceneState& scene);
bool removeAllSpheres(SceneState& scene);
bool removeAllMeshObjects(SceneState& scene);
bool clearSceneObjects(SceneState& scene);
bool restoreDefaultSceneObjects(SceneState& scene);
bool applySceneEnvironmentMode(SceneState& scene, int environmentMode);
bool setSelectedMeshPosition(SceneState& scene, float3 position);
bool setSelectedMeshRotation(SceneState& scene, float3 rotation);
bool setSelectedMeshScale(SceneState& scene, float3 scale);
void moveLight(SceneState& scene, const float3 delta);
void setSceneExposure(SceneState& scene, float value);
void setSceneSkyIntensity(SceneState& scene, float value);
void setSceneLightIntensity(SceneState& scene, float value);
void adjustSceneExposure(SceneState& scene, float delta);
void adjustSceneSkyIntensity(SceneState& scene, float delta);
void adjustSceneLightIntensity(SceneState& scene, float delta);
