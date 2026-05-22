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
    MeshData mesh;
    float3 position = make_float3(0.0f, 0.0f, 0.0f);
    float3 rotation = make_float3(0.0f, 0.0f, 0.0f);
    float3 scale = make_float3(1.0f, 1.0f, 1.0f);
    std::array<float, 12> transform{
        1.0f, 0.0f, 0.0f, 0.0f,
        0.0f, 1.0f, 0.0f, 0.0f,
        0.0f, 0.0f, 1.0f, 0.0f};
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
    int selectedSphere = 0;
    int selectedMeshObject = 0;
    int selectedMeshMaterial = 0;
};

SceneState makeDefaultScene();
void clampScene(SceneState& scene);
void moveSelectedSphere(SceneState& scene, const float3 delta);
void toggleSelectedMaterial(SceneState& scene);
void cycleSelectedSphereMaterialPreset(SceneState& scene);
void selectNextMeshObject(SceneState& scene);
void cycleSelectedMeshMaterialPreset(SceneState& scene);
void moveLight(SceneState& scene, const float3 delta);
