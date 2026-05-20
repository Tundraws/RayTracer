#pragma once

#include "../common/rtx_shared.h"

#include <vector>

struct SphereGeometry
{
    float3 center;
    float radius;
};

struct SceneState
{
    std::vector<SphereGeometry> spheres;
    std::vector<SphereMaterial> materials;
    float3 lightPosition;
    int lightType = LightPoint;
    float lightRadius = 4.0f;
    int selectedSphere = 0;
};

SceneState makeDefaultScene();
void clampScene(SceneState& scene);
void moveSelectedSphere(SceneState& scene, const float3 delta);
void toggleSelectedMaterial(SceneState& scene);
void moveLight(SceneState& scene, const float3 delta);
void toggleLightType(SceneState& scene);
void changeLightRadius(SceneState& scene, float delta);
