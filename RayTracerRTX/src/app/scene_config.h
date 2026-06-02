#pragma once

#include "camera.h"
#include "scene.h"

#include <filesystem>
#include <string>
#include <vector>

struct MeshTransformConfig
{
    float3 position = make_float3(0.0f, 0.0f, 0.0f);
    float3 rotation = make_float3(0.0f, 0.0f, 0.0f);
    float3 scale = make_float3(1.0f, 1.0f, 1.0f);
};

struct SceneMaterialConfig
{
    std::string name;
    int materialType = MaterialDiffuse;
    bool usedFallbackType = false;
    float3 baseColor = make_float3(0.8f, 0.8f, 0.78f);
    float3 specularColor = make_float3(1.0f, 1.0f, 1.0f);
    float roughness = 0.4f;
    float metallic = 0.0f;
    float ior = 1.5f;
    float alpha = 1.0f;
    std::string texturePath;
    std::string normalTexturePath;
    std::string metallicTexturePath;
    std::string roughnessTexturePath;
};

struct MeshObjectConfig
{
    std::filesystem::path meshPath;
    std::string primitive;
    std::string name;
    MeshTransformConfig transform;
    std::string materialOverride;
};

struct SphereConfig
{
    std::string name;
    float3 position = make_float3(0.0f, 1.0f, 0.0f);
    float radius = 1.0f;
    std::string materialOverride;
};

struct SceneGroupConfig
{
    std::string name;
    std::vector<SceneObjectRef> objects;
    float3 position = make_float3(0.0f, 0.0f, 0.0f);
    float3 rotation = make_float3(0.0f, 0.0f, 0.0f);
    float3 scale = make_float3(1.0f, 1.0f, 1.0f);
};

struct SceneConfig
{
    std::vector<MeshObjectConfig> meshObjects;
    bool hasMeshObjects = false;
    std::vector<SphereConfig> spheres;
    bool hasSpheres = false;
    std::vector<SceneGroupConfig> groups;
    std::vector<SceneMaterialConfig> materials;
    std::vector<std::string> sphereMaterialRefs;
    bool hasCamera = false;
    CameraState camera;
    bool hasLightPosition = false;
    float3 lightPosition = make_float3(10.0f, 14.0f, -10.0f);
    bool hasExposure = false;
    float exposure = 0.82f;
    bool hasSkyIntensity = false;
    float skyIntensity = 0.78f;
    bool hasSkyHorizonColor = false;
    float3 skyHorizonColor = make_float3(0.62f, 0.70f, 0.78f);
    bool hasSkyZenithColor = false;
    float3 skyZenithColor = make_float3(0.12f, 0.18f, 0.30f);
    bool hasSkyGradientBlend = false;
    float skyGradientBlend = 1.0f;
    bool hasLightIntensity = false;
    float lightIntensity = 0.95f;
    bool hasAreaLightRadius = false;
    float areaLightRadius = 0.0f;
    bool hasEnvironmentIntensity = false;
    float environmentIntensity = 1.0f;
    std::string environmentType = "gradient";
    std::filesystem::path environmentPath;
};

struct SceneConfigResult
{
    bool ok = false;
    SceneConfig config;
    std::string error;
};

struct SceneBuildResult
{
    bool ok = false;
    SceneState scene;
    CameraState camera;
    std::string error;
    std::vector<std::string> warnings;
};

class AssetCache;

SceneConfigResult loadSceneConfigFile(const std::filesystem::path& path);
SceneBuildResult buildSceneFromConfig(const SceneConfig& config, const std::filesystem::path& baseDirectory, AssetCache& assets);
SceneBuildResult buildSceneFromConfig(const SceneConfig& config, const std::filesystem::path& baseDirectory);
SceneBuildResult buildSceneFromMeshPath(const std::filesystem::path& meshPath, AssetCache& assets);
SceneBuildResult buildSceneFromMeshPath(const std::filesystem::path& meshPath);
SceneBuildResult buildDefaultSceneInput();
bool reloadScenePresetFromConfig(
    const std::filesystem::path& configPath,
    SceneBuildResult& preset,
    SceneState& scene,
    CameraState& camera,
    AssetCache& assets,
    bool& accumulationResetRequested,
    std::string& error);
bool applyScenePresetByIndex(const std::vector<SceneBuildResult>& presets, int index, SceneState& scene, CameraState& camera);
bool saveScenePresetByIndex(std::vector<SceneBuildResult>& presets, int index, const SceneState& scene, const CameraState& camera);
bool saveSceneToConfigFile(const std::filesystem::path& path, const SceneState& scene, const CameraState& camera, std::string& error);
bool resetSceneViewFromPreset(const SceneBuildResult& preset, SceneState& scene, CameraState& camera);
