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

struct MeshObjectConfig
{
    std::filesystem::path meshPath;
    MeshTransformConfig transform;
};

struct SceneConfig
{
    std::vector<MeshObjectConfig> meshObjects;
    bool hasCamera = false;
    CameraState camera;
    bool hasLightPosition = false;
    float3 lightPosition = make_float3(10.0f, 14.0f, -10.0f);
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

SceneConfigResult loadSceneConfigFile(const std::filesystem::path& path);
SceneBuildResult buildSceneFromConfig(const SceneConfig& config, const std::filesystem::path& baseDirectory);
SceneBuildResult buildSceneFromMeshPath(const std::filesystem::path& meshPath);
SceneBuildResult buildDefaultSceneInput();
