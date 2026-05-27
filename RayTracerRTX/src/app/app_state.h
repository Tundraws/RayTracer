#pragma once

#include "asset_cache.h"
#include "camera.h"
#include "scene_config.h"

#include <chrono>
#include <filesystem>
#include <string>
#include <vector>

enum EditorObjectKind
{
    EditorObjectSphere = 0,
    EditorObjectMesh = 1,
    EditorObjectGroup = 2
};

enum HierarchySelectionKind
{
    HierarchySelectionScene = 0,
    HierarchySelectionCamera = 1,
    HierarchySelectionLight = 2,
    HierarchySelectionSphere = 3,
    HierarchySelectionMesh = 4,
    HierarchySelectionGroup = 5
};

struct AppState
{
    CameraState camera;
    InputState input;
    SceneState scene;
    bool cursorCaptured = true;
    int renderMode = RenderModeRealtime;
    int renderQuality = RenderQualityHigh;
    bool denoiserEnabled = false;
    bool denoiserAvailable = false;
    unsigned int progressiveSamples = 0;
    bool renderingPaused = false;
    bool imguiPanelVisible = true;
    std::vector<SceneBuildResult> scenePresets;
    std::vector<SceneBuildResult> scenePresetDefaults;
    std::vector<std::wstring> scenePresetNames;
    std::vector<std::filesystem::path> scenePresetConfigPaths;
    AssetCache assetCache;
    int scenePresetIndex = 0;
    std::string lastUiMessage;
    bool lastUiMessageIsError = false;
    bool rendererSceneRebuildRequested = false;
    bool imguiPanelPinnedRight = true;
    float imguiPanelWidth = 360.0f;
    float imguiPanelHeight = 680.0f;
    int editorPrimitiveToAdd = BuiltInMeshCube;
    int editorObjectKind = EditorObjectSphere;
    int environmentMode = SceneEnvironmentOpen;
    float roomWidth = 18.0f;
    float roomDepth = 18.0f;
    float roomHeight = 9.0f;
    int hierarchySelectionKind = HierarchySelectionScene;
    int hierarchySelectionIndex = 0;
    std::vector<int> groupSelectionSpheres;
    std::vector<int> groupSelectionMeshes;
    std::vector<SceneState> undoStack;
};

struct FrameStats
{
    int frameCount = 0;
    double hostAccumMs = 0.0;
    double gpuAccumMs = 0.0;
    double fps = 0.0;
    double avgHostMs = 0.0;
    double avgGpuMs = 0.0;
    std::chrono::steady_clock::time_point lastUpdate = std::chrono::steady_clock::now();
};
