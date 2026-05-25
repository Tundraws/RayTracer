#include "application.h"

#include "../gpu/optix_renderer.h"
#include "asset_cache.h"
#include "camera.h"
#include "logger.h"
#include "material.h"
#include "scene.h"
#include "scene_config.h"

#include "../../third_party/imgui/backends/imgui_impl_glfw.h"
#include "../../third_party/imgui/backends/imgui_impl_opengl2.h"
#include "../../third_party/imgui/imgui.h"

#ifndef NOMINMAX
#define NOMINMAX
#endif
#include <windows.h>
#include <commdlg.h>

#include <GLFW/glfw3.h>
#define GLFW_EXPOSE_NATIVE_WIN32
#include <GLFW/glfw3native.h>
#include <GL/gl.h>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cctype>
#include <cstdio>
#include <cwctype>
#include <filesystem>
#include <iomanip>
#include <iostream>
#include <optional>
#include <sstream>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#ifndef RAYTRACERRTX_SOURCE_DIR
#define RAYTRACERRTX_SOURCE_DIR ""
#endif

namespace
{
int gWidth = 800;
int gHeight = 600;
float gImguiPanelX = 0.0f;
float gImguiPanelY = 0.0f;
float gImguiPanelWidth = 0.0f;
float gImguiPanelHeight = 0.0f;

enum EditorObjectKind
{
    EditorObjectSphere = 0,
    EditorObjectMesh = 1
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

void addScenePreset(AppState& appState, SceneBuildResult preset, std::wstring name, std::filesystem::path configPath = {});

float3 add3(const float3 a, const float3 b)
{
    return make_float3(a.x + b.x, a.y + b.y, a.z + b.z);
}

float3 sub3(const float3 a, const float3 b)
{
    return make_float3(a.x - b.x, a.y - b.y, a.z - b.z);
}

float3 mul3(const float3 a, const float value)
{
    return make_float3(a.x * value, a.y * value, a.z * value);
}

float dot3(const float3 a, const float3 b)
{
    return a.x * b.x + a.y * b.y + a.z * b.z;
}

float3 normalize3(const float3 v)
{
    const float length = std::sqrt(dot3(v, v));
    if (length <= 0.000001f)
    {
        return make_float3(0.0f, 0.0f, 0.0f);
    }

    return make_float3(v.x / length, v.y / length, v.z / length);
}

float clampf(const float value, const float minValue, const float maxValue)
{
    return value < minValue ? minValue : (value > maxValue ? maxValue : value);
}

float3 clamp3(const float3 value, const float3 minValue, const float3 maxValue)
{
    return make_float3(
        clampf(value.x, minValue.x, maxValue.x),
        clampf(value.y, minValue.y, maxValue.y),
        clampf(value.z, minValue.z, maxValue.z));
}

const char* u8c(const char8_t* text)
{
    return reinterpret_cast<const char*>(text);
}

std::string wideToUtf8(const std::wstring& text)
{
    if (text.empty())
    {
        return {};
    }

    const int size = WideCharToMultiByte(CP_UTF8, 0, text.c_str(), static_cast<int>(text.size()), nullptr, 0, nullptr, nullptr);
    if (size <= 0)
    {
        return {};
    }

    std::string result(static_cast<size_t>(size), '\0');
    WideCharToMultiByte(CP_UTF8, 0, text.c_str(), static_cast<int>(text.size()), result.data(), size, nullptr, nullptr);
    return result;
}

const wchar_t* qualityNameW(const int quality)
{
    switch (clampRenderQuality(quality))
    {
    case RenderQualityLow:
        return L"\u041D\u0418\u0417\u041A\u041E\u0415";
    case RenderQualityMedium:
        return L"\u0421\u0420\u0415\u0414\u041D\u0415\u0415";
    case RenderQualityPathTracing:
        return L"\u041D\u0410\u041A\u041E\u041F\u041B\u0415\u041D\u0418\u0415";
    default:
        return L"\u0412\u042B\u0421\u041E\u041A\u041E\u0415";
    }
}

const char* qualityNameUtf8(const int quality)
{
    switch (clampRenderQuality(quality))
    {
    case RenderQualityLow:
        return u8c(u8"\u041D\u0438\u0437\u043A\u043E\u0435");
    case RenderQualityMedium:
        return u8c(u8"\u0421\u0440\u0435\u0434\u043D\u0435\u0435");
    case RenderQualityPathTracing:
        return u8c(u8"\u041D\u0430\u043A\u043E\u043F\u043B\u0435\u043D\u0438\u0435");
    default:
        return u8c(u8"\u0412\u044B\u0441\u043E\u043A\u043E\u0435");
    }
}

const char* materialTypeNameUtf8(const int materialType)
{
    switch (materialType)
    {
    case MaterialMirror:
        return u8c(u8"\u0417\u0435\u0440\u043A\u0430\u043B\u044C\u043D\u044B\u0439");
    case MaterialMetal:
        return u8c(u8"\u041C\u0435\u0442\u0430\u043B\u043B");
    case MaterialDielectric:
        return u8c(u8"\u0421\u0442\u0435\u043A\u043B\u043E");
    default:
        return u8c(u8"\u041C\u0430\u0442\u043E\u0432\u044B\u0439");
    }
}

const char* builtInPrimitiveNameUtf8(const int primitiveType)
{
    switch (primitiveType)
    {
    case -1:
        return u8c(u8"\u0421\u0444\u0435\u0440\u0430");
    case BuiltInMeshPyramid:
        return u8c(u8"\u041F\u0438\u0440\u0430\u043C\u0438\u0434\u0430");
    case BuiltInMeshPlane:
        return u8c(u8"\u041F\u043B\u043E\u0441\u043A\u043E\u0441\u0442\u044C/\u043F\u0430\u043D\u0435\u043B\u044C");
    case BuiltInMeshCube:
    default:
        return u8c(u8"\u041A\u0443\u0431");
    }
}

bool builtInPrimitiveCombo(const char* id, int& primitiveType)
{
    const int primitiveTypes[] = {-1, BuiltInMeshCube, BuiltInMeshPyramid, BuiltInMeshPlane};
    bool changed = false;
    if (ImGui::BeginCombo(id, builtInPrimitiveNameUtf8(primitiveType)))
    {
        for (const int type : primitiveTypes)
        {
            const bool selected = primitiveType == type;
            if (ImGui::Selectable(builtInPrimitiveNameUtf8(type), selected))
            {
                primitiveType = type;
                changed = true;
            }
            if (selected)
            {
                ImGui::SetItemDefaultFocus();
            }
        }
        ImGui::EndCombo();
    }
    return changed;
}

bool materialTypeCombo(const char* id, int& materialType)
{
    const int materialTypes[] = {MaterialDiffuse, MaterialMirror, MaterialMetal, MaterialDielectric};
    bool changed = false;
    if (ImGui::BeginCombo(id, materialTypeNameUtf8(materialType)))
    {
        for (const int type : materialTypes)
        {
            const bool selected = materialType == type;
            if (ImGui::Selectable(materialTypeNameUtf8(type), selected))
            {
                materialType = type;
                changed = true;
            }
            if (selected)
            {
                ImGui::SetItemDefaultFocus();
            }
        }
        ImGui::EndCombo();
    }
    return changed;
}

bool qualityCombo(const char* id, int& quality)
{
    const int qualities[] = {RenderQualityLow, RenderQualityMedium, RenderQualityHigh, RenderQualityPathTracing};
    int selectedQuality = clampRenderQuality(quality);
    bool changed = false;
    if (ImGui::BeginCombo(id, qualityNameUtf8(selectedQuality)))
    {
        for (const int candidate : qualities)
        {
            const bool selected = selectedQuality == candidate;
            if (ImGui::Selectable(qualityNameUtf8(candidate), selected))
            {
                quality = candidate;
                changed = true;
            }
            if (selected)
            {
                ImGui::SetItemDefaultFocus();
            }
        }
        ImGui::EndCombo();
    }
    return changed;
}

std::string meshObjectLabel(const MeshObject& object, const int index)
{
    if (!object.displayName.empty())
    {
        return object.displayName;
    }
    if (!object.assetReference.empty())
    {
        return object.assetReference;
    }
    return std::string(u8c(u8"\u041C\u043E\u0434\u0435\u043B\u044C ")) + std::to_string(index + 1);
}

void applyQualityMode(AppState& appState)
{
    appState.renderQuality = clampRenderQuality(appState.renderQuality);
    if (renderQualityUsesPathTracing(appState.renderQuality))
    {
        appState.renderMode = RenderModeProgressive;
        appState.denoiserEnabled = renderQualityUsesDenoiser(appState.renderQuality);
    }
    else
    {
        appState.renderMode = RenderModeRealtime;
        appState.denoiserEnabled = false;
    }
    appState.progressiveSamples = 0;
}

struct HudTextCache
{
    HFONT font = nullptr;
};

HudTextCache& hudTextCache()
{
    static HudTextCache cache;
    return cache;
}

void releaseHudTextCache(HudTextCache& cache)
{
    if (cache.font != nullptr)
    {
        DeleteObject(cache.font);
        cache.font = nullptr;
    }
}

void ensureHudTextCache(int width, int height)
{
    HudTextCache& cache = hudTextCache();
    if (cache.font != nullptr)
    {
        return;
    }

    (void)width;
    (void)height;

    HDC screenDC = GetDC(nullptr);
    cache.font = CreateFontW(
        -18,
        0,
        0,
        0,
        FW_SEMIBOLD,
        FALSE,
        FALSE,
        FALSE,
        DEFAULT_CHARSET,
        OUT_DEFAULT_PRECIS,
        CLIP_DEFAULT_PRECIS,
        CLEARTYPE_QUALITY,
        DEFAULT_PITCH | FF_SWISS,
        L"Arial");
    ReleaseDC(nullptr, screenDC);
}

void drawHud(
    GLFWwindow* window,
    const SceneState& scene,
    const FrameStats& stats,
    const int renderMode,
    const int renderQuality,
    const bool denoiserEnabled,
    const bool denoiserAvailable,
    const unsigned int progressiveSamples,
    const std::wstring& presetName)
{
    if (window == nullptr)
    {
        return;
    }

    HWND hwnd = glfwGetWin32Window(window);
    if (hwnd == nullptr)
    {
        return;
    }

    HDC dc = GetDC(hwnd);
    if (dc == nullptr)
    {
        return;
    }

    struct DcGuard
    {
        HDC dc;
        HWND hwnd;
        ~DcGuard()
        {
            if (dc != nullptr)
            {
                ReleaseDC(hwnd, dc);
            }
        }
    } guard{dc, hwnd};

    ensureHudTextCache(gWidth, gHeight);
    HudTextCache& cache = hudTextCache();
    if (cache.font == nullptr)
    {
        return;
    }

    const int panelX = 18;
    const int panelY = 18;

    const int oldBkMode = SetBkMode(dc, TRANSPARENT);
    const COLORREF oldTextColor = SetTextColor(dc, RGB(0, 0, 0));
    HGDIOBJ oldFont = SelectObject(dc, cache.font);

    std::wostringstream line1;
    line1 << L"FPS: " << std::fixed << std::setprecision(1) << stats.fps
          << L"   GPU: " << stats.avgGpuMs << L" \u043C\u0441";

    const bool hasSphere = scene.selectedSphere >= 0 && scene.selectedSphere < static_cast<int>(scene.materials.size());
    const bool hasMesh = scene.selectedMeshObject >= 0 && scene.selectedMeshObject < static_cast<int>(scene.meshObjects.size());
    const MeshObject* meshObject = hasMesh ? &scene.meshObjects[static_cast<size_t>(scene.selectedMeshObject)] : nullptr;
    const bool hasMeshMaterial = meshObject != nullptr &&
        scene.selectedMeshMaterial >= 0 &&
        scene.selectedMeshMaterial < static_cast<int>(meshObject->mesh.materials.size());
    const MeshMaterial* meshMaterial = hasMeshMaterial ? &meshObject->mesh.materials[static_cast<size_t>(scene.selectedMeshMaterial)] : nullptr;

    std::wostringstream line2;
    line2 << L"\u0421\u0426\u0415\u041D\u0410: " << presetName;

    std::wostringstream line3;
    line3 << L"\u041E\u0411\u042A\u0415\u041A\u0422: "
          << (hasMesh ? L"\u0441\u0435\u0442\u043A\u0430 " : L"\u0441\u0435\u0442\u043A\u0430 ")
          << (hasMesh ? scene.selectedMeshObject + 1 : 0) << L"/" << scene.meshObjects.size()
          << L"   \u041C\u0410\u0422\u0415\u0420\u0418\u0410\u041B: "
          << (meshMaterial != nullptr ? materialNameW(meshMaterial->materialType) :
              (hasSphere ? materialNameW(scene.materials[scene.selectedSphere].materialType) : L"N/A"));

    std::wostringstream lineMode;
    lineMode << L"\u0420\u0415\u0416\u0418\u041C: " << (renderMode == RenderModeProgressive ? L"\u043D\u0430\u043A\u043E\u043F\u043B\u0435\u043D\u0438\u0435" : L"\u0440\u0435\u0430\u043B\u044C\u043D\u043E\u0435 \u0432\u0440\u0435\u043C\u044F")
             << L"   \u041A\u0410\u0427\u0415\u0421\u0422\u0412\u041E: " << qualityNameW(renderQuality)
             << L"   \u0421\u042D\u041C\u041F\u041B\u042B: " << progressiveSamples
             << L"   \u0428\u0423\u041C: " << (denoiserEnabled ? (denoiserAvailable ? L"\u0432\u043A\u043B" : L"\u043D\u0435\u0434\u043E\u0441\u0442\u0443\u043F\u0435\u043D") : L"\u0432\u044B\u043A\u043B");

    const std::wstring line4 = lineMode.str();
    std::wostringstream lineTuning;
    lineTuning << L"\u042D\u041A\u0421\u041F\u041E\u0417\u0418\u0426\u0418\u042F: " << std::setprecision(2) << scene.exposure
               << L"   \u041D\u0415\u0411\u041E: " << scene.skyIntensity
               << L"   \u0421\u0412\u0415\u0422: " << scene.lightIntensity;
    const std::wstring line5 = lineTuning.str();
    const std::wstring line6 = L"G \u0441\u0446\u0435\u043D\u0430   C \u0441\u0431\u0440\u043E\u0441   M/V \u043C\u0430\u0442\u0435\u0440\u0438\u0430\u043B   Q \u043A\u0430\u0447\u0435\u0441\u0442\u0432\u043E   P \u043D\u0430\u043A\u043E\u043F\u043B\u0435\u043D\u0438\u0435";
    const std::wstring line7 = L"4/5 \u044D\u043A\u0441\u043F\u043E\u0437\u0438\u0446\u0438\u044F   6/7 \u043D\u0435\u0431\u043E   8/9 \u0441\u0432\u0435\u0442";

    const std::wstring text1 = line1.str();
    const std::wstring text2 = line2.str();
    const std::wstring text3 = line3.str();
    TextOutW(dc, panelX, panelY, text1.c_str(), static_cast<int>(text1.size()));
    TextOutW(dc, panelX, panelY + 22, text2.c_str(), static_cast<int>(text2.size()));
    TextOutW(dc, panelX, panelY + 44, text3.c_str(), static_cast<int>(text3.size()));
    TextOutW(dc, panelX, panelY + 66, line4.c_str(), static_cast<int>(line4.size()));
    TextOutW(dc, panelX, panelY + 88, line5.c_str(), static_cast<int>(line5.size()));
    TextOutW(dc, panelX, panelY + 110, line6.c_str(), static_cast<int>(line6.size()));
    TextOutW(dc, panelX, panelY + 132, line7.c_str(), static_cast<int>(line7.size()));

    SelectObject(dc, oldFont);
    SetTextColor(dc, oldTextColor);
    SetBkMode(dc, oldBkMode);
}

void syncSelectedMeshMaterialToCombined(SceneState& scene)
{
    if (scene.selectedMeshObject < 0 ||
        scene.selectedMeshObject >= static_cast<int>(scene.meshObjects.size()) ||
        scene.selectedMeshMaterial < 0)
    {
        return;
    }

    size_t combinedMaterialIndex = 0;
    for (int i = 0; i < scene.selectedMeshObject && i < static_cast<int>(scene.meshObjects.size()); ++i)
    {
        combinedMaterialIndex += scene.meshObjects[static_cast<size_t>(i)].mesh.materials.size();
    }
    MeshObject& object = scene.meshObjects[static_cast<size_t>(scene.selectedMeshObject)];
    if (object.mesh.materials.empty() || combinedMaterialIndex >= scene.mesh.materials.size())
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

void applySelectedMeshMaterialToWholeObject(SceneState& scene)
{
    if (scene.selectedMeshObject < 0 ||
        scene.selectedMeshObject >= static_cast<int>(scene.meshObjects.size()))
    {
        return;
    }

    MeshObject& object = scene.meshObjects[static_cast<size_t>(scene.selectedMeshObject)];
    if (scene.selectedMeshMaterial < 0 ||
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

bool hasPresetIndex(const AppState& appState, const int index)
{
    return index >= 0 && index < static_cast<int>(appState.scenePresets.size());
}

void saveCurrentScenePreset(AppState& appState)
{
    if (!hasPresetIndex(appState, appState.scenePresetIndex))
    {
        return;
    }

    saveScenePresetByIndex(appState.scenePresets, appState.scenePresetIndex, appState.scene, appState.camera);
}

bool applyScenePreset(AppState& appState, const int index)
{
    if (!hasPresetIndex(appState, index))
    {
        return false;
    }

    saveCurrentScenePreset(appState);
    if (!applyScenePresetByIndex(appState.scenePresets, index, appState.scene, appState.camera))
    {
        return false;
    }

    appState.scenePresetIndex = index;
    appState.progressiveSamples = 0;
    appState.rendererSceneRebuildRequested = true;
    return true;
}

bool resetCurrentPresetView(AppState& appState)
{
    if (appState.scenePresetIndex < 0 ||
        appState.scenePresetIndex >= static_cast<int>(appState.scenePresetDefaults.size()))
    {
        return false;
    }

    const bool reset = resetSceneViewFromPreset(
        appState.scenePresetDefaults[static_cast<size_t>(appState.scenePresetIndex)],
        appState.scene,
        appState.camera);
    if (reset)
    {
        saveCurrentScenePreset(appState);
        appState.progressiveSamples = 0;
        appState.rendererSceneRebuildRequested = true;
    }
    return reset;
}

std::optional<std::filesystem::path> openMeshFileDialog(GLFWwindow* window)
{
    wchar_t fileName[MAX_PATH] = L"";

    OPENFILENAMEW ofn{};
    ofn.lStructSize = sizeof(ofn);
    ofn.hwndOwner = window != nullptr ? glfwGetWin32Window(window) : nullptr;
    ofn.lpstrTitle = L"\u0412\u044B\u0431\u0435\u0440\u0438\u0442\u0435 OBJ \u0438\u043B\u0438 glTF \u043C\u043E\u0434\u0435\u043B\u044C";
    ofn.lpstrFilter =
        L"3D \u043C\u043E\u0434\u0435\u043B\u0438 (*.obj;*.gltf;*.glb)\0*.obj;*.gltf;*.glb\0"
        L"OBJ (*.obj)\0*.obj\0"
        L"glTF (*.gltf;*.glb)\0*.gltf;*.glb\0"
        L"\u0412\u0441\u0435 \u0444\u0430\u0439\u043B\u044B (*.*)\0*.*\0";
    ofn.lpstrFile = fileName;
    ofn.nMaxFile = MAX_PATH;
    ofn.Flags = OFN_FILEMUSTEXIST | OFN_PATHMUSTEXIST | OFN_NOCHANGEDIR;

    if (GetOpenFileNameW(&ofn) == TRUE)
    {
        return std::filesystem::path(fileName);
    }
    return std::nullopt;
}

bool loadUserMeshPreset(AppState& appState, const std::filesystem::path& meshPath)
{
    SceneBuildResult loaded = buildSceneFromMeshPath(meshPath, appState.assetCache);
    if (!loaded.ok)
    {
        appState.lastUiMessage = loaded.error.empty() ? "Не удалось загрузить модель." : loaded.error;
        appState.lastUiMessageIsError = true;
        return false;
    }

    saveCurrentScenePreset(appState);
    std::wstring presetName = L"\u041C\u043E\u0434\u0435\u043B\u044C: ";
    presetName += meshPath.filename().wstring().empty() ? meshPath.wstring() : meshPath.filename().wstring();
    addScenePreset(appState, std::move(loaded), presetName);
    const int newPresetIndex = static_cast<int>(appState.scenePresets.size()) - 1;
    if (!applyScenePresetByIndex(appState.scenePresets, newPresetIndex, appState.scene, appState.camera))
    {
        appState.lastUiMessage = "Модель загрузилась, но не удалось применить сцену.";
        appState.lastUiMessageIsError = true;
        return false;
    }

    appState.scenePresetIndex = newPresetIndex;
    appState.progressiveSamples = 0;
    appState.rendererSceneRebuildRequested = true;
    appState.lastUiMessage = "Модель загружена: " + meshPath.filename().string();
    appState.lastUiMessageIsError = false;
    return true;
}

bool reloadCurrentSceneConfig(AppState& appState)
{
    if (!hasPresetIndex(appState, appState.scenePresetIndex) ||
        appState.scenePresetIndex >= static_cast<int>(appState.scenePresetConfigPaths.size()))
    {
        appState.lastUiMessage = "РЈ С‚РµРєСѓС‰РµР№ СЃС†РµРЅС‹ РЅРµС‚ JSON-С„Р°Р№Р»Р° РґР»СЏ РїРµСЂРµР·Р°РіСЂСѓР·РєРё.";
        appState.lastUiMessageIsError = true;
        return false;
    }

    const std::filesystem::path configPath = appState.scenePresetConfigPaths[static_cast<size_t>(appState.scenePresetIndex)];
    if (configPath.empty())
    {
        appState.lastUiMessage = "РЈ С‚РµРєСѓС‰РµР№ СЃС†РµРЅС‹ РЅРµС‚ JSON-С„Р°Р№Р»Р° РґР»СЏ РїРµСЂРµР·Р°РіСЂСѓР·РєРё.";
        appState.lastUiMessageIsError = true;
        return false;
    }

    bool resetAccumulation = false;
    std::string error;
    SceneBuildResult& preset = appState.scenePresets[static_cast<size_t>(appState.scenePresetIndex)];
    const bool reloaded = reloadScenePresetFromConfig(
        configPath,
        preset,
        appState.scene,
        appState.camera,
        appState.assetCache,
        resetAccumulation,
        error);
    if (!reloaded)
    {
        appState.lastUiMessage = error.empty() ? "РќРµ СѓРґР°Р»РѕСЃСЊ РїРµСЂРµР·Р°РіСЂСѓР·РёС‚СЊ СЃС†РµРЅСѓ." : error;
        appState.lastUiMessageIsError = true;
        return false;
    }

    appState.scenePresetDefaults[static_cast<size_t>(appState.scenePresetIndex)] = preset;
    if (resetAccumulation)
    {
        appState.progressiveSamples = 0;
    }
    appState.rendererSceneRebuildRequested = true;
    appState.lastUiMessage = "РЎС†РµРЅР° РїРµСЂРµР·Р°РіСЂСѓР¶РµРЅР°: " + configPath.filename().string();
    appState.lastUiMessageIsError = false;
    return true;
}

void drawImguiPanel(AppState& appState, const FrameStats& stats, GLFWwindow* window)
{
    if (!appState.imguiPanelVisible)
    {
        return;
    }

    SceneState& scene = appState.scene;
    const ImVec2 displaySize = ImGui::GetIO().DisplaySize;
    const float displayWidth = std::max(1.0f, displaySize.x);
    const float displayHeight = std::max(1.0f, displaySize.y);
    const float margin = 16.0f;
    const float minPanelWidth = std::min(320.0f, displayWidth - margin * 2.0f);
    const float maxPanelWidth = std::min(520.0f, displayWidth - margin * 2.0f);
    const float minPanelHeight = std::min(380.0f, displayHeight - margin * 2.0f);
    const float maxPanelHeight = std::max(minPanelHeight, displayHeight - margin * 2.0f);
    appState.imguiPanelWidth = clampf(appState.imguiPanelWidth, minPanelWidth, maxPanelWidth);
    appState.imguiPanelHeight = clampf(appState.imguiPanelHeight, minPanelHeight, maxPanelHeight);

    if (appState.imguiPanelPinnedRight)
    {
        const float panelX = std::max(margin, displayWidth - appState.imguiPanelWidth - margin);
        ImGui::SetNextWindowPos(ImVec2(panelX, margin), ImGuiCond_Always);
    }
    else
    {
        ImGui::SetNextWindowPos(ImVec2(std::max(margin, displayWidth - appState.imguiPanelWidth - margin), margin), ImGuiCond_FirstUseEver);
    }
    ImGui::SetNextWindowSize(ImVec2(appState.imguiPanelWidth, appState.imguiPanelHeight), ImGuiCond_FirstUseEver);
    ImGui::SetNextWindowSizeConstraints(ImVec2(minPanelWidth, minPanelHeight), ImVec2(maxPanelWidth, maxPanelHeight));
    ImGuiWindowFlags panelFlags = ImGuiWindowFlags_NoCollapse | ImGuiWindowFlags_AlwaysVerticalScrollbar;
    if (appState.imguiPanelPinnedRight)
    {
        panelFlags |= ImGuiWindowFlags_NoMove;
    }
    if (appState.cursorCaptured)
    {
        panelFlags |= ImGuiWindowFlags_NoInputs;
    }
    ImGui::Begin(
        u8c(u8"\u041F\u0430\u043D\u0435\u043B\u044C \u0441\u0446\u0435\u043D\u044B"),
        &appState.imguiPanelVisible,
        panelFlags);

    const ImVec2 panelPos = ImGui::GetWindowPos();
    const ImVec2 panelSize = ImGui::GetWindowSize();
    appState.imguiPanelWidth = panelSize.x;
    appState.imguiPanelHeight = panelSize.y;
    gImguiPanelX = panelPos.x;
    gImguiPanelY = panelPos.y;
    gImguiPanelWidth = panelSize.x;
    gImguiPanelHeight = panelSize.y;
    ImGui::PushItemWidth(std::min(260.0f, std::max(170.0f, panelSize.x - 56.0f)));

    ImGui::Text("FPS %.1f | GPU %.2f ms", stats.fps, stats.avgGpuMs);
    ImGui::SameLine();
    if (ImGui::Button(appState.imguiPanelPinnedRight ? u8c(u8"\u041E\u0442\u043A\u0440\u0435\u043F\u0438\u0442\u044C") : u8c(u8"\u041F\u0440\u0438\u0436\u0430\u0442\u044C \u0441\u043F\u0440\u0430\u0432\u0430")))
    {
        appState.imguiPanelPinnedRight = !appState.imguiPanelPinnedRight;
    }

    ImGui::Text("%s: %s | %s: %s",
        u8c(u8"\u0420\u0435\u0436\u0438\u043C"),
        appState.renderMode == RenderModeProgressive ? u8c(u8"\u043D\u0430\u043A\u043E\u043F\u043B\u0435\u043D\u0438\u0435") : u8c(u8"\u0440\u0435\u0430\u043B\u044C\u043D\u043E\u0435 \u0432\u0440\u0435\u043C\u044F"),
        u8c(u8"\u041A\u0430\u0447\u0435\u0441\u0442\u0432\u043E"),
        qualityNameUtf8(appState.renderQuality));

    MeshObject* selectedMeshObject = nullptr;
    MeshMaterial* meshMaterial = nullptr;
    const auto refreshMeshSelection = [&]()
    {
        selectedMeshObject = nullptr;
        meshMaterial = nullptr;
        if (scene.selectedMeshObject >= 0 && scene.selectedMeshObject < static_cast<int>(scene.meshObjects.size()))
        {
            MeshObject& object = scene.meshObjects[static_cast<size_t>(scene.selectedMeshObject)];
            selectedMeshObject = &object;
            if (scene.selectedMeshMaterial >= 0 && scene.selectedMeshMaterial < static_cast<int>(object.mesh.materials.size()))
            {
                meshMaterial = &object.mesh.materials[static_cast<size_t>(scene.selectedMeshMaterial)];
            }
        }
    };
    refreshMeshSelection();

    const auto drawSphereMaterialEditor = [&]()
    {
        if (scene.selectedSphere < 0 ||
            scene.selectedSphere >= static_cast<int>(scene.spheres.size()) ||
            scene.selectedSphere >= static_cast<int>(scene.materials.size()))
        {
            ImGui::TextWrapped("%s", u8c(u8"\u0421\u0444\u0435\u0440\u0430 \u043D\u0435 \u0432\u044B\u0431\u0440\u0430\u043D\u0430."));
            return;
        }

        SphereMaterial& material = scene.materials[static_cast<size_t>(scene.selectedSphere)];
        bool changed = false;
        int materialType = material.materialType;
        if (materialTypeCombo(u8c(u8"\u0422\u0438\u043F##sphere_material_type"), materialType))
        {
            setSelectedSphereMaterialType(scene, materialType);
            changed = true;
        }

        float color[3] = {material.color.x, material.color.y, material.color.z};
        if (ImGui::ColorEdit3(material.materialType == MaterialDielectric ? u8c(u8"\u041E\u0442\u0442\u0435\u043D\u043E\u043A##sphere_color") : u8c(u8"\u0426\u0432\u0435\u0442##sphere_color"), color, ImGuiColorEditFlags_NoInputs))
        {
            setSelectedSphereColor(scene, make_float3(color[0], color[1], color[2]));
            changed = true;
        }

        if (material.materialType == MaterialDielectric)
        {
            changed = ImGui::SliderFloat(u8c(u8"\u041C\u0443\u0442\u043D\u043E\u0441\u0442\u044C##sphere_glass_roughness"), &material.roughness, 0.02f, 0.6f, "%.2f") || changed;
            changed = ImGui::SliderFloat(u8c(u8"IOR##sphere_ior"), &material.ior, 1.01f, 2.8f, "%.2f") || changed;
            float transparency = 1.0f - material.alpha;
            if (ImGui::SliderFloat(u8c(u8"\u041F\u0440\u043E\u0437\u0440\u0430\u0447\u043D\u043E\u0441\u0442\u044C##sphere_transparency"), &transparency, 0.0f, 1.0f, "%.2f"))
            {
                material.alpha = 1.0f - transparency;
                changed = true;
            }
        }
        else if (material.materialType == MaterialMirror)
        {
            changed = ImGui::SliderFloat(u8c(u8"\u0420\u0430\u0437\u043C\u044B\u0442\u0438\u0435##sphere_mirror_roughness"), &material.roughness, 0.02f, 0.35f, "%.2f") || changed;
        }
        else if (material.materialType == MaterialMetal)
        {
            changed = ImGui::SliderFloat(u8c(u8"\u0428\u0435\u0440\u043E\u0445\u043E\u0432\u0430\u0442\u043E\u0441\u0442\u044C##sphere_metal_roughness"), &material.roughness, 0.02f, 1.0f, "%.2f") || changed;
        }
        else
        {
            changed = ImGui::SliderFloat(u8c(u8"\u0428\u0435\u0440\u043E\u0445\u043E\u0432\u0430\u0442\u043E\u0441\u0442\u044C##sphere_matte_roughness"), &material.roughness, 0.15f, 1.0f, "%.2f") || changed;
        }
        if (changed)
        {
            material.roughness = clampf(material.roughness, 0.02f, 1.0f);
            material.ior = clampf(material.ior, 1.01f, 2.8f);
            material.alpha = clampf(material.alpha, 0.0f, 1.0f);
            appState.progressiveSamples = 0;
        }
    };

    const auto drawMeshMaterialEditor = [&]()
    {
        if (meshMaterial == nullptr)
        {
            ImGui::TextWrapped("%s", u8c(u8"\u041F\u043E\u043B\u0438\u0433\u043E\u043D\u0430\u043B\u044C\u043D\u0430\u044F \u043C\u043E\u0434\u0435\u043B\u044C \u043D\u0435 \u0432\u044B\u0431\u0440\u0430\u043D\u0430."));
            return;
        }

        bool changed = false;
        int materialType = meshMaterial->materialType;
        if (materialTypeCombo(u8c(u8"\u0422\u0438\u043F##mesh_material_type"), materialType))
        {
            setSelectedMeshMaterialType(scene, materialType);
            changed = true;
        }

        float color[3] = {meshMaterial->color.x, meshMaterial->color.y, meshMaterial->color.z};
        if (ImGui::ColorEdit3(meshMaterial->materialType == MaterialDielectric ? u8c(u8"\u041E\u0442\u0442\u0435\u043D\u043E\u043A##mesh_color") : u8c(u8"\u0426\u0432\u0435\u0442##mesh_color"), color, ImGuiColorEditFlags_NoInputs))
        {
            meshMaterial->color = make_float3(color[0], color[1], color[2]);
            changed = true;
        }
        if (meshMaterial->textureIndex >= 0)
        {
            bool textureEnabled = meshMaterial->textureEnabled != 0;
            if (ImGui::Checkbox(u8c(u8"\u0418\u0441\u043F\u043E\u043B\u044C\u0437\u043E\u0432\u0430\u0442\u044C \u0442\u0435\u043A\u0441\u0442\u0443\u0440\u0443"), &textureEnabled))
            {
                meshMaterial->textureEnabled = textureEnabled ? 1 : 0;
                changed = true;
            }
        }

        if (meshMaterial->materialType == MaterialDielectric)
        {
            changed = ImGui::SliderFloat(u8c(u8"\u041C\u0443\u0442\u043D\u043E\u0441\u0442\u044C##mesh_glass_roughness"), &meshMaterial->roughness, 0.02f, 0.6f, "%.2f") || changed;
            changed = ImGui::SliderFloat(u8c(u8"IOR##mesh_ior"), &meshMaterial->ior, 1.01f, 2.8f, "%.2f") || changed;
            float transparency = 1.0f - meshMaterial->alpha;
            if (ImGui::SliderFloat(u8c(u8"\u041F\u0440\u043E\u0437\u0440\u0430\u0447\u043D\u043E\u0441\u0442\u044C##mesh_transparency"), &transparency, 0.0f, 1.0f, "%.2f"))
            {
                meshMaterial->alpha = 1.0f - transparency;
                changed = true;
            }
        }
        else if (meshMaterial->materialType == MaterialMirror)
        {
            changed = ImGui::SliderFloat(u8c(u8"\u0420\u0430\u0437\u043C\u044B\u0442\u0438\u0435##mesh_mirror_roughness"), &meshMaterial->roughness, 0.02f, 0.35f, "%.2f") || changed;
        }
        else if (meshMaterial->materialType == MaterialMetal)
        {
            changed = ImGui::SliderFloat(u8c(u8"\u0428\u0435\u0440\u043E\u0445\u043E\u0432\u0430\u0442\u043E\u0441\u0442\u044C##mesh_metal_roughness"), &meshMaterial->roughness, 0.02f, 1.0f, "%.2f") || changed;
        }
        else
        {
            changed = ImGui::SliderFloat(u8c(u8"\u0428\u0435\u0440\u043E\u0445\u043E\u0432\u0430\u0442\u043E\u0441\u0442\u044C##mesh_matte_roughness"), &meshMaterial->roughness, 0.15f, 1.0f, "%.2f") || changed;
        }
        if (changed)
        {
            meshMaterial->roughness = clampf(meshMaterial->roughness, 0.02f, 1.0f);
            meshMaterial->ior = clampf(meshMaterial->ior, 1.01f, 2.8f);
            meshMaterial->alpha = clampf(meshMaterial->alpha, 0.0f, 1.0f);
            applySelectedMeshMaterialToWholeObject(scene);
            appState.progressiveSamples = 0;
            appState.rendererSceneRebuildRequested = true;
        }
    };

    if (ImGui::BeginTabBar("ScenePanelTabs", ImGuiTabBarFlags_FittingPolicyScroll))
    {
        if (ImGui::BeginTabItem(u8c(u8"\u0421\u0446\u0435\u043D\u0430")))
        {
            std::vector<std::string> presetNames;
            presetNames.reserve(appState.scenePresetNames.size());
            for (const std::wstring& name : appState.scenePresetNames)
            {
                presetNames.push_back(wideToUtf8(name));
            }
            const char* currentPreset = appState.scenePresetIndex >= 0 && appState.scenePresetIndex < static_cast<int>(presetNames.size())
                ? presetNames[static_cast<size_t>(appState.scenePresetIndex)].c_str()
                : u8c(u8"\u0421\u0432\u043E\u044F \u0441\u0446\u0435\u043D\u0430");
            if (ImGui::BeginCombo(u8c(u8"\u041F\u0440\u0435\u0441\u0435\u0442"), currentPreset))
            {
                for (int i = 0; i < static_cast<int>(presetNames.size()); ++i)
                {
                    const bool selected = i == appState.scenePresetIndex;
                    const std::string presetLabel = presetNames[static_cast<size_t>(i)] + "##preset_" + std::to_string(i);
                    if (ImGui::Selectable(presetLabel.c_str(), selected))
                    {
                        applyScenePreset(appState, i);
                    }
                    if (selected)
                    {
                        ImGui::SetItemDefaultFocus();
                    }
                }
                ImGui::EndCombo();
            }
            if (ImGui::Button(u8c(u8"\u0421\u0431\u0440\u043E\u0441 \u043A\u0430\u043C\u0435\u0440\u044B/\u0441\u0432\u0435\u0442\u0430")) && resetCurrentPresetView(appState))
            {
                appState.progressiveSamples = 0;
            }
            const bool canReloadSceneConfig =
                hasPresetIndex(appState, appState.scenePresetIndex) &&
                appState.scenePresetIndex < static_cast<int>(appState.scenePresetConfigPaths.size()) &&
                !appState.scenePresetConfigPaths[static_cast<size_t>(appState.scenePresetIndex)].empty();
            if (!canReloadSceneConfig)
            {
                ImGui::BeginDisabled();
            }
            if (ImGui::Button(u8c(u8"\u041F\u0435\u0440\u0435\u0437\u0430\u0433\u0440\u0443\u0437\u0438\u0442\u044C JSON")))
            {
                reloadCurrentSceneConfig(appState);
            }
            if (!canReloadSceneConfig)
            {
                ImGui::EndDisabled();
            }
            if (ImGui::Button(u8c(u8"\u0417\u0430\u0433\u0440\u0443\u0437\u0438\u0442\u044C OBJ/glTF...")))
            {
                if (const std::optional<std::filesystem::path> meshPath = openMeshFileDialog(window))
                {
                    loadUserMeshPreset(appState, *meshPath);
                }
            }
            if (!appState.lastUiMessage.empty())
            {
                ImGui::TextWrapped("%s: %s",
                    appState.lastUiMessageIsError ? u8c(u8"\u041E\u0448\u0438\u0431\u043A\u0430") : u8c(u8"\u0421\u0442\u0430\u0442\u0443\u0441"),
                    appState.lastUiMessage.c_str());
            }
            ImGui::EndTabItem();
        }

        if (ImGui::BeginTabItem(u8c(u8"\u0420\u0435\u0434\u0430\u043A\u0442\u043E\u0440")))
        {
            builtInPrimitiveCombo(u8c(u8"\u0414\u043E\u0431\u0430\u0432\u0438\u0442\u044C##editor_add_type"), appState.editorPrimitiveToAdd);
            if (ImGui::Button(u8c(u8"\u0414\u043E\u0431\u0430\u0432\u0438\u0442\u044C \u043E\u0431\u044A\u0435\u043A\u0442")))
            {
                bool added = false;
                if (appState.editorPrimitiveToAdd == -1)
                {
                    added = addSphere(scene);
                }
                else
                {
                    added = addBuiltInMeshPrimitive(scene, appState.editorPrimitiveToAdd);
                    appState.editorObjectKind = EditorObjectMesh;
                    appState.rendererSceneRebuildRequested = added || appState.rendererSceneRebuildRequested;
                    refreshMeshSelection();
                }
                appState.progressiveSamples = added ? 0 : appState.progressiveSamples;
            }
            ImGui::SeparatorText(u8c(u8"\u0412\u044B\u0431\u0440\u0430\u043D\u043D\u044B\u0439 \u043E\u0431\u044A\u0435\u043A\u0442"));
            const std::string editorLabel = appState.editorObjectKind == EditorObjectMesh && !scene.meshObjects.empty()
                ? meshObjectLabel(scene.meshObjects[static_cast<size_t>(std::max(0, std::min(scene.selectedMeshObject, static_cast<int>(scene.meshObjects.size()) - 1)))], scene.selectedMeshObject)
                : std::string(u8c(u8"\u0421\u0444\u0435\u0440\u0430 ")) + std::to_string(scene.selectedSphere + 1);
            if (ImGui::BeginCombo(u8c(u8"\u041E\u0431\u044A\u0435\u043A\u0442##editor_selected"), editorLabel.c_str()))
            {
                for (int i = 0; i < static_cast<int>(scene.spheres.size()); ++i)
                {
                    const std::string label = std::string(u8c(u8"\u0421\u0444\u0435\u0440\u0430 ")) + std::to_string(i + 1);
                    const bool selected = appState.editorObjectKind == EditorObjectSphere && scene.selectedSphere == i;
                    if (ImGui::Selectable(label.c_str(), selected))
                    {
                        appState.editorObjectKind = EditorObjectSphere;
                        scene.selectedSphere = i;
                        clampScene(scene);
                    }
                }
                for (int i = 0; i < static_cast<int>(scene.meshObjects.size()); ++i)
                {
                    const std::string label = meshObjectLabel(scene.meshObjects[static_cast<size_t>(i)], i);
                    const bool selected = appState.editorObjectKind == EditorObjectMesh && scene.selectedMeshObject == i;
                    if (ImGui::Selectable(label.c_str(), selected))
                    {
                        appState.editorObjectKind = EditorObjectMesh;
                        scene.selectedMeshObject = i;
                        scene.selectedMeshMaterial = 0;
                        clampScene(scene);
                        refreshMeshSelection();
                    }
                }
                ImGui::EndCombo();
            }
            ImGui::Separator();

            if (!scene.spheres.empty())
            {
                const std::string selectedSphereLabel = std::string(u8c(u8"\u0421\u0444\u0435\u0440\u0430 ")) + std::to_string(scene.selectedSphere + 1);
                if (ImGui::BeginCombo(u8c(u8"\u0421\u0444\u0435\u0440\u0430"), selectedSphereLabel.c_str()))
                {
                    for (int i = 0; i < static_cast<int>(scene.spheres.size()); ++i)
                    {
                        const std::string label = std::string(u8c(u8"\u0421\u0444\u0435\u0440\u0430 ")) + std::to_string(i + 1);
                        const bool selected = i == scene.selectedSphere;
                        if (ImGui::Selectable(label.c_str(), selected))
                        {
                            scene.selectedSphere = i;
                            clampScene(scene);
                        }
                        if (selected)
                        {
                            ImGui::SetItemDefaultFocus();
                        }
                    }
                    ImGui::EndCombo();
                }
                if (ImGui::Button(u8c(u8"\u0414\u043E\u0431\u0430\u0432\u0438\u0442\u044C")))
                {
                    appState.editorObjectKind = EditorObjectSphere;
                    appState.progressiveSamples = addSphere(scene) ? 0 : appState.progressiveSamples;
                }
                ImGui::SameLine();
                const bool canRemoveSphere = scene.spheres.size() > 1;
                if (!canRemoveSphere)
                {
                    ImGui::BeginDisabled();
                }
                if (ImGui::Button(u8c(u8"\u0423\u0434\u0430\u043B\u0438\u0442\u044C")))
                {
                    appState.progressiveSamples = removeSelectedSphere(scene) ? 0 : appState.progressiveSamples;
                    appState.editorObjectKind = EditorObjectSphere;
                }
                if (!canRemoveSphere)
                {
                    ImGui::EndDisabled();
                }
                clampScene(scene);
                if (scene.selectedSphere >= 0 && scene.selectedSphere < static_cast<int>(scene.spheres.size()))
                {
                    SphereGeometry& sphere = scene.spheres[static_cast<size_t>(scene.selectedSphere)];
                    float pos[3] = {sphere.center.x, sphere.center.y, sphere.center.z};
                    if (ImGui::DragFloat3(u8c(u8"\u041F\u043E\u0437\u0438\u0446\u0438\u044F##sphere_pos"), pos, 0.05f, -50.0f, 50.0f, "%.2f"))
                    {
                        const float oldRadius = sphere.radius;
                        sphere.center = make_float3(pos[0], pos[1], pos[2]);
                        sphere.radius = oldRadius;
                        clampScene(scene);
                        appState.progressiveSamples = 0;
                    }
                    float radius = sphere.radius;
                    if (ImGui::SliderFloat(u8c(u8"\u0420\u0430\u0434\u0438\u0443\u0441"), &radius, 0.25f, 5.0f, "%.2f"))
                    {
                        setSelectedSphereRadius(scene, radius);
                        appState.progressiveSamples = 0;
                    }
                }
            }

            ImGui::SeparatorText(u8c(u8"\u041F\u043E\u043B\u0438\u0433\u043E\u043D\u0430\u043B\u044C\u043D\u0430\u044F \u043C\u043E\u0434\u0435\u043B\u044C"));
            const std::string selectedMeshLabel = scene.meshObjects.empty()
                ? std::string(u8c(u8"\u041D\u0435\u0442"))
                : std::string(u8c(u8"\u041C\u043E\u0434\u0435\u043B\u044C ")) + std::to_string(scene.selectedMeshObject + 1);
            if (ImGui::BeginCombo(u8c(u8"\u041C\u043E\u0434\u0435\u043B\u044C"), selectedMeshLabel.c_str()))
            {
                for (int i = 0; i < static_cast<int>(scene.meshObjects.size()); ++i)
                {
                    const std::string label = meshObjectLabel(scene.meshObjects[static_cast<size_t>(i)], i);
                    const bool selected = i == scene.selectedMeshObject;
                    if (ImGui::Selectable(label.c_str(), selected))
                    {
                        scene.selectedMeshObject = i;
                        appState.editorObjectKind = EditorObjectMesh;
                        scene.selectedMeshMaterial = 0;
                        clampScene(scene);
                        refreshMeshSelection();
                    }
                    if (selected)
                    {
                        ImGui::SetItemDefaultFocus();
                    }
                }
                ImGui::EndCombo();
            }
            if (selectedMeshObject != nullptr)
            {
                char nameBuffer[128]{};
                const std::string currentName = meshObjectLabel(*selectedMeshObject, scene.selectedMeshObject);
                std::snprintf(nameBuffer, sizeof(nameBuffer), "%s", currentName.c_str());
                if (ImGui::InputText(u8c(u8"\u0418\u043C\u044F##mesh_name"), nameBuffer, sizeof(nameBuffer)))
                {
                    selectedMeshObject->displayName = nameBuffer;
                }
                const bool canRemoveMesh = scene.meshObjects.size() > 1;
                if (!canRemoveMesh)
                {
                    ImGui::BeginDisabled();
                }
                if (ImGui::Button(u8c(u8"\u0423\u0434\u0430\u043B\u0438\u0442\u044C \u043C\u043E\u0434\u0435\u043B\u044C")))
                {
                    const bool removed = removeSelectedMeshObject(scene);
                    appState.progressiveSamples = removed ? 0 : appState.progressiveSamples;
                    appState.rendererSceneRebuildRequested = removed || appState.rendererSceneRebuildRequested;
                    refreshMeshSelection();
                }
                if (!canRemoveMesh)
                {
                    ImGui::EndDisabled();
                }
                float position[3] = {selectedMeshObject->position.x, selectedMeshObject->position.y, selectedMeshObject->position.z};
                float rotation[3] = {selectedMeshObject->rotation.x, selectedMeshObject->rotation.y, selectedMeshObject->rotation.z};
                float scale[3] = {selectedMeshObject->scale.x, selectedMeshObject->scale.y, selectedMeshObject->scale.z};
                bool transformChanged = false;
                if (ImGui::DragFloat3(u8c(u8"\u041F\u043E\u0437\u0438\u0446\u0438\u044F##mesh_pos"), position, 0.08f, -50.0f, 50.0f, "%.2f"))
                {
                    transformChanged = setSelectedMeshPosition(scene, make_float3(position[0], position[1], position[2])) || transformChanged;
                }
                if (ImGui::DragFloat3(u8c(u8"\u041F\u043E\u0432\u043E\u0440\u043E\u0442##mesh_rot"), rotation, 0.8f, -360.0f, 360.0f, "%.1f"))
                {
                    transformChanged = setSelectedMeshRotation(scene, make_float3(rotation[0], rotation[1], rotation[2])) || transformChanged;
                }
                if (ImGui::DragFloat3(u8c(u8"\u041C\u0430\u0441\u0448\u0442\u0430\u0431##mesh_scale"), scale, 0.10f, 0.05f, 100.0f, "%.2f"))
                {
                    transformChanged = setSelectedMeshScale(scene, make_float3(scale[0], scale[1], scale[2])) || transformChanged;
                }
                if (transformChanged)
                {
                    appState.progressiveSamples = 0;
                }
            }
            ImGui::EndTabItem();
        }

        if (ImGui::BeginTabItem(u8c(u8"\u041C\u0430\u0442\u0435\u0440\u0438\u0430\u043B\u044B")))
        {
            if (ImGui::CollapsingHeader(u8c(u8"\u0421\u0444\u0435\u0440\u0430"), ImGuiTreeNodeFlags_DefaultOpen))
            {
                drawSphereMaterialEditor();
            }
            if (ImGui::CollapsingHeader(u8c(u8"\u041C\u043E\u0434\u0435\u043B\u044C"), ImGuiTreeNodeFlags_DefaultOpen))
            {
                drawMeshMaterialEditor();
            }
            ImGui::EndTabItem();
        }

        if (ImGui::BeginTabItem(u8c(u8"\u0421\u0432\u0435\u0442")))
        {
            bool changed = false;
            float lightPosition[3] = {scene.lightPosition.x, scene.lightPosition.y, scene.lightPosition.z};
            if (ImGui::DragFloat3(u8c(u8"\u041F\u043E\u0437\u0438\u0446\u0438\u044F"), lightPosition, 0.05f, -40.0f, 40.0f, "%.2f"))
            {
                scene.lightPosition = make_float3(lightPosition[0], std::max(6.0f, lightPosition[1]), lightPosition[2]);
                clampScene(scene);
                changed = true;
            }
            changed = ImGui::SliderFloat(u8c(u8"\u0421\u0438\u043B\u0430 \u0441\u0432\u0435\u0442\u0430"), &scene.lightIntensity, 0.0f, 5.0f, "%.2f") || changed;
            changed = ImGui::SliderFloat(u8c(u8"\u0420\u0430\u0437\u043C\u0435\u0440 area light"), &scene.areaLightRadius, 0.0f, 8.0f, "%.2f") || changed;
            changed = ImGui::SliderFloat(u8c(u8"\u042F\u0440\u043A\u043E\u0441\u0442\u044C \u043D\u0435\u0431\u0430"), &scene.skyIntensity, 0.0f, 3.0f, "%.2f") || changed;
            changed = ImGui::SliderFloat(u8c(u8"\u041E\u043A\u0440\u0443\u0436\u0435\u043D\u0438\u0435"), &scene.environmentIntensity, 0.0f, 4.0f, "%.2f") || changed;
            changed = ImGui::SliderFloat(u8c(u8"\u042D\u043A\u0441\u043F\u043E\u0437\u0438\u0446\u0438\u044F"), &scene.exposure, 0.1f, 2.5f, "%.2f") || changed;
            if (changed)
            {
                setSceneExposure(scene, scene.exposure);
                setSceneSkyIntensity(scene, scene.skyIntensity);
                setSceneLightIntensity(scene, scene.lightIntensity);
                scene.areaLightRadius = clampf(scene.areaLightRadius, 0.0f, 8.0f);
                scene.environmentIntensity = clampf(scene.environmentIntensity, 0.0f, 4.0f);
                appState.progressiveSamples = 0;
            }
            ImGui::EndTabItem();
        }

        if (ImGui::BeginTabItem(u8c(u8"\u041A\u0430\u0447\u0435\u0441\u0442\u0432\u043E")))
        {
            int selectedQuality = appState.renderQuality;
            if (qualityCombo(u8c(u8"\u0420\u0435\u0436\u0438\u043C##quality_mode"), selectedQuality))
            {
                appState.renderQuality = selectedQuality;
                applyQualityMode(appState);
            }
            bool progressive = appState.renderMode == RenderModeProgressive;
            if (ImGui::Checkbox(u8c(u8"\u0420\u0435\u0436\u0438\u043C \u043D\u0430\u043A\u043E\u043F\u043B\u0435\u043D\u0438\u044F"), &progressive))
            {
                appState.renderMode = progressive ? RenderModeProgressive : RenderModeRealtime;
                appState.progressiveSamples = 0;
            }
            if (ImGui::Checkbox(u8c(u8"\u0428\u0443\u043C\u043E\u043F\u043E\u0434\u0430\u0432\u0438\u0442\u0435\u043B\u044C"), &appState.denoiserEnabled))
            {
                appState.progressiveSamples = 0;
            }
            ImGui::TextWrapped("%s", u8c(u8"\u0420\u0435\u0436\u0438\u043C \u043D\u0430\u043A\u043E\u043F\u043B\u0435\u043D\u0438\u044F \u043C\u043E\u0436\u0435\u0442 \u0431\u044B\u0442\u044C \u043C\u0435\u0434\u043B\u0435\u043D\u043D\u044B\u043C \u0438 \u0448\u0443\u043C\u043D\u044B\u043C. \u0414\u043B\u044F \u043E\u0431\u044B\u0447\u043D\u043E\u0433\u043E \u043F\u043E\u043A\u0430\u0437\u0430 \u043B\u0443\u0447\u0448\u0435 High/Medium."));
            ImGui::EndTabItem();
        }

        if (ImGui::BeginTabItem(u8c(u8"\u0414\u0438\u0430\u0433\u043D\u043E\u0441\u0442\u0438\u043A\u0430")))
        {
            size_t meshMaterialCount = 0;
            size_t triangleCount = 0;
            for (const MeshObject& object : scene.meshObjects)
            {
                meshMaterialCount += object.mesh.materials.size();
                triangleCount += object.mesh.triangles.size();
            }
            ImGui::Text("%s: %zu", u8c(u8"\u0421\u0444\u0435\u0440\u044B"), scene.spheres.size());
            ImGui::Text("%s: %zu", u8c(u8"\u041C\u043E\u0434\u0435\u043B\u0438"), scene.meshObjects.size());
            ImGui::Text("%s: %zu", u8c(u8"\u041C\u0430\u0442\u0435\u0440\u0438\u0430\u043B\u044B"), scene.materials.size() + meshMaterialCount);
            ImGui::Text("%s: %zu", u8c(u8"\u0422\u0440\u0435\u0443\u0433\u043E\u043B\u044C\u043D\u0438\u043A\u0438"), triangleCount);
            ImGui::TextWrapped("%s: %s", u8c(u8"\u041B\u043E\u0433"), getLogFilePath().string().c_str());
            if (!appState.lastUiMessage.empty())
            {
                ImGui::TextWrapped("%s: %s",
                    appState.lastUiMessageIsError ? u8c(u8"\u041E\u0448\u0438\u0431\u043A\u0430") : u8c(u8"\u0421\u0442\u0430\u0442\u0443\u0441"),
                    appState.lastUiMessage.c_str());
            }
            ImGui::TextWrapped("%s", u8c(u8"\u041B\u041A\u041C: \u043A\u0443\u0440\u0441\u043E\u0440/\u043A\u0430\u043C\u0435\u0440\u0430. H: \u0441\u043A\u0440\u044B\u0442\u044C \u043F\u0430\u043D\u0435\u043B\u044C."));
            ImGui::EndTabItem();
        }
        ImGui::EndTabBar();
    }
    ImGui::PopItemWidth();
    ImGui::End();
}

void mouseCallback(GLFWwindow* window, double xpos, double ypos)
{
    auto* state = static_cast<AppState*>(glfwGetWindowUserPointer(window));
    if (state == nullptr || !state->cursorCaptured)
    {
        return;
    }

    if (state->input.firstMouse)
    {
        state->input.lastX = xpos;
        state->input.lastY = ypos;
        state->input.firstMouse = false;
    }

    const float xoffset = static_cast<float>(xpos - state->input.lastX);
    const float yoffset = static_cast<float>(state->input.lastY - ypos);
    state->input.lastX = xpos;
    state->input.lastY = ypos;

    constexpr float sensitivity = 0.18f;
    state->camera.yaw += xoffset * sensitivity;
    state->camera.pitch += yoffset * sensitivity;

    if (state->camera.pitch > 89.0f)
    {
        state->camera.pitch = 89.0f;
    }
    if (state->camera.pitch < -89.0f)
    {
        state->camera.pitch = -89.0f;
    }
}

void mouseButtonCallback(GLFWwindow* window, int button, int action, int)
{
    if (action != GLFW_PRESS)
    {
        return;
    }

    auto* state = static_cast<AppState*>(glfwGetWindowUserPointer(window));
    if (state == nullptr)
    {
        return;
    }

    if (button != GLFW_MOUSE_BUTTON_LEFT)
    {
        return;
    }

    if (!state->cursorCaptured && ImGui::GetCurrentContext() != nullptr && ImGui::GetIO().WantCaptureMouse)
    {
        return;
    }

    if (!state->cursorCaptured && state->imguiPanelVisible)
    {
        double x = 0.0;
        double y = 0.0;
        glfwGetCursorPos(window, &x, &y);
        const bool insidePanel =
            static_cast<float>(x) >= gImguiPanelX &&
            static_cast<float>(x) <= gImguiPanelX + gImguiPanelWidth &&
            static_cast<float>(y) >= gImguiPanelY &&
            static_cast<float>(y) <= gImguiPanelY + gImguiPanelHeight;
        if (insidePanel)
        {
            return;
        }
    }

    state->cursorCaptured = state->cursorCaptured ? false : true;
    glfwSetInputMode(window, GLFW_CURSOR, state->cursorCaptured ? GLFW_CURSOR_DISABLED : GLFW_CURSOR_NORMAL);
    state->input.firstMouse = true;
}

void processInput(GLFWwindow* window, AppState& appState, float deltaTimeSec)
{
    CameraState& camera = appState.camera;
    SceneState& scene = appState.scene;

    float3 forward{};
    float3 right{};
    float3 up{};
    float scale = 0.0f;
    float aspect = 0.0f;
    updateCameraBasis(camera, gWidth, gHeight, forward, right, up, scale, aspect);

    float3 cameraRight = make_float3(right.x, 0.0f, right.z);
    float3 cameraForward = make_float3(forward.x, 0.0f, forward.z);
    cameraRight = normalize3(cameraRight);
    cameraForward = normalize3(cameraForward);
    if (dot3(cameraRight, cameraRight) <= 0.000001f)
    {
        cameraRight = make_float3(1.0f, 0.0f, 0.0f);
    }
    if (dot3(cameraForward, cameraForward) <= 0.000001f)
    {
        cameraForward = make_float3(0.0f, 0.0f, 1.0f);
    }

    if (glfwGetKey(window, GLFW_KEY_ESCAPE) == GLFW_PRESS)
    {
        glfwSetWindowShouldClose(window, GLFW_TRUE);
    }

    const float dt = clampf(deltaTimeSec, 0.0f, 0.05f);
    const float cameraStep = 8.5f * dt;
    if (glfwGetKey(window, GLFW_KEY_W) == GLFW_PRESS)
    {
        camera.position = add3(camera.position, mul3(forward, cameraStep));
    }
    if (glfwGetKey(window, GLFW_KEY_S) == GLFW_PRESS)
    {
        camera.position = sub3(camera.position, mul3(forward, cameraStep));
    }
    if (glfwGetKey(window, GLFW_KEY_A) == GLFW_PRESS)
    {
        camera.position = sub3(camera.position, mul3(right, cameraStep));
    }
    if (glfwGetKey(window, GLFW_KEY_D) == GLFW_PRESS)
    {
        camera.position = add3(camera.position, mul3(right, cameraStep));
    }
    if (glfwGetKey(window, GLFW_KEY_SPACE) == GLFW_PRESS)
    {
        camera.position = add3(camera.position, mul3(up, cameraStep));
    }

    if (glfwGetKey(window, GLFW_KEY_1) == GLFW_PRESS)
    {
        scene.selectedSphere = 0;
    }
    if (glfwGetKey(window, GLFW_KEY_2) == GLFW_PRESS)
    {
        scene.selectedSphere = 1;
    }
    if (glfwGetKey(window, GLFW_KEY_3) == GLFW_PRESS)
    {
        scene.selectedSphere = 2;
    }

    const auto moveSelectedEditorObject = [&](const float3 delta)
    {
        if (appState.editorObjectKind == EditorObjectMesh &&
            scene.selectedMeshObject >= 0 &&
            scene.selectedMeshObject < static_cast<int>(scene.meshObjects.size()))
        {
            const MeshObject& object = scene.meshObjects[static_cast<size_t>(scene.selectedMeshObject)];
            if (setSelectedMeshPosition(scene, add3(object.position, delta)))
            {
                appState.progressiveSamples = 0;
            }
        }
        else
        {
            moveSelectedSphere(scene, delta);
            appState.progressiveSamples = 0;
        }
    };

    const float objectStep = 7.5f * dt;
    const float verticalStep = 6.5f * dt;
    if (glfwGetKey(window, GLFW_KEY_LEFT) == GLFW_PRESS)
    {
        moveSelectedEditorObject(mul3(cameraRight, -objectStep));
    }
    if (glfwGetKey(window, GLFW_KEY_RIGHT) == GLFW_PRESS)
    {
        moveSelectedEditorObject(mul3(cameraRight, objectStep));
    }
    if (glfwGetKey(window, GLFW_KEY_UP) == GLFW_PRESS)
    {
        moveSelectedEditorObject(mul3(cameraForward, objectStep));
    }
    if (glfwGetKey(window, GLFW_KEY_DOWN) == GLFW_PRESS)
    {
        moveSelectedEditorObject(mul3(cameraForward, -objectStep));
    }
    if (glfwGetKey(window, GLFW_KEY_R) == GLFW_PRESS)
    {
        moveSelectedEditorObject(make_float3(0.0f, verticalStep, 0.0f));
    }
    if (glfwGetKey(window, GLFW_KEY_F) == GLFW_PRESS)
    {
        moveSelectedEditorObject(make_float3(0.0f, -verticalStep, 0.0f));
    }

    const float lightStep = 7.0f * dt;
    if (glfwGetKey(window, GLFW_KEY_J) == GLFW_PRESS)
    {
        moveLight(scene, make_float3(-lightStep, 0.0f, 0.0f));
    }
    if (glfwGetKey(window, GLFW_KEY_L) == GLFW_PRESS)
    {
        moveLight(scene, make_float3(lightStep, 0.0f, 0.0f));
    }
    if (glfwGetKey(window, GLFW_KEY_I) == GLFW_PRESS)
    {
        moveLight(scene, make_float3(0.0f, 0.0f, -lightStep));
    }
    if (glfwGetKey(window, GLFW_KEY_K) == GLFW_PRESS)
    {
        moveLight(scene, make_float3(0.0f, 0.0f, lightStep));
    }
    if (glfwGetKey(window, GLFW_KEY_U) == GLFW_PRESS)
    {
        moveLight(scene, make_float3(0.0f, lightStep, 0.0f));
    }
    if (glfwGetKey(window, GLFW_KEY_O) == GLFW_PRESS)
    {
        moveLight(scene, make_float3(0.0f, -lightStep, 0.0f));
    }
    static bool mWasDown = false;
    const bool mIsDown = glfwGetKey(window, GLFW_KEY_M) == GLFW_PRESS;
    if (mIsDown && !mWasDown)
    {
        cycleSelectedSphereMaterialPreset(scene);
    }
    mWasDown = mIsDown;

    static bool bWasDown = false;
    const bool bIsDown = glfwGetKey(window, GLFW_KEY_B) == GLFW_PRESS;
    if (bIsDown && !bWasDown)
    {
        selectNextMeshObject(scene);
        appState.progressiveSamples = 0;
    }
    bWasDown = bIsDown;

    static bool vWasDown = false;
    const bool vIsDown = glfwGetKey(window, GLFW_KEY_V) == GLFW_PRESS;
    if (vIsDown && !vWasDown)
    {
        cycleSelectedMeshMaterialPreset(scene);
        appState.progressiveSamples = 0;
    }
    vWasDown = vIsDown;

    static bool gWasDown = false;
    const bool gIsDown = glfwGetKey(window, GLFW_KEY_G) == GLFW_PRESS;
    if (gIsDown && !gWasDown && !appState.scenePresets.empty())
    {
        const int nextPresetIndex = (appState.scenePresetIndex + 1) % static_cast<int>(appState.scenePresets.size());
        applyScenePreset(appState, nextPresetIndex);
    }
    gWasDown = gIsDown;

    static bool cWasDown = false;
    const bool cIsDown = glfwGetKey(window, GLFW_KEY_C) == GLFW_PRESS;
    if (cIsDown && !cWasDown)
    {
        resetCurrentPresetView(appState);
    }
    cWasDown = cIsDown;

    static bool f5WasDown = false;
    const bool f5IsDown = glfwGetKey(window, GLFW_KEY_F5) == GLFW_PRESS;
    if (f5IsDown && !f5WasDown)
    {
        reloadCurrentSceneConfig(appState);
    }
    f5WasDown = f5IsDown;

    static bool qWasDown = false;
    const bool qIsDown = glfwGetKey(window, GLFW_KEY_Q) == GLFW_PRESS;
    if (qIsDown && !qWasDown)
    {
        appState.renderQuality = nextRenderQuality(appState.renderQuality);
        applyQualityMode(appState);
    }
    qWasDown = qIsDown;

    static bool pWasDown = false;
    const bool pIsDown = glfwGetKey(window, GLFW_KEY_P) == GLFW_PRESS;
    if (pIsDown && !pWasDown)
    {
        appState.renderMode = appState.renderMode == RenderModeProgressive ? RenderModeRealtime : RenderModeProgressive;
        appState.progressiveSamples = 0;
    }
    pWasDown = pIsDown;

    static bool nWasDown = false;
    const bool nIsDown = glfwGetKey(window, GLFW_KEY_N) == GLFW_PRESS;
    if (nIsDown && !nWasDown)
    {
        appState.denoiserEnabled = !appState.denoiserEnabled;
        appState.progressiveSamples = 0;
    }
    nWasDown = nIsDown;

    static bool hWasDown = false;
    const bool hIsDown = glfwGetKey(window, GLFW_KEY_H) == GLFW_PRESS;
    if (hIsDown && !hWasDown)
    {
        appState.imguiPanelVisible = !appState.imguiPanelVisible;
    }
    hWasDown = hIsDown;

    static bool key4WasDown = false;
    const bool key4IsDown = glfwGetKey(window, GLFW_KEY_4) == GLFW_PRESS;
    if (key4IsDown && !key4WasDown)
    {
        adjustSceneExposure(scene, -0.05f);
        appState.progressiveSamples = 0;
    }
    key4WasDown = key4IsDown;

    static bool key5WasDown = false;
    const bool key5IsDown = glfwGetKey(window, GLFW_KEY_5) == GLFW_PRESS;
    if (key5IsDown && !key5WasDown)
    {
        adjustSceneExposure(scene, 0.05f);
        appState.progressiveSamples = 0;
    }
    key5WasDown = key5IsDown;

    static bool key6WasDown = false;
    const bool key6IsDown = glfwGetKey(window, GLFW_KEY_6) == GLFW_PRESS;
    if (key6IsDown && !key6WasDown)
    {
        adjustSceneSkyIntensity(scene, -0.05f);
        appState.progressiveSamples = 0;
    }
    key6WasDown = key6IsDown;

    static bool key7WasDown = false;
    const bool key7IsDown = glfwGetKey(window, GLFW_KEY_7) == GLFW_PRESS;
    if (key7IsDown && !key7WasDown)
    {
        adjustSceneSkyIntensity(scene, 0.05f);
        appState.progressiveSamples = 0;
    }
    key7WasDown = key7IsDown;

    static bool key8WasDown = false;
    const bool key8IsDown = glfwGetKey(window, GLFW_KEY_8) == GLFW_PRESS;
    if (key8IsDown && !key8WasDown)
    {
        adjustSceneLightIntensity(scene, -0.1f);
        appState.progressiveSamples = 0;
    }
    key8WasDown = key8IsDown;

    static bool key9WasDown = false;
    const bool key9IsDown = glfwGetKey(window, GLFW_KEY_9) == GLFW_PRESS;
    if (key9IsDown && !key9WasDown)
    {
        adjustSceneLightIntensity(scene, 0.1f);
        appState.progressiveSamples = 0;
    }
    key9WasDown = key9IsDown;

}

std::filesystem::path findSceneAsset(const std::string& fileName)
{
    const std::filesystem::path sourceDir = RAYTRACERRTX_SOURCE_DIR;
    const std::filesystem::path fromSource = sourceDir.empty()
        ? std::filesystem::path{}
        : sourceDir.parent_path() / "assets" / "scenes" / fileName;
    const std::filesystem::path candidates[] = {
        fromSource,
        std::filesystem::path("RayTracerRTX") / "assets" / "scenes" / fileName,
        std::filesystem::path("assets") / "scenes" / fileName
    };

    for (const std::filesystem::path& candidate : candidates)
    {
        if (!candidate.empty() && std::filesystem::exists(candidate))
        {
            return candidate;
        }
    }
    return {};
}

void addScenePreset(AppState& appState, SceneBuildResult preset, std::wstring name, std::filesystem::path configPath)
{
    if (!preset.ok)
    {
        return;
    }
    clampScene(preset.scene);
    appState.scenePresetDefaults.push_back(preset);
    appState.scenePresets.push_back(std::move(preset));
    appState.scenePresetNames.push_back(std::move(name));
    appState.scenePresetConfigPaths.push_back(std::move(configPath));
}

void addSceneConfigPreset(AppState& appState, const std::string& fileName, const std::wstring& name)
{
    const std::filesystem::path path = findSceneAsset(fileName);
    if (path.empty())
    {
        return;
    }

    const SceneConfigResult config = loadSceneConfigFile(path);
    if (!config.ok)
    {
        return;
    }
    addScenePreset(appState, buildSceneFromConfig(config.config, path.parent_path(), appState.assetCache), name, path);
}
} // namespace

void run_optix_app(const ApplicationOptions& options)
{
    clearLogFile();
    logInfo("RayTracerRTX started. Log file: " + getLogFilePath().string());

    AppState appState;
    SceneBuildResult initial = buildDefaultSceneInput();
    if (!options.sceneConfigPath.empty())
    {
        const SceneConfigResult config = loadSceneConfigFile(options.sceneConfigPath);
        if (config.ok)
        {
            initial = buildSceneFromConfig(config.config, options.sceneConfigPath.parent_path(), appState.assetCache);
        }
        else
        {
            logError(config.error + " Using default scene.");
        }
    }
    else if (!options.meshPath.empty())
    {
        SceneBuildResult meshScene = buildSceneFromMeshPath(options.meshPath, appState.assetCache);
        if (meshScene.ok)
        {
            initial = std::move(meshScene);
        }
        else
        {
            logError(meshScene.error + " Using default scene.");
        }
    }

    if (!initial.ok)
    {
        logError(initial.error + " Using default scene.");
        initial = buildDefaultSceneInput();
    }
    if (!glfwInit())
    {
        throw std::runtime_error("Failed to initialize GLFW.");
    }

    GLFWmonitor* monitor = glfwGetPrimaryMonitor();
    const GLFWvidmode* mode = monitor != nullptr ? glfwGetVideoMode(monitor) : nullptr;
    const int windowWidth = mode != nullptr ? std::min(1600, std::max(1280, mode->width - 120)) : 1440;
    const int windowHeight = mode != nullptr ? std::min(900, std::max(720, mode->height - 140)) : 810;
    const int windowX = mode != nullptr ? std::max(0, (mode->width - windowWidth) / 2) : 100;
    const int windowY = mode != nullptr ? std::max(0, (mode->height - windowHeight) / 2) : 100;

    glfwWindowHint(GLFW_DECORATED, GLFW_TRUE);
    glfwWindowHint(GLFW_RESIZABLE, GLFW_FALSE);
    GLFWwindow* window = glfwCreateWindow(windowWidth, windowHeight, "RayTracerRTX OptiX", nullptr, nullptr);
    if (window == nullptr)
    {
        glfwTerminate();
        throw std::runtime_error("Failed to create GLFW window.");
    }

    glfwMakeContextCurrent(window);
    glfwSwapInterval(1);
    glfwSetWindowPos(window, windowX, windowY);
    glfwGetFramebufferSize(window, &gWidth, &gHeight);
    glViewport(0, 0, gWidth, gHeight);

    appState.camera = initial.camera;
    appState.scene = initial.scene;
    const bool hasExplicitInput = !options.sceneConfigPath.empty() || !options.meshPath.empty();
    addScenePreset(appState, initial, hasExplicitInput ? L"\u0412\u0445\u043E\u0434\u043D\u0430\u044F \u0441\u0446\u0435\u043D\u0430" : L"\u0411\u0430\u0437\u043E\u0432\u0430\u044F \u0441\u0446\u0435\u043D\u0430");
    if (hasExplicitInput)
    {
        addScenePreset(appState, buildDefaultSceneInput(), L"\u0411\u0430\u0437\u043E\u0432\u0430\u044F \u0441\u0446\u0435\u043D\u0430");
    }
    addSceneConfigPreset(appState, "material_showcase_scene.json", L"\u041C\u0430\u0442\u0435\u0440\u0438\u0430\u043B\u044B");
    addSceneConfigPreset(appState, "textured_cube_scene.json", L"\u041F\u043E\u043B\u0438\u0433\u043E\u043D\u0430\u043B\u044C\u043D\u0430\u044F \u043C\u043E\u0434\u0435\u043B\u044C");
    glfwSetWindowUserPointer(window, &appState);
    glfwSetCursorPosCallback(window, mouseCallback);
    glfwSetMouseButtonCallback(window, mouseButtonCallback);
    glfwSetInputMode(window, GLFW_CURSOR, GLFW_CURSOR_DISABLED);

    IMGUI_CHECKVERSION();
    ImGui::CreateContext();
    ImGuiIO& imguiIo = ImGui::GetIO();
    imguiIo.ConfigFlags |= ImGuiConfigFlags_NavEnableKeyboard;
    imguiIo.IniFilename = nullptr;
    imguiIo.Fonts->AddFontFromFileTTF(
        "C:\\Windows\\Fonts\\segoeui.ttf",
        16.0f,
        nullptr,
        imguiIo.Fonts->GetGlyphRangesCyrillic());
    ImGui::StyleColorsDark();
    ImGui_ImplGlfw_InitForOpenGL(window, true);
    ImGui_ImplOpenGL2_Init();

    OptixRenderer renderer;
    renderer.setRenderSize(gWidth, gHeight);
    renderer.initialize(appState.scene);

    std::vector<uchar4> pixels(gWidth * gHeight);
    FrameStats stats;
    auto lastFrameTime = std::chrono::steady_clock::now();

    while (!glfwWindowShouldClose(window))
    {
        const auto frameNow = std::chrono::steady_clock::now();
        const float deltaTimeSec = static_cast<float>(std::chrono::duration<double>(frameNow - lastFrameTime).count());
        lastFrameTime = frameNow;

        if (glfwGetKey(window, GLFW_KEY_ESCAPE) == GLFW_PRESS)
        {
            glfwSetWindowShouldClose(window, GLFW_TRUE);
        }

        processInput(window, appState, deltaTimeSec);
        if (appState.rendererSceneRebuildRequested)
        {
            renderer.destroy();
            renderer.setRenderSize(gWidth, gHeight);
            renderer.initialize(appState.scene);
            renderer.setRenderQuality(appState.renderQuality);
            renderer.setRenderMode(appState.renderMode);
            renderer.setDenoiserEnabled(appState.denoiserEnabled);
            appState.progressiveSamples = 0;
            appState.rendererSceneRebuildRequested = false;
        }

        const auto hostFrameStart = std::chrono::steady_clock::now();
        float gpuTimeMs = 0.0f;
        renderer.setRenderQuality(appState.renderQuality);
        renderer.setRenderMode(appState.renderMode);
        renderer.setDenoiserEnabled(appState.denoiserEnabled);
        renderer.renderFrame(appState.scene, appState.camera, pixels, &gpuTimeMs);
        appState.progressiveSamples = renderer.getAccumulationSampleCount();
        appState.denoiserAvailable = renderer.isDenoiserAvailable();
        const auto hostFrameEnd = std::chrono::steady_clock::now();

        const double hostFrameMs = std::chrono::duration<double, std::milli>(hostFrameEnd - hostFrameStart).count();
        stats.frameCount += 1;
        stats.hostAccumMs += hostFrameMs;
        stats.gpuAccumMs += static_cast<double>(gpuTimeMs);

        const auto now = std::chrono::steady_clock::now();
        const double elapsedSec = std::chrono::duration<double>(now - stats.lastUpdate).count();
        if (elapsedSec > 0.0)
        {
            stats.fps = static_cast<double>(stats.frameCount) / elapsedSec;
            stats.avgHostMs = stats.hostAccumMs / static_cast<double>(stats.frameCount);
            stats.avgGpuMs = stats.gpuAccumMs / static_cast<double>(stats.frameCount);
        }
        if (elapsedSec >= 1.0)
        {
            const bool hasSelectedSphere =
                appState.scene.selectedSphere >= 0 &&
                appState.scene.selectedSphere < static_cast<int>(appState.scene.spheres.size()) &&
                appState.scene.selectedSphere < static_cast<int>(appState.scene.materials.size());
            std::ostringstream title;
            title << std::fixed << std::setprecision(1)
                  << "RayTracerRTX OptiX | FPS " << stats.fps
                  << " | Frame " << stats.avgHostMs << " ms"
                  << " | GPU " << stats.avgGpuMs << " ms";
            if (hasSelectedSphere)
            {
                title << " | Sphere " << (appState.scene.selectedSphere + 1)
                      << " " << materialName(appState.scene.materials[appState.scene.selectedSphere].materialType);
            }
            title
                  << " | Light (" << appState.scene.lightPosition.x << ", "
                  << appState.scene.lightPosition.y << ", "
                  << appState.scene.lightPosition.z << ")";

            glfwSetWindowTitle(window, title.str().c_str());
        }

        if (elapsedSec >= 1.0)
        {
            stats.frameCount = 0;
            stats.hostAccumMs = 0.0;
            stats.gpuAccumMs = 0.0;
            stats.lastUpdate = now;
        }

        glClear(GL_COLOR_BUFFER_BIT);
        glDrawPixels(gWidth, gHeight, GL_RGBA, GL_UNSIGNED_BYTE, pixels.data());
        ImGui_ImplOpenGL2_NewFrame();
        ImGui_ImplGlfw_NewFrame();
        ImGui::NewFrame();
        drawImguiPanel(appState, stats, window);
        ImGui::Render();
        ImGui_ImplOpenGL2_RenderDrawData(ImGui::GetDrawData());
        glfwSwapBuffers(window);
        drawHud(
            window,
            appState.scene,
            stats,
            appState.renderMode,
            appState.renderQuality,
            appState.denoiserEnabled,
            appState.denoiserAvailable,
            appState.progressiveSamples,
            appState.scenePresetIndex >= 0 && appState.scenePresetIndex < static_cast<int>(appState.scenePresetNames.size())
                ? appState.scenePresetNames[static_cast<size_t>(appState.scenePresetIndex)]
                : std::wstring{L"\u0421\u0432\u043E\u044F \u0441\u0446\u0435\u043D\u0430"});
        glfwPollEvents();
    }

    renderer.destroy();
    ImGui_ImplOpenGL2_Shutdown();
    ImGui_ImplGlfw_Shutdown();
    ImGui::DestroyContext();
    glfwDestroyWindow(window);
    glfwTerminate();
}
