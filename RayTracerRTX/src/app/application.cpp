#include "application.h"

#include "../gpu/optix_renderer.h"
#include "app_state.h"
#include "asset_cache.h"
#include "camera.h"
#include "logger.h"
#include "material.h"
#include "renderer_controller.h"
#include "renderer_statistics.h"
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

void addScenePreset(AppState& appState, SceneBuildResult preset, std::wstring name, std::filesystem::path configPath = {});
std::filesystem::path defaultSavedScenePath();

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

std::wstring utf8ToWide(const std::string& text)
{
    if (text.empty())
    {
        return {};
    }

    const int size = MultiByteToWideChar(CP_UTF8, 0, text.c_str(), static_cast<int>(text.size()), nullptr, 0);
    if (size <= 0)
    {
        return {};
    }

    std::wstring result(static_cast<size_t>(size), L'\0');
    MultiByteToWideChar(CP_UTF8, 0, text.c_str(), static_cast<int>(text.size()), result.data(), size);
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

const char* environmentModeNameUtf8(const int mode)
{
    switch (mode)
    {
    case SceneEnvironmentRoom:
        return u8c(u8"\u041A\u043E\u043C\u043D\u0430\u0442\u0430");
    case SceneEnvironmentEmpty:
        return u8c(u8"\u041F\u0443\u0441\u0442\u0430\u044F \u0441\u0446\u0435\u043D\u0430");
    case SceneEnvironmentOpen:
    default:
        return u8c(u8"\u041E\u0442\u043A\u0440\u044B\u0442\u0430\u044F \u0441\u0446\u0435\u043D\u0430");
    }
}

bool environmentModeCombo(const char* id, int& mode)
{
    const int modes[] = {SceneEnvironmentOpen, SceneEnvironmentRoom, SceneEnvironmentEmpty};
    bool changed = false;
    if (ImGui::BeginCombo(id, environmentModeNameUtf8(mode)))
    {
        for (const int candidate : modes)
        {
            const bool selected = mode == candidate;
            if (ImGui::Selectable(environmentModeNameUtf8(candidate), selected))
            {
                mode = candidate;
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

const char* renderModeNameUtf8(const int mode)
{
    return mode == RenderModeProgressive ? u8c(u8"\u0420\u0435\u0436\u0438\u043C \u043D\u0430\u043A\u043E\u043F\u043B\u0435\u043D\u0438\u044F") : u8c(u8"\u0420\u0435\u0430\u043B\u044C\u043D\u043E\u0435 \u0432\u0440\u0435\u043C\u044F");
}

bool renderModeCombo(const char* id, int& mode)
{
    const int modes[] = {RenderModeRealtime, RenderModeProgressive};
    bool changed = false;
    if (ImGui::BeginCombo(id, renderModeNameUtf8(mode)))
    {
        for (const int candidate : modes)
        {
            const bool selected = mode == candidate;
            if (ImGui::Selectable(renderModeNameUtf8(candidate), selected))
            {
                mode = candidate;
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

std::string selectedSceneObjectLabel(const SceneState& scene, const int selectionKind)
{
    if (selectionKind == SceneHierarchySelectionSphere &&
        scene.selectedSphere >= 0 &&
        scene.selectedSphere < static_cast<int>(scene.spheres.size()))
    {
        return std::string(u8c(u8"\u0421\u0444\u0435\u0440\u0430 ")) + std::to_string(scene.selectedSphere + 1);
    }

    if (selectionKind == SceneHierarchySelectionMesh &&
        scene.selectedMeshObject >= 0 &&
        scene.selectedMeshObject < static_cast<int>(scene.meshObjects.size()))
    {
        const MeshObject& object = scene.meshObjects[static_cast<size_t>(scene.selectedMeshObject)];
        const bool isEnvironment = object.assetReference.rfind("environment:", 0) == 0;
        return (isEnvironment ? std::string(u8c(u8"\u041F\u0430\u043D\u0435\u043B\u044C: ")) : std::string{}) +
            meshObjectLabel(object, scene.selectedMeshObject);
    }

    if (selectionKind == SceneHierarchySelectionGroup &&
        scene.selectedGroup >= 0 &&
        scene.selectedGroup < static_cast<int>(scene.groups.size()))
    {
        const SceneGroup& group = scene.groups[static_cast<size_t>(scene.selectedGroup)];
        return group.name.empty()
            ? std::string(u8c(u8"\u0413\u0440\u0443\u043F\u043F\u0430 ")) + std::to_string(scene.selectedGroup + 1)
            : std::string(u8c(u8"\u0413\u0440\u0443\u043F\u043F\u0430: ")) + group.name;
    }

    if (selectionKind == SceneHierarchySelectionCamera)
    {
        return u8c(u8"\u041A\u0430\u043C\u0435\u0440\u0430");
    }
    if (selectionKind == SceneHierarchySelectionLight)
    {
        return u8c(u8"\u0421\u0432\u0435\u0442");
    }
    return u8c(u8"\u0421\u0446\u0435\u043D\u0430");
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
    const int hierarchySelectionKind,
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

    const bool hasSphere = hierarchySelectionKind == SceneHierarchySelectionSphere &&
        scene.selectedSphere >= 0 &&
        scene.selectedSphere < static_cast<int>(scene.materials.size());
    const bool hasMesh = hierarchySelectionKind == SceneHierarchySelectionMesh &&
        scene.selectedMeshObject >= 0 &&
        scene.selectedMeshObject < static_cast<int>(scene.meshObjects.size());
    const MeshObject* meshObject = hasMesh ? &scene.meshObjects[static_cast<size_t>(scene.selectedMeshObject)] : nullptr;
    const bool hasMeshMaterial = meshObject != nullptr &&
        scene.selectedMeshMaterial >= 0 &&
        scene.selectedMeshMaterial < static_cast<int>(meshObject->mesh.materials.size());
    const MeshMaterial* meshMaterial = hasMeshMaterial ? &meshObject->mesh.materials[static_cast<size_t>(scene.selectedMeshMaterial)] : nullptr;

    std::wostringstream line2;
    line2 << L"\u0421\u0426\u0415\u041D\u0410: " << presetName;

    const std::wstring selectedObject = utf8ToWide(selectedSceneObjectLabel(scene, hierarchySelectionKind));
    std::wostringstream line3;
    line3 << L"\u0412\u042B\u0411\u0420\u0410\u041D\u041E: " << selectedObject
          << L"   \u041C\u0410\u0422\u0415\u0420\u0418\u0410\u041B: "
          << (meshMaterial != nullptr ? materialNameW(meshMaterial->materialType) :
              (hasSphere ? materialNameW(scene.materials[scene.selectedSphere].materialType) : L"N/A"));

    std::wostringstream lineMode;
    lineMode << L"\u0420\u0415\u0416\u0418\u041C: " << (renderMode == RenderModeProgressive ? L"\u043D\u0430\u043A\u043E\u043F\u043B\u0435\u043D\u0438\u0435" : L"\u0440\u0435\u0430\u043B\u044C\u043D\u043E\u0435 \u0432\u0440\u0435\u043C\u044F")
             << L"   \u041A\u0410\u0427\u0415\u0421\u0422\u0412\u041E: " << qualityNameW(renderQuality);
    if (renderMode == RenderModeProgressive)
    {
        lineMode << L"   \u0421\u042D\u041C\u041F\u041B\u042B: " << progressiveSamples;
    }
    const std::wstring line4 = lineMode.str();

    const std::wstring text1 = line1.str();
    const std::wstring text2 = line2.str();
    const std::wstring text3 = line3.str();
    TextOutW(dc, panelX, panelY, text1.c_str(), static_cast<int>(text1.size()));
    TextOutW(dc, panelX, panelY + 22, text2.c_str(), static_cast<int>(text2.size()));
    TextOutW(dc, panelX, panelY + 44, text3.c_str(), static_cast<int>(text3.size()));
    TextOutW(dc, panelX, panelY + 66, line4.c_str(), static_cast<int>(line4.size()));

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
    appState.undoStack.clear();
    invalidateAccumulation(appState);
    requestRendererSceneRebuild(appState);
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
        invalidateAccumulation(appState);
        requestRendererSceneRebuild(appState);
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
    if (loaded.scene.meshObjects.empty())
    {
        appState.lastUiMessage = "В загруженном файле не найдены mesh-объекты.";
        appState.lastUiMessageIsError = true;
        return false;
    }

    SceneEditor editor(appState.scene);
    const SceneState before = appState.scene;
    MeshObject object = std::move(loaded.scene.meshObjects.front());
    object.assetReference = meshPath.string();
    object.displayName = meshPath.filename().string().empty() ? meshPath.string() : meshPath.filename().string();
    applySceneEditResultWithUndo(appState, before, editor.addMeshObject(std::move(object)));
    saveCurrentScenePreset(appState);
    appState.hierarchySelectionKind = SceneHierarchySelectionMesh;
    appState.hierarchySelectionIndex = appState.scene.selectedMeshObject;
    appState.editorObjectKind = EditorObjectMesh;
    appState.lastUiMessage = "Модель добавлена в текущую сцену: " + meshPath.filename().string();
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
        invalidateAccumulation(appState);
    }
    appState.undoStack.clear();
    requestRendererSceneRebuild(appState);
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
    SceneEditor editor(scene);
    const ImVec2 displaySize = ImGui::GetIO().DisplaySize;
    const float displayWidth = std::max(1.0f, displaySize.x);
    const float displayHeight = std::max(1.0f, displaySize.y);
    const float margin = 16.0f;
    const float minPanelWidth = std::min(320.0f, displayWidth - margin * 2.0f);
    const float maxPanelWidth = std::min(640.0f, displayWidth - margin * 2.0f);
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
    const RendererStatistics rendererStatistics = computeRendererStatistics(scene, appState.progressiveSamples);

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
        int materialType = material.materialType;
        if (materialTypeCombo(u8c(u8"\u0422\u0438\u043F##sphere_material_type"), materialType))
        {
            const SceneState before = scene;
            applySceneEditResultWithUndo(appState, before, editor.setSelectedSphereMaterialType(materialType));
            return;
        }
        if (ImGui::Button(u8c(u8"\u0421\u0431\u0440\u043E\u0441\u0438\u0442\u044C \u043C\u0430\u0442\u0435\u0440\u0438\u0430\u043B##reset_sphere_material")))
        {
            const SceneState before = scene;
            applySceneEditResultWithUndo(appState, before, editor.resetSelectedSphereMaterial());
            return;
        }

        float color[3] = {material.color.x, material.color.y, material.color.z};
        float roughness = material.roughness;
        float ior = material.ior;
        float alpha = material.alpha;
        bool changed = false;

        ImGui::SeparatorText(u8c(u8"\u041F\u043E\u0432\u0435\u0440\u0445\u043D\u043E\u0441\u0442\u044C"));
        if (material.materialType == MaterialDiffuse || material.materialType == MaterialMetal || material.materialType == MaterialDielectric)
        {
            changed = ImGui::ColorEdit3(material.materialType == MaterialDielectric ? u8c(u8"\u041E\u0442\u0442\u0435\u043D\u043E\u043A##sphere_color") : u8c(u8"\u0426\u0432\u0435\u0442##sphere_color"), color, ImGuiColorEditFlags_NoInputs) || changed;
        }

        ImGui::SeparatorText(u8c(u8"\u041E\u0442\u0440\u0430\u0436\u0435\u043D\u0438\u0435"));
        if (material.materialType == MaterialDielectric)
        {
            changed = ImGui::SliderFloat(u8c(u8"\u041C\u0443\u0442\u043D\u043E\u0441\u0442\u044C##sphere_glass_roughness"), &roughness, 0.02f, 0.6f, "%.2f") || changed;
            ImGui::SeparatorText(u8c(u8"\u041F\u0440\u043E\u0437\u0440\u0430\u0447\u043D\u043E\u0441\u0442\u044C/\u0441\u0442\u0435\u043A\u043B\u043E"));
            changed = ImGui::SliderFloat(u8c(u8"\u041F\u043E\u043A\u0430\u0437\u0430\u0442\u0435\u043B\u044C \u043F\u0440\u0435\u043B\u043E\u043C\u043B\u0435\u043D\u0438\u044F##sphere_ior"), &ior, 1.01f, 2.8f, "%.2f") || changed;
            float transparency = 1.0f - alpha;
            if (ImGui::SliderFloat(u8c(u8"\u041F\u0440\u043E\u0437\u0440\u0430\u0447\u043D\u043E\u0441\u0442\u044C##sphere_transparency"), &transparency, 0.0f, 1.0f, "%.2f"))
            {
                alpha = 1.0f - transparency;
                changed = true;
            }
        }
        else if (material.materialType == MaterialMirror)
        {
            ImGui::TextWrapped("%s", u8c(u8"\u041E\u0442\u0440\u0430\u0436\u0435\u043D\u0438\u0435: \u0432\u044B\u0441\u043E\u043A\u043E\u0435"));
            changed = ImGui::SliderFloat(u8c(u8"\u0420\u0430\u0437\u043C\u044B\u0442\u0438\u0435##sphere_mirror_roughness"), &roughness, 0.02f, 0.35f, "%.2f") || changed;
        }
        else if (material.materialType == MaterialMetal)
        {
            ImGui::TextWrapped("metallic value: 1.00");
            changed = ImGui::SliderFloat(u8c(u8"\u0428\u0435\u0440\u043E\u0445\u043E\u0432\u0430\u0442\u043E\u0441\u0442\u044C##sphere_metal_roughness"), &roughness, 0.02f, 1.0f, "%.2f") || changed;
        }
        else
        {
            changed = ImGui::SliderFloat(u8c(u8"\u0428\u0435\u0440\u043E\u0445\u043E\u0432\u0430\u0442\u043E\u0441\u0442\u044C##sphere_matte_roughness"), &roughness, 0.15f, 1.0f, "%.2f") || changed;
        }

        ImGui::SeparatorText(u8c(u8"\u0422\u0435\u043A\u0441\u0442\u0443\u0440\u044B"));
        ImGui::TextWrapped("%s", u8c(u8"\u0414\u043B\u044F \u0441\u0444\u0435\u0440 \u0442\u0435\u043A\u0441\u0442\u0443\u0440\u044B \u043D\u0435 \u0437\u0430\u0434\u0430\u044E\u0442\u0441\u044F."));

        if (changed)
        {
            const SceneState before = scene;
            applySceneEditResultWithUndo(appState, before, editor.setSelectedSphereMaterialProperties(make_float3(color[0], color[1], color[2]), roughness, ior, alpha));
        }
    };

    const auto drawMeshMaterialEditor = [&]()
    {
        if (meshMaterial == nullptr)
        {
            ImGui::TextWrapped("%s", u8c(u8"\u041F\u043E\u043B\u0438\u0433\u043E\u043D\u0430\u043B\u044C\u043D\u0430\u044F \u043C\u043E\u0434\u0435\u043B\u044C \u043D\u0435 \u0432\u044B\u0431\u0440\u0430\u043D\u0430."));
            return;
        }

        const auto drawTextureSlot = [&](const char* label, const int textureIndex, const std::string& path)
        {
            const MeshTextureMetadata metadata = selectedMeshObject != nullptr
                ? getMeshTextureMetadata(selectedMeshObject->mesh, textureIndex, path)
                : MeshTextureMetadata{};
            if (!metadata.hasPath)
            {
                ImGui::Text("%s: %s", label, u8c(u8"\u043D\u0435\u0442"));
                return;
            }
            if (!metadata.loaded)
            {
                ImGui::Text("%s: %s", label, u8c(u8"\u0443\u043A\u0430\u0437\u0430\u043D\u0430, \u043D\u043E \u043D\u0435 \u0437\u0430\u0433\u0440\u0443\u0436\u0435\u043D\u0430"));
                return;
            }
            ImGui::Text("%s: %ux%u, %u ch", label, metadata.width, metadata.height, metadata.channels);
        };

        bool changed = false;
        int materialType = meshMaterial->materialType;
        if (materialTypeCombo(u8c(u8"\u0422\u0438\u043F##mesh_material_type"), materialType))
        {
            const SceneState before = scene;
            applySceneEditResultWithUndo(appState, before, editor.setSelectedMeshMaterialType(materialType));
            refreshMeshSelection();
            return;
        }
        if (ImGui::Button(u8c(u8"\u0421\u0431\u0440\u043E\u0441\u0438\u0442\u044C \u043C\u0430\u0442\u0435\u0440\u0438\u0430\u043B##reset_mesh_material")))
        {
            const SceneState before = scene;
            applySceneEditResultWithUndo(appState, before, editor.resetSelectedMeshMaterial());
            refreshMeshSelection();
            return;
        }

        float color[3] = {meshMaterial->color.x, meshMaterial->color.y, meshMaterial->color.z};
        float roughness = meshMaterial->roughness;
        float ior = meshMaterial->ior;
        float alpha = meshMaterial->alpha;
        bool textureEnabled = meshMaterial->textureEnabled != 0;

        ImGui::SeparatorText(u8c(u8"\u041F\u043E\u0432\u0435\u0440\u0445\u043D\u043E\u0441\u0442\u044C"));
        if (meshMaterial->materialType == MaterialDiffuse || meshMaterial->materialType == MaterialMetal || meshMaterial->materialType == MaterialDielectric)
        {
            changed = ImGui::ColorEdit3(meshMaterial->materialType == MaterialDielectric ? u8c(u8"\u041E\u0442\u0442\u0435\u043D\u043E\u043A##mesh_color") : u8c(u8"\u0426\u0432\u0435\u0442##mesh_color"), color, ImGuiColorEditFlags_NoInputs) || changed;
        }

        ImGui::SeparatorText(u8c(u8"\u041E\u0442\u0440\u0430\u0436\u0435\u043D\u0438\u0435"));
        if (meshMaterial->materialType == MaterialDielectric)
        {
            changed = ImGui::SliderFloat(u8c(u8"\u041C\u0443\u0442\u043D\u043E\u0441\u0442\u044C##mesh_glass_roughness"), &roughness, 0.02f, 0.6f, "%.2f") || changed;
            ImGui::SeparatorText(u8c(u8"\u041F\u0440\u043E\u0437\u0440\u0430\u0447\u043D\u043E\u0441\u0442\u044C/\u0441\u0442\u0435\u043A\u043B\u043E"));
            changed = ImGui::SliderFloat(u8c(u8"\u041F\u043E\u043A\u0430\u0437\u0430\u0442\u0435\u043B\u044C \u043F\u0440\u0435\u043B\u043E\u043C\u043B\u0435\u043D\u0438\u044F##mesh_ior"), &ior, 1.01f, 2.8f, "%.2f") || changed;
            float transparency = 1.0f - alpha;
            if (ImGui::SliderFloat(u8c(u8"\u041F\u0440\u043E\u0437\u0440\u0430\u0447\u043D\u043E\u0441\u0442\u044C##mesh_transparency"), &transparency, 0.0f, 1.0f, "%.2f"))
            {
                alpha = 1.0f - transparency;
                changed = true;
            }
        }
        else if (meshMaterial->materialType == MaterialMirror)
        {
            ImGui::TextWrapped("%s", u8c(u8"\u041E\u0442\u0440\u0430\u0436\u0435\u043D\u0438\u0435: \u0432\u044B\u0441\u043E\u043A\u043E\u0435"));
            changed = ImGui::SliderFloat(u8c(u8"\u0420\u0430\u0437\u043C\u044B\u0442\u0438\u0435##mesh_mirror_roughness"), &roughness, 0.02f, 0.35f, "%.2f") || changed;
        }
        else if (meshMaterial->materialType == MaterialMetal)
        {
            ImGui::TextWrapped("metallic value: 1.00");
            changed = ImGui::SliderFloat(u8c(u8"\u0428\u0435\u0440\u043E\u0445\u043E\u0432\u0430\u0442\u043E\u0441\u0442\u044C##mesh_metal_roughness"), &roughness, 0.02f, 1.0f, "%.2f") || changed;
        }
        else
        {
            changed = ImGui::SliderFloat(u8c(u8"\u0428\u0435\u0440\u043E\u0445\u043E\u0432\u0430\u0442\u043E\u0441\u0442\u044C##mesh_matte_roughness"), &roughness, 0.15f, 1.0f, "%.2f") || changed;
        }

        ImGui::SeparatorText(u8c(u8"\u0422\u0435\u043A\u0441\u0442\u0443\u0440\u044B"));
        if (meshMaterial->textureIndex >= 0)
        {
            changed = ImGui::Checkbox(u8c(u8"\u0418\u0441\u043F\u043E\u043B\u044C\u0437\u043E\u0432\u0430\u0442\u044C base color texture"), &textureEnabled) || changed;
        }
        drawTextureSlot("base color texture", meshMaterial->textureIndex, meshMaterial->texturePath);
        drawTextureSlot("normal map", meshMaterial->normalTextureIndex, meshMaterial->normalTexturePath);
        drawTextureSlot("roughness map", meshMaterial->roughnessTextureIndex, meshMaterial->roughnessTexturePath);
        drawTextureSlot("metallic map", meshMaterial->metallicTextureIndex, meshMaterial->metallicTexturePath);
        ImGui::TextWrapped("%s", u8c(u8"\u041F\u0443\u0442\u0438 \u0442\u0435\u043A\u0441\u0442\u0443\u0440 \u0437\u0430\u0434\u0430\u044E\u0442\u0441\u044F \u0432 OBJ/MTL, glTF/GLB \u0438\u043B\u0438 JSON. \u0412 UI \u043E\u043D\u0438 \u043F\u043E\u043A\u0430 \u0442\u043E\u043B\u044C\u043A\u043E \u043E\u0442\u043E\u0431\u0440\u0430\u0436\u0430\u044E\u0442\u0441\u044F."));

        if (changed)
        {
            const SceneState before = scene;
            applySceneEditResultWithUndo(appState, before, editor.setSelectedMeshMaterialProperties(make_float3(color[0], color[1], color[2]), roughness, ior, alpha, textureEnabled));
            refreshMeshSelection();
        }
    };

    const auto selectHierarchy = [&](const int kind, const int index)
    {
        if (selectHierarchyObject(scene, kind, index))
        {
            appState.hierarchySelectionKind = kind;
            appState.hierarchySelectionIndex = index;
            if (kind == SceneHierarchySelectionSphere)
            {
                appState.editorObjectKind = EditorObjectSphere;
            }
            else if (kind == SceneHierarchySelectionMesh)
            {
                appState.editorObjectKind = EditorObjectMesh;
                refreshMeshSelection();
            }
            else if (kind == SceneHierarchySelectionGroup)
            {
                appState.editorObjectKind = EditorObjectGroup;
            }
        }
    };

    const auto syncEditorKindFromHierarchy = [&]()
    {
        if (appState.hierarchySelectionKind == SceneHierarchySelectionSphere)
        {
            appState.editorObjectKind = EditorObjectSphere;
            appState.hierarchySelectionIndex = scene.selectedSphere;
        }
        else if (appState.hierarchySelectionKind == SceneHierarchySelectionMesh)
        {
            appState.editorObjectKind = EditorObjectMesh;
            appState.hierarchySelectionIndex = scene.selectedMeshObject;
            refreshMeshSelection();
        }
        else if (appState.hierarchySelectionKind == SceneHierarchySelectionGroup)
        {
            appState.editorObjectKind = EditorObjectGroup;
            appState.hierarchySelectionIndex = scene.selectedGroup;
        }
    };

    const auto selectSafeObjectAfterDelete = [&]()
    {
        clampScene(scene);
        if (appState.hierarchySelectionKind == SceneHierarchySelectionSphere && scene.spheres.empty())
        {
            appState.hierarchySelectionKind = scene.meshObjects.empty() ? SceneHierarchySelectionScene : SceneHierarchySelectionMesh;
        }
        else if (appState.hierarchySelectionKind == SceneHierarchySelectionMesh && scene.meshObjects.empty())
        {
            appState.hierarchySelectionKind = scene.spheres.empty() ? SceneHierarchySelectionScene : SceneHierarchySelectionSphere;
        }
        else if (appState.hierarchySelectionKind == SceneHierarchySelectionGroup && scene.groups.empty())
        {
            if (!scene.meshObjects.empty())
            {
                appState.hierarchySelectionKind = SceneHierarchySelectionMesh;
            }
            else
            {
                appState.hierarchySelectionKind = scene.spheres.empty() ? SceneHierarchySelectionScene : SceneHierarchySelectionSphere;
            }
        }
        syncEditorKindFromHierarchy();
    };

    const float hierarchyWidth = std::min(190.0f, std::max(150.0f, panelSize.x * 0.36f));
    const float childHeight = std::max(300.0f, panelSize.y - 92.0f);
    const auto selectionContains = [](const std::vector<int>& values, const int value)
    {
        return std::find(values.begin(), values.end(), value) != values.end();
    };
    const auto toggleSelectionValue = [](std::vector<int>& values, const int value, const bool enabled)
    {
        const auto found = std::find(values.begin(), values.end(), value);
        if (enabled)
        {
            if (found == values.end())
            {
                values.push_back(value);
            }
        }
        else if (found != values.end())
        {
            values.erase(found);
        }
    };

    ImGui::BeginChild("HierarchyPanel", ImVec2(hierarchyWidth, childHeight), true);
    ImGui::TextUnformatted(u8c(u8"\u0418\u0435\u0440\u0430\u0440\u0445\u0438\u044F"));
    if (ImGui::Selectable(u8c(u8"\u0421\u0446\u0435\u043D\u0430"), appState.hierarchySelectionKind == HierarchySelectionScene))
    {
        selectHierarchy(SceneHierarchySelectionScene, 0);
    }
    if (ImGui::Selectable(u8c(u8"\u041A\u0430\u043C\u0435\u0440\u0430"), appState.hierarchySelectionKind == HierarchySelectionCamera))
    {
        selectHierarchy(SceneHierarchySelectionCamera, 0);
    }
    if (ImGui::Selectable(u8c(u8"\u0421\u0432\u0435\u0442"), appState.hierarchySelectionKind == HierarchySelectionLight))
    {
        selectHierarchy(SceneHierarchySelectionLight, 0);
    }
    if (ImGui::TreeNodeEx(u8c(u8"\u041C\u043E\u0434\u0435\u043B\u0438"), ImGuiTreeNodeFlags_DefaultOpen))
    {
        for (int i = 0; i < static_cast<int>(scene.spheres.size()); ++i)
        {
            const int groupIndex = findObjectGroupIndex(scene, SceneObjectRef{SceneHierarchySelectionSphere, i});
            bool checked = selectionContains(appState.groupSelectionSpheres, i);
            if (groupIndex >= 0)
            {
                ImGui::BeginDisabled();
            }
            if (ImGui::Checkbox(("##group_select_sphere_" + std::to_string(i)).c_str(), &checked))
            {
                toggleSelectionValue(appState.groupSelectionSpheres, i, checked);
            }
            if (groupIndex >= 0)
            {
                ImGui::EndDisabled();
            }
            ImGui::SameLine();
            const std::string label = std::string(u8c(u8"\u0421\u0444\u0435\u0440\u0430 ")) + std::to_string(i + 1) + "##hier_sphere_" + std::to_string(i);
            const bool selected = appState.hierarchySelectionKind == HierarchySelectionSphere && scene.selectedSphere == i;
            if (ImGui::Selectable(label.c_str(), selected))
            {
                selectHierarchy(SceneHierarchySelectionSphere, i);
            }
        }
        for (int i = 0; i < static_cast<int>(scene.meshObjects.size()); ++i)
        {
            const int groupIndex = findObjectGroupIndex(scene, SceneObjectRef{SceneHierarchySelectionMesh, i});
            bool checked = selectionContains(appState.groupSelectionMeshes, i);
            if (groupIndex >= 0)
            {
                ImGui::BeginDisabled();
            }
            if (ImGui::Checkbox(("##group_select_mesh_" + std::to_string(i)).c_str(), &checked))
            {
                toggleSelectionValue(appState.groupSelectionMeshes, i, checked);
            }
            if (groupIndex >= 0)
            {
                ImGui::EndDisabled();
            }
            ImGui::SameLine();
            const MeshObject& object = scene.meshObjects[static_cast<size_t>(i)];
            const bool isEnvironment = object.assetReference.rfind("environment:", 0) == 0;
            const std::string label = (isEnvironment ? std::string(u8c(u8"\u041F\u0430\u043D\u0435\u043B\u044C: ")) : std::string{}) + meshObjectLabel(object, i) + "##hier_mesh_" + std::to_string(i);
            const bool selected = appState.hierarchySelectionKind == HierarchySelectionMesh && scene.selectedMeshObject == i;
            if (ImGui::Selectable(label.c_str(), selected))
            {
                selectHierarchy(SceneHierarchySelectionMesh, i);
            }
        }
        ImGui::TreePop();
    }
    if (ImGui::TreeNodeEx(u8c(u8"\u0413\u0440\u0443\u043F\u043F\u044B"), ImGuiTreeNodeFlags_DefaultOpen))
    {
        for (int i = 0; i < static_cast<int>(scene.groups.size()); ++i)
        {
            const SceneGroup& group = scene.groups[static_cast<size_t>(i)];
            const std::string visibleName = group.name.empty()
                ? std::string(u8c(u8"\u0413\u0440\u0443\u043F\u043F\u0430 ")) + std::to_string(i + 1)
                : group.name;
            const std::string label = visibleName + " (" + std::to_string(group.objects.size()) + ")##hier_group_" + std::to_string(i);
            const bool selected = appState.hierarchySelectionKind == HierarchySelectionGroup && scene.selectedGroup == i;
            if (ImGui::Selectable(label.c_str(), selected))
            {
                selectHierarchy(SceneHierarchySelectionGroup, i);
            }
        }
        ImGui::TreePop();
    }
    ImGui::EndChild();
    ImGui::SameLine();

    ImGui::BeginChild("PropertiesPanel", ImVec2(0.0f, childHeight), true);
    ImGui::TextUnformatted(u8c(u8"\u0421\u0432\u043E\u0439\u0441\u0442\u0432\u0430"));
    if (appState.hierarchySelectionKind == HierarchySelectionScene)
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
            }
            ImGui::EndCombo();
        }
        if (ImGui::Button(u8c(u8"\u0421\u0431\u0440\u043E\u0441 \u043A\u0430\u043C\u0435\u0440\u044B/\u0441\u0432\u0435\u0442\u0430")) && resetCurrentPresetView(appState))
        {
            applySceneEditResult(appState, makeCameraDirty());
        }
        if (ImGui::Button(u8c(u8"\u0417\u0430\u0433\u0440\u0443\u0437\u0438\u0442\u044C OBJ/glTF...")))
        {
            if (const std::optional<std::filesystem::path> meshPath = openMeshFileDialog(window))
            {
                loadUserMeshPreset(appState, *meshPath);
            }
        }
        ImGui::SameLine();
        if (ImGui::Button(u8c(u8"\u0421\u043E\u0445\u0440\u0430\u043D\u0438\u0442\u044C JSON")))
        {
            const std::filesystem::path savePath = defaultSavedScenePath();
            std::error_code ec;
            std::filesystem::create_directories(savePath.parent_path(), ec);
            std::string error;
            if (saveSceneToConfigFile(savePath, scene, appState.camera, error))
            {
                appState.lastUiMessage = "Сцена сохранена: " + savePath.string();
                appState.lastUiMessageIsError = false;
            }
            else
            {
                appState.lastUiMessage = error.empty() ? "Не удалось сохранить сцену." : error;
                appState.lastUiMessageIsError = true;
                logError(appState.lastUiMessage);
            }
        }
        ImGui::SeparatorText(u8c(u8"\u041E\u043A\u0440\u0443\u0436\u0435\u043D\u0438\u0435"));
        environmentModeCombo(u8c(u8"\u0420\u0435\u0436\u0438\u043C##environment_mode"), appState.environmentMode);
        if (ImGui::Button(u8c(u8"\u041F\u0440\u0438\u043C\u0435\u043D\u0438\u0442\u044C")))
        {
            const SceneState before = scene;
            applySceneEditResultWithUndo(appState, before, editor.applyEnvironmentMode(appState.environmentMode));
            refreshMeshSelection();
        }
        if (appState.environmentMode == SceneEnvironmentRoom)
        {
            float roomSize[3] = {appState.roomWidth, appState.roomDepth, appState.roomHeight};
            if (ImGui::DragFloat3(u8c(u8"\u0420\u0430\u0437\u043C\u0435\u0440 \u043A\u043E\u043C\u043D\u0430\u0442\u044B##room_size"), roomSize, 0.5f, 4.0f, 80.0f, "%.2f"))
            {
                appState.roomWidth = clampf(roomSize[0], 4.0f, 80.0f);
                appState.roomDepth = clampf(roomSize[1], 4.0f, 80.0f);
                appState.roomHeight = clampf(roomSize[2], 2.0f, 40.0f);
                const SceneState before = scene;
                applySceneEditResultWithUndo(appState, before, editor.applyRoomDimensions(appState.roomWidth, appState.roomDepth, appState.roomHeight));
                refreshMeshSelection();
            }
        }
        float exposure = scene.exposure;
        if (ImGui::SliderFloat(u8c(u8"\u042D\u043A\u0441\u043F\u043E\u0437\u0438\u0446\u0438\u044F"), &exposure, 0.1f, 2.5f, "%.2f"))
        {
            const SceneState before = scene;
            applySceneEditResultWithUndo(appState, before, editor.setExposure(exposure));
        }
        float skyIntensity = scene.skyIntensity;
        if (ImGui::SliderFloat(u8c(u8"\u041D\u0435\u0431\u043E"), &skyIntensity, 0.0f, 3.0f, "%.2f"))
        {
            const SceneState before = scene;
            applySceneEditResultWithUndo(appState, before, editor.setSkyIntensity(skyIntensity));
        }
        float horizonColor[3] = {scene.skyHorizonColor.x, scene.skyHorizonColor.y, scene.skyHorizonColor.z};
        if (ImGui::ColorEdit3(u8c(u8"\u0413\u043E\u0440\u0438\u0437\u043E\u043D\u0442"), horizonColor, ImGuiColorEditFlags_Float))
        {
            const SceneState before = scene;
            applySceneEditResultWithUndo(appState, before, editor.setSkyHorizonColor(make_float3(horizonColor[0], horizonColor[1], horizonColor[2])));
        }
        float zenithColor[3] = {scene.skyZenithColor.x, scene.skyZenithColor.y, scene.skyZenithColor.z};
        if (ImGui::ColorEdit3(u8c(u8"\u0412\u0435\u0440\u0445 \u043D\u0435\u0431\u0430"), zenithColor, ImGuiColorEditFlags_Float))
        {
            const SceneState before = scene;
            applySceneEditResultWithUndo(appState, before, editor.setSkyZenithColor(make_float3(zenithColor[0], zenithColor[1], zenithColor[2])));
        }

        ImGui::SeparatorText(u8c(u8"\u041E\u0431\u044A\u0435\u043A\u0442\u044B"));
        builtInPrimitiveCombo(u8c(u8"\u0414\u043E\u0431\u0430\u0432\u0438\u0442\u044C##primitive_add_combo"), appState.editorPrimitiveToAdd);
        if (ImGui::Button(u8c(u8"\u0414\u043E\u0431\u0430\u0432\u0438\u0442\u044C##add_object")))
        {
            const SceneState before = scene;
            const SceneEditResult result = appState.editorPrimitiveToAdd == -1
                ? editor.addSphere()
                : editor.addMeshPrimitive(appState.editorPrimitiveToAdd);
            applySceneEditResultWithUndo(appState, before, result);
            refreshMeshSelection();
        }
        ImGui::SameLine();
        if (ImGui::Button(u8c(u8"\u041E\u0447\u0438\u0441\u0442\u0438\u0442\u044C##clear_scene")))
        {
            const SceneState before = scene;
            applySceneEditResultWithUndo(appState, before, editor.clearScene());
            refreshMeshSelection();
        }
        if (scene.meshObjects.empty() && scene.spheres.empty())
        {
            ImGui::BeginDisabled();
        }
        if (ImGui::Button(u8c(u8"\u041F\u0440\u0435\u0434\u044B\u0434\u0443\u0449\u0430\u044F \u043C\u043E\u0434\u0435\u043B\u044C (B)##select_prev_object")))
        {
            selectPreviousSceneObject(scene, appState.hierarchySelectionKind);
            syncEditorKindFromHierarchy();
            applySceneEditResult(appState, makeRenderSettingsDirty());
            refreshMeshSelection();
        }
        if (scene.meshObjects.empty() && scene.spheres.empty())
        {
            ImGui::EndDisabled();
        }

        std::vector<SceneObjectRef> groupRefs;
        for (const int sphereIndex : appState.groupSelectionSpheres)
        {
            if (sphereIndex >= 0 &&
                sphereIndex < static_cast<int>(scene.spheres.size()) &&
                findObjectGroupIndex(scene, SceneObjectRef{SceneHierarchySelectionSphere, sphereIndex}) < 0)
            {
                groupRefs.push_back(SceneObjectRef{SceneHierarchySelectionSphere, sphereIndex});
            }
        }
        for (const int meshIndex : appState.groupSelectionMeshes)
        {
            if (meshIndex >= 0 &&
                meshIndex < static_cast<int>(scene.meshObjects.size()) &&
                findObjectGroupIndex(scene, SceneObjectRef{SceneHierarchySelectionMesh, meshIndex}) < 0)
            {
                groupRefs.push_back(SceneObjectRef{SceneHierarchySelectionMesh, meshIndex});
            }
        }
        ImGui::Text("%s: %zu", u8c(u8"\u0412\u044B\u0431\u0440\u0430\u043D\u043E \u0434\u043B\u044F \u0433\u0440\u0443\u043F\u043F\u044B"), groupRefs.size());
        if (groupRefs.size() < 2)
        {
            ImGui::BeginDisabled();
        }
        if (ImGui::Button(u8c(u8"\u0421\u043E\u0437\u0434\u0430\u0442\u044C \u0433\u0440\u0443\u043F\u043F\u0443##create_group")))
        {
            const SceneState before = scene;
            applySceneEditResultWithUndo(appState, before, editor.createGroup(groupRefs));
            appState.groupSelectionSpheres.clear();
            appState.groupSelectionMeshes.clear();
            if (!scene.groups.empty())
            {
                selectHierarchy(SceneHierarchySelectionGroup, scene.selectedGroup);
            }
        }
        if (groupRefs.size() < 2)
        {
            ImGui::EndDisabled();
        }
        ImGui::SameLine();
        if (ImGui::Button(u8c(u8"\u0421\u0431\u0440\u043E\u0441\u0438\u0442\u044C \u043E\u0442\u043C\u0435\u0442\u043A\u0438##clear_group_selection")))
        {
            appState.groupSelectionSpheres.clear();
            appState.groupSelectionMeshes.clear();
        }

        ImGui::SeparatorText(u8c(u8"\u0420\u0435\u043D\u0434\u0435\u0440"));
        ImGui::Text("FPS: %.1f", stats.fps);
        ImGui::Text("CPU frame: %.2f ms", stats.avgHostMs);
        ImGui::Text("GPU: %.2f ms", stats.avgGpuMs);
        ImGui::Text("%s: %zu", u8c(u8"\u0421\u0444\u0435\u0440\u044B"), rendererStatistics.sphereCount);
        ImGui::Text("%s: %zu", u8c(u8"\u041C\u043E\u0434\u0435\u043B\u0438"), rendererStatistics.meshObjectCount);
        ImGui::Text("%s: %zu", u8c(u8"\u0422\u0440\u0435\u0443\u0433\u043E\u043B\u044C\u043D\u0438\u043A\u0438"), rendererStatistics.triangleCount);
        ImGui::Text("%s: %zu", u8c(u8"\u041C\u0430\u0442\u0435\u0440\u0438\u0430\u043B\u044B"), rendererStatistics.materialCount);
        ImGui::Text("%s: %u", u8c(u8"\u0421\u044D\u043C\u043F\u043B\u044B"), rendererStatistics.accumulationSamples);
        ImGui::Text("%s: %s", u8c(u8"\u0428\u0443\u043C\u043E\u043F\u043E\u0434\u0430\u0432\u0438\u0442\u0435\u043B\u044C"),
            appState.denoiserEnabled ? (appState.denoiserAvailable ? u8c(u8"\u0432\u043A\u043B") : u8c(u8"\u043D\u0435\u0434\u043E\u0441\u0442\u0443\u043F\u0435\u043D")) : u8c(u8"\u0432\u044B\u043A\u043B"));
        ImGui::TextWrapped("%s", u8c(u8"VRAM: \u043D\u0435 \u043E\u0442\u043E\u0431\u0440\u0430\u0436\u0430\u0435\u0442\u0441\u044F, \u0447\u0442\u043E\u0431\u044B UI \u043D\u0435 \u0437\u0430\u0432\u0438\u0441\u0435\u043B \u043E\u0442 CUDA runtime \u0432 \u043F\u0430\u043D\u0435\u043B\u0438."));
        int selectedRenderMode = appState.renderMode;
        if (renderModeCombo(u8c(u8"\u0420\u0435\u0436\u0438\u043C##renderer_mode"), selectedRenderMode))
        {
            appState.renderMode = selectedRenderMode;
            applySceneEditResult(appState, makeRenderSettingsDirty());
        }
        int selectedQuality = appState.renderQuality;
        if (qualityCombo(u8c(u8"\u041A\u0430\u0447\u0435\u0441\u0442\u0432\u043E##scene_quality"), selectedQuality))
        {
            appState.renderQuality = selectedQuality;
            applyQualityMode(appState);
        }
        if (ImGui::Checkbox(u8c(u8"\u0428\u0443\u043C\u043E\u043F\u043E\u0434\u0430\u0432\u0438\u0442\u0435\u043B\u044C##renderer_denoiser"), &appState.denoiserEnabled))
        {
            applySceneEditResult(appState, makeRenderSettingsDirty());
        }
        if (ImGui::Checkbox(u8c(u8"\u041F\u0430\u0443\u0437\u0430 \u0440\u0435\u043D\u0434\u0435\u0440\u0430##renderer_pause"), &appState.renderingPaused))
        {
            applySceneEditResult(appState, makeRenderSettingsDirty());
        }
        if (ImGui::Button(u8c(u8"\u0421\u0431\u0440\u043E\u0441\u0438\u0442\u044C \u043D\u0430\u043A\u043E\u043F\u043B\u0435\u043D\u0438\u0435##reset_accumulation")))
        {
            invalidateAccumulation(appState);
        }
    }
    else if (appState.hierarchySelectionKind == HierarchySelectionCamera)
    {
        ImGui::Text("Position: %.2f %.2f %.2f", appState.camera.position.x, appState.camera.position.y, appState.camera.position.z);
        ImGui::Text("Yaw/Pitch/FOV: %.1f %.1f %.1f", appState.camera.yaw, appState.camera.pitch, appState.camera.fov);
        ImGui::TextWrapped("%s", u8c(u8"\u041A\u0430\u043C\u0435\u0440\u0430 \u0434\u0432\u0438\u0433\u0430\u0435\u0442\u0441\u044F WASD/Space \u0438 \u043C\u044B\u0448\u044C\u044E."));
    }
    else if (appState.hierarchySelectionKind == HierarchySelectionLight)
    {
        float lightPosition[3] = {scene.lightPosition.x, scene.lightPosition.y, scene.lightPosition.z};
        if (ImGui::DragFloat3(u8c(u8"\u041F\u043E\u0437\u0438\u0446\u0438\u044F"), lightPosition, 0.05f, -40.0f, 40.0f, "%.2f"))
        {
            const SceneState before = scene;
            applySceneEditResultWithUndo(appState, before, editor.setLightPosition(make_float3(lightPosition[0], std::max(6.0f, lightPosition[1]), lightPosition[2])));
        }
        float lightIntensity = scene.lightIntensity;
        if (ImGui::SliderFloat(u8c(u8"\u0421\u0438\u043B\u0430"), &lightIntensity, 0.0f, 5.0f, "%.2f"))
        {
            const SceneState before = scene;
            applySceneEditResultWithUndo(appState, before, editor.setLightIntensity(lightIntensity));
        }
        float areaLightRadius = scene.areaLightRadius;
        if (ImGui::SliderFloat(u8c(u8"\u0420\u0430\u0437\u043C\u0435\u0440 \u0438\u0441\u0442\u043E\u0447\u043D\u0438\u043A\u0430"), &areaLightRadius, 0.0f, 8.0f, "%.2f"))
        {
            const SceneState before = scene;
            scene.areaLightRadius = clampf(areaLightRadius, 0.0f, 8.0f);
            applySceneEditResultWithUndo(appState, before, makeLightingDirty());
        }
    }
    else if (appState.hierarchySelectionKind == HierarchySelectionGroup)
    {
        if (!scene.groups.empty() &&
            scene.selectedGroup >= 0 &&
            scene.selectedGroup < static_cast<int>(scene.groups.size()))
        {
            SceneGroup& group = scene.groups[static_cast<size_t>(scene.selectedGroup)];
            char nameBuffer[128]{};
            std::snprintf(nameBuffer, sizeof(nameBuffer), "%s", group.name.c_str());
            if (ImGui::InputText(u8c(u8"\u0418\u043C\u044F##group_name"), nameBuffer, sizeof(nameBuffer)))
            {
                group.name = nameBuffer;
            }

            ImGui::Text("%s: %zu", u8c(u8"\u041E\u0431\u044A\u0435\u043A\u0442\u043E\u0432"), group.objects.size());
            float position[3] = {group.position.x, group.position.y, group.position.z};
            float rotation[3] = {group.rotation.x, group.rotation.y, group.rotation.z};
            float scale[3] = {group.scale.x, group.scale.y, group.scale.z};
            if (ImGui::DragFloat3(u8c(u8"\u041F\u043E\u0437\u0438\u0446\u0438\u044F##group_pos"), position, 0.08f, -50.0f, 50.0f, "%.2f"))
            {
                const SceneState before = scene;
                applySceneEditResultWithUndo(appState, before, editor.setSelectedGroupPosition(make_float3(position[0], position[1], position[2])));
            }
            if (ImGui::DragFloat3(u8c(u8"\u041F\u043E\u0432\u043E\u0440\u043E\u0442##group_rot"), rotation, 0.8f, -360.0f, 360.0f, "%.1f"))
            {
                const SceneState before = scene;
                applySceneEditResultWithUndo(appState, before, editor.setSelectedGroupRotation(make_float3(rotation[0], rotation[1], rotation[2])));
            }
            if (ImGui::DragFloat3(u8c(u8"\u041C\u0430\u0441\u0448\u0442\u0430\u0431##group_scale"), scale, 0.05f, 0.05f, 100.0f, "%.2f"))
            {
                const SceneState before = scene;
                applySceneEditResultWithUndo(appState, before, editor.setSelectedGroupScale(make_float3(scale[0], scale[1], scale[2])));
            }

            if (ImGui::Button(u8c(u8"\u0420\u0430\u0437\u0433\u0440\u0443\u043F\u043F\u0438\u0440\u043E\u0432\u0430\u0442\u044C##ungroup")))
            {
                const SceneState before = scene;
                applySceneEditResultWithUndo(appState, before, editor.ungroupSelectedGroup());
                selectSafeObjectAfterDelete();
            }
            ImGui::SameLine();
            if (ImGui::Button(u8c(u8"\u0423\u0434\u0430\u043B\u0438\u0442\u044C \u0433\u0440\u0443\u043F\u043F\u0443 \u0438 \u043E\u0431\u044A\u0435\u043A\u0442\u044B##delete_group")))
            {
                const SceneState before = scene;
                applySceneEditResultWithUndo(appState, before, editor.removeSelectedGroup());
                appState.groupSelectionSpheres.clear();
                appState.groupSelectionMeshes.clear();
                selectSafeObjectAfterDelete();
                ImGui::EndChild();
                ImGui::PopItemWidth();
                if (!appState.lastUiMessage.empty())
                {
                    ImGui::TextWrapped("%s: %s",
                        appState.lastUiMessageIsError ? u8c(u8"\u041E\u0448\u0438\u0431\u043A\u0430") : u8c(u8"\u0421\u0442\u0430\u0442\u0443\u0441"),
                        appState.lastUiMessage.c_str());
                }
                ImGui::End();
                return;
            }
        }
    }
    else if (appState.hierarchySelectionKind == HierarchySelectionSphere)
    {
        if (!scene.spheres.empty())
        {
            clampScene(scene);
            SphereGeometry& sphere = scene.spheres[static_cast<size_t>(scene.selectedSphere)];
            if (ImGui::Button(u8c(u8"\u0423\u0434\u0430\u043B\u0438\u0442\u044C \u0441\u0444\u0435\u0440\u0443")))
            {
                const SceneState before = scene;
                applySceneEditResultWithUndo(appState, before, editor.deleteSelectedSphere());
                selectSafeObjectAfterDelete();
                ImGui::EndChild();
                ImGui::PopItemWidth();
                if (!appState.lastUiMessage.empty())
                {
                    ImGui::TextWrapped("%s: %s",
                        appState.lastUiMessageIsError ? u8c(u8"\u041E\u0448\u0438\u0431\u043A\u0430") : u8c(u8"\u0421\u0442\u0430\u0442\u0443\u0441"),
                        appState.lastUiMessage.c_str());
                }
                ImGui::End();
                return;
            }
            float pos[3] = {sphere.center.x, sphere.center.y, sphere.center.z};
            if (ImGui::DragFloat3(u8c(u8"\u041F\u043E\u0437\u0438\u0446\u0438\u044F##sphere_pos"), pos, 0.05f, -50.0f, 50.0f, "%.2f"))
            {
                const SceneState before = scene;
                const float oldRadius = sphere.radius;
                sphere.center = make_float3(pos[0], pos[1], pos[2]);
                sphere.radius = oldRadius;
                clampScene(scene);
                applySceneEditResultWithUndo(appState, before, makeTransformDirty());
            }
            float radius = sphere.radius;
            if (ImGui::SliderFloat(u8c(u8"\u0420\u0430\u0434\u0438\u0443\u0441"), &radius, 0.25f, 5.0f, "%.2f"))
            {
                const SceneState before = scene;
                applySceneEditResultWithUndo(appState, before, editor.setSelectedSphereRadius(radius));
            }
            drawSphereMaterialEditor();
        }
    }
    else if (appState.hierarchySelectionKind == HierarchySelectionMesh)
    {
        refreshMeshSelection();
        if (selectedMeshObject != nullptr)
        {
            char nameBuffer[128]{};
            const std::string currentName = meshObjectLabel(*selectedMeshObject, scene.selectedMeshObject);
            std::snprintf(nameBuffer, sizeof(nameBuffer), "%s", currentName.c_str());
            if (ImGui::InputText(u8c(u8"\u0418\u043C\u044F##mesh_name"), nameBuffer, sizeof(nameBuffer)))
            {
                selectedMeshObject->displayName = nameBuffer;
            }
            if (ImGui::Button(u8c(u8"\u0423\u0434\u0430\u043B\u0438\u0442\u044C \u043C\u043E\u0434\u0435\u043B\u044C")))
            {
                const SceneState before = scene;
                applySceneEditResultWithUndo(appState, before, editor.deleteSelectedMeshObject());
                selectSafeObjectAfterDelete();
                ImGui::EndChild();
                ImGui::PopItemWidth();
                if (!appState.lastUiMessage.empty())
                {
                    ImGui::TextWrapped("%s: %s",
                        appState.lastUiMessageIsError ? u8c(u8"\u041E\u0448\u0438\u0431\u043A\u0430") : u8c(u8"\u0421\u0442\u0430\u0442\u0443\u0441"),
                        appState.lastUiMessage.c_str());
                }
                ImGui::End();
                return;
            }
            float position[3] = {selectedMeshObject->position.x, selectedMeshObject->position.y, selectedMeshObject->position.z};
            float rotation[3] = {selectedMeshObject->rotation.x, selectedMeshObject->rotation.y, selectedMeshObject->rotation.z};
            float scale[3] = {selectedMeshObject->scale.x, selectedMeshObject->scale.y, selectedMeshObject->scale.z};
            if (ImGui::DragFloat3(u8c(u8"\u041F\u043E\u0437\u0438\u0446\u0438\u044F##mesh_pos"), position, 0.08f, -50.0f, 50.0f, "%.2f"))
            {
                const SceneState before = scene;
                applySceneEditResultWithUndo(appState, before, editor.setSelectedMeshPosition(make_float3(position[0], position[1], position[2])));
            }
            if (ImGui::DragFloat3(u8c(u8"\u041F\u043E\u0432\u043E\u0440\u043E\u0442##mesh_rot"), rotation, 0.8f, -360.0f, 360.0f, "%.1f"))
            {
                const SceneState before = scene;
                applySceneEditResultWithUndo(appState, before, editor.setSelectedMeshRotation(make_float3(rotation[0], rotation[1], rotation[2])));
            }
            if (ImGui::DragFloat3(u8c(u8"\u041C\u0430\u0441\u0448\u0442\u0430\u0431##mesh_scale"), scale, 0.10f, 0.05f, 100.0f, "%.2f"))
            {
                const SceneState before = scene;
                applySceneEditResultWithUndo(appState, before, editor.setSelectedMeshScale(make_float3(scale[0], scale[1], scale[2])));
            }
            drawMeshMaterialEditor();
        }
    }

    if (!appState.lastUiMessage.empty())
    {
        ImGui::TextWrapped("%s: %s",
            appState.lastUiMessageIsError ? u8c(u8"\u041E\u0448\u0438\u0431\u043A\u0430") : u8c(u8"\u0421\u0442\u0430\u0442\u0443\u0441"),
            appState.lastUiMessage.c_str());
    }
    ImGui::EndChild();
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
    SceneEditor editor(scene);

    const bool imguiCapturesKeyboard =
        ImGui::GetCurrentContext() != nullptr &&
        (ImGui::GetIO().WantCaptureKeyboard || ImGui::GetIO().WantTextInput);
    if (imguiCapturesKeyboard)
    {
        return;
    }

    static bool zWasDown = false;
    const bool zIsDown = glfwGetKey(window, GLFW_KEY_Z) == GLFW_PRESS;
    const bool ctrlIsDown =
        glfwGetKey(window, GLFW_KEY_LEFT_CONTROL) == GLFW_PRESS ||
        glfwGetKey(window, GLFW_KEY_RIGHT_CONTROL) == GLFW_PRESS;
    if (zIsDown && ctrlIsDown && !zWasDown)
    {
        undoLastSceneEdit(appState);
    }
    zWasDown = zIsDown;
    if (ctrlIsDown)
    {
        return;
    }

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
    bool cameraChanged = false;
    if (glfwGetKey(window, GLFW_KEY_W) == GLFW_PRESS)
    {
        camera.position = add3(camera.position, mul3(forward, cameraStep));
        cameraChanged = true;
    }
    if (glfwGetKey(window, GLFW_KEY_S) == GLFW_PRESS)
    {
        camera.position = sub3(camera.position, mul3(forward, cameraStep));
        cameraChanged = true;
    }
    if (glfwGetKey(window, GLFW_KEY_A) == GLFW_PRESS)
    {
        camera.position = sub3(camera.position, mul3(right, cameraStep));
        cameraChanged = true;
    }
    if (glfwGetKey(window, GLFW_KEY_D) == GLFW_PRESS)
    {
        camera.position = add3(camera.position, mul3(right, cameraStep));
        cameraChanged = true;
    }
    if (glfwGetKey(window, GLFW_KEY_SPACE) == GLFW_PRESS)
    {
        camera.position = add3(camera.position, mul3(up, cameraStep));
        cameraChanged = true;
    }
    if (cameraChanged)
    {
        applySceneEditResult(appState, makeCameraDirty());
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
            const SceneState before = scene;
            applySceneEditResultWithUndo(appState, before, editor.setSelectedMeshPosition(add3(object.position, delta)));
        }
        else if (appState.editorObjectKind == EditorObjectGroup &&
            scene.selectedGroup >= 0 &&
            scene.selectedGroup < static_cast<int>(scene.groups.size()))
        {
            const SceneGroup& group = scene.groups[static_cast<size_t>(scene.selectedGroup)];
            const SceneState before = scene;
            applySceneEditResultWithUndo(appState, before, editor.setSelectedGroupPosition(add3(group.position, delta)));
        }
        else
        {
            const SceneState before = scene;
            applySceneEditResultWithUndo(appState, before, editor.moveSelectedSphere(delta));
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
        const SceneState before = scene;
        applySceneEditResultWithUndo(appState, before, editor.moveLight(make_float3(-lightStep, 0.0f, 0.0f)));
    }
    if (glfwGetKey(window, GLFW_KEY_L) == GLFW_PRESS)
    {
        const SceneState before = scene;
        applySceneEditResultWithUndo(appState, before, editor.moveLight(make_float3(lightStep, 0.0f, 0.0f)));
    }
    if (glfwGetKey(window, GLFW_KEY_I) == GLFW_PRESS)
    {
        const SceneState before = scene;
        applySceneEditResultWithUndo(appState, before, editor.moveLight(make_float3(0.0f, 0.0f, -lightStep)));
    }
    if (glfwGetKey(window, GLFW_KEY_K) == GLFW_PRESS)
    {
        const SceneState before = scene;
        applySceneEditResultWithUndo(appState, before, editor.moveLight(make_float3(0.0f, 0.0f, lightStep)));
    }
    if (glfwGetKey(window, GLFW_KEY_U) == GLFW_PRESS)
    {
        const SceneState before = scene;
        applySceneEditResultWithUndo(appState, before, editor.moveLight(make_float3(0.0f, lightStep, 0.0f)));
    }
    if (glfwGetKey(window, GLFW_KEY_O) == GLFW_PRESS)
    {
        const SceneState before = scene;
        applySceneEditResultWithUndo(appState, before, editor.moveLight(make_float3(0.0f, -lightStep, 0.0f)));
    }
    static bool mWasDown = false;
    const bool mIsDown = glfwGetKey(window, GLFW_KEY_M) == GLFW_PRESS;
    if (mIsDown && !mWasDown)
    {
        const SceneState before = scene;
        applySceneEditResultWithUndo(appState, before, editor.cycleSelectedSphereMaterialPreset());
    }
    mWasDown = mIsDown;

    static bool bWasDown = false;
    const bool bIsDown = glfwGetKey(window, GLFW_KEY_B) == GLFW_PRESS;
    if (bIsDown && !bWasDown)
    {
        selectPreviousSceneObject(scene, appState.hierarchySelectionKind);
        if (appState.hierarchySelectionKind == SceneHierarchySelectionSphere)
        {
            appState.editorObjectKind = EditorObjectSphere;
            appState.hierarchySelectionIndex = scene.selectedSphere;
        }
        else if (appState.hierarchySelectionKind == SceneHierarchySelectionMesh)
        {
            appState.editorObjectKind = EditorObjectMesh;
            appState.hierarchySelectionIndex = scene.selectedMeshObject;
        }
        else if (appState.hierarchySelectionKind == SceneHierarchySelectionGroup)
        {
            appState.editorObjectKind = EditorObjectGroup;
            appState.hierarchySelectionIndex = scene.selectedGroup;
        }
        applySceneEditResult(appState, makeRenderSettingsDirty());
    }
    bWasDown = bIsDown;

    static bool vWasDown = false;
    const bool vIsDown = glfwGetKey(window, GLFW_KEY_V) == GLFW_PRESS;
    if (vIsDown && !vWasDown)
    {
        const SceneState before = scene;
        applySceneEditResultWithUndo(appState, before, editor.cycleSelectedMeshMaterialPreset());
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
        applySceneEditResult(appState, makeRenderSettingsDirty());
    }
    pWasDown = pIsDown;

    static bool nWasDown = false;
    const bool nIsDown = glfwGetKey(window, GLFW_KEY_N) == GLFW_PRESS;
    if (nIsDown && !nWasDown)
    {
        appState.denoiserEnabled = !appState.denoiserEnabled;
        applySceneEditResult(appState, makeRenderSettingsDirty());
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
        const SceneState before = scene;
        applySceneEditResultWithUndo(appState, before, editor.adjustExposure(-0.05f));
    }
    key4WasDown = key4IsDown;

    static bool key5WasDown = false;
    const bool key5IsDown = glfwGetKey(window, GLFW_KEY_5) == GLFW_PRESS;
    if (key5IsDown && !key5WasDown)
    {
        const SceneState before = scene;
        applySceneEditResultWithUndo(appState, before, editor.adjustExposure(0.05f));
    }
    key5WasDown = key5IsDown;

    static bool key6WasDown = false;
    const bool key6IsDown = glfwGetKey(window, GLFW_KEY_6) == GLFW_PRESS;
    if (key6IsDown && !key6WasDown)
    {
        const SceneState before = scene;
        applySceneEditResultWithUndo(appState, before, editor.adjustSkyIntensity(-0.05f));
    }
    key6WasDown = key6IsDown;

    static bool key7WasDown = false;
    const bool key7IsDown = glfwGetKey(window, GLFW_KEY_7) == GLFW_PRESS;
    if (key7IsDown && !key7WasDown)
    {
        const SceneState before = scene;
        applySceneEditResultWithUndo(appState, before, editor.adjustSkyIntensity(0.05f));
    }
    key7WasDown = key7IsDown;

    static bool key8WasDown = false;
    const bool key8IsDown = glfwGetKey(window, GLFW_KEY_8) == GLFW_PRESS;
    if (key8IsDown && !key8WasDown)
    {
        const SceneState before = scene;
        applySceneEditResultWithUndo(appState, before, editor.adjustLightIntensity(-0.1f));
    }
    key8WasDown = key8IsDown;

    static bool key9WasDown = false;
    const bool key9IsDown = glfwGetKey(window, GLFW_KEY_9) == GLFW_PRESS;
    if (key9IsDown && !key9WasDown)
    {
        const SceneState before = scene;
        applySceneEditResultWithUndo(appState, before, editor.adjustLightIntensity(0.1f));
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

std::filesystem::path defaultSavedScenePath()
{
    const std::filesystem::path sourceDir = RAYTRACERRTX_SOURCE_DIR;
    if (!sourceDir.empty())
    {
        return sourceDir.parent_path() / "assets" / "scenes" / "saved_scene.json";
    }
    return std::filesystem::path("RayTracerRTX") / "assets" / "scenes" / "saved_scene.json";
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
    addSceneConfigPreset(appState, "floating_sphere_scene.json", L"\u0421\u0444\u0435\u0440\u0430 \u0432 \u0432\u043E\u0437\u0434\u0443\u0445\u0435");
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
            invalidateAccumulation(appState);
            appState.rendererSceneRebuildRequested = false;
        }

        const auto hostFrameStart = std::chrono::steady_clock::now();
        float gpuTimeMs = 0.0f;
        if (!appState.renderingPaused)
        {
            renderer.setRenderQuality(appState.renderQuality);
            renderer.setRenderMode(appState.renderMode);
            renderer.setDenoiserEnabled(appState.denoiserEnabled);
            renderer.renderFrame(appState.scene, appState.camera, pixels, &gpuTimeMs);
            appState.progressiveSamples = renderer.getAccumulationSampleCount();
            appState.denoiserAvailable = renderer.isDenoiserAvailable();
        }
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
            appState.hierarchySelectionKind,
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
