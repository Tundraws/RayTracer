#include "application.h"

#include "../gpu/optix_renderer.h"
#include "camera.h"
#include "material.h"
#include "scene.h"

#ifndef NOMINMAX
#define NOMINMAX
#endif
#include <windows.h>

#include <GLFW/glfw3.h>
#define GLFW_EXPOSE_NATIVE_WIN32
#include <GLFW/glfw3native.h>
#include <GL/gl.h>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cctype>
#include <cwctype>
#include <iomanip>
#include <iostream>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

namespace
{
int gWidth = 800;
int gHeight = 600;

struct AppState
{
    CameraState camera;
    InputState input;
    SceneState scene = makeDefaultScene();
    bool cursorCaptured = true;
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

void drawHud(GLFWwindow* window, const SceneState& scene, const FrameStats& stats)
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
          << L"   FRAME: " << stats.avgHostMs << L" MS   GPU: " << stats.avgGpuMs << L" MS";

    std::wostringstream line2;
    line2 << L"\u0421\u0424\u0415\u0420\u0410: " << (scene.selectedSphere + 1)
          << L"   \u041c\u0410\u0422\u0415\u0420\u0418\u0410\u041b: " << materialNameW(scene.materials[scene.selectedSphere].materialType)
          << L"   \u0421\u0412\u0415\u0422: " << std::fixed << std::setprecision(1)
          << scene.lightPosition.x << L" " << scene.lightPosition.y << L" " << scene.lightPosition.z;

    const std::wstring line3 = L"\u0421\u0424\u0415\u0420\u042b: 1-3 \u0412\u042b\u0411\u041e\u0420";
    const std::wstring line4 = L"\u0414\u0412\u0418\u0416\u0415\u041d\u0418\u0415 \u0421\u0424\u0415\u0420\u042b: \u0421\u0422\u0420\u0415\u041b\u041a\u0418 - X/Z";
    const std::wstring line5 = L"PGUP/PGDN \u0418\u041b\u0418 R/F - Y";
    const std::wstring line6 = L"\u0421\u0412\u0415\u0422: J/L - X, I/K - Z, U/O - Y";
    const std::wstring line7 = L"\u041c\u0410\u0422\u0415\u0420\u0418\u0410\u041b: M   \u041a\u0410\u041c\u0415\u0420\u0410: WASD/QE + \u041c\u042b\u0428\u042c";

    const std::wstring text1 = line1.str();
    const std::wstring text2 = line2.str();
    TextOutW(dc, panelX, panelY, text1.c_str(), static_cast<int>(text1.size()));
    TextOutW(dc, panelX, panelY + 22, text2.c_str(), static_cast<int>(text2.size()));
    TextOutW(dc, panelX, panelY + 44, line3.c_str(), static_cast<int>(line3.size()));
    TextOutW(dc, panelX, panelY + 66, line4.c_str(), static_cast<int>(line4.size()));
    TextOutW(dc, panelX, panelY + 88, line5.c_str(), static_cast<int>(line5.size()));
    TextOutW(dc, panelX, panelY + 110, line6.c_str(), static_cast<int>(line6.size()));
    TextOutW(dc, panelX, panelY + 132, line7.c_str(), static_cast<int>(line7.size()));

    SelectObject(dc, oldFont);
    SetTextColor(dc, oldTextColor);
    SetBkMode(dc, oldBkMode);
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

    constexpr float sensitivity = 0.12f;
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
    if (button != GLFW_MOUSE_BUTTON_LEFT || action != GLFW_PRESS)
    {
        return;
    }

    auto* state = static_cast<AppState*>(glfwGetWindowUserPointer(window));
    if (state == nullptr)
    {
        return;
    }

    state->cursorCaptured = !state->cursorCaptured;
    glfwSetInputMode(window, GLFW_CURSOR, state->cursorCaptured ? GLFW_CURSOR_DISABLED : GLFW_CURSOR_NORMAL);
    state->input.firstMouse = true;
}

void processInput(GLFWwindow* window, AppState& appState)
{
    CameraState& camera = appState.camera;
    SceneState& scene = appState.scene;

    float3 forward{};
    float3 right{};
    float3 up{};
    float scale = 0.0f;
    float aspect = 0.0f;
    updateCameraBasis(camera, gWidth, gHeight, forward, right, up, scale, aspect);

    float3 cameraRight = make_float3(-right.x, 0.0f, -right.z);
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

    constexpr float speed = 0.12f;
    if (glfwGetKey(window, GLFW_KEY_W) == GLFW_PRESS)
    {
        camera.position = add3(camera.position, mul3(forward, speed));
    }
    if (glfwGetKey(window, GLFW_KEY_S) == GLFW_PRESS)
    {
        camera.position = sub3(camera.position, mul3(forward, speed));
    }
    if (glfwGetKey(window, GLFW_KEY_A) == GLFW_PRESS)
    {
        camera.position = sub3(camera.position, mul3(right, speed));
    }
    if (glfwGetKey(window, GLFW_KEY_D) == GLFW_PRESS)
    {
        camera.position = add3(camera.position, mul3(right, speed));
    }
    if (glfwGetKey(window, GLFW_KEY_Q) == GLFW_PRESS)
    {
        camera.position = add3(camera.position, mul3(up, speed));
    }
    if (glfwGetKey(window, GLFW_KEY_E) == GLFW_PRESS)
    {
        camera.position = sub3(camera.position, mul3(up, speed));
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

    constexpr float sphereStep = 0.06f;
    constexpr float verticalStep = 0.10f;
    if (glfwGetKey(window, GLFW_KEY_LEFT) == GLFW_PRESS)
    {
        moveSelectedSphere(scene, mul3(cameraRight, -sphereStep));
    }
    if (glfwGetKey(window, GLFW_KEY_RIGHT) == GLFW_PRESS)
    {
        moveSelectedSphere(scene, mul3(cameraRight, sphereStep));
    }
    if (glfwGetKey(window, GLFW_KEY_UP) == GLFW_PRESS)
    {
        moveSelectedSphere(scene, mul3(cameraForward, sphereStep));
    }
    if (glfwGetKey(window, GLFW_KEY_DOWN) == GLFW_PRESS)
    {
        moveSelectedSphere(scene, mul3(cameraForward, -sphereStep));
    }
    if (glfwGetKey(window, GLFW_KEY_PAGE_UP) == GLFW_PRESS)
    {
        moveSelectedSphere(scene, make_float3(0.0f, verticalStep, 0.0f));
    }
    if (glfwGetKey(window, GLFW_KEY_PAGE_DOWN) == GLFW_PRESS)
    {
        moveSelectedSphere(scene, make_float3(0.0f, -verticalStep, 0.0f));
    }
    if (glfwGetKey(window, GLFW_KEY_R) == GLFW_PRESS)
    {
        moveSelectedSphere(scene, make_float3(0.0f, verticalStep, 0.0f));
    }
    if (glfwGetKey(window, GLFW_KEY_F) == GLFW_PRESS)
    {
        moveSelectedSphere(scene, make_float3(0.0f, -verticalStep, 0.0f));
    }

    constexpr float lightStep = 0.07f;
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
        toggleSelectedMaterial(scene);
    }
    mWasDown = mIsDown;
}
} // namespace

void run_optix_app()
{
    if (!glfwInit())
    {
        throw std::runtime_error("Не удалось инициализировать GLFW.");
    }

    GLFWmonitor* monitor = glfwGetPrimaryMonitor();
    const GLFWvidmode* mode = monitor != nullptr ? glfwGetVideoMode(monitor) : nullptr;
    const int windowWidth = 1280;
    const int windowHeight = 720;
    const int windowX = mode != nullptr ? std::max(0, (mode->width - windowWidth) / 2) : 100;
    const int windowY = mode != nullptr ? std::max(0, (mode->height - windowHeight) / 2) : 100;

    glfwWindowHint(GLFW_DECORATED, GLFW_TRUE);
    GLFWwindow* window = glfwCreateWindow(windowWidth, windowHeight, "RayTracerRTX OptiX", nullptr, nullptr);
    if (window == nullptr)
    {
        glfwTerminate();
        throw std::runtime_error("Не удалось создать окно GLFW.");
    }

    glfwMakeContextCurrent(window);
    glfwSwapInterval(1);
    glfwSetWindowPos(window, windowX, windowY);
    glfwGetFramebufferSize(window, &gWidth, &gHeight);
    glViewport(0, 0, gWidth, gHeight);

    AppState appState;
    glfwSetWindowUserPointer(window, &appState);
    glfwSetCursorPosCallback(window, mouseCallback);
    glfwSetMouseButtonCallback(window, mouseButtonCallback);
    glfwSetInputMode(window, GLFW_CURSOR, GLFW_CURSOR_DISABLED);

    OptixRenderer renderer;
    renderer.setRenderSize(gWidth, gHeight);
    renderer.initialize();

    std::vector<uchar4> pixels(gWidth * gHeight);
    FrameStats stats;

    while (!glfwWindowShouldClose(window))
    {
        if (glfwGetKey(window, GLFW_KEY_ESCAPE) == GLFW_PRESS)
        {
            glfwSetWindowShouldClose(window, GLFW_TRUE);
        }

        processInput(window, appState);

        const auto hostFrameStart = std::chrono::steady_clock::now();
        float gpuTimeMs = 0.0f;
        renderer.renderFrame(appState.scene, appState.camera, pixels, &gpuTimeMs);
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
            std::ostringstream title;
            title << std::fixed << std::setprecision(1)
                  << "RayTracerRTX OptiX | FPS " << stats.fps
                  << " | Frame " << stats.avgHostMs << " ms"
                  << " | GPU " << stats.avgGpuMs << " ms"
                  << " | Sphere " << (appState.scene.selectedSphere + 1)
                  << " " << materialName(appState.scene.materials[appState.scene.selectedSphere].materialType)
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
        glfwSwapBuffers(window);
        drawHud(window, appState.scene, stats);
        glfwPollEvents();
    }

    renderer.destroy();
    glfwDestroyWindow(window);
    glfwTerminate();
}
