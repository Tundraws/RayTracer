#include "optix_renderer.h"

#include "optix_device_programs.h"

#ifndef NOMINMAX
#define NOMINMAX
#endif
#include <windows.h>

#include <GLFW/glfw3.h>
#define GLFW_EXPOSE_NATIVE_WIN32
#include <GLFW/glfw3native.h>
#include <GL/gl.h>

#include <cuda.h>
#include <cuda_runtime.h>
#include <nvrtc.h>

#include <optix.h>
#include <optix_function_table_definition.h>
#include <optix_stack_size.h>
#include <optix_stubs.h>

#include <cmath>
#include <chrono>
#include <algorithm>
#include <array>
#include <cwctype>
#include <cstdint>
#include <cctype>
#include <iomanip>
#include <iostream>
#include <sstream>
#include <stdexcept>
#include <cstring>
#include <string>
#include <vector>

namespace
{
int gWidth = 800;
int gHeight = 600;
constexpr float kPi = 3.14159265358979323846f;

enum MaterialType
{
    MaterialDiffuse = 0,
    MaterialMirror = 1
};

enum RayType
{
    RayTypeRadiance = 0,
    RayTypeShadow = 1,
    RayTypeCount = 2
};

struct SphereGeometry
{
    float3 center;
    float radius;
};

struct SphereMaterial
{
    float3 color;
    int materialType;
};

struct SceneState
{
    std::vector<SphereGeometry> spheres;
    std::vector<SphereMaterial> materials;
    float3 lightPosition;
    int selectedSphere = 0;
};

SceneState makeDefaultScene();

struct LaunchParams
{
    uchar4* image;
    unsigned int imageWidth;
    unsigned int imageHeight;
    OptixTraversableHandle handle;
    float3 cameraPosition;
    float3 cameraForward;
    float3 cameraRight;
    float3 cameraUp;
    float cameraScale;
    float cameraAspect;
    float3 lightPosition;
    SphereMaterial* materials;
    int sphereCount;
    int maxDepth;
};

template <typename T>
struct SbtRecord
{
    __align__(OPTIX_SBT_RECORD_ALIGNMENT) char header[OPTIX_SBT_RECORD_HEADER_SIZE];
    T data;
};

struct EmptyData
{
};

using RaygenRecord = SbtRecord<EmptyData>;
using MissRecord = SbtRecord<EmptyData>;
using HitgroupRecord = SbtRecord<EmptyData>;

struct CameraState
{
    float3 position = make_float3(0.0f, 1.3f, -6.0f);
    float yaw = 90.0f;
    float pitch = -10.0f;
    float fov = 60.0f;
};

struct InputState
{
    double lastX = gWidth / 2.0;
    double lastY = gHeight / 2.0;
    bool firstMouse = true;
};

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

struct ProgramGroups
{
    OptixProgramGroup raygen = nullptr;
    OptixProgramGroup missRadiance = nullptr;
    OptixProgramGroup missShadow = nullptr;
    OptixProgramGroup hitRadiance = nullptr;
    OptixProgramGroup hitShadow = nullptr;
};

struct OptixRenderer
{
    void initialize();
    void renderFrame(const SceneState& scene, const CameraState& camera, std::vector<uchar4>& hostPixels, float* gpuTimeMs = nullptr);
    void destroy();

private:
    void createContext();
    void createScene();
    void createModule();
    void createProgramGroups();
    void createPipeline();
    void createSbt();
    void rebuildAccelerationStructure();

    OptixDeviceContext context = nullptr;
    OptixModule module = nullptr;
    OptixModule sphereModule = nullptr;
    ProgramGroups programGroups;
    OptixPipeline pipeline = nullptr;
    OptixShaderBindingTable sbt = {};
    OptixPipelineCompileOptions pipelineCompileOptions = {};

    CUstream stream = nullptr;
    CUdeviceptr dSphereCenters = 0;
    CUdeviceptr dSphereRadii = 0;
    CUdeviceptr dMaterials = 0;
    CUdeviceptr dSphereGasBuffer = 0;
    CUdeviceptr dLaunchParams = 0;
    OptixTraversableHandle sphereGasHandle = 0;
    std::vector<uint32_t> sphereFlags;
    OptixBuildInput sphereBuildInput = {};
    OptixAccelBuildOptions sphereAccelOptions = {};
    OptixAccelBufferSizes sphereGasSizes = {};
    cudaEvent_t frameStart = nullptr;
    cudaEvent_t frameStop = nullptr;

    uchar4* dFrameBuffer = nullptr;
};

inline void cudaCheck(cudaError_t result, const char* expression, const char* file, int line)
{
    if (result != cudaSuccess)
    {
        std::ostringstream out;
        out << "CUDA error: " << expression << " failed at " << file << ":" << line
            << " with code " << static_cast<int>(result) << " (" << cudaGetErrorString(result) << ")";
        throw std::runtime_error(out.str());
    }
}

inline void optixCheck(OptixResult result, const char* expression, const char* file, int line)
{
    if (result != OPTIX_SUCCESS)
    {
        std::ostringstream out;
        out << "OptiX error: " << expression << " failed at " << file << ":" << line
            << " with code " << static_cast<int>(result);
        throw std::runtime_error(out.str());
    }
}

inline void nvrtcCheck(nvrtcResult result, const char* expression, const char* file, int line)
{
    if (result != NVRTC_SUCCESS)
    {
        std::ostringstream out;
        out << "NVRTC error: " << expression << " failed at " << file << ":" << line
            << " with code " << static_cast<int>(result) << " (" << nvrtcGetErrorString(result) << ")";
        throw std::runtime_error(out.str());
    }
}

#define CUDA_CHECK(expr) cudaCheck((expr), #expr, __FILE__, __LINE__)
#define OPTIX_CHECK(expr) optixCheck((expr), #expr, __FILE__, __LINE__)
#define NVRTC_CHECK(expr) nvrtcCheck((expr), #expr, __FILE__, __LINE__)

void contextLogCallback(unsigned int level, const char* tag, const char* message, void*)
{
    std::cerr << "[" << std::setw(2) << level << "][" << tag << "] " << message << '\n';
}

std::string getShortPath(const std::string& path)
{
    const DWORD required = GetShortPathNameA(path.c_str(), nullptr, 0);
    if (required == 0)
    {
        return path;
    }

    std::string shortPath(required, '\0');
    char* buffer = shortPath.empty() ? nullptr : &shortPath[0];
    const DWORD written = GetShortPathNameA(path.c_str(), buffer, required);
    if (written == 0)
    {
        return path;
    }

    shortPath.resize(written);
    return shortPath;
}

std::string compileDeviceProgram()
{
    int device = 0;
    cudaDeviceProp props{};
    CUDA_CHECK(cudaGetDevice(&device));
    CUDA_CHECK(cudaGetDeviceProperties(&props, device));

    const std::string cudaInclude = getShortPath("C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v13.1\\include");
    const std::string optixInclude = getShortPath("C:\\ProgramData\\NVIDIA Corporation\\OptiX SDK 9.1.0\\include");
    const std::string architecture = "--gpu-architecture=compute_" + std::to_string(props.major) + std::to_string(props.minor);
    const std::string includeCuda = "-I" + cudaInclude;
    const std::string includeOptix = "-I" + optixInclude;

    const std::vector<const char*> options = {
        "--std=c++14",
        architecture.c_str(),
        includeCuda.c_str(),
        includeOptix.c_str(),
        "--use_fast_math",
        "--relocatable-device-code=true",
        "--device-as-default-execution-space",
        "--optix-ir"
    };

    nvrtcProgram program = nullptr;
    NVRTC_CHECK(nvrtcCreateProgram(
        &program,
        gpu::kOptixDeviceProgram,
        "optix_programs.cu",
        0,
        nullptr,
        nullptr));

    const nvrtcResult compileResult = nvrtcCompileProgram(
        program,
        static_cast<int>(options.size()),
        options.data());

    size_t logSize = 0;
    NVRTC_CHECK(nvrtcGetProgramLogSize(program, &logSize));
    std::string log(logSize, '\0');
    if (logSize > 1)
    {
        NVRTC_CHECK(nvrtcGetProgramLog(program, &log[0]));
        std::cout << log << '\n';
    }

    if (compileResult != NVRTC_SUCCESS)
    {
        nvrtcDestroyProgram(&program);
        throw std::runtime_error("Не удалось скомпилировать OptiX device-программы.");
    }

    size_t irSize = 0;
    NVRTC_CHECK(nvrtcGetOptiXIRSize(program, &irSize));
    std::string optixIr(irSize, '\0');
    NVRTC_CHECK(nvrtcGetOptiXIR(program, &optixIr[0]));
    NVRTC_CHECK(nvrtcDestroyProgram(&program));
    return optixIr;
}

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

float3 cross3(const float3 a, const float3 b)
{
    return make_float3(
        a.y * b.z - a.z * b.y,
        a.z * b.x - a.x * b.z,
        a.x * b.y - a.y * b.x);
}

float3 normalize3(const float3 v)
{
    const float length = std::sqrt(dot3(v, v));
    if (length <= 0.0f)
    {
        return make_float3(0.0f, 0.0f, 0.0f);
    }
    return mul3(v, 1.0f / length);
}

float3 add3(const float3 a, const float3 b);
float3 sub3(const float3 a, const float3 b);
float3 mul3(const float3 a, const float value);

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

SceneState makeDefaultScene()
{
    SceneState scene;
    scene.spheres = {
        {make_float3(0.0f, 0.0f, 0.0f), 1.0f},
        {make_float3(2.2f, -0.2f, 1.5f), 0.8f},
        {make_float3(-2.2f, -0.2f, 1.5f), 0.8f},
        {make_float3(0.0f, -1001.0f, 0.0f), 1000.0f}
    };

    scene.materials = {
        {make_float3(1.0f, 1.0f, 1.0f), MaterialMirror},
        {make_float3(0.82f, 0.70f, 0.60f), MaterialDiffuse},
        {make_float3(0.60f, 0.80f, 0.75f), MaterialDiffuse},
        {make_float3(0.88f, 0.88f, 0.90f), MaterialDiffuse}
    };

    scene.lightPosition = make_float3(7.0f, 10.0f, -10.0f);
    scene.selectedSphere = 0;
    return scene;
}

void clampScene(SceneState& scene)
{
    const float3 sphereMin = make_float3(-3.5f, 0.0f, -4.0f);
    const float3 sphereMax = make_float3(3.5f, 3.0f, 3.0f);
    const float floorY = scene.spheres.back().center.y + scene.spheres.back().radius;

    for (size_t i = 0; i + 1 < scene.spheres.size(); ++i)
    {
        SphereGeometry& sphere = scene.spheres[i];
        const float3 minBounds = make_float3(sphereMin.x, floorY + sphere.radius, sphereMin.z);
        const float3 maxBounds = make_float3(sphereMax.x, sphereMax.y, sphereMax.z);
        sphere.center = clamp3(sphere.center, minBounds, maxBounds);
    }

    scene.lightPosition = clamp3(
        scene.lightPosition,
        make_float3(-10.0f, 1.0f, -15.0f),
        make_float3(10.0f, 18.0f, 15.0f));

    if (scene.selectedSphere < 0)
    {
        scene.selectedSphere = 0;
    }
    if (scene.selectedSphere > static_cast<int>(scene.spheres.size()) - 2)
    {
        scene.selectedSphere = static_cast<int>(scene.spheres.size()) - 2;
    }
}

void moveSelectedSphere(SceneState& scene, const float3 delta)
{
    const int index = scene.selectedSphere;
    if (index < 0 || index >= static_cast<int>(scene.spheres.size()) - 1)
    {
        return;
    }

    scene.spheres[index].center = add3(scene.spheres[index].center, delta);
    clampScene(scene);
}

void toggleSelectedMaterial(SceneState& scene)
{
    const int index = scene.selectedSphere;
    if (index < 0 || index >= static_cast<int>(scene.materials.size()) - 1)
    {
        return;
    }

    SphereMaterial& material = scene.materials[index];
    material.materialType = (material.materialType == MaterialDiffuse) ? MaterialMirror : MaterialDiffuse;
}

void moveLight(SceneState& scene, const float3 delta)
{
    scene.lightPosition = add3(scene.lightPosition, delta);
    clampScene(scene);
}

const char* materialName(int materialType)
{
    return materialType == MaterialMirror ? "mirror" : "diffuse";
}

const wchar_t* materialNameW(int materialType)
{
    return materialType == MaterialMirror ? L"\u0417\u0415\u0420\u041A\u0410\u041B\u041E" : L"\u041C\u0410\u0422\u041E\u0412\u042B\u0419";
}

std::array<uint8_t, 7> glyphFor(char input)
{
    const char c = static_cast<char>(std::toupper(static_cast<unsigned char>(input)));
    switch (c)
    {
    case 'A': return {0x0E, 0x11, 0x11, 0x1F, 0x11, 0x11, 0x11};
    case 'B': return {0x1E, 0x11, 0x11, 0x1E, 0x11, 0x11, 0x1E};
    case 'C': return {0x0E, 0x11, 0x10, 0x10, 0x10, 0x11, 0x0E};
    case 'D': return {0x1E, 0x11, 0x11, 0x11, 0x11, 0x11, 0x1E};
    case 'E': return {0x1F, 0x10, 0x10, 0x1E, 0x10, 0x10, 0x1F};
    case 'F': return {0x1F, 0x10, 0x10, 0x1E, 0x10, 0x10, 0x10};
    case 'G': return {0x0E, 0x11, 0x10, 0x13, 0x11, 0x11, 0x0E};
    case 'H': return {0x11, 0x11, 0x11, 0x1F, 0x11, 0x11, 0x11};
    case 'I': return {0x1F, 0x04, 0x04, 0x04, 0x04, 0x04, 0x1F};
    case 'J': return {0x07, 0x02, 0x02, 0x02, 0x12, 0x12, 0x0C};
    case 'K': return {0x11, 0x12, 0x14, 0x18, 0x14, 0x12, 0x11};
    case 'L': return {0x10, 0x10, 0x10, 0x10, 0x10, 0x10, 0x1F};
    case 'M': return {0x11, 0x1B, 0x15, 0x15, 0x11, 0x11, 0x11};
    case 'N': return {0x11, 0x19, 0x15, 0x13, 0x11, 0x11, 0x11};
    case 'O': return {0x0E, 0x11, 0x11, 0x11, 0x11, 0x11, 0x0E};
    case 'P': return {0x1E, 0x11, 0x11, 0x1E, 0x10, 0x10, 0x10};
    case 'Q': return {0x0E, 0x11, 0x11, 0x11, 0x15, 0x12, 0x0D};
    case 'R': return {0x1E, 0x11, 0x11, 0x1E, 0x14, 0x12, 0x11};
    case 'S': return {0x0F, 0x10, 0x10, 0x0E, 0x01, 0x01, 0x1E};
    case 'T': return {0x1F, 0x04, 0x04, 0x04, 0x04, 0x04, 0x04};
    case 'U': return {0x11, 0x11, 0x11, 0x11, 0x11, 0x11, 0x0E};
    case 'V': return {0x11, 0x11, 0x11, 0x11, 0x0A, 0x0A, 0x04};
    case 'W': return {0x11, 0x11, 0x11, 0x15, 0x15, 0x1B, 0x11};
    case 'X': return {0x11, 0x0A, 0x04, 0x04, 0x0A, 0x11, 0x11};
    case 'Y': return {0x11, 0x0A, 0x04, 0x04, 0x04, 0x04, 0x04};
    case 'Z': return {0x1F, 0x01, 0x02, 0x04, 0x08, 0x10, 0x1F};
    case '0': return {0x0E, 0x11, 0x13, 0x15, 0x19, 0x11, 0x0E};
    case '1': return {0x04, 0x0C, 0x04, 0x04, 0x04, 0x04, 0x0E};
    case '2': return {0x0E, 0x11, 0x01, 0x02, 0x04, 0x08, 0x1F};
    case '3': return {0x1E, 0x01, 0x01, 0x0E, 0x01, 0x01, 0x1E};
    case '4': return {0x02, 0x06, 0x0A, 0x12, 0x1F, 0x02, 0x02};
    case '5': return {0x1F, 0x10, 0x10, 0x1E, 0x01, 0x01, 0x1E};
    case '6': return {0x0E, 0x10, 0x10, 0x1E, 0x11, 0x11, 0x0E};
    case '7': return {0x1F, 0x01, 0x02, 0x04, 0x08, 0x08, 0x08};
    case '8': return {0x0E, 0x11, 0x11, 0x0E, 0x11, 0x11, 0x0E};
    case '9': return {0x0E, 0x11, 0x11, 0x0F, 0x01, 0x01, 0x0E};
    case '.': return {0x00, 0x00, 0x00, 0x00, 0x00, 0x0C, 0x0C};
    case ':': return {0x00, 0x0C, 0x0C, 0x00, 0x0C, 0x0C, 0x00};
    case '-': return {0x00, 0x00, 0x00, 0x1F, 0x00, 0x00, 0x00};
    case '/': return {0x01, 0x02, 0x04, 0x08, 0x10, 0x00, 0x00};
    case '(': return {0x02, 0x04, 0x08, 0x08, 0x08, 0x04, 0x02};
    case ')': return {0x08, 0x04, 0x02, 0x02, 0x02, 0x04, 0x08};
    case ',': return {0x00, 0x00, 0x00, 0x00, 0x0C, 0x0C, 0x08};
    case ' ': return {0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00};
    default: return {0x1F, 0x11, 0x02, 0x04, 0x08, 0x00, 0x08};
    }
}

std::array<uint8_t, 7> glyphFor(wchar_t input)
{
    switch (input)
    {
    case L'\u0410': return glyphFor('A');
    case L'\u0411': return {0x1F, 0x10, 0x10, 0x1E, 0x11, 0x11, 0x1E};
    case L'\u0412': return glyphFor('B');
    case L'\u0413': return {0x1F, 0x10, 0x10, 0x10, 0x10, 0x10, 0x10};
    case L'\u0414': return {0x0E, 0x11, 0x11, 0x11, 0x11, 0x11, 0x1F};
    case L'\u0415': return glyphFor('E');
    case L'\u0401': return glyphFor('E');
    case L'\u0416': return {0x11, 0x0A, 0x04, 0x1F, 0x04, 0x0A, 0x11};
    case L'\u0417': return {0x1E, 0x01, 0x01, 0x0E, 0x01, 0x01, 0x1E};
    case L'\u0418': return {0x11, 0x19, 0x15, 0x13, 0x11, 0x11, 0x11};
    case L'\u0419': return {0x11, 0x19, 0x15, 0x13, 0x11, 0x11, 0x11};
    case L'\u041A': return glyphFor('K');
    case L'\u041B': return {0x04, 0x0A, 0x11, 0x11, 0x11, 0x11, 0x11};
    case L'\u041C': return glyphFor('M');
    case L'\u041D': return glyphFor('H');
    case L'\u041E': return glyphFor('O');
    case L'\u041F': return {0x1F, 0x11, 0x11, 0x11, 0x11, 0x11, 0x11};
    case L'\u0420': return glyphFor('P');
    case L'\u0421': return glyphFor('C');
    case L'\u0422': return glyphFor('T');
    case L'\u0423': return glyphFor('Y');
    case L'\u0424': return {0x0E, 0x15, 0x15, 0x1F, 0x05, 0x05, 0x0E};
    case L'\u0425': return glyphFor('X');
    case L'\u0426': return {0x11, 0x11, 0x11, 0x11, 0x11, 0x1F, 0x01};
    case L'\u0427': return {0x11, 0x11, 0x11, 0x0F, 0x01, 0x01, 0x01};
    case L'\u0428': return {0x11, 0x11, 0x11, 0x15, 0x15, 0x15, 0x1F};
    case L'\u0429': return {0x11, 0x11, 0x11, 0x15, 0x15, 0x1F, 0x01};
    case L'\u042A': return {0x18, 0x10, 0x10, 0x1E, 0x11, 0x11, 0x1E};
    case L'\u042B': return {0x11, 0x11, 0x11, 0x1E, 0x11, 0x11, 0x1E};
    case L'\u042C': return {0x10, 0x10, 0x10, 0x1E, 0x11, 0x11, 0x1E};
    case L'\u042D': return {0x1E, 0x01, 0x01, 0x0F, 0x01, 0x01, 0x1E};
    case L'\u042E': return {0x15, 0x15, 0x15, 0x1F, 0x15, 0x15, 0x15};
    case L'\u042F': return {0x0F, 0x11, 0x11, 0x0F, 0x05, 0x09, 0x11};
    case L'0': return glyphFor('0');
    case L'1': return glyphFor('1');
    case L'2': return glyphFor('2');
    case L'3': return glyphFor('3');
    case L'4': return glyphFor('4');
    case L'5': return glyphFor('5');
    case L'6': return glyphFor('6');
    case L'7': return glyphFor('7');
    case L'8': return glyphFor('8');
    case L'9': return glyphFor('9');
    case L'.': return glyphFor('.');
    case L':': return glyphFor(':');
    case L'-': return glyphFor('-');
    case L'/': return glyphFor('/');
    case L'(': return glyphFor('(');
    case L')': return glyphFor(')');
    case L',': return glyphFor(',');
    case L' ': return glyphFor(' ');
    default: return glyphFor('?');
    }
}

void fillRect(std::vector<uchar4>& pixels, int x, int y, int w, int h, uchar4 color)
{
    const int x2 = std::min(x + w, gWidth);
    const int y2 = std::min(y + h, gHeight);
    for (int py = std::max(0, y); py < y2; ++py)
    {
        for (int px = std::max(0, x); px < x2; ++px)
        {
            pixels[py * gWidth + px] = color;
        }
    }
}

struct HudTextCache
{
    HDC dc = nullptr;
    HBITMAP bitmap = nullptr;
    HGDIOBJ oldBitmap = nullptr;
    HFONT font = nullptr;
    void* bits = nullptr;
    int width = 0;
    int height = 0;
};

HudTextCache& hudTextCache()
{
    static HudTextCache cache;
    return cache;
}

void releaseHudTextCache(HudTextCache& cache)
{
    if (cache.dc != nullptr)
    {
        if (cache.oldBitmap != nullptr)
        {
            SelectObject(cache.dc, cache.oldBitmap);
            cache.oldBitmap = nullptr;
        }
        if (cache.bitmap != nullptr)
        {
            DeleteObject(cache.bitmap);
            cache.bitmap = nullptr;
        }
        if (cache.font != nullptr)
        {
            DeleteObject(cache.font);
            cache.font = nullptr;
        }
        DeleteDC(cache.dc);
        cache.dc = nullptr;
    }
    cache.bits = nullptr;
    cache.width = 0;
    cache.height = 0;
}

void ensureHudTextCache(int width, int height)
{
    HudTextCache& cache = hudTextCache();
    if (cache.dc != nullptr && cache.width == width && cache.height == height)
    {
        return;
    }

    releaseHudTextCache(cache);

    HDC screenDC = GetDC(nullptr);
    cache.dc = CreateCompatibleDC(screenDC);

    BITMAPINFO bmi{};
    bmi.bmiHeader.biSize = sizeof(BITMAPINFOHEADER);
    bmi.bmiHeader.biWidth = width;
    bmi.bmiHeader.biHeight = -height;
    bmi.bmiHeader.biPlanes = 1;
    bmi.bmiHeader.biBitCount = 32;
    bmi.bmiHeader.biCompression = BI_RGB;

    cache.bitmap = CreateDIBSection(screenDC, &bmi, DIB_RGB_COLORS, &cache.bits, nullptr, 0);
    cache.oldBitmap = SelectObject(cache.dc, cache.bitmap);
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
    cache.width = width;
    cache.height = height;

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
        HWND hwnd;
        HDC dc;
        ~DcGuard()
        {
            if (dc != nullptr)
            {
                ReleaseDC(hwnd, dc);
            }
        }
    } guard{hwnd, dc};

    static HFONT font = CreateFontW(
        -18,
        0,
        0,
        0,
        FW_NORMAL,
        FALSE,
        FALSE,
        FALSE,
        DEFAULT_CHARSET,
        OUT_DEFAULT_PRECIS,
        CLIP_DEFAULT_PRECIS,
        CLEARTYPE_QUALITY,
        DEFAULT_PITCH | FF_SWISS,
        L"Arial");

    const HGDIOBJ oldFont = SelectObject(dc, font);
    SetBkMode(dc, TRANSPARENT);
    SetTextAlign(dc, TA_LEFT | TA_TOP);
    SetTextColor(dc, RGB(0, 0, 0));

    const int panelX = 14;
    const int panelY = 14;
    std::wostringstream line1;
    line1 << std::fixed << std::setprecision(1)
          << L"\u041A\u0410\u0414\u0420 " << stats.fps
          << L" FPS   \u0425\u041E\u0421\u0422 " << stats.avgHostMs << L" \u041C\u0421   \u0413\u041F\u0423 " << stats.avgGpuMs << L" \u041C\u0421";

    std::wostringstream line2;
    line2 << L"\u0421\u0424\u0415\u0420\u0410 " << (scene.selectedSphere + 1)
          << L"   \u041C\u0410\u0422\u0415\u0420\u0418\u0410\u041B: " << materialNameW(scene.materials[scene.selectedSphere].materialType)
          << L"   \u0421\u0412\u0415\u0422: " << std::fixed << std::setprecision(1)
          << scene.lightPosition.x << L" " << scene.lightPosition.y << L" " << scene.lightPosition.z;

    const std::wstring line3 = L"\u0421\u0424\u0415\u0420\u042B: 1-3 \u0412\u042B\u0411\u041E\u0420";
    const std::wstring line4 = L"\u0414\u0412\u0418\u0416\u0415\u041D\u0418\u0415 \u0421\u0424\u0415\u0420\u042B: \u0421\u0422\u0420\u0415\u041B\u041A\u0418 - X/Z";
    const std::wstring line5 = L"PGUP/PGDN \u0438\u043B\u0438 R/F - Y";
    const std::wstring line6 = L"\u0421\u0412\u0415\u0422: J/L - X, I/K - Z, U/O - Y";
    const std::wstring line7 = L"\u041C\u0410\u0422\u0415\u0420\u0418\u0410\u041B: M   \u041A\u0410\u041C\u0415\u0420\u0410: WASD/QE + \u041C\u042B\u0428\u042C";

    TextOutW(dc, panelX, panelY, line1.str().c_str(), static_cast<int>(line1.str().size()));
    TextOutW(dc, panelX, panelY + 22, line2.str().c_str(), static_cast<int>(line2.str().size()));
    TextOutW(dc, panelX, panelY + 44, line3.c_str(), static_cast<int>(line3.size()));
    TextOutW(dc, panelX, panelY + 66, line4.c_str(), static_cast<int>(line4.size()));
    TextOutW(dc, panelX, panelY + 88, line5.c_str(), static_cast<int>(line5.size()));
    TextOutW(dc, panelX, panelY + 110, line6.c_str(), static_cast<int>(line6.size()));
    TextOutW(dc, panelX, panelY + 132, line7.c_str(), static_cast<int>(line7.size()));

    SelectObject(dc, oldFont);
}

void updateCameraBasis(
    const CameraState& camera,
    float3& forward,
    float3& right,
    float3& up,
    float& scale,
    float& aspect)
{
    const float yaw = camera.yaw * kPi / 180.0f;
    const float pitch = camera.pitch * kPi / 180.0f;

    forward = normalize3(make_float3(
        std::cos(yaw) * std::cos(pitch),
        std::sin(pitch),
        std::sin(yaw) * std::cos(pitch)));

    const float3 worldUp = make_float3(0.0f, 1.0f, 0.0f);
    right = normalize3(cross3(forward, worldUp));
    up = normalize3(cross3(right, forward));
    aspect = static_cast<float>(gWidth) / static_cast<float>(gHeight);
    scale = std::tan((camera.fov * 0.5f) * kPi / 180.0f);
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
    updateCameraBasis(camera, forward, right, up, scale, aspect);

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
        moveSelectedSphere(scene, make_float3(-sphereStep, 0.0f, 0.0f));
    }
    if (glfwGetKey(window, GLFW_KEY_RIGHT) == GLFW_PRESS)
    {
        moveSelectedSphere(scene, make_float3(sphereStep, 0.0f, 0.0f));
    }
    if (glfwGetKey(window, GLFW_KEY_UP) == GLFW_PRESS)
    {
        moveSelectedSphere(scene, make_float3(0.0f, 0.0f, -sphereStep));
    }
    if (glfwGetKey(window, GLFW_KEY_DOWN) == GLFW_PRESS)
    {
        moveSelectedSphere(scene, make_float3(0.0f, 0.0f, sphereStep));
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

void OptixRenderer::initialize()
{
    createContext();
    CUDA_CHECK(cudaStreamCreate(&stream));
    CUDA_CHECK(cudaEventCreate(&frameStart));
    CUDA_CHECK(cudaEventCreate(&frameStop));
    createScene();
    createModule();
    createProgramGroups();
    createPipeline();
    createSbt();
    CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&dFrameBuffer), gWidth * gHeight * sizeof(uchar4)));
    CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&dLaunchParams), sizeof(LaunchParams)));
}

void OptixRenderer::createContext()
{
    CUDA_CHECK(cudaFree(nullptr));
    OPTIX_CHECK(optixInit());

    CUcontext cuContext = nullptr;
    OptixDeviceContextOptions options{};
    options.logCallbackFunction = contextLogCallback;
    options.logCallbackLevel = 4;
    OPTIX_CHECK(optixDeviceContextCreate(cuContext, &options, &context));
}

void OptixRenderer::createScene()
{
    const SceneState scene = makeDefaultScene();

    sphereFlags.assign(scene.spheres.size(), OPTIX_GEOMETRY_FLAG_NONE);
    std::vector<float3> centers;
    std::vector<float> radii;
    centers.reserve(scene.spheres.size());
    radii.reserve(scene.spheres.size());

    for (const SphereGeometry& sphere : scene.spheres)
    {
        centers.push_back(sphere.center);
        radii.push_back(sphere.radius);
    }

    CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&dSphereCenters), centers.size() * sizeof(float3)));
    CUDA_CHECK(cudaMemcpy(reinterpret_cast<void*>(dSphereCenters), centers.data(), centers.size() * sizeof(float3), cudaMemcpyHostToDevice));

    CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&dSphereRadii), radii.size() * sizeof(float)));
    CUDA_CHECK(cudaMemcpy(reinterpret_cast<void*>(dSphereRadii), radii.data(), radii.size() * sizeof(float), cudaMemcpyHostToDevice));

    CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&dMaterials), scene.materials.size() * sizeof(SphereMaterial)));
    CUDA_CHECK(cudaMemcpy(reinterpret_cast<void*>(dMaterials), scene.materials.data(), scene.materials.size() * sizeof(SphereMaterial), cudaMemcpyHostToDevice));

    sphereAccelOptions.buildFlags = OPTIX_BUILD_FLAG_ALLOW_COMPACTION | OPTIX_BUILD_FLAG_ALLOW_RANDOM_VERTEX_ACCESS;
    sphereAccelOptions.operation = OPTIX_BUILD_OPERATION_BUILD;
    sphereBuildInput = {};
    sphereBuildInput.type = OPTIX_BUILD_INPUT_TYPE_SPHERES;
    sphereBuildInput.sphereArray.vertexBuffers = &dSphereCenters;
    sphereBuildInput.sphereArray.radiusBuffers = &dSphereRadii;
    sphereBuildInput.sphereArray.numVertices = static_cast<unsigned int>(scene.spheres.size());
    sphereBuildInput.sphereArray.singleRadius = 0;
    sphereBuildInput.sphereArray.radiusStrideInBytes = sizeof(float);
    sphereBuildInput.sphereArray.flags = sphereFlags.data();
    sphereBuildInput.sphereArray.numSbtRecords = 1;

    OPTIX_CHECK(optixAccelComputeMemoryUsage(context, &sphereAccelOptions, &sphereBuildInput, 1, &sphereGasSizes));

    CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&dSphereGasBuffer), sphereGasSizes.outputSizeInBytes));

    rebuildAccelerationStructure();
}

void OptixRenderer::rebuildAccelerationStructure()
{
    CUdeviceptr dTempBuffer = 0;
    const size_t tempSize = sphereGasSizes.tempSizeInBytes;
    CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&dTempBuffer), tempSize));

    sphereAccelOptions.operation = OPTIX_BUILD_OPERATION_BUILD;
    OPTIX_CHECK(optixAccelBuild(
        context,
        stream,
        &sphereAccelOptions,
        &sphereBuildInput,
        1,
        dTempBuffer,
        sphereGasSizes.tempSizeInBytes,
        dSphereGasBuffer,
        sphereGasSizes.outputSizeInBytes,
        &sphereGasHandle,
        nullptr,
        0));

    CUDA_CHECK(cudaStreamSynchronize(stream));
    CUDA_CHECK(cudaFree(reinterpret_cast<void*>(dTempBuffer)));
}

void OptixRenderer::createModule()
{
    const std::string optixIr = compileDeviceProgram();

    OptixModuleCompileOptions moduleOptions{};
    moduleOptions.maxRegisterCount = OPTIX_COMPILE_DEFAULT_MAX_REGISTER_COUNT;
    moduleOptions.optLevel = OPTIX_COMPILE_OPTIMIZATION_DEFAULT;
    moduleOptions.debugLevel = OPTIX_COMPILE_DEBUG_LEVEL_MINIMAL;

    pipelineCompileOptions.usesMotionBlur = false;
    pipelineCompileOptions.traversableGraphFlags = OPTIX_TRAVERSABLE_GRAPH_FLAG_ALLOW_SINGLE_GAS;
    pipelineCompileOptions.numPayloadValues = 4;
    pipelineCompileOptions.numAttributeValues = 2;
    pipelineCompileOptions.exceptionFlags = OPTIX_EXCEPTION_FLAG_NONE;
    pipelineCompileOptions.pipelineLaunchParamsVariableName = "params";
    pipelineCompileOptions.usesPrimitiveTypeFlags = OPTIX_PRIMITIVE_TYPE_FLAGS_SPHERE;

    char log[4096]{};
    size_t logSize = sizeof(log);
    OPTIX_CHECK(optixModuleCreate(context, &moduleOptions, &pipelineCompileOptions, optixIr.data(), optixIr.size(), log, &logSize, &module));

    OptixBuiltinISOptions builtinOptions{};
    builtinOptions.builtinISModuleType = OPTIX_PRIMITIVE_TYPE_SPHERE;
    builtinOptions.usesMotionBlur = false;
    OPTIX_CHECK(optixBuiltinISModuleGet(context, &moduleOptions, &pipelineCompileOptions, &builtinOptions, &sphereModule));
}

void OptixRenderer::createProgramGroups()
{
    OptixProgramGroupOptions options{};
    char log[4096]{};
    size_t logSize = sizeof(log);

    OptixProgramGroupDesc raygenDesc{};
    raygenDesc.kind = OPTIX_PROGRAM_GROUP_KIND_RAYGEN;
    raygenDesc.raygen.module = module;
    raygenDesc.raygen.entryFunctionName = "__raygen__rg";
    OPTIX_CHECK(optixProgramGroupCreate(context, &raygenDesc, 1, &options, log, &logSize, &programGroups.raygen));

    OptixProgramGroupDesc missRadianceDesc{};
    missRadianceDesc.kind = OPTIX_PROGRAM_GROUP_KIND_MISS;
    missRadianceDesc.miss.module = module;
    missRadianceDesc.miss.entryFunctionName = "__miss__radiance";
    logSize = sizeof(log);
    OPTIX_CHECK(optixProgramGroupCreate(context, &missRadianceDesc, 1, &options, log, &logSize, &programGroups.missRadiance));

    OptixProgramGroupDesc missShadowDesc{};
    missShadowDesc.kind = OPTIX_PROGRAM_GROUP_KIND_MISS;
    missShadowDesc.miss.module = module;
    missShadowDesc.miss.entryFunctionName = "__miss__shadow";
    logSize = sizeof(log);
    OPTIX_CHECK(optixProgramGroupCreate(context, &missShadowDesc, 1, &options, log, &logSize, &programGroups.missShadow));

    OptixProgramGroupDesc hitRadianceDesc{};
    hitRadianceDesc.kind = OPTIX_PROGRAM_GROUP_KIND_HITGROUP;
    hitRadianceDesc.hitgroup.moduleCH = module;
    hitRadianceDesc.hitgroup.entryFunctionNameCH = "__closesthit__radiance";
    hitRadianceDesc.hitgroup.moduleIS = sphereModule;
    hitRadianceDesc.hitgroup.entryFunctionNameIS = nullptr;
    logSize = sizeof(log);
    OPTIX_CHECK(optixProgramGroupCreate(context, &hitRadianceDesc, 1, &options, log, &logSize, &programGroups.hitRadiance));

    OptixProgramGroupDesc hitShadowDesc{};
    hitShadowDesc.kind = OPTIX_PROGRAM_GROUP_KIND_HITGROUP;
    hitShadowDesc.hitgroup.moduleCH = module;
    hitShadowDesc.hitgroup.entryFunctionNameCH = "__closesthit__shadow";
    hitShadowDesc.hitgroup.moduleIS = sphereModule;
    hitShadowDesc.hitgroup.entryFunctionNameIS = nullptr;
    logSize = sizeof(log);
    OPTIX_CHECK(optixProgramGroupCreate(context, &hitShadowDesc, 1, &options, log, &logSize, &programGroups.hitShadow));
}

void OptixRenderer::createPipeline()
{
    std::vector<OptixProgramGroup> groups = {
        programGroups.raygen,
        programGroups.missRadiance,
        programGroups.missShadow,
        programGroups.hitRadiance,
        programGroups.hitShadow
    };

    OptixPipelineLinkOptions linkOptions{};
    linkOptions.maxTraceDepth = 2;
    linkOptions.maxTraversableGraphDepth = 1;

    char log[4096]{};
    size_t logSize = sizeof(log);
    OPTIX_CHECK(optixPipelineCreate(context, &pipelineCompileOptions, &linkOptions, groups.data(), static_cast<unsigned int>(groups.size()), log, &logSize, &pipeline));

    OptixStackSizes stackSizes{};
    for (OptixProgramGroup group : groups)
    {
        OPTIX_CHECK(optixUtilAccumulateStackSizes(group, &stackSizes, pipeline));
    }

    uint32_t directCallableStackSizeFromTraversal = 0;
    uint32_t directCallableStackSizeFromState = 0;
    uint32_t continuationStackSize = 0;
    OPTIX_CHECK(optixUtilComputeStackSizes(&stackSizes, 2, 0, 0, &directCallableStackSizeFromTraversal, &directCallableStackSizeFromState, &continuationStackSize));
    OPTIX_CHECK(optixPipelineSetStackSize(pipeline, directCallableStackSizeFromTraversal, directCallableStackSizeFromState, continuationStackSize, 1));
}

void OptixRenderer::createSbt()
{
    RaygenRecord raygenRecord{};
    OPTIX_CHECK(optixSbtRecordPackHeader(programGroups.raygen, &raygenRecord));
    CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&sbt.raygenRecord), sizeof(RaygenRecord)));
    CUDA_CHECK(cudaMemcpy(reinterpret_cast<void*>(sbt.raygenRecord), &raygenRecord, sizeof(RaygenRecord), cudaMemcpyHostToDevice));

    std::vector<MissRecord> missRecords(2);
    OPTIX_CHECK(optixSbtRecordPackHeader(programGroups.missRadiance, &missRecords[0]));
    OPTIX_CHECK(optixSbtRecordPackHeader(programGroups.missShadow, &missRecords[1]));
    CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&sbt.missRecordBase), missRecords.size() * sizeof(MissRecord)));
    CUDA_CHECK(cudaMemcpy(reinterpret_cast<void*>(sbt.missRecordBase), missRecords.data(), missRecords.size() * sizeof(MissRecord), cudaMemcpyHostToDevice));
    sbt.missRecordStrideInBytes = sizeof(MissRecord);
    sbt.missRecordCount = static_cast<unsigned int>(missRecords.size());

    std::vector<HitgroupRecord> hitRecords(2);
    OPTIX_CHECK(optixSbtRecordPackHeader(programGroups.hitRadiance, &hitRecords[0]));
    OPTIX_CHECK(optixSbtRecordPackHeader(programGroups.hitShadow, &hitRecords[1]));
    CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&sbt.hitgroupRecordBase), hitRecords.size() * sizeof(HitgroupRecord)));
    CUDA_CHECK(cudaMemcpy(reinterpret_cast<void*>(sbt.hitgroupRecordBase), hitRecords.data(), hitRecords.size() * sizeof(HitgroupRecord), cudaMemcpyHostToDevice));
    sbt.hitgroupRecordStrideInBytes = sizeof(HitgroupRecord);
    sbt.hitgroupRecordCount = static_cast<unsigned int>(hitRecords.size());
}

void OptixRenderer::renderFrame(const SceneState& scene, const CameraState& camera, std::vector<uchar4>& hostPixels, float* gpuTimeMs)
{
    float3 forward{};
    float3 right{};
    float3 up{};
    float scale = 0.0f;
    float aspect = 0.0f;
    updateCameraBasis(camera, forward, right, up, scale, aspect);

    std::vector<float3> centers;
    std::vector<float> radii;
    centers.reserve(scene.spheres.size());
    radii.reserve(scene.spheres.size());

    for (const SphereGeometry& sphere : scene.spheres)
    {
        centers.push_back(sphere.center);
        radii.push_back(sphere.radius);
    }

    CUDA_CHECK(cudaEventRecord(frameStart, stream));

    CUDA_CHECK(cudaMemcpyAsync(
        reinterpret_cast<void*>(dSphereCenters),
        centers.data(),
        centers.size() * sizeof(float3),
        cudaMemcpyHostToDevice,
        stream));
    CUDA_CHECK(cudaMemcpyAsync(
        reinterpret_cast<void*>(dSphereRadii),
        radii.data(),
        radii.size() * sizeof(float),
        cudaMemcpyHostToDevice,
        stream));
    CUDA_CHECK(cudaMemcpyAsync(
        reinterpret_cast<void*>(dMaterials),
        scene.materials.data(),
        scene.materials.size() * sizeof(SphereMaterial),
        cudaMemcpyHostToDevice,
        stream));

    rebuildAccelerationStructure();

    LaunchParams params{};
    params.image = dFrameBuffer;
    params.imageWidth = gWidth;
    params.imageHeight = gHeight;
    params.handle = sphereGasHandle;
    params.cameraPosition = camera.position;
    params.cameraForward = forward;
    params.cameraRight = right;
    params.cameraUp = up;
    params.cameraScale = scale;
    params.cameraAspect = aspect;
    params.lightPosition = scene.lightPosition;
    params.materials = reinterpret_cast<SphereMaterial*>(dMaterials);
    params.sphereCount = static_cast<int>(scene.spheres.size());
    params.maxDepth = 1;

    CUDA_CHECK(cudaMemcpyAsync(reinterpret_cast<void*>(dLaunchParams), &params, sizeof(LaunchParams), cudaMemcpyHostToDevice, stream));
    OPTIX_CHECK(optixLaunch(pipeline, stream, dLaunchParams, sizeof(LaunchParams), &sbt, gWidth, gHeight, 1));
    CUDA_CHECK(cudaMemcpyAsync(hostPixels.data(), dFrameBuffer, hostPixels.size() * sizeof(uchar4), cudaMemcpyDeviceToHost, stream));
    CUDA_CHECK(cudaEventRecord(frameStop, stream));
    CUDA_CHECK(cudaStreamSynchronize(stream));

    if (gpuTimeMs != nullptr)
    {
        CUDA_CHECK(cudaEventElapsedTime(gpuTimeMs, frameStart, frameStop));
    }
}

void OptixRenderer::destroy()
{
    if (dFrameBuffer != nullptr)
    {
        cudaFree(dFrameBuffer);
        dFrameBuffer = nullptr;
    }
    if (dLaunchParams != 0)
    {
        cudaFree(reinterpret_cast<void*>(dLaunchParams));
        dLaunchParams = 0;
    }
    if (dSphereGasBuffer != 0)
    {
        cudaFree(reinterpret_cast<void*>(dSphereGasBuffer));
        dSphereGasBuffer = 0;
    }
    if (sbt.hitgroupRecordBase != 0)
    {
        cudaFree(reinterpret_cast<void*>(sbt.hitgroupRecordBase));
        sbt.hitgroupRecordBase = 0;
    }
    if (sbt.missRecordBase != 0)
    {
        cudaFree(reinterpret_cast<void*>(sbt.missRecordBase));
        sbt.missRecordBase = 0;
    }
    if (sbt.raygenRecord != 0)
    {
        cudaFree(reinterpret_cast<void*>(sbt.raygenRecord));
        sbt.raygenRecord = 0;
    }
    if (pipeline != nullptr)
    {
        optixPipelineDestroy(pipeline);
        pipeline = nullptr;
    }
    if (programGroups.hitShadow != nullptr)
    {
        optixProgramGroupDestroy(programGroups.hitShadow);
        programGroups.hitShadow = nullptr;
    }
    if (programGroups.hitRadiance != nullptr)
    {
        optixProgramGroupDestroy(programGroups.hitRadiance);
        programGroups.hitRadiance = nullptr;
    }
    if (programGroups.missShadow != nullptr)
    {
        optixProgramGroupDestroy(programGroups.missShadow);
        programGroups.missShadow = nullptr;
    }
    if (programGroups.missRadiance != nullptr)
    {
        optixProgramGroupDestroy(programGroups.missRadiance);
        programGroups.missRadiance = nullptr;
    }
    if (programGroups.raygen != nullptr)
    {
        optixProgramGroupDestroy(programGroups.raygen);
        programGroups.raygen = nullptr;
    }
    if (module != nullptr)
    {
        optixModuleDestroy(module);
        module = nullptr;
    }
    if (sphereModule != nullptr)
    {
        optixModuleDestroy(sphereModule);
        sphereModule = nullptr;
    }
    if (dMaterials != 0)
    {
        cudaFree(reinterpret_cast<void*>(dMaterials));
        dMaterials = 0;
    }
    if (dSphereRadii != 0)
    {
        cudaFree(reinterpret_cast<void*>(dSphereRadii));
        dSphereRadii = 0;
    }
    if (dSphereCenters != 0)
    {
        cudaFree(reinterpret_cast<void*>(dSphereCenters));
        dSphereCenters = 0;
    }
    if (stream != nullptr)
    {
        cudaStreamDestroy(stream);
        stream = nullptr;
    }
    if (frameStop != nullptr)
    {
        cudaEventDestroy(frameStop);
        frameStop = nullptr;
    }
    if (frameStart != nullptr)
    {
        cudaEventDestroy(frameStart);
        frameStart = nullptr;
    }
    if (context != nullptr)
    {
        optixDeviceContextDestroy(context);
        context = nullptr;
    }
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

        drawHud(window, appState.scene, stats);

        glClear(GL_COLOR_BUFFER_BIT);
        glDrawPixels(gWidth, gHeight, GL_RGBA, GL_UNSIGNED_BYTE, pixels.data());
        glfwSwapBuffers(window);
        glfwPollEvents();
    }

    renderer.destroy();
    glfwDestroyWindow(window);
    glfwTerminate();
}
