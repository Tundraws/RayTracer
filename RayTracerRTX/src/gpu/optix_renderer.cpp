#include "optix_renderer.h"

#include "optix_device_programs.h"
#include "../app/camera.h"
#include "../app/scene.h"

#ifndef NOMINMAX
#define NOMINMAX
#endif
#include <windows.h>

#include <cuda.h>
#include <cuda_runtime.h>
#include <nvrtc.h>

#include <optix.h>
#include <optix_function_table_definition.h>
#include <optix_stack_size.h>
#include <optix_stubs.h>

#include <array>
#include <cstdlib>
#include <iomanip>
#include <iostream>
#include <sstream>
#include <stdexcept>
#include <cstring>
#include <string>
#include <utility>
#include <vector>

namespace
{
int gWidth = 800;
int gHeight = 600;
constexpr unsigned int kMaxReflectionDepth = 6;
constexpr unsigned int kMaxTraceDepth = kMaxReflectionDepth + 2;

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

#ifndef RAYTRACERRTX_CUDA_INCLUDE_DIR
#define RAYTRACERRTX_CUDA_INCLUDE_DIR ""
#endif

#ifndef RAYTRACERRTX_OPTIX_INCLUDE_DIR
#define RAYTRACERRTX_OPTIX_INCLUDE_DIR ""
#endif

#ifndef RAYTRACERRTX_SOURCE_DIR
#define RAYTRACERRTX_SOURCE_DIR ""
#endif

void contextLogCallback(unsigned int level, const char* tag, const char* message, void*)
{
    std::cerr << "[" << std::setw(2) << level << "][" << tag << "] " << message << '\n';
}

std::string getEnvString(const char* name)
{
    char* value = nullptr;
    size_t length = 0;
    if (_dupenv_s(&value, &length, name) != 0 || value == nullptr)
    {
        return {};
    }

    std::string result(value);
    std::free(value);
    return result;
}

std::string getConfiguredPath(const char* macroValue, const char* envName, const char* suffix)
{
    if (macroValue != nullptr && macroValue[0] != '\0')
    {
        return macroValue;
    }

    const std::string envValue = getEnvString(envName);
    if (envValue.empty())
    {
        return {};
    }

    return suffix != nullptr && suffix[0] != '\0'
        ? envValue + suffix
        : envValue;
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

    const std::string cudaIncludePath = getConfiguredPath(RAYTRACERRTX_CUDA_INCLUDE_DIR, "CUDA_PATH", "\\include");
    const std::string optixIncludePath = getConfiguredPath(RAYTRACERRTX_OPTIX_INCLUDE_DIR, "OPTIX_SDK_DIR", "\\include");
    const std::string projectIncludePath = getConfiguredPath(RAYTRACERRTX_SOURCE_DIR, "RAYTRACERRTX_SOURCE_DIR", "");
    if (cudaIncludePath.empty() || optixIncludePath.empty() || projectIncludePath.empty())
    {
        throw std::runtime_error("CUDA, OptiX and project include paths must be configured in the Visual Studio project or environment.");
    }

    const std::string cudaInclude = getShortPath(cudaIncludePath);
    const std::string optixInclude = getShortPath(optixIncludePath);
    const std::string projectInclude = getShortPath(projectIncludePath);
    const std::string architecture = "--gpu-architecture=compute_" + std::to_string(props.major) + std::to_string(props.minor);
    const std::string includeCuda = "-I" + cudaInclude;
    const std::string includeOptix = "-I" + optixInclude;
    const std::string includeProject = "-I" + projectInclude;

    const std::vector<const char*> options = {
        "--std=c++14",
        architecture.c_str(),
        includeCuda.c_str(),
        includeOptix.c_str(),
        includeProject.c_str(),
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

} // namespace

void OptixRenderer::setRenderSize(int width, int height)
{
    gWidth = width;
    gHeight = height;
}

void OptixRenderer::initialize()
{
    initialize(makeDefaultScene());
}

void OptixRenderer::initialize(const SceneState& initialScene)
{
    createContext();
    CUDA_CHECK(cudaStreamCreate(&stream));
    CUDA_CHECK(cudaEventCreate(&frameStart));
    CUDA_CHECK(cudaEventCreate(&frameStop));
    createScene(initialScene);
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
    options.logCallbackLevel = 2;
    OPTIX_CHECK(optixDeviceContextCreate(cuContext, &options, &context));
}

void OptixRenderer::createScene(const SceneState& scene)
{
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

    std::vector<MeshObject> renderMeshObjects = scene.meshObjects;
    if (renderMeshObjects.empty() && !isEmptyMesh(scene.mesh))
    {
        MeshObject legacyMeshObject;
        legacyMeshObject.assetReference = "legacy SceneState::mesh";
        legacyMeshObject.mesh = scene.mesh;
        renderMeshObjects.push_back(std::move(legacyMeshObject));
    }

    std::vector<MeshVertexGpu> meshVertices;
    std::vector<MeshTriangleGpu> meshTriangles;
    std::vector<MeshMaterialGpu> meshMaterials;
    std::vector<MeshObjectGpu> meshObjectsGpu;
    std::vector<std::vector<uint3>> meshIndexBuffers;
    std::vector<uchar4> meshTexturePixels;
    meshInstanceTransforms.clear();

    for (const MeshObject& object : renderMeshObjects)
    {
        if (isEmptyMesh(object.mesh) || !hasValidMeshMaterialIndices(object.mesh))
        {
            continue;
        }

        const unsigned int vertexOffset = static_cast<unsigned int>(meshVertices.size());
        const unsigned int triangleOffset = static_cast<unsigned int>(meshTriangles.size());
        const unsigned int materialOffset = static_cast<unsigned int>(meshMaterials.size());
        for (const MeshVertex& vertex : object.mesh.vertices)
        {
            meshVertices.push_back(MeshVertexGpu{vertex.position, vertex.normal, vertex.texcoord, vertex.tangent, vertex.hasTexcoord});
        }

        std::vector<uint3> objectIndices;
        objectIndices.reserve(object.mesh.triangles.size());
        for (const MeshTriangle& triangle : object.mesh.triangles)
        {
            objectIndices.push_back(make_uint3(
                vertexOffset + triangle.i0,
                vertexOffset + triangle.i1,
                vertexOffset + triangle.i2));
            meshTriangles.push_back(MeshTriangleGpu{
                vertexOffset + triangle.i0,
                vertexOffset + triangle.i1,
                vertexOffset + triangle.i2,
                materialOffset + triangle.materialIndex});
        }

        for (const MeshMaterial& material : object.mesh.materials)
        {
            MeshMaterialGpu materialGpu{
                material.color,
                material.materialType,
                material.specularColor,
                material.roughness,
                material.ior,
                material.alpha};
            if (material.textureIndex >= 0 && static_cast<size_t>(material.textureIndex) < object.mesh.textures.size())
            {
                const MeshTexture& texture = object.mesh.textures[static_cast<size_t>(material.textureIndex)];
                if (!texture.pixels.empty() && texture.width > 0 && texture.height > 0)
                {
                    materialGpu.hasTexture = 1;
                    materialGpu.textureOffset = static_cast<unsigned int>(meshTexturePixels.size());
                    materialGpu.textureWidth = texture.width;
                    materialGpu.textureHeight = texture.height;
                    meshTexturePixels.insert(meshTexturePixels.end(), texture.pixels.begin(), texture.pixels.end());
                }
            }
            if (material.normalTextureIndex >= 0 && static_cast<size_t>(material.normalTextureIndex) < object.mesh.textures.size())
            {
                const MeshTexture& texture = object.mesh.textures[static_cast<size_t>(material.normalTextureIndex)];
                if (!texture.pixels.empty() && texture.width > 0 && texture.height > 0)
                {
                    materialGpu.hasNormalTexture = 1;
                    materialGpu.normalTextureOffset = static_cast<unsigned int>(meshTexturePixels.size());
                    materialGpu.normalTextureWidth = texture.width;
                    materialGpu.normalTextureHeight = texture.height;
                    meshTexturePixels.insert(meshTexturePixels.end(), texture.pixels.begin(), texture.pixels.end());
                }
            }
            meshMaterials.push_back(materialGpu);
        }

        meshObjectsGpu.push_back(MeshObjectGpu{triangleOffset, static_cast<unsigned int>(object.mesh.triangles.size())});
        meshIndexBuffers.push_back(std::move(objectIndices));
        meshInstanceTransforms.push_back(object.transform);
    }
    if (meshMaterials.empty())
    {
        meshMaterials.push_back(MeshMaterialGpu{});
        meshMaterials.back().color = make_float3(0.8f, 0.8f, 0.78f);
    }
    if (meshTexturePixels.empty())
    {
        meshTexturePixels.push_back(make_uchar4(255, 255, 255, 255));
    }
    meshTexturePixelCount = static_cast<unsigned int>(meshTexturePixels.size());
    meshObjectCount = static_cast<unsigned int>(meshObjectsGpu.size());

    CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&dMeshVertices), meshVertices.size() * sizeof(MeshVertexGpu)));
    CUDA_CHECK(cudaMemcpy(reinterpret_cast<void*>(dMeshVertices), meshVertices.data(), meshVertices.size() * sizeof(MeshVertexGpu), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&dMeshTriangles), meshTriangles.size() * sizeof(MeshTriangleGpu)));
    CUDA_CHECK(cudaMemcpy(reinterpret_cast<void*>(dMeshTriangles), meshTriangles.data(), meshTriangles.size() * sizeof(MeshTriangleGpu), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&dMeshMaterials), meshMaterials.size() * sizeof(MeshMaterialGpu)));
    CUDA_CHECK(cudaMemcpy(reinterpret_cast<void*>(dMeshMaterials), meshMaterials.data(), meshMaterials.size() * sizeof(MeshMaterialGpu), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&dMeshObjects), meshObjectsGpu.size() * sizeof(MeshObjectGpu)));
    CUDA_CHECK(cudaMemcpy(reinterpret_cast<void*>(dMeshObjects), meshObjectsGpu.data(), meshObjectsGpu.size() * sizeof(MeshObjectGpu), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&dMeshTexturePixels), meshTexturePixels.size() * sizeof(uchar4)));
    CUDA_CHECK(cudaMemcpy(reinterpret_cast<void*>(dMeshTexturePixels), meshTexturePixels.data(), meshTexturePixels.size() * sizeof(uchar4), cudaMemcpyHostToDevice));

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

    const std::array<float3, 4> planeVertices = {
        make_float3(-80.0f, 0.0f, -80.0f),
        make_float3(80.0f, 0.0f, -80.0f),
        make_float3(80.0f, 0.0f, 80.0f),
        make_float3(-80.0f, 0.0f, 80.0f)
    };
    const std::array<uint3, 2> planeIndices = {
        make_uint3(0u, 1u, 2u),
        make_uint3(0u, 2u, 3u)
    };

    CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&dPlaneVertices), planeVertices.size() * sizeof(float3)));
    CUDA_CHECK(cudaMemcpy(reinterpret_cast<void*>(dPlaneVertices), planeVertices.data(), planeVertices.size() * sizeof(float3), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&dPlaneIndices), planeIndices.size() * sizeof(uint3)));
    CUDA_CHECK(cudaMemcpy(reinterpret_cast<void*>(dPlaneIndices), planeIndices.data(), planeIndices.size() * sizeof(uint3), cudaMemcpyHostToDevice));

    planeFlags.assign(1, OPTIX_GEOMETRY_FLAG_NONE);
    planeAccelOptions.buildFlags = OPTIX_BUILD_FLAG_ALLOW_COMPACTION;
    planeAccelOptions.operation = OPTIX_BUILD_OPERATION_BUILD;
    planeBuildInput = {};
    planeBuildInput.type = OPTIX_BUILD_INPUT_TYPE_TRIANGLES;
    planeBuildInput.triangleArray.vertexFormat = OPTIX_VERTEX_FORMAT_FLOAT3;
    planeBuildInput.triangleArray.vertexStrideInBytes = sizeof(float3);
    planeBuildInput.triangleArray.numVertices = static_cast<unsigned int>(planeVertices.size());
    planeBuildInput.triangleArray.vertexBuffers = &dPlaneVertices;
    planeBuildInput.triangleArray.indexFormat = OPTIX_INDICES_FORMAT_UNSIGNED_INT3;
    planeBuildInput.triangleArray.indexStrideInBytes = sizeof(uint3);
    planeBuildInput.triangleArray.numIndexTriplets = static_cast<unsigned int>(planeIndices.size());
    planeBuildInput.triangleArray.indexBuffer = dPlaneIndices;
    planeBuildInput.triangleArray.flags = planeFlags.data();
    planeBuildInput.triangleArray.numSbtRecords = 1;

    OPTIX_CHECK(optixAccelComputeMemoryUsage(context, &planeAccelOptions, &planeBuildInput, 1, &planeGasSizes));
    CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&dPlaneGasBuffer), planeGasSizes.outputSizeInBytes));

    meshFlags.clear();
    meshBuildInputs.clear();
    meshAccelOptions.clear();
    meshGasSizes.clear();
    dMeshIndexBuffers.clear();
    dMeshGasBuffers.clear();
    meshGasHandles.assign(meshIndexBuffers.size(), 0);

    for (const std::vector<uint3>& objectIndices : meshIndexBuffers)
    {
        CUdeviceptr dObjectIndices = 0;
        CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&dObjectIndices), objectIndices.size() * sizeof(uint3)));
        CUDA_CHECK(cudaMemcpy(reinterpret_cast<void*>(dObjectIndices), objectIndices.data(), objectIndices.size() * sizeof(uint3), cudaMemcpyHostToDevice));
        dMeshIndexBuffers.push_back(dObjectIndices);

        meshFlags.push_back(std::vector<uint32_t>{OPTIX_GEOMETRY_FLAG_NONE});

        OptixAccelBuildOptions accelOptions{};
        accelOptions.buildFlags = OPTIX_BUILD_FLAG_ALLOW_COMPACTION | OPTIX_BUILD_FLAG_ALLOW_RANDOM_VERTEX_ACCESS;
        accelOptions.operation = OPTIX_BUILD_OPERATION_BUILD;
        meshAccelOptions.push_back(accelOptions);

        OptixBuildInput buildInput{};
        buildInput.type = OPTIX_BUILD_INPUT_TYPE_TRIANGLES;
        buildInput.triangleArray.vertexFormat = OPTIX_VERTEX_FORMAT_FLOAT3;
        buildInput.triangleArray.vertexStrideInBytes = sizeof(MeshVertexGpu);
        buildInput.triangleArray.numVertices = static_cast<unsigned int>(meshVertices.size());
        buildInput.triangleArray.vertexBuffers = &dMeshVertices;
        buildInput.triangleArray.indexFormat = OPTIX_INDICES_FORMAT_UNSIGNED_INT3;
        buildInput.triangleArray.indexStrideInBytes = sizeof(uint3);
        buildInput.triangleArray.numIndexTriplets = static_cast<unsigned int>(objectIndices.size());
        buildInput.triangleArray.indexBuffer = dObjectIndices;
        buildInput.triangleArray.flags = meshFlags.back().data();
        buildInput.triangleArray.numSbtRecords = 1;
        meshBuildInputs.push_back(buildInput);

        OptixAccelBufferSizes gasSizes{};
        OPTIX_CHECK(optixAccelComputeMemoryUsage(context, &meshAccelOptions.back(), &meshBuildInputs.back(), 1, &gasSizes));
        meshGasSizes.push_back(gasSizes);

        CUdeviceptr dGasBuffer = 0;
        CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&dGasBuffer), gasSizes.outputSizeInBytes));
        dMeshGasBuffers.push_back(dGasBuffer);
    }

    iasAccelOptions.buildFlags = OPTIX_BUILD_FLAG_NONE;
    iasAccelOptions.operation = OPTIX_BUILD_OPERATION_BUILD;
    iasBuildInput = {};
    iasBuildInput.type = OPTIX_BUILD_INPUT_TYPE_INSTANCES;
    iasBuildInput.instanceArray.numInstances = 2u + meshObjectCount;
    CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&dIasInstances), iasBuildInput.instanceArray.numInstances * sizeof(OptixInstance)));
    iasBuildInput.instanceArray.instances = dIasInstances;
    OPTIX_CHECK(optixAccelComputeMemoryUsage(context, &iasAccelOptions, &iasBuildInput, 1, &iasSizes));
    CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&dIasBuffer), iasSizes.outputSizeInBytes));

    rebuildAccelerationStructure();
}

void OptixRenderer::rebuildAccelerationStructure()
{
    CUdeviceptr dSphereTempBuffer = 0;
    CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&dSphereTempBuffer), sphereGasSizes.tempSizeInBytes));

    sphereAccelOptions.operation = OPTIX_BUILD_OPERATION_BUILD;
    OPTIX_CHECK(optixAccelBuild(
        context,
        stream,
        &sphereAccelOptions,
        &sphereBuildInput,
        1,
        dSphereTempBuffer,
        sphereGasSizes.tempSizeInBytes,
        dSphereGasBuffer,
        sphereGasSizes.outputSizeInBytes,
        &sphereGasHandle,
        nullptr,
        0));

    if (planeGasHandle == 0)
    {
        CUdeviceptr dPlaneTempBuffer = 0;
        CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&dPlaneTempBuffer), planeGasSizes.tempSizeInBytes));
        planeAccelOptions.operation = OPTIX_BUILD_OPERATION_BUILD;
        OPTIX_CHECK(optixAccelBuild(
            context,
            stream,
            &planeAccelOptions,
            &planeBuildInput,
            1,
            dPlaneTempBuffer,
            planeGasSizes.tempSizeInBytes,
            dPlaneGasBuffer,
            planeGasSizes.outputSizeInBytes,
            &planeGasHandle,
            nullptr,
            0));
        CUDA_CHECK(cudaFree(reinterpret_cast<void*>(dPlaneTempBuffer)));
    }

    for (size_t i = 0; i < meshBuildInputs.size(); ++i)
    {
        CUdeviceptr dMeshTempBuffer = 0;
        CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&dMeshTempBuffer), meshGasSizes[i].tempSizeInBytes));
        meshAccelOptions[i].operation = OPTIX_BUILD_OPERATION_BUILD;
        OPTIX_CHECK(optixAccelBuild(
            context,
            stream,
            &meshAccelOptions[i],
            &meshBuildInputs[i],
            1,
            dMeshTempBuffer,
            meshGasSizes[i].tempSizeInBytes,
            dMeshGasBuffers[i],
            meshGasSizes[i].outputSizeInBytes,
            &meshGasHandles[i],
            nullptr,
            0));
        CUDA_CHECK(cudaFree(reinterpret_cast<void*>(dMeshTempBuffer)));
    }

    std::vector<OptixInstance> instances(2u + meshObjectCount);
    for (OptixInstance& instance : instances)
    {
        std::memset(&instance, 0, sizeof(OptixInstance));
        instance.transform[0] = 1.0f;
        instance.transform[5] = 1.0f;
        instance.transform[10] = 1.0f;
        instance.visibilityMask = 255;
        instance.flags = OPTIX_INSTANCE_FLAG_NONE;
    }

    instances[0].instanceId = 0u;
    instances[0].sbtOffset = 0u;
    instances[0].traversableHandle = sphereGasHandle;
    instances[1].instanceId = 1u;
    instances[1].sbtOffset = 2u;
    instances[1].traversableHandle = planeGasHandle;
    for (unsigned int i = 0; i < meshObjectCount; ++i)
    {
        OptixInstance& instance = instances[2u + i];
        instance.instanceId = 2u + i;
        instance.sbtOffset = 4u;
        instance.traversableHandle = meshGasHandles[i];
        if (i < meshInstanceTransforms.size())
        {
            std::copy(meshInstanceTransforms[i].begin(), meshInstanceTransforms[i].end(), instance.transform);
        }
    }

    CUDA_CHECK(cudaMemcpyAsync(
        reinterpret_cast<void*>(dIasInstances),
        instances.data(),
        instances.size() * sizeof(OptixInstance),
        cudaMemcpyHostToDevice,
        stream));

    CUdeviceptr dIasTempBuffer = 0;
    CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&dIasTempBuffer), iasSizes.tempSizeInBytes));
    iasAccelOptions.operation = OPTIX_BUILD_OPERATION_BUILD;
    OPTIX_CHECK(optixAccelBuild(
        context,
        stream,
        &iasAccelOptions,
        &iasBuildInput,
        1,
        dIasTempBuffer,
        iasSizes.tempSizeInBytes,
        dIasBuffer,
        iasSizes.outputSizeInBytes,
        &iasHandle,
        nullptr,
        0));

    CUDA_CHECK(cudaStreamSynchronize(stream));
    CUDA_CHECK(cudaFree(reinterpret_cast<void*>(dIasTempBuffer)));
    CUDA_CHECK(cudaFree(reinterpret_cast<void*>(dSphereTempBuffer)));
}

void OptixRenderer::createModule()
{
    const std::string optixIr = compileDeviceProgram();

    OptixModuleCompileOptions moduleOptions{};
    moduleOptions.maxRegisterCount = OPTIX_COMPILE_DEFAULT_MAX_REGISTER_COUNT;
    moduleOptions.optLevel = OPTIX_COMPILE_OPTIMIZATION_DEFAULT;
    moduleOptions.debugLevel = OPTIX_COMPILE_DEBUG_LEVEL_NONE;

    pipelineCompileOptions.usesMotionBlur = false;
    pipelineCompileOptions.traversableGraphFlags = OPTIX_TRAVERSABLE_GRAPH_FLAG_ALLOW_SINGLE_LEVEL_INSTANCING;
    pipelineCompileOptions.numPayloadValues = 4;
    pipelineCompileOptions.numAttributeValues = 2;
    pipelineCompileOptions.exceptionFlags = OPTIX_EXCEPTION_FLAG_NONE;
    pipelineCompileOptions.pipelineLaunchParamsVariableName = "params";
    pipelineCompileOptions.usesPrimitiveTypeFlags = OPTIX_PRIMITIVE_TYPE_FLAGS_SPHERE | OPTIX_PRIMITIVE_TYPE_FLAGS_TRIANGLE;

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
    OPTIX_CHECK(optixProgramGroupCreate(context, &hitRadianceDesc, 1, &options, log, &logSize, &programGroups.hitSphereRadiance));

    OptixProgramGroupDesc hitShadowDesc{};
    hitShadowDesc.kind = OPTIX_PROGRAM_GROUP_KIND_HITGROUP;
    hitShadowDesc.hitgroup.moduleCH = module;
    hitShadowDesc.hitgroup.entryFunctionNameCH = "__closesthit__shadow";
    hitShadowDesc.hitgroup.moduleIS = sphereModule;
    hitShadowDesc.hitgroup.entryFunctionNameIS = nullptr;
    logSize = sizeof(log);
    OPTIX_CHECK(optixProgramGroupCreate(context, &hitShadowDesc, 1, &options, log, &logSize, &programGroups.hitSphereShadow));

    OptixProgramGroupDesc hitPlaneRadianceDesc{};
    hitPlaneRadianceDesc.kind = OPTIX_PROGRAM_GROUP_KIND_HITGROUP;
    hitPlaneRadianceDesc.hitgroup.moduleCH = module;
    hitPlaneRadianceDesc.hitgroup.entryFunctionNameCH = "__closesthit__radiance_plane";
    hitPlaneRadianceDesc.hitgroup.moduleIS = nullptr;
    hitPlaneRadianceDesc.hitgroup.entryFunctionNameIS = nullptr;
    logSize = sizeof(log);
    OPTIX_CHECK(optixProgramGroupCreate(context, &hitPlaneRadianceDesc, 1, &options, log, &logSize, &programGroups.hitPlaneRadiance));

    OptixProgramGroupDesc hitPlaneShadowDesc{};
    hitPlaneShadowDesc.kind = OPTIX_PROGRAM_GROUP_KIND_HITGROUP;
    hitPlaneShadowDesc.hitgroup.moduleCH = module;
    hitPlaneShadowDesc.hitgroup.entryFunctionNameCH = "__closesthit__shadow_plane";
    hitPlaneShadowDesc.hitgroup.moduleIS = nullptr;
    hitPlaneShadowDesc.hitgroup.entryFunctionNameIS = nullptr;
    logSize = sizeof(log);
    OPTIX_CHECK(optixProgramGroupCreate(context, &hitPlaneShadowDesc, 1, &options, log, &logSize, &programGroups.hitPlaneShadow));

    OptixProgramGroupDesc hitMeshRadianceDesc{};
    hitMeshRadianceDesc.kind = OPTIX_PROGRAM_GROUP_KIND_HITGROUP;
    hitMeshRadianceDesc.hitgroup.moduleCH = module;
    hitMeshRadianceDesc.hitgroup.entryFunctionNameCH = "__closesthit__radiance_mesh";
    hitMeshRadianceDesc.hitgroup.moduleIS = nullptr;
    hitMeshRadianceDesc.hitgroup.entryFunctionNameIS = nullptr;
    logSize = sizeof(log);
    OPTIX_CHECK(optixProgramGroupCreate(context, &hitMeshRadianceDesc, 1, &options, log, &logSize, &programGroups.hitMeshRadiance));

    OptixProgramGroupDesc hitMeshShadowDesc{};
    hitMeshShadowDesc.kind = OPTIX_PROGRAM_GROUP_KIND_HITGROUP;
    hitMeshShadowDesc.hitgroup.moduleCH = module;
    hitMeshShadowDesc.hitgroup.entryFunctionNameCH = "__closesthit__shadow_mesh";
    hitMeshShadowDesc.hitgroup.moduleIS = nullptr;
    hitMeshShadowDesc.hitgroup.entryFunctionNameIS = nullptr;
    logSize = sizeof(log);
    OPTIX_CHECK(optixProgramGroupCreate(context, &hitMeshShadowDesc, 1, &options, log, &logSize, &programGroups.hitMeshShadow));
}

void OptixRenderer::createPipeline()
{
    std::vector<OptixProgramGroup> groups = {
        programGroups.raygen,
        programGroups.missRadiance,
        programGroups.missShadow,
        programGroups.hitSphereRadiance,
        programGroups.hitSphereShadow,
        programGroups.hitPlaneRadiance,
        programGroups.hitPlaneShadow,
        programGroups.hitMeshRadiance,
        programGroups.hitMeshShadow
    };

    OptixPipelineLinkOptions linkOptions{};
    linkOptions.maxTraceDepth = kMaxTraceDepth;
    linkOptions.maxTraversableGraphDepth = 2;

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
    OPTIX_CHECK(optixUtilComputeStackSizes(&stackSizes, kMaxTraceDepth, 0, 0, &directCallableStackSizeFromTraversal, &directCallableStackSizeFromState, &continuationStackSize));
    OPTIX_CHECK(optixPipelineSetStackSize(pipeline, directCallableStackSizeFromTraversal, directCallableStackSizeFromState, continuationStackSize, 2));
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

    std::vector<HitgroupRecord> hitRecords(6);
    OPTIX_CHECK(optixSbtRecordPackHeader(programGroups.hitSphereRadiance, &hitRecords[0]));
    OPTIX_CHECK(optixSbtRecordPackHeader(programGroups.hitSphereShadow, &hitRecords[1]));
    OPTIX_CHECK(optixSbtRecordPackHeader(programGroups.hitPlaneRadiance, &hitRecords[2]));
    OPTIX_CHECK(optixSbtRecordPackHeader(programGroups.hitPlaneShadow, &hitRecords[3]));
    OPTIX_CHECK(optixSbtRecordPackHeader(programGroups.hitMeshRadiance, &hitRecords[4]));
    OPTIX_CHECK(optixSbtRecordPackHeader(programGroups.hitMeshShadow, &hitRecords[5]));
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
    updateCameraBasis(camera, gWidth, gHeight, forward, right, up, scale, aspect);

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
    params.handle = iasHandle;
    params.cameraPosition = camera.position;
    params.cameraForward = forward;
    params.cameraRight = right;
    params.cameraUp = up;
    params.cameraScale = scale;
    params.cameraAspect = aspect;
    params.lightPosition = scene.lightPosition;
    params.materials = reinterpret_cast<SphereMaterial*>(dMaterials);
    params.sphereCount = static_cast<int>(scene.spheres.size());
    params.meshVertices = reinterpret_cast<MeshVertexGpu*>(dMeshVertices);
    params.meshTriangles = reinterpret_cast<MeshTriangleGpu*>(dMeshTriangles);
    params.meshMaterials = reinterpret_cast<MeshMaterialGpu*>(dMeshMaterials);
    params.meshObjects = reinterpret_cast<MeshObjectGpu*>(dMeshObjects);
    params.meshTexturePixels = reinterpret_cast<uchar4*>(dMeshTexturePixels);
    params.meshVertexCount = 0u;
    params.meshTriangleCount = 0u;
    params.meshMaterialCount = 0u;
    for (const MeshObject& object : scene.meshObjects)
    {
        params.meshVertexCount += static_cast<unsigned int>(object.mesh.vertices.size());
        params.meshTriangleCount += static_cast<unsigned int>(object.mesh.triangles.size());
        params.meshMaterialCount += static_cast<unsigned int>(object.mesh.materials.size());
    }
    if (scene.meshObjects.empty())
    {
        params.meshVertexCount = static_cast<unsigned int>(scene.mesh.vertices.size());
        params.meshTriangleCount = static_cast<unsigned int>(scene.mesh.triangles.size());
        params.meshMaterialCount = static_cast<unsigned int>(scene.mesh.materials.size());
    }
    params.meshObjectCount = meshObjectCount;
    params.meshTexturePixelCount = meshTexturePixelCount;
    params.maxDepth = kMaxReflectionDepth;

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
    if (dPlaneGasBuffer != 0)
    {
        cudaFree(reinterpret_cast<void*>(dPlaneGasBuffer));
        dPlaneGasBuffer = 0;
    }
    for (CUdeviceptr buffer : dMeshGasBuffers)
    {
        if (buffer != 0)
        {
            cudaFree(reinterpret_cast<void*>(buffer));
        }
    }
    dMeshGasBuffers.clear();
    if (dIasBuffer != 0)
    {
        cudaFree(reinterpret_cast<void*>(dIasBuffer));
        dIasBuffer = 0;
    }
    if (dIasInstances != 0)
    {
        cudaFree(reinterpret_cast<void*>(dIasInstances));
        dIasInstances = 0;
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
    if (programGroups.hitPlaneShadow != nullptr)
    {
        optixProgramGroupDestroy(programGroups.hitPlaneShadow);
        programGroups.hitPlaneShadow = nullptr;
    }
    if (programGroups.hitMeshShadow != nullptr)
    {
        optixProgramGroupDestroy(programGroups.hitMeshShadow);
        programGroups.hitMeshShadow = nullptr;
    }
    if (programGroups.hitMeshRadiance != nullptr)
    {
        optixProgramGroupDestroy(programGroups.hitMeshRadiance);
        programGroups.hitMeshRadiance = nullptr;
    }
    if (programGroups.hitPlaneRadiance != nullptr)
    {
        optixProgramGroupDestroy(programGroups.hitPlaneRadiance);
        programGroups.hitPlaneRadiance = nullptr;
    }
    if (programGroups.hitSphereShadow != nullptr)
    {
        optixProgramGroupDestroy(programGroups.hitSphereShadow);
        programGroups.hitSphereShadow = nullptr;
    }
    if (programGroups.hitSphereRadiance != nullptr)
    {
        optixProgramGroupDestroy(programGroups.hitSphereRadiance);
        programGroups.hitSphereRadiance = nullptr;
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
    if (dMeshMaterials != 0)
    {
        cudaFree(reinterpret_cast<void*>(dMeshMaterials));
        dMeshMaterials = 0;
    }
    if (dMeshObjects != 0)
    {
        cudaFree(reinterpret_cast<void*>(dMeshObjects));
        dMeshObjects = 0;
        meshObjectCount = 0;
    }
    if (dMeshTexturePixels != 0)
    {
        cudaFree(reinterpret_cast<void*>(dMeshTexturePixels));
        dMeshTexturePixels = 0;
        meshTexturePixelCount = 0;
    }
    if (dMeshTriangles != 0)
    {
        cudaFree(reinterpret_cast<void*>(dMeshTriangles));
        dMeshTriangles = 0;
    }
    for (CUdeviceptr buffer : dMeshIndexBuffers)
    {
        if (buffer != 0)
        {
            cudaFree(reinterpret_cast<void*>(buffer));
        }
    }
    dMeshIndexBuffers.clear();
    if (dMeshVertices != 0)
    {
        cudaFree(reinterpret_cast<void*>(dMeshVertices));
        dMeshVertices = 0;
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
    if (dPlaneIndices != 0)
    {
        cudaFree(reinterpret_cast<void*>(dPlaneIndices));
        dPlaneIndices = 0;
    }
    if (dPlaneVertices != 0)
    {
        cudaFree(reinterpret_cast<void*>(dPlaneVertices));
        dPlaneVertices = 0;
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
