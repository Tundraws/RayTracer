#pragma once

#include <cuda_runtime.h>
#include <optix.h>

#include <vector>

struct ProgramGroups
{
    OptixProgramGroup raygen = nullptr;
    OptixProgramGroup missRadiance = nullptr;
    OptixProgramGroup missShadow = nullptr;
    OptixProgramGroup hitRadiance = nullptr;
    OptixProgramGroup hitShadow = nullptr;
};

class OptixRenderer
{
public:
    void setRenderSize(int width, int height);
    void initialize();
    void renderFrame(const struct SceneState& scene, const struct CameraState& camera, std::vector<uchar4>& hostPixels, float* gpuTimeMs = nullptr);
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
