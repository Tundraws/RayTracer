#pragma once

#include "../common/rtx_shared.h"

#include <cuda_runtime.h>
#include <optix.h>

#include <array>
#include <cstddef>
#include <vector>

struct ProgramGroups
{
    OptixProgramGroup raygen = nullptr;
    OptixProgramGroup missRadiance = nullptr;
    OptixProgramGroup missShadow = nullptr;
    OptixProgramGroup hitSphereRadiance = nullptr;
    OptixProgramGroup hitSphereShadow = nullptr;
    OptixProgramGroup hitPlaneRadiance = nullptr;
    OptixProgramGroup hitPlaneShadow = nullptr;
    OptixProgramGroup hitMeshRadiance = nullptr;
    OptixProgramGroup hitMeshShadow = nullptr;
};

class OptixRenderer
{
public:
    void setRenderSize(int width, int height);
    void setRenderMode(int mode);
    int getRenderMode() const;
    void setRenderQuality(int quality);
    int getRenderQuality() const;
    void setDenoiserEnabled(bool enabled);
    bool isDenoiserEnabled() const;
    bool isDenoiserAvailable() const;
    void resetAccumulation();
    unsigned int getAccumulationSampleCount() const;
    void initialize();
    void initialize(const struct SceneState& initialScene);
    void renderFrame(const struct SceneState& scene, const struct CameraState& camera, std::vector<uchar4>& hostPixels, float* gpuTimeMs = nullptr);
    void destroy();

private:
    void createContext();
    void createScene(const struct SceneState& scene);
    void createModule();
    void createProgramGroups();
    void createPipeline();
    void createSbt();
    void ensureSphereResources(const struct SceneState& scene);
    void ensureEnvironmentResources(const struct SceneState& scene);
    void syncMeshInstanceTransforms(const struct SceneState& scene);
    void rebuildAccelerationStructure();
    void rebuildSphereAccelerationStructure();
    void rebuildInstanceAccelerationStructure();
    void initializeDenoiser();
    void releaseDenoiser();
    bool applyDenoiser(std::vector<uchar4>& hostPixels);

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
    CUdeviceptr dPlaneVertices = 0;
    CUdeviceptr dPlaneIndices = 0;
    CUdeviceptr dMeshVertices = 0;
    CUdeviceptr dMeshTriangles = 0;
    CUdeviceptr dMeshMaterials = 0;
    CUdeviceptr dMeshObjects = 0;
    CUdeviceptr dMeshTexturePixels = 0;
    CUdeviceptr dEnvironmentPixels = 0;
    CUdeviceptr dSphereGasBuffer = 0;
    CUdeviceptr dPlaneGasBuffer = 0;
    std::vector<CUdeviceptr> dMeshIndexBuffers;
    std::vector<CUdeviceptr> dMeshGasBuffers;
    CUdeviceptr dIasBuffer = 0;
    CUdeviceptr dIasInstances = 0;
    CUdeviceptr dLaunchParams = 0;
    CUdeviceptr dAccumulationBuffer = 0;
    CUdeviceptr dDenoisedBuffer = 0;
    CUdeviceptr dDenoiserState = 0;
    CUdeviceptr dDenoiserScratch = 0;
    OptixTraversableHandle sphereGasHandle = 0;
    OptixTraversableHandle planeGasHandle = 0;
    std::vector<OptixTraversableHandle> meshGasHandles;
    OptixTraversableHandle iasHandle = 0;
    std::vector<uint32_t> sphereFlags;
    std::vector<uint32_t> planeFlags;
    std::vector<std::vector<uint32_t>> meshFlags;
    std::vector<OptixBuildInput> meshBuildInputs;
    std::vector<OptixAccelBuildOptions> meshAccelOptions;
    std::vector<OptixAccelBufferSizes> meshGasSizes;
    std::vector<std::array<float, 12>> meshInstanceTransforms;
    std::size_t sphereBufferCapacity = 0;
    std::size_t materialBufferCapacity = 0;
    std::size_t environmentPixelCapacity = 0;
    unsigned int meshTexturePixelCount = 0;
    unsigned int meshObjectCount = 0;
    unsigned int accumulationSampleCount = 0;
    int renderMode = RenderModeRealtime;
    int renderQuality = RenderQualityHigh;
    bool showGroundPlane = false;
    bool denoiserEnabled = false;
    bool denoiserAvailable = false;
    std::size_t lastAccumulationSignature = 0;
    std::size_t lastSphereGeometrySignature = 0;
    std::size_t lastInstanceSignature = 0;
    OptixDenoiser denoiser = nullptr;
    OptixDenoiserSizes denoiserSizes = {};
    OptixBuildInput sphereBuildInput = {};
    OptixBuildInput planeBuildInput = {};
    OptixBuildInput iasBuildInput = {};
    OptixAccelBuildOptions sphereAccelOptions = {};
    OptixAccelBuildOptions planeAccelOptions = {};
    OptixAccelBuildOptions iasAccelOptions = {};
    OptixAccelBufferSizes sphereGasSizes = {};
    OptixAccelBufferSizes planeGasSizes = {};
    OptixAccelBufferSizes iasSizes = {};
    cudaEvent_t frameStart = nullptr;
    cudaEvent_t frameStop = nullptr;

    uchar4* dFrameBuffer = nullptr;
};
