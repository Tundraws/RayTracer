#pragma once

#include <cuda_runtime.h>
#include <optix.h>

#if !defined(__CUDACC_RTC__)
#include <cmath>
#endif

enum MaterialType
{
    MaterialDiffuse = 0,
    MaterialMirror = 1,
    MaterialMetal = 2,
    MaterialDielectric = 3
};

#if !defined(__CUDACC_RTC__)
inline float clamp01(const float value)
{
    return value < 0.0f ? 0.0f : (value > 1.0f ? 1.0f : value);
}

inline float3 reinhardToneMap(const float3 color)
{
    return make_float3(
        color.x / (1.0f + color.x),
        color.y / (1.0f + color.y),
        color.z / (1.0f + color.z));
}

inline float3 gammaCorrect(const float3 color, const float gamma = 2.2f)
{
    const float invGamma = 1.0f / gamma;
    return make_float3(
        std::pow(clamp01(color.x), invGamma),
        std::pow(clamp01(color.y), invGamma),
        std::pow(clamp01(color.z), invGamma));
}

inline float3 toneMapAndGammaCorrect(const float3 color)
{
    return gammaCorrect(reinhardToneMap(color));
}
#endif

struct SphereMaterial
{
    float3 color{};
    int materialType = MaterialDiffuse;
    float3 specularColor{1.0f, 1.0f, 1.0f};
    float roughness = 0.35f;
    float ior = 1.5f;
    float alpha = 1.0f;
};

struct MeshVertexGpu
{
    float3 position;
    float3 normal;
};

struct MeshTriangleGpu
{
    unsigned int i0;
    unsigned int i1;
    unsigned int i2;
    unsigned int materialIndex;
};

struct MeshMaterialGpu
{
    float3 color{};
    int materialType = MaterialDiffuse;
    float3 specularColor{1.0f, 1.0f, 1.0f};
    float roughness = 0.35f;
    float ior = 1.5f;
    float alpha = 1.0f;
};

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
    MeshVertexGpu* meshVertices;
    MeshTriangleGpu* meshTriangles;
    MeshMaterialGpu* meshMaterials;
    unsigned int meshVertexCount;
    unsigned int meshTriangleCount;
    unsigned int meshMaterialCount;
    int maxDepth;
};
