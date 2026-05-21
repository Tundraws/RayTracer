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

inline float ggxClampDot(const float value)
{
    return clamp01(value);
}

inline float ggxClampRoughness(const float roughness)
{
    return roughness < 0.045f ? 0.045f : (roughness > 1.0f ? 1.0f : roughness);
}

inline float ggxDistribution(const float nDotH, const float roughness)
{
    const float alpha = ggxClampRoughness(roughness);
    const float a2 = alpha * alpha * alpha * alpha;
    const float ndh = ggxClampDot(nDotH);
    const float denom = ndh * ndh * (a2 - 1.0f) + 1.0f;
    constexpr float pi = 3.14159265358979323846f;
    constexpr float epsilon = 1e-5f;
    return a2 / (pi * denom * denom + epsilon);
}

inline float ggxGeometrySchlick(const float nDotV, const float roughness)
{
    const float r = ggxClampRoughness(roughness) + 1.0f;
    const float k = (r * r) / 8.0f;
    const float ndv = ggxClampDot(nDotV);
    constexpr float epsilon = 1e-5f;
    return ndv / (ndv * (1.0f - k) + k + epsilon);
}

inline float ggxGeometrySmith(const float nDotV, const float nDotL, const float roughness)
{
    return ggxGeometrySchlick(nDotV, roughness) * ggxGeometrySchlick(nDotL, roughness);
}

inline float ggxFresnelSchlick(const float cosTheta, const float f0)
{
    const float m = clamp01(1.0f - cosTheta);
    const float m2 = m * m;
    return f0 + (1.0f - f0) * m2 * m2 * m;
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
    float2 texcoord;
    float3 tangent;
    int hasTexcoord;
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
    int hasTexture = 0;
    unsigned int textureOffset = 0;
    unsigned int textureWidth = 0;
    unsigned int textureHeight = 0;
    int hasNormalTexture = 0;
    unsigned int normalTextureOffset = 0;
    unsigned int normalTextureWidth = 0;
    unsigned int normalTextureHeight = 0;
};

struct MeshObjectGpu
{
    unsigned int triangleOffset = 0;
    unsigned int triangleCount = 0;
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
    MeshObjectGpu* meshObjects;
    uchar4* meshTexturePixels;
    unsigned int meshVertexCount;
    unsigned int meshTriangleCount;
    unsigned int meshMaterialCount;
    unsigned int meshObjectCount;
    unsigned int meshTexturePixelCount;
    int maxDepth;
};
