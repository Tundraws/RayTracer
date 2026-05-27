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

enum RenderMode
{
    RenderModeRealtime = 0,
    RenderModeProgressive = 1
};

enum RenderQuality
{
    RenderQualityLow = 0,
    RenderQualityMedium = 1,
    RenderQualityHigh = 2,
    RenderQualityPathTracing = 3
};

#if !defined(__CUDACC_RTC__)
inline int clampRenderQuality(const int quality)
{
    return quality < RenderQualityLow || quality > RenderQualityPathTracing
        ? RenderQualityMedium
        : quality;
}

inline int nextRenderQuality(const int quality)
{
    const int current = clampRenderQuality(quality);
    return current == RenderQualityPathTracing ? RenderQualityLow : current + 1;
}

inline int renderQualityMaxDepth(const int quality)
{
    switch (clampRenderQuality(quality))
    {
    case RenderQualityLow:
        return 1;
    case RenderQualityHigh:
        return 5;
    case RenderQualityPathTracing:
        return 6;
    default:
        return 3;
    }
}

inline int renderQualitySamplesPerPixel(const int quality)
{
    switch (clampRenderQuality(quality))
    {
    case RenderQualityLow:
        return 1;
    case RenderQualityHigh:
        return 4;
    case RenderQualityPathTracing:
        return 2;
    default:
        return 2;
    }
}

inline bool renderQualityShadowsEnabled(const int quality)
{
    return clampRenderQuality(quality) != RenderQualityLow;
}

inline bool renderQualityUsesPathTracing(const int quality)
{
    return clampRenderQuality(quality) == RenderQualityPathTracing;
}

inline bool renderQualityUsesDenoiser(const int quality)
{
    return clampRenderQuality(quality) == RenderQualityPathTracing;
}

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

inline float clampSceneExposure(const float value)
{
    return value < 0.1f ? 0.1f : (value > 2.5f ? 2.5f : value);
}

inline float clampSceneLightIntensity(const float value)
{
    return value < 0.0f ? 0.0f : (value > 5.0f ? 5.0f : value);
}

inline float clampSceneSkyIntensity(const float value)
{
    return value < 0.0f ? 0.0f : (value > 3.0f ? 3.0f : value);
}

inline float clampSceneAreaLightRadius(const float value)
{
    return value < 0.0f ? 0.0f : (value > 12.0f ? 12.0f : value);
}

inline float clampSceneEnvironmentIntensity(const float value)
{
    return value < 0.0f ? 0.0f : (value > 4.0f ? 4.0f : value);
}

inline float3 toneMapAndGammaCorrect(const float3 color, const float exposure = 1.0f)
{
    return gammaCorrect(reinhardToneMap(make_float3(
        color.x * clampSceneExposure(exposure),
        color.y * clampSceneExposure(exposure),
        color.z * clampSceneExposure(exposure))));
}

inline float ggxClampDot(const float value)
{
    return clamp01(value);
}

inline float ggxClampRoughness(const float roughness)
{
    return roughness < 0.045f ? 0.045f : (roughness > 1.0f ? 1.0f : roughness);
}

inline float clampMaterialRoughnessShared(const float roughness)
{
    return roughness < 0.02f ? 0.02f : (roughness > 1.0f ? 1.0f : roughness);
}

inline float clampMaterialIorShared(const float ior)
{
    return ior < 1.01f ? 1.01f : (ior > 2.8f ? 2.8f : ior);
}

inline float dielectricF0FromIor(const float ior)
{
    const float safeIor = clampMaterialIorShared(ior);
    const float f0 = (safeIor - 1.0f) / (safeIor + 1.0f);
    return f0 * f0;
}

inline float dielectricFresnelSchlick(const float cosTheta, const float ior)
{
    const float f0 = dielectricF0FromIor(ior);
    const float m = clamp01(1.0f - cosTheta);
    const float m2 = m * m;
    return f0 + (1.0f - f0) * m2 * m2 * m;
}

inline float dotShared(const float3 a, const float3 b)
{
    return a.x * b.x + a.y * b.y + a.z * b.z;
}

inline float3 addShared(const float3 a, const float3 b)
{
    return make_float3(a.x + b.x, a.y + b.y, a.z + b.z);
}

inline float3 mulShared(const float3 a, const float b)
{
    return make_float3(a.x * b, a.y * b, a.z * b);
}

inline float3 normalizeShared(const float3 v)
{
    const float len = std::sqrt(dotShared(v, v));
    return len > 0.0f ? mulShared(v, 1.0f / len) : make_float3(0.0f, 0.0f, 0.0f);
}

inline bool refractDirection(const float3 incident, const float3 normal, const float eta, float3& refracted)
{
    const float safeEta = eta < 0.01f ? 0.01f : (eta > 8.0f ? 8.0f : eta);
    const float cosi = ggxClampDot(-dotShared(incident, normal));
    const float sin2Theta = 1.0f - cosi * cosi;
    const float k = 1.0f - safeEta * safeEta * sin2Theta;
    if (k < 0.0f)
    {
        refracted = make_float3(0.0f, 0.0f, 0.0f);
        return false;
    }

    refracted = normalizeShared(addShared(mulShared(incident, safeEta), mulShared(normal, safeEta * cosi - std::sqrt(k))));
    return true;
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
    int hasMetallicTexture = 0;
    unsigned int metallicTextureOffset = 0;
    unsigned int metallicTextureWidth = 0;
    unsigned int metallicTextureHeight = 0;
    int hasRoughnessTexture = 0;
    unsigned int roughnessTextureOffset = 0;
    unsigned int roughnessTextureWidth = 0;
    unsigned int roughnessTextureHeight = 0;
};

struct MeshObjectGpu
{
    unsigned int triangleOffset = 0;
    unsigned int triangleCount = 0;
};

struct LaunchParams
{
    uchar4* image;
    float4* accumulation;
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
    float exposure;
    float skyIntensity;
    float3 skyHorizonColor;
    float3 skyZenithColor;
    float lightIntensity;
    float areaLightRadius;
    float environmentIntensity;
    SphereMaterial* materials;
    int sphereCount;
    MeshVertexGpu* meshVertices;
    MeshTriangleGpu* meshTriangles;
    MeshMaterialGpu* meshMaterials;
    MeshObjectGpu* meshObjects;
    uchar4* meshTexturePixels;
    uchar4* environmentPixels;
    unsigned int meshVertexCount;
    unsigned int meshTriangleCount;
    unsigned int meshMaterialCount;
    unsigned int meshObjectCount;
    unsigned int meshTexturePixelCount;
    unsigned int environmentWidth;
    unsigned int environmentHeight;
    unsigned int environmentPixelCount;
    int maxDepth;
    int renderMode;
    int renderQuality;
    int shadowEnabled;
    int samplesPerPixel;
    unsigned int accumulationSample;
};
