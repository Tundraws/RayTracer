#pragma once

namespace gpu
{
static const char* kOptixDeviceProgram = R"(
#include <optix.h>
#include <cuda_runtime.h>
#include <common/rtx_shared.h>

extern "C" __constant__ LaunchParams params;

enum RayType
{
    RAY_TYPE_RADIANCE = 0,
    RAY_TYPE_SHADOW = 1,
    RAY_TYPE_COUNT = 2
};

static __forceinline__ __device__ float3 make_vec(const float x, const float y, const float z)
{
    return make_float3(x, y, z);
}

static __forceinline__ __device__ float3 add3(const float3 a, const float3 b)
{
    return make_vec(a.x + b.x, a.y + b.y, a.z + b.z);
}

static __forceinline__ __device__ float3 sub3(const float3 a, const float3 b)
{
    return make_vec(a.x - b.x, a.y - b.y, a.z - b.z);
}

static __forceinline__ __device__ float3 mul3(const float3 a, const float b)
{
    return make_vec(a.x * b, a.y * b, a.z * b);
}

static __forceinline__ __device__ float dot3(const float3 a, const float3 b)
{
    return a.x * b.x + a.y * b.y + a.z * b.z;
}

static __forceinline__ __device__ float3 cross3(const float3 a, const float3 b)
{
    return make_vec(
        a.y * b.z - a.z * b.y,
        a.z * b.x - a.x * b.z,
        a.x * b.y - a.y * b.x);
}

static __forceinline__ __device__ float3 normalize3(const float3 v)
{
    const float len = sqrtf(dot3(v, v));
    return len > 0.0f ? mul3(v, 1.0f / len) : make_vec(0.0f, 0.0f, 0.0f);
}

static __forceinline__ __device__ float3 reflect3(const float3 i, const float3 n)
{
    return sub3(i, mul3(n, 2.0f * dot3(i, n)));
}

static __forceinline__ __device__ bool refract3(const float3 i, const float3 n, const float eta, float3& refracted)
{
    const float safeEta = fminf(fmaxf(eta, 0.01f), 8.0f);
    const float cosi = fminf(fmaxf(-dot3(i, n), 0.0f), 1.0f);
    const float sin2Theta = 1.0f - cosi * cosi;
    const float k = 1.0f - safeEta * safeEta * sin2Theta;
    if (k < 0.0f)
    {
        refracted = make_vec(0.0f, 0.0f, 0.0f);
        return false;
    }
    refracted = normalize3(add3(mul3(i, safeEta), mul3(n, safeEta * cosi - sqrtf(k))));
    return true;
}

static __forceinline__ __device__ float3 clamp3(const float3 v, const float minValue, const float maxValue)
{
    return make_vec(
        fminf(fmaxf(v.x, minValue), maxValue),
        fminf(fmaxf(v.y, minValue), maxValue),
        fminf(fmaxf(v.z, minValue), maxValue));
}

static __forceinline__ __device__ float saturate1(const float value)
{
    return fminf(fmaxf(value, 0.0f), 1.0f);
}

static __forceinline__ __device__ float3 lerp3(const float3 a, const float3 b, const float t)
{
    return add3(mul3(a, 1.0f - t), mul3(b, t));
}

static __forceinline__ __device__ float3 sampleEnvironmentMap(const float3 rayDir)
{
    if (params.environmentPixels == nullptr ||
        params.environmentWidth == 0u ||
        params.environmentHeight == 0u ||
        params.environmentPixelCount == 0u)
    {
        return make_vec(0.0f, 0.0f, 0.0f);
    }

    const float u = atan2f(rayDir.z, rayDir.x) * (1.0f / 6.2831853f) + 0.5f;
    const float v = acosf(fminf(fmaxf(rayDir.y, -1.0f), 1.0f)) * (1.0f / 3.14159265f);
    const unsigned int x = static_cast<unsigned int>(fminf(u * static_cast<float>(params.environmentWidth), static_cast<float>(params.environmentWidth - 1u)));
    const unsigned int y = static_cast<unsigned int>(fminf(v * static_cast<float>(params.environmentHeight), static_cast<float>(params.environmentHeight - 1u)));
    const unsigned int offset = y * params.environmentWidth + x;
    if (offset >= params.environmentPixelCount)
    {
        return make_vec(0.0f, 0.0f, 0.0f);
    }

    const uchar4 pixel = params.environmentPixels[offset];
    return make_vec(
        static_cast<float>(pixel.x) / 255.0f,
        static_cast<float>(pixel.y) / 255.0f,
        static_cast<float>(pixel.z) / 255.0f);
}

static __forceinline__ __device__ float3 environmentColor(const float3 rayDir)
{
    const float3 mapped = sampleEnvironmentMap(rayDir);
    if (params.environmentPixelCount > 0u)
    {
        return mul3(mapped, params.skyIntensity * params.environmentIntensity);
    }

    const float t = saturate1(0.5f * (rayDir.y + 1.0f));
    const float horizonGlow = expf(-6.0f * fabsf(rayDir.y));
    const float sun = powf(fmaxf(dot3(rayDir, normalize3(make_vec(0.35f, 0.55f, -0.75f))), 0.0f), 96.0f);
    const float3 ground = make_vec(0.20f, 0.22f, 0.21f);
    const float3 horizon = make_vec(0.62f, 0.70f, 0.78f);
    const float3 zenith = make_vec(0.12f, 0.18f, 0.30f);
    const float3 sky = lerp3(horizon, zenith, t * t);
    const float3 base = rayDir.y < 0.0f ? lerp3(ground, horizon, saturate1(rayDir.y + 1.0f)) : sky;
    return mul3(
        add3(add3(base, mul3(make_vec(0.95f, 0.74f, 0.42f), 0.18f * horizonGlow)), mul3(make_vec(1.0f, 0.86f, 0.58f), 1.2f * sun)),
        params.skyIntensity * params.environmentIntensity);
}

static __forceinline__ __device__ float3 reinhardToneMapDevice(const float3 color)
{
    return make_vec(
        color.x / (1.0f + color.x),
        color.y / (1.0f + color.y),
        color.z / (1.0f + color.z));
}

static __forceinline__ __device__ float3 gammaCorrectDevice(const float3 color)
{
    const float invGamma = 1.0f / 2.2f;
    return make_vec(
        powf(saturate1(color.x), invGamma),
        powf(saturate1(color.y), invGamma),
        powf(saturate1(color.z), invGamma));
}

static __forceinline__ __device__ float3 postProcessColor(const float3 color)
{
    return gammaCorrectDevice(reinhardToneMapDevice(clamp3(mul3(color, params.exposure), 0.0f, 12.0f)));
}

static __forceinline__ __device__ float fresnelSchlick(const float cosTheta, const float f0)
{
    const float m = saturate1(1.0f - cosTheta);
    const float m2 = m * m;
    return f0 + (1.0f - f0) * m2 * m2 * m;
}

static __forceinline__ __device__ float safeMaterialRoughness(const float roughness)
{
    return fminf(fmaxf(roughness, 0.02f), 1.0f);
}

static __forceinline__ __device__ float safeMaterialIor(const float ior)
{
    return fminf(fmaxf(ior, 1.01f), 2.8f);
}

static __forceinline__ __device__ float dielectricF0(const float ior)
{
    const float safeIor = safeMaterialIor(ior);
    const float f0 = (safeIor - 1.0f) / (safeIor + 1.0f);
    return f0 * f0;
}

static __forceinline__ __device__ float dielectricFresnel(const float cosTheta, const float ior)
{
    return fresnelSchlick(cosTheta, dielectricF0(ior));
}

static __forceinline__ __device__ float ggxDistributionDevice(const float nDotH, const float roughness)
{
    const float alpha = fmaxf(saturate1(roughness), 0.045f);
    const float a2 = alpha * alpha * alpha * alpha;
    const float ndh = saturate1(nDotH);
    const float denom = ndh * ndh * (a2 - 1.0f) + 1.0f;
    return a2 / (3.14159265f * denom * denom + 1e-5f);
}

static __forceinline__ __device__ float ggxGeometrySchlickDevice(const float nDotV, const float roughness)
{
    const float r = fmaxf(saturate1(roughness), 0.045f) + 1.0f;
    const float k = (r * r) / 8.0f;
    const float ndv = saturate1(nDotV);
    return ndv / (ndv * (1.0f - k) + k + 1e-5f);
}

static __forceinline__ __device__ float ggxGeometrySmithDevice(const float nDotV, const float nDotL, const float roughness)
{
    return ggxGeometrySchlickDevice(nDotV, roughness) * ggxGeometrySchlickDevice(nDotL, roughness);
}

static __forceinline__ __device__ float3 ggxDirectLight(
    const float3 baseColor,
    const float3 specularColor,
    const float3 normal,
    const float3 viewDir,
    const float3 lightDir,
    const float roughness,
    const float metallic)
{
    const float nDotL = saturate1(dot3(normal, lightDir));
    const float nDotV = saturate1(dot3(normal, viewDir));
    if (nDotL <= 0.0f || nDotV <= 0.0f)
    {
        return make_vec(0.0f, 0.0f, 0.0f);
    }

    const float3 halfDir = normalize3(add3(lightDir, viewDir));
    const float nDotH = saturate1(dot3(normal, halfDir));
    const float vDotH = saturate1(dot3(viewDir, halfDir));
    const float D = ggxDistributionDevice(nDotH, roughness);
    const float G = ggxGeometrySmithDevice(nDotV, nDotL, roughness);

    const float3 dielectricF0 = make_vec(
        fminf(fmaxf(specularColor.x * 0.04f, 0.02f), 0.18f),
        fminf(fmaxf(specularColor.y * 0.04f, 0.02f), 0.18f),
        fminf(fmaxf(specularColor.z * 0.04f, 0.02f), 0.18f));
    const float3 f0 = lerp3(dielectricF0, baseColor, metallic);
    const float fresnelFactor = fresnelSchlick(vDotH, 0.0f);
    const float3 F = add3(f0, mul3(sub3(make_vec(1.0f, 1.0f, 1.0f), f0), fresnelFactor));
    const float denominator = fmaxf(4.0f * nDotV * nDotL, 1e-4f);
    const float3 specular = mul3(F, (D * G) / denominator);
    const float3 kd = mul3(sub3(make_vec(1.0f, 1.0f, 1.0f), F), 1.0f - metallic);
    const float3 diffuse = mul3(make_vec(kd.x * baseColor.x, kd.y * baseColor.y, kd.z * baseColor.z), 1.0f / 3.14159265f);
    return mul3(add3(diffuse, specular), nDotL);
}

static __forceinline__ __device__ float3 roughReflectionDir(const float3 reflectedDir, const float3 normal, const float roughness)
{
    const float r = safeMaterialRoughness(roughness);
    const float spread = saturate1(r * r);
    const float3 helper = fabsf(normal.y) < 0.999f ? make_vec(0.0f, 1.0f, 0.0f) : make_vec(1.0f, 0.0f, 0.0f);
    const float3 tangent = normalize3(cross3(helper, normal));
    const float3 bitangent = normalize3(cross3(normal, tangent));
    const float phaseRaw = sinf(dot3(reflectedDir, make_vec(12.9898f, 78.233f, 37.719f))) * 43758.5453f;
    const float phase = phaseRaw - floorf(phaseRaw);
    const float angle = 6.2831853f * phase;
    const float3 lobeOffset = add3(mul3(tangent, cosf(angle)), mul3(bitangent, sinf(angle)));
    const float3 broadened = normalize3(add3(add3(reflectedDir, mul3(lobeOffset, 0.85f * spread)), mul3(normal, 0.18f * spread)));
    return normalize3(lerp3(reflectedDir, broadened, spread));
}

)"
R"(
static __forceinline__ __device__ float3 sampleDiffuseTexture(const MeshMaterialGpu material, const float2 texcoord)
{
    if (material.hasTexture == 0 || material.textureWidth == 0u || material.textureHeight == 0u || params.meshTexturePixels == nullptr)
    {
        return material.color;
    }

    float u = texcoord.x - floorf(texcoord.x);
    float v = texcoord.y - floorf(texcoord.y);
    if (u < 0.0f)
    {
        u += 1.0f;
    }
    if (v < 0.0f)
    {
        v += 1.0f;
    }

    const unsigned int x = static_cast<unsigned int>(fminf(u * static_cast<float>(material.textureWidth), static_cast<float>(material.textureWidth - 1u)));
    const unsigned int y = static_cast<unsigned int>(fminf((1.0f - v) * static_cast<float>(material.textureHeight), static_cast<float>(material.textureHeight - 1u)));
    const unsigned int offset = material.textureOffset + y * material.textureWidth + x;
    if (offset >= params.meshTexturePixelCount)
    {
        return material.color;
    }

    const uchar4 pixel = params.meshTexturePixels[offset];
    return make_vec(
        static_cast<float>(pixel.x) / 255.0f,
        static_cast<float>(pixel.y) / 255.0f,
        static_cast<float>(pixel.z) / 255.0f);
}

static __forceinline__ __device__ float3 sampleNormalTexture(
    const MeshMaterialGpu material,
    const float2 texcoord,
    const float3 normal,
    const float3 tangent,
    const int hasTexcoord)
{
    if (hasTexcoord == 0 ||
        material.hasNormalTexture == 0 ||
        material.normalTextureWidth == 0u ||
        material.normalTextureHeight == 0u ||
        params.meshTexturePixels == nullptr ||
        dot3(tangent, tangent) <= 1e-6f)
    {
        return normal;
    }

    float u = texcoord.x - floorf(texcoord.x);
    float v = texcoord.y - floorf(texcoord.y);
    if (u < 0.0f)
    {
        u += 1.0f;
    }
    if (v < 0.0f)
    {
        v += 1.0f;
    }

    const unsigned int x = static_cast<unsigned int>(fminf(u * static_cast<float>(material.normalTextureWidth), static_cast<float>(material.normalTextureWidth - 1u)));
    const unsigned int y = static_cast<unsigned int>(fminf((1.0f - v) * static_cast<float>(material.normalTextureHeight), static_cast<float>(material.normalTextureHeight - 1u)));
    const unsigned int offset = material.normalTextureOffset + y * material.normalTextureWidth + x;
    if (offset >= params.meshTexturePixelCount)
    {
        return normal;
    }

    const uchar4 pixel = params.meshTexturePixels[offset];
    const float3 tangentSpaceNormal = normalize3(make_vec(
        static_cast<float>(pixel.x) / 255.0f * 2.0f - 1.0f,
        static_cast<float>(pixel.y) / 255.0f * 2.0f - 1.0f,
        static_cast<float>(pixel.z) / 255.0f * 2.0f - 1.0f));
    const float3 t = normalize3(sub3(tangent, mul3(normal, dot3(normal, tangent))));
    const float3 b = normalize3(cross3(normal, t));
    return normalize3(add3(add3(mul3(t, tangentSpaceNormal.x), mul3(b, tangentSpaceNormal.y)), mul3(normal, tangentSpaceNormal.z)));
}

static __forceinline__ __device__ void setRadiancePayload(const float3 color)
{
    optixSetPayload_0(__float_as_uint(color.x));
    optixSetPayload_1(__float_as_uint(color.y));
    optixSetPayload_2(__float_as_uint(color.z));
}

static __forceinline__ __device__ void setShadowPayload(const bool visible)
{
    optixSetPayload_0(visible ? 1u : 0u);
}

static __forceinline__ __device__ bool traceShadow(
    const OptixTraversableHandle handle,
    const float3 origin,
    const float3 direction,
    const float tmin,
    const float tmax)
{
    unsigned int visible = 1u;
    optixTrace(
        handle,
        origin,
        direction,
        tmin,
        tmax,
        0.0f,
        OptixVisibilityMask(255),
        OPTIX_RAY_FLAG_DISABLE_ANYHIT,
        RAY_TYPE_SHADOW,
        RAY_TYPE_COUNT,
        RAY_TYPE_SHADOW,
        visible);
    return visible != 0u;
}

static __forceinline__ __device__ float3 traceRadiance(
    const OptixTraversableHandle handle,
    const float3 origin,
    const float3 direction,
    const float tmin,
    const float tmax,
    const unsigned int depth,
    const unsigned int seed)
{
    unsigned int p0 = 0u;
    unsigned int p1 = 0u;
    unsigned int p2 = 0u;
    unsigned int p3 = depth;
    unsigned int p4 = seed;
    optixTrace(
        handle,
        origin,
        direction,
        tmin,
        tmax,
        0.0f,
        OptixVisibilityMask(255),
        OPTIX_RAY_FLAG_DISABLE_ANYHIT,
        RAY_TYPE_RADIANCE,
        RAY_TYPE_COUNT,
        RAY_TYPE_RADIANCE,
        p0, p1, p2, p3, p4);
    return make_vec(__uint_as_float(p0), __uint_as_float(p1), __uint_as_float(p2));
}

static __forceinline__ __device__ unsigned int hash32(unsigned int x)
{
    x ^= x >> 16;
    x *= 0x7feb352du;
    x ^= x >> 15;
    x *= 0x846ca68bu;
    x ^= x >> 16;
    return x;
}

static __forceinline__ __device__ float random01(unsigned int& seed)
{
    seed = hash32(seed);
    return static_cast<float>(seed & 0x00ffffffu) / 16777216.0f;
}

static __forceinline__ __device__ float3 cosineHemisphereDirection(const float3 normal, unsigned int& seed)
{
    const float r1 = random01(seed);
    const float r2 = random01(seed);
    const float phi = 6.2831853f * r1;
    const float radius = sqrtf(r2);
    const float x = cosf(phi) * radius;
    const float z = sinf(phi) * radius;
    const float y = sqrtf(fmaxf(0.0f, 1.0f - r2));
    const float3 helper = fabsf(normal.y) < 0.999f ? make_vec(0.0f, 1.0f, 0.0f) : make_vec(1.0f, 0.0f, 0.0f);
    const float3 tangent = normalize3(cross3(helper, normal));
    const float3 bitangent = normalize3(cross3(normal, tangent));
    return normalize3(add3(add3(mul3(tangent, x), mul3(normal, y)), mul3(bitangent, z)));
}

static __forceinline__ __device__ int areaLightSampleCount()
{
    if (params.areaLightRadius <= 0.001f)
    {
        return 1;
    }
    if (params.renderQuality == RenderQualityLow)
    {
        return 1;
    }
    if (params.renderQuality == RenderQualityHigh || params.renderQuality == RenderQualityPathTracing)
    {
        return 4;
    }
    return 2;
}

static __forceinline__ __device__ float3 areaLightSamplePosition(const float3 hitPoint, const int sampleIndex, const int sampleCount)
{
    if (params.areaLightRadius <= 0.001f || sampleCount <= 1)
    {
        return params.lightPosition;
    }

    const float3 centerDir = normalize3(sub3(params.lightPosition, hitPoint));
    const float3 helper = fabsf(centerDir.y) < 0.999f ? make_vec(0.0f, 1.0f, 0.0f) : make_vec(1.0f, 0.0f, 0.0f);
    const float3 tangent = normalize3(cross3(helper, centerDir));
    const float3 bitangent = normalize3(cross3(centerDir, tangent));
    const float phaseRaw = sinf(dot3(hitPoint, make_vec(12.9898f, 78.233f, 37.719f))) * 43758.5453f;
    const float phase = phaseRaw - floorf(phaseRaw);
    const float angle = 6.2831853f * (phase + (static_cast<float>(sampleIndex) + 0.5f) * 0.6180339f);
    const float radius = params.areaLightRadius * sqrtf((static_cast<float>(sampleIndex) + 0.5f) / static_cast<float>(sampleCount));
    return add3(params.lightPosition, add3(mul3(tangent, cosf(angle) * radius), mul3(bitangent, sinf(angle) * radius)));
}

static __forceinline__ __device__ void evaluateDirectLight(
    const float3 hitPoint,
    const float3 normal,
    float3& lightDir,
    float& lightDistance,
    float& visibility,
    float& ndotl)
{
    const int sampleCount = areaLightSampleCount();
    float3 lightDirSum = make_vec(0.0f, 0.0f, 0.0f);
    float distanceSum = 0.0f;
    float visibilitySum = 0.0f;
    float ndotlSum = 0.0f;

    for (int i = 0; i < sampleCount; ++i)
    {
        const float3 samplePosition = areaLightSamplePosition(hitPoint, i, sampleCount);
        const float3 lightVector = sub3(samplePosition, hitPoint);
        const float distance = sqrtf(dot3(lightVector, lightVector));
        const float3 dir = distance > 0.0f ? mul3(lightVector, 1.0f / distance) : make_vec(0.0f, 0.0f, 0.0f);
        const bool visible = params.shadowEnabled == 0
            ? true
            : traceShadow(
                params.handle,
                add3(hitPoint, mul3(normal, 0.002f)),
                dir,
                0.001f,
                distance - 0.01f);
        lightDirSum = add3(lightDirSum, dir);
        distanceSum += distance;
        visibilitySum += visible ? 1.0f : 0.0f;
        ndotlSum += fmaxf(dot3(normal, dir), 0.0f);
    }

    const float invSamples = 1.0f / static_cast<float>(sampleCount);
    lightDir = normalize3(mul3(lightDirSum, invSamples));
    lightDistance = distanceSum * invSamples;
    visibility = visibilitySum * invSamples;
    ndotl = ndotlSum * invSamples;
}

static __forceinline__ __device__ float3 progressiveBounce(
    const float3 hitPoint,
    const float3 normal,
    const float3 rayDirection,
    const float3 baseColor,
    const MeshMaterialGpu material,
    const unsigned int depth)
{
    if (depth >= static_cast<unsigned int>(params.maxDepth))
    {
        return make_vec(0.0f, 0.0f, 0.0f);
    }

    unsigned int seed = optixGetPayload_4() ^ (depth * 0x9e3779b9u);
    float3 bounceDir{};
    float3 throughput = baseColor;
    if (material.materialType == MaterialMirror || material.materialType == MaterialMetal)
    {
        const float3 reflected = normalize3(reflect3(rayDirection, normal));
        const float3 diffuseLobe = cosineHemisphereDirection(normal, seed);
        bounceDir = roughReflectionDir(lerp3(reflected, diffuseLobe, saturate1(material.roughness)), normal, material.roughness);
        const float3 tint = material.materialType == MaterialMetal ? material.color : material.specularColor;
        throughput = make_vec(tint.x * baseColor.x, tint.y * baseColor.y, tint.z * baseColor.z);
    }
    else if (material.materialType == MaterialDielectric)
    {
        const float frontFace = dot3(rayDirection, normal) < 0.0f ? 1.0f : 0.0f;
        const float3 orientedNormal = frontFace > 0.5f ? normal : mul3(normal, -1.0f);
        const float eta = frontFace > 0.5f ? 1.0f / safeMaterialIor(material.ior) : safeMaterialIor(material.ior);
        const float cosTheta = saturate1(dot3(mul3(rayDirection, -1.0f), orientedNormal));
        const float fresnel = dielectricFresnel(cosTheta, material.ior);
        const float chooseReflection = random01(seed);
        float3 refractedDir{};
        if (chooseReflection < fresnel || !refract3(rayDirection, orientedNormal, eta, refractedDir))
        {
            bounceDir = roughReflectionDir(normalize3(reflect3(rayDirection, orientedNormal)), orientedNormal, material.roughness);
        }
        else
        {
            bounceDir = refractedDir;
        }
        throughput = lerp3(make_vec(1.0f, 1.0f, 1.0f), material.color, 0.22f);
    }
    else
    {
        bounceDir = cosineHemisphereDirection(normal, seed);
    }

    const float3 bounced = traceRadiance(
        params.handle,
        add3(hitPoint, mul3(normal, 0.003f)),
        bounceDir,
        0.001f,
        1e20f,
        depth + 1u,
        seed);
    return make_vec(bounced.x * throughput.x, bounced.y * throughput.y, bounced.z * throughput.z);
}

static __forceinline__ __device__ uchar4 toColor(const float3 color)
{
    const float3 c = postProcessColor(color);
    return make_uchar4(
        static_cast<unsigned char>(c.x * 255.0f),
        static_cast<unsigned char>(c.y * 255.0f),
        static_cast<unsigned char>(c.z * 255.0f),
        255);
}

)"
R"(
static __forceinline__ __device__ float3 shadeMaterial(
    const float3 hitPoint,
    float3 normal,
    const float2 texcoord,
    const float3 rayDirection,
    const unsigned int depth,
    const MeshMaterialGpu material)
{
    if (dot3(normal, rayDirection) > 0.0f)
    {
        normal = mul3(normal, -1.0f);
    }

    float3 lightDir{};
    float lightDistance = 0.0f;
    float visibility = 1.0f;
    float ndotl = 0.0f;
    evaluateDirectLight(hitPoint, normal, lightDir, lightDistance, visibility, ndotl);
    const float3 env = environmentColor(normal);

    const float ambient = 0.14f;
    const float diffuseShadowFloor = 0.34f;
    const float mirrorShadowFloor = 0.50f;
    const float roughness = safeMaterialRoughness(material.roughness);
    const float shadowFloor = material.materialType == MaterialMirror || material.materialType == MaterialMetal || material.materialType == MaterialDielectric
        ? mirrorShadowFloor
        : diffuseShadowFloor;
    const float shadowFactor = shadowFloor + (1.0f - shadowFloor) * visibility;
    const float diffuse = ndotl * shadowFactor * params.lightIntensity;
    const float3 viewDir = mul3(rayDirection, -1.0f);
    const float3 halfDir = normalize3(add3(lightDir, viewDir));
    const float specularPower = fmaxf(8.0f, 160.0f * (1.0f - roughness));
    const float specular = visibility * powf(fmaxf(dot3(normal, halfDir), 0.0f), specularPower) * params.lightIntensity;
    const bool reflectiveMaterial = material.materialType == MaterialMirror || material.materialType == MaterialMetal || material.materialType == MaterialDielectric;
    const float diffuseSpecularWeight = reflectiveMaterial ? 0.0f : 0.06f;
    const float diffuseLightWeight = reflectiveMaterial ? 1.0f : 0.88f;
    const float3 baseColor = sampleDiffuseTexture(material, texcoord);
    const float3 diffuseColor = reflectiveMaterial
        ? baseColor
        : clamp3(baseColor, 0.0f, 0.96f);
    const float metallic = material.materialType == MaterialMetal ? 1.0f : 0.0f;
    const float mirrorGgxBoost = material.materialType == MaterialMirror ? 1.35f : 1.0f;
    const float3 ggxLight = mul3(
        ggxDirectLight(diffuseColor, material.specularColor, normal, viewDir, lightDir, roughness, metallic),
        shadowFactor * params.lightIntensity * 1.35f * mirrorGgxBoost);

    float3 localColor = add3(
        add3(mul3(diffuseColor, ambient + diffuseLightWeight * diffuse * 0.32f), ggxLight),
        mul3(make_vec(1.0f, 1.0f, 1.0f), diffuseSpecularWeight * specular));
    localColor = add3(localColor, mul3(make_vec(diffuseColor.x * env.x, diffuseColor.y * env.y, diffuseColor.z * env.z), 0.10f));

    if (material.materialType == MaterialMirror)
    {
        if (depth < static_cast<unsigned int>(params.maxDepth))
        {
            const float3 reflectedDir = roughReflectionDir(normalize3(reflect3(rayDirection, normal)), normal, roughness);
            const float3 reflectedColor = traceRadiance(
                params.handle,
                add3(hitPoint, mul3(normal, 0.002f)),
                reflectedDir,
                0.001f,
                1e20f,
                depth + 1u,
                optixGetPayload_4() ^ 0x9e3779b9u);
            const float mirrorHighlight = 0.55f;
            const float3 highlight = mul3(make_vec(1.0f, 1.0f, 1.0f), mirrorHighlight * specular);
            localColor = add3(reflectedColor, highlight);
        }
        else
        {
            localColor = make_vec(0.231f, 0.251f, 0.251f);
        }
    }
    else if (material.materialType == MaterialMetal)
    {
        if (depth < static_cast<unsigned int>(params.maxDepth))
        {
            const float3 reflectedDir = roughReflectionDir(normalize3(reflect3(rayDirection, normal)), normal, roughness);
            const float3 reflectedColor = traceRadiance(
                params.handle,
                add3(hitPoint, mul3(normal, 0.002f)),
                reflectedDir,
                0.001f,
                1e20f,
                depth + 1u,
                optixGetPayload_4() ^ 0x85ebca6bu);
            const float3 specularTint = lerp3(material.specularColor, material.color, 0.65f);
            const float reflectionWeight = 0.65f + 0.25f * (1.0f - roughness);
            localColor = add3(
                mul3(make_vec(reflectedColor.x * specularTint.x, reflectedColor.y * specularTint.y, reflectedColor.z * specularTint.z), reflectionWeight),
                mul3(material.color, (ambient + diffuse * 0.28f) * roughness));
        }
    }
    else if (material.materialType == MaterialDielectric)
    {
        if (depth < static_cast<unsigned int>(params.maxDepth))
        {
            const float frontFace = dot3(rayDirection, normal) < 0.0f ? 1.0f : 0.0f;
            const float eta = frontFace > 0.5f ? 1.0f / safeMaterialIor(material.ior) : safeMaterialIor(material.ior);
            const float3 orientedNormal = frontFace > 0.5f ? normal : mul3(normal, -1.0f);
            const float cosTheta = saturate1(dot3(mul3(rayDirection, -1.0f), orientedNormal));
            const float fresnel = dielectricFresnel(cosTheta, material.ior);

            const float3 reflectedDir = roughReflectionDir(normalize3(reflect3(rayDirection, orientedNormal)), orientedNormal, roughness);
            const float3 reflectedColor = traceRadiance(
                params.handle,
                add3(hitPoint, mul3(orientedNormal, 0.002f)),
                reflectedDir,
                0.001f,
                1e20f,
                depth + 1u,
                optixGetPayload_4() ^ 0xc2b2ae35u);

            float3 transmittedColor = localColor;
            float3 refractedDir{};
            const bool refracted = refract3(rayDirection, orientedNormal, eta, refractedDir);
            if (refracted)
            {
                transmittedColor = traceRadiance(
                    params.handle,
                    sub3(hitPoint, mul3(orientedNormal, 0.002f)),
                    refractedDir,
                    0.001f,
                    1e20f,
                    depth + 1u,
                    optixGetPayload_4() ^ 0x27d4eb2fu);
            }

            const float reflectionMix = refracted ? fresnel : 1.0f;
            const float opacity = saturate1(material.alpha);
            const float3 tint = lerp3(make_vec(1.0f, 1.0f, 1.0f), material.color, 0.35f);
            const float3 glassColor = lerp3(
                make_vec(transmittedColor.x * tint.x, transmittedColor.y * tint.y, transmittedColor.z * tint.z),
                reflectedColor,
                fminf(1.0f, reflectionMix + roughness * 0.18f));
            localColor = lerp3(glassColor, localColor, opacity * 0.18f);
        }
    }

    if (params.renderMode == RenderModeProgressive)
    {
        const float3 indirect = progressiveBounce(hitPoint, normal, rayDirection, diffuseColor, material, depth);
        localColor = add3(mul3(localColor, 0.72f), mul3(indirect, 0.28f));
    }

    return localColor;
}

)"
R"(
extern "C" __global__ void __raygen__rg()
{
    const uint3 idx = optixGetLaunchIndex();
    const uint3 dim = optixGetLaunchDimensions();

    const float2 offsets[4] = {
        make_float2(0.25f, 0.25f),
        make_float2(0.75f, 0.25f),
        make_float2(0.25f, 0.75f),
        make_float2(0.75f, 0.75f)
    };
    const int primarySampleCount = params.samplesPerPixel < 1
        ? 1
        : (params.samplesPerPixel > 8 ? 8 : params.samplesPerPixel);

    float3 color = make_vec(0.0f, 0.0f, 0.0f);
    for (int sample = 0; sample < primarySampleCount; ++sample)
    {
        unsigned int seed = hash32(idx.x + idx.y * dim.x + params.accumulationSample * 9781u + static_cast<unsigned int>(sample) * 6271u);
        const float jitterX = params.renderMode == RenderModeProgressive ? random01(seed) : offsets[sample].x;
        const float jitterY = params.renderMode == RenderModeProgressive ? random01(seed) : offsets[sample].y;
        const float u = (static_cast<float>(idx.x) + jitterX) / static_cast<float>(dim.x);
        const float v = (static_cast<float>(idx.y) + jitterY) / static_cast<float>(dim.y);

        const float px = (2.0f * u - 1.0f) * params.cameraAspect * params.cameraScale;
        const float py = (2.0f * v - 1.0f) * params.cameraScale;

        const float3 direction = normalize3(
            add3(
                add3(params.cameraForward, mul3(params.cameraRight, px)),
                mul3(params.cameraUp, py)));

        color = add3(color, traceRadiance(
            params.handle,
            params.cameraPosition,
            direction,
            0.001f,
            1e20f,
            0u,
            seed));
    }
    color = mul3(color, 1.0f / static_cast<float>(primarySampleCount));

    const unsigned int pixelIndex = idx.y * params.imageWidth + idx.x;
    if (params.renderMode == RenderModeProgressive && params.accumulation != nullptr)
    {
        const float sampleCount = static_cast<float>(params.accumulationSample);
        const float invCount = 1.0f / (sampleCount + 1.0f);
        const float4 previous = params.accumulation[pixelIndex];
        const float3 accumulated = make_vec(
            (previous.x * sampleCount + color.x) * invCount,
            (previous.y * sampleCount + color.y) * invCount,
            (previous.z * sampleCount + color.z) * invCount);
        params.accumulation[pixelIndex] = make_float4(accumulated.x, accumulated.y, accumulated.z, 1.0f);
        params.image[pixelIndex] = toColor(accumulated);
    }
    else
    {
        params.image[pixelIndex] = toColor(color);
    }

}

extern "C" __global__ void __miss__radiance()
{
    const float3 rayDir = normalize3(optixGetWorldRayDirection());
    setRadiancePayload(environmentColor(rayDir));
}

extern "C" __global__ void __miss__shadow()
{
    setShadowPayload(true);
}

extern "C" __global__ void __closesthit__radiance()
{
    const unsigned int depth = optixGetPayload_3();
    const unsigned int primitiveIndex = optixGetPrimitiveIndex();
    const OptixTraversableHandle gasHandle = optixGetGASTraversableHandle();

    const float3 rayOrigin = optixGetWorldRayOrigin();
    const float3 rayDirection = normalize3(optixGetWorldRayDirection());
    const float tHit = optixGetRayTmax();
    const float3 hitPoint = add3(rayOrigin, mul3(rayDirection, tHit));

    float4 sphereData;
    optixGetSphereData(gasHandle, primitiveIndex, 0u, 0.0f, &sphereData);

    const float3 objectPoint = optixTransformPointFromWorldToObjectSpace(hitPoint);
    float3 normal = normalize3(
        optixTransformNormalFromObjectToWorldSpace(
            sub3(objectPoint, make_vec(sphereData.x, sphereData.y, sphereData.z))));

    SphereMaterial material = params.materials[primitiveIndex];
    MeshMaterialGpu materialGpu{};
    materialGpu.color = material.color;
    materialGpu.materialType = material.materialType;
    materialGpu.specularColor = material.specularColor;
    materialGpu.roughness = material.roughness;
    materialGpu.ior = material.ior;
    materialGpu.alpha = material.alpha;
    setRadiancePayload(shadeMaterial(hitPoint, normal, make_float2(0.0f, 0.0f), rayDirection, depth, materialGpu));
}

extern "C" __global__ void __closesthit__shadow()
{
    setShadowPayload(false);
}

extern "C" __global__ void __closesthit__radiance_plane()
{
    const float3 rayOrigin = optixGetWorldRayOrigin();
    const float3 rayDirection = normalize3(optixGetWorldRayDirection());
    const float tHit = optixGetRayTmax();
    const float3 hitPoint = add3(rayOrigin, mul3(rayDirection, tHit));
    const float3 normal = make_vec(0.0f, 1.0f, 0.0f);

    float3 lightDir{};
    float lightDistance = 0.0f;
    float visibility = 1.0f;
    float ndotl = 0.0f;
    evaluateDirectLight(hitPoint, normal, lightDir, lightDistance, visibility, ndotl);

    const float ambient = 0.18f;
    const float planeShadowFloor = 0.32f;
    const float shadowFactor = planeShadowFloor + (1.0f - planeShadowFloor) * visibility;
    const float diffuse = ndotl * shadowFactor;
    const float3 viewDir = mul3(rayDirection, -1.0f);
    const float3 halfDir = normalize3(add3(lightDir, viewDir));
    const float specular = visibility * powf(fmaxf(dot3(normal, halfDir), 0.0f), 56.0f);

    const float3 baseColor = make_vec(0.95f, 0.95f, 0.96f);
    float3 localColor = add3(
        mul3(baseColor, ambient + 0.85f * diffuse),
        mul3(make_vec(1.0f, 1.0f, 1.0f), 0.20f * specular));

    const float3 toCamera = sub3(hitPoint, params.cameraPosition);
    const float distanceToCamera = sqrtf(dot3(toCamera, toCamera));
    const float fog = saturate1((distanceToCamera - 18.0f) / 70.0f);
    const float fogCurve = fog * fog * (3.0f - 2.0f * fog);
    const float3 fadeToSky = environmentColor(normalize3(sub3(hitPoint, params.cameraPosition)));
    localColor = lerp3(localColor, fadeToSky, 0.30f * fogCurve);

    setRadiancePayload(localColor);
}

extern "C" __global__ void __closesthit__shadow_plane()
{
    setShadowPayload(false);
}

extern "C" __global__ void __closesthit__radiance_mesh()
{
    const unsigned int depth = optixGetPayload_3();
    const unsigned int primitiveIndex = optixGetPrimitiveIndex();
    const unsigned int instanceId = optixGetInstanceId();
    const unsigned int meshObjectIndex = instanceId >= 2u ? instanceId - 2u : 0u;
    MeshObjectGpu object{};
    if (params.meshObjects != nullptr && meshObjectIndex < params.meshObjectCount)
    {
        object = params.meshObjects[meshObjectIndex];
    }

    const MeshTriangleGpu triangle = params.meshTriangles[object.triangleOffset + primitiveIndex];
    const MeshVertexGpu v0 = params.meshVertices[triangle.i0];
    const MeshVertexGpu v1 = params.meshVertices[triangle.i1];
    const MeshVertexGpu v2 = params.meshVertices[triangle.i2];

    const float2 bary = optixGetTriangleBarycentrics();
    const float w0 = 1.0f - bary.x - bary.y;
    const float w1 = bary.x;
    const float w2 = bary.y;
    float3 normal = normalize3(add3(add3(mul3(v0.normal, w0), mul3(v1.normal, w1)), mul3(v2.normal, w2)));
    normal = normalize3(optixTransformNormalFromObjectToWorldSpace(normal));
    float3 tangent = normalize3(add3(add3(mul3(v0.tangent, w0), mul3(v1.tangent, w1)), mul3(v2.tangent, w2)));
    tangent = normalize3(optixTransformVectorFromObjectToWorldSpace(tangent));
    const int hasTexcoord = v0.hasTexcoord != 0 && v1.hasTexcoord != 0 && v2.hasTexcoord != 0 ? 1 : 0;
    const float2 texcoord = make_float2(
        v0.texcoord.x * w0 + v1.texcoord.x * w1 + v2.texcoord.x * w2,
        v0.texcoord.y * w0 + v1.texcoord.y * w1 + v2.texcoord.y * w2);

    const float3 rayOrigin = optixGetWorldRayOrigin();
    const float3 rayDirection = normalize3(optixGetWorldRayDirection());
    const float tHit = optixGetRayTmax();
    const float3 hitPoint = add3(rayOrigin, mul3(rayDirection, tHit));

    MeshMaterialGpu material = params.meshMaterials[0];
    if (triangle.materialIndex < params.meshMaterialCount)
    {
        material = params.meshMaterials[triangle.materialIndex];
    }
    normal = sampleNormalTexture(material, texcoord, normal, tangent, hasTexcoord);

    setRadiancePayload(shadeMaterial(hitPoint, normal, texcoord, rayDirection, depth, material));
}

extern "C" __global__ void __closesthit__shadow_mesh()
{
    setShadowPayload(false);
}
)";
}
