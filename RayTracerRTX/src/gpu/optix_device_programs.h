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
    const float cosi = fminf(fmaxf(dot3(i, n), -1.0f), 1.0f);
    const float k = 1.0f - eta * eta * (1.0f - cosi * cosi);
    if (k < 0.0f)
    {
        return false;
    }
    refracted = normalize3(add3(mul3(i, eta), mul3(n, eta * -cosi - sqrtf(k))));
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

static __forceinline__ __device__ float3 environmentColor(const float3 rayDir)
{
    const float t = saturate1(0.5f * (rayDir.y + 1.0f));
    const float horizonGlow = expf(-6.0f * fabsf(rayDir.y));
    const float sun = powf(fmaxf(dot3(rayDir, normalize3(make_vec(0.35f, 0.55f, -0.75f))), 0.0f), 96.0f);
    const float3 ground = make_vec(0.20f, 0.22f, 0.21f);
    const float3 horizon = make_vec(0.62f, 0.70f, 0.78f);
    const float3 zenith = make_vec(0.12f, 0.18f, 0.30f);
    const float3 sky = lerp3(horizon, zenith, t * t);
    const float3 base = rayDir.y < 0.0f ? lerp3(ground, horizon, saturate1(rayDir.y + 1.0f)) : sky;
    return add3(add3(base, mul3(make_vec(0.95f, 0.74f, 0.42f), 0.18f * horizonGlow)), mul3(make_vec(1.0f, 0.86f, 0.58f), 1.2f * sun));
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
    return gammaCorrectDevice(reinhardToneMapDevice(clamp3(color, 0.0f, 16.0f)));
}

static __forceinline__ __device__ float fresnelSchlick(const float cosTheta, const float f0)
{
    const float m = saturate1(1.0f - cosTheta);
    const float m2 = m * m;
    return f0 + (1.0f - f0) * m2 * m2 * m;
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
    const float blend = saturate1(roughness * roughness);
    return normalize3(lerp3(reflectedDir, normal, 0.45f * blend));
}

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
    const unsigned int depth)
{
    unsigned int p0 = 0u;
    unsigned int p1 = 0u;
    unsigned int p2 = 0u;
    unsigned int p3 = depth;
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
        p0, p1, p2, p3);
    return make_vec(__uint_as_float(p0), __uint_as_float(p1), __uint_as_float(p2));
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

    const float3 lightVector = sub3(params.lightPosition, hitPoint);
    const float lightDistance = sqrtf(dot3(lightVector, lightVector));
    const float3 lightDir = lightDistance > 0.0f ? mul3(lightVector, 1.0f / lightDistance) : make_vec(0.0f, 0.0f, 0.0f);

    const bool visible = traceShadow(
        params.handle,
        add3(hitPoint, mul3(normal, 0.002f)),
        lightDir,
        0.001f,
        lightDistance - 0.01f);
    const float visibility = visible ? 1.0f : 0.0f;
    const float3 env = environmentColor(normal);

    const float ambient = 0.14f;
    const float diffuseShadowFloor = 0.34f;
    const float mirrorShadowFloor = 0.50f;
    const float roughness = fmaxf(saturate1(material.roughness), 0.02f);
    const float shadowFloor = material.materialType == MaterialMirror || material.materialType == MaterialMetal || material.materialType == MaterialDielectric
        ? mirrorShadowFloor
        : diffuseShadowFloor;
    const float shadowFactor = shadowFloor + (1.0f - shadowFloor) * visibility;
    const float ndotl = fmaxf(dot3(normal, lightDir), 0.0f);
    const float diffuse = ndotl * shadowFactor;
    const float3 viewDir = mul3(rayDirection, -1.0f);
    const float3 halfDir = normalize3(add3(lightDir, viewDir));
    const float specularPower = fmaxf(8.0f, 160.0f * (1.0f - roughness));
    const float specular = visibility * powf(fmaxf(dot3(normal, halfDir), 0.0f), specularPower);
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
        shadowFactor * 1.35f * mirrorGgxBoost);

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
                depth + 1u);
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
                depth + 1u);
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
            const float eta = frontFace > 0.5f ? 1.0f / fmaxf(material.ior, 1.01f) : fmaxf(material.ior, 1.01f);
            const float3 orientedNormal = frontFace > 0.5f ? normal : mul3(normal, -1.0f);
            const float cosTheta = fminf(dot3(mul3(rayDirection, -1.0f), orientedNormal), 1.0f);
            const float f0 = (material.ior - 1.0f) / (material.ior + 1.0f);
            const float fresnel = fresnelSchlick(cosTheta, f0 * f0);

            const float3 reflectedDir = roughReflectionDir(normalize3(reflect3(rayDirection, orientedNormal)), orientedNormal, roughness);
            const float3 reflectedColor = traceRadiance(
                params.handle,
                add3(hitPoint, mul3(orientedNormal, 0.002f)),
                reflectedDir,
                0.001f,
                1e20f,
                depth + 1u);

            float3 transmittedColor = localColor;
            float3 refractedDir{};
            if (refract3(rayDirection, orientedNormal, eta, refractedDir))
            {
                transmittedColor = traceRadiance(
                    params.handle,
                    sub3(hitPoint, mul3(orientedNormal, 0.002f)),
                    refractedDir,
                    0.001f,
                    1e20f,
                    depth + 1u);
            }

            const float opacity = saturate1(material.alpha);
            const float3 tint = lerp3(make_vec(1.0f, 1.0f, 1.0f), material.color, 0.35f);
            const float3 glassColor = lerp3(
                make_vec(transmittedColor.x * tint.x, transmittedColor.y * tint.y, transmittedColor.z * tint.z),
                reflectedColor,
                fminf(1.0f, fresnel + roughness * 0.25f));
            localColor = lerp3(glassColor, localColor, opacity * 0.18f);
        }
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
    const int primarySampleCount = 4;

    float3 color = make_vec(0.0f, 0.0f, 0.0f);
    for (int sample = 0; sample < primarySampleCount; ++sample)
    {
        const float u = (static_cast<float>(idx.x) + offsets[sample].x) / static_cast<float>(dim.x);
        const float v = (static_cast<float>(idx.y) + offsets[sample].y) / static_cast<float>(dim.y);

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
            0u));
    }
    color = mul3(color, 1.0f / static_cast<float>(primarySampleCount));

    params.image[idx.y * params.imageWidth + idx.x] = toColor(color);

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

    const float3 lightVector = sub3(params.lightPosition, hitPoint);
    const float lightDistance = sqrtf(dot3(lightVector, lightVector));
    const float3 lightDir = lightDistance > 0.0f ? mul3(lightVector, 1.0f / lightDistance) : make_vec(0.0f, 0.0f, 0.0f);

    const bool visible = traceShadow(
        params.handle,
        add3(hitPoint, mul3(normal, 0.002f)),
        lightDir,
        0.001f,
        lightDistance - 0.01f);
    const float visibility = visible ? 1.0f : 0.0f;

    const float ambient = 0.18f;
    const float planeShadowFloor = 0.32f;
    const float shadowFactor = planeShadowFloor + (1.0f - planeShadowFloor) * visibility;
    const float ndotl = fmaxf(dot3(normal, lightDir), 0.0f);
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
