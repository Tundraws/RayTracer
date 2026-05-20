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

static __forceinline__ __device__ float hashToUnit(const unsigned int seed)
{
    unsigned int x = seed;
    x ^= x >> 16;
    x *= 0x7feb352du;
    x ^= x >> 15;
    x *= 0x846ca68bu;
    x ^= x >> 16;
    return static_cast<float>(x & 0x00ffffffu) / 16777215.0f;
}

static __forceinline__ __device__ float2 areaLightSample(const int index, const float3 hitPoint)
{
    const int gridSize = 8;
    const int sx = index % gridSize;
    const int sy = index / gridSize;
    const unsigned int baseSeed =
        static_cast<unsigned int>(fabsf(hitPoint.x) * 37.0f) ^
        (static_cast<unsigned int>(fabsf(hitPoint.y) * 53.0f) << 8) ^
        (static_cast<unsigned int>(fabsf(hitPoint.z) * 71.0f) << 16);
    const float jx = hashToUnit(baseSeed + static_cast<unsigned int>(index * 2 + 1));
    const float jy = hashToUnit(baseSeed + static_cast<unsigned int>(index * 2 + 2));
    return make_float2(
        (static_cast<float>(sx) + jx) / static_cast<float>(gridSize) - 0.5f,
        (static_cast<float>(sy) + jy) / static_cast<float>(gridSize) - 0.5f);
}

static __forceinline__ __device__ float3 lightSamplePosition(const float3 hitPoint, const int sampleIndex)
{
    if (params.lightType != LightArea)
    {
        return params.lightPosition;
    }

    const float3 toLight = normalize3(sub3(params.lightPosition, hitPoint));
    float3 tangent = normalize3(cross3(make_vec(0.0f, 1.0f, 0.0f), toLight));
    if (dot3(tangent, tangent) <= 0.000001f)
    {
        tangent = make_vec(1.0f, 0.0f, 0.0f);
    }
    const float3 bitangent = normalize3(cross3(toLight, tangent));
    const float2 sample = areaLightSample(sampleIndex, hitPoint);
    return add3(
        params.lightPosition,
        add3(
            mul3(tangent, sample.x * params.lightRadius),
            mul3(bitangent, sample.y * params.lightRadius)));
}

static __forceinline__ __device__ float evaluateVisibility(
    const OptixTraversableHandle handle,
    const float3 hitPoint,
    const float3 normal,
    float3* outLightDir,
    float* outNdotL)
{
    const int sampleCount = params.lightType == LightArea ? 64 : 1;
    float visibility = 0.0f;
    float3 lightDirSum = make_vec(0.0f, 0.0f, 0.0f);
    float ndotlSum = 0.0f;

    for (int i = 0; i < sampleCount; ++i)
    {
        const float3 samplePosition = lightSamplePosition(hitPoint, i);
        const float3 lightVector = sub3(samplePosition, hitPoint);
        const float lightDistance = sqrtf(dot3(lightVector, lightVector));
        const float3 lightDir = lightDistance > 0.0f ? mul3(lightVector, 1.0f / lightDistance) : make_vec(0.0f, 0.0f, 0.0f);
        const float ndotl = fmaxf(dot3(normal, lightDir), 0.0f);

        const bool visible = traceShadow(
            handle,
            add3(hitPoint, mul3(normal, 0.002f)),
            lightDir,
            0.001f,
            lightDistance - 0.01f);

        if (visible)
        {
            visibility += 1.0f;
            lightDirSum = add3(lightDirSum, lightDir);
            ndotlSum += ndotl;
        }
    }

    const float invSamples = 1.0f / static_cast<float>(sampleCount);
    *outLightDir = visibility > 0.0f ? normalize3(lightDirSum) : normalize3(sub3(params.lightPosition, hitPoint));
    const float rawVisibility = visibility * invSamples;
    const float smoothedVisibility = rawVisibility * rawVisibility * (3.0f - 2.0f * rawVisibility);
    *outNdotL = ndotlSum * invSamples * (params.lightType == LightArea ? 1.08f : 1.0f);
    return smoothedVisibility;
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
    const float3 c = clamp3(color, 0.0f, 1.0f);
    return make_uchar4(
        static_cast<unsigned char>(c.x * 255.0f),
        static_cast<unsigned char>(c.y * 255.0f),
        static_cast<unsigned char>(c.z * 255.0f),
        255);
}

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

    float3 color = make_vec(0.0f, 0.0f, 0.0f);
    for (int sample = 0; sample < 4; ++sample)
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
    color = mul3(color, 0.25f);

    params.image[idx.y * params.imageWidth + idx.x] = toColor(color);

}

extern "C" __global__ void __miss__radiance()
{
    const float3 rayDir = normalize3(optixGetWorldRayDirection());
    const float t = saturate1(0.5f * (rayDir.y + 1.0f));
    const float tt = t * t;
    const float3 horizon = make_vec(0.231f, 0.251f, 0.251f);
    const float3 zenith = make_vec(0.333f, 0.341f, 0.341f);
    const float3 sky = add3(mul3(horizon, 1.0f - tt), mul3(zenith, tt));
    setRadiancePayload(sky);
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

    if (dot3(normal, rayDirection) > 0.0f)
    {
        normal = mul3(normal, -1.0f);
    }

    SphereMaterial material = params.materials[primitiveIndex];
    float3 lightDir = make_vec(0.0f, 1.0f, 0.0f);
    float ndotl = 0.0f;
    const float visibility = evaluateVisibility(params.handle, hitPoint, normal, &lightDir, &ndotl);

    const float ambient = 0.18f;
    const float diffuseShadowFloor = params.lightType == LightArea ? 0.32f : 0.48f;
    const float mirrorShadowFloor = params.lightType == LightArea ? 0.50f : 0.62f;
    const float shadowFloor = material.materialType == MaterialMirror ? mirrorShadowFloor : diffuseShadowFloor;
    const float shadowFactor = shadowFloor + (1.0f - shadowFloor) * visibility;
    const float diffuse = ndotl * shadowFactor;
    const float3 viewDir = mul3(rayDirection, -1.0f);
    const float3 halfDir = normalize3(add3(lightDir, viewDir));
    const float specularPower = material.materialType == MaterialMirror ? 96.0f : 128.0f;
    const float specular = visibility * powf(fmaxf(dot3(normal, halfDir), 0.0f), specularPower);
    const float diffuseSpecularWeight = material.materialType == MaterialMirror ? 0.0f : 0.06f;
    const float diffuseLightWeight = material.materialType == MaterialMirror ? 1.0f : 0.72f;
    const float3 diffuseColor = material.materialType == MaterialMirror
        ? material.color
        : clamp3(material.color, 0.0f, 0.86f);

    float3 localColor = add3(
        mul3(diffuseColor, ambient + diffuseLightWeight * diffuse),
        mul3(make_vec(1.0f, 1.0f, 1.0f), diffuseSpecularWeight * specular));

    if (material.materialType == MaterialMirror)
    {
        if (depth < static_cast<unsigned int>(params.maxDepth))
        {
            const float3 reflectedDir = normalize3(reflect3(rayDirection, normal));
            const float3 reflectedColor = traceRadiance(
                params.handle,
                add3(hitPoint, mul3(normal, 0.002f)),
                reflectedDir,
                0.001f,
                1e20f,
                depth + 1u);
            const float mirrorHighlight = params.lightType == LightArea ? 0.32f : 0.55f;
            const float3 highlight = mul3(make_vec(1.0f, 1.0f, 1.0f), mirrorHighlight * specular);
            localColor = add3(reflectedColor, highlight);
        }
        else
        {
            localColor = make_vec(0.231f, 0.251f, 0.251f);
        }
    }

    setRadiancePayload(localColor);
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

    float3 lightDir = make_vec(0.0f, 1.0f, 0.0f);
    float ndotl = 0.0f;
    const float visibility = evaluateVisibility(params.handle, hitPoint, normal, &lightDir, &ndotl);

    const float ambient = 0.18f;
    const float planeShadowFloor = params.lightType == LightArea ? 0.38f : 0.55f;
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
    const float3 fadeToSky = make_vec(0.231f, 0.251f, 0.251f);
    localColor = lerp3(localColor, fadeToSky, 0.30f * fogCurve);

    setRadiancePayload(localColor);
}

extern "C" __global__ void __closesthit__shadow_plane()
{
    setShadowPayload(false);
}
)";
}
