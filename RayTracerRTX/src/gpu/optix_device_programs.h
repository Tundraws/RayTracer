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

    const float u = (static_cast<float>(idx.x) + 0.5f) / static_cast<float>(dim.x);
    const float v = (static_cast<float>(idx.y) + 0.5f) / static_cast<float>(dim.y);

    const float px = (2.0f * u - 1.0f) * params.cameraAspect * params.cameraScale;
    const float py = (2.0f * v - 1.0f) * params.cameraScale;

    const float3 direction = normalize3(
        add3(
            add3(params.cameraForward, mul3(params.cameraRight, px)),
            mul3(params.cameraUp, py)));

    const float3 color = traceRadiance(
        params.handle,
        params.cameraPosition,
        direction,
        0.001f,
        1e20f,
        0u);

    params.image[idx.y * params.imageWidth + idx.x] = toColor(color);
}

extern "C" __global__ void __miss__radiance()
{
    const float3 rayDir = normalize3(optixGetWorldRayDirection());
    const float t = clamp3(make_vec(0.5f * (rayDir.y + 1.0f), 0.0f, 0.0f), 0.0f, 1.0f).x;
    const float3 horizon = make_vec(0.96f, 0.84f, 0.70f);
    const float3 zenith = make_vec(0.24f, 0.44f, 0.78f);
    const float3 sky = add3(mul3(horizon, 1.0f - t), mul3(zenith, t));

    const float3 sunDir = normalize3(make_vec(-0.35f, 0.82f, -0.45f));
    const float sunAmount = powf(fmaxf(dot3(rayDir, sunDir), 0.0f), 96.0f);
    const float glowAmount = powf(fmaxf(dot3(rayDir, sunDir), 0.0f), 12.0f);
    const float3 sunGlow = add3(
        mul3(make_vec(1.00f, 0.82f, 0.58f), 0.55f * glowAmount),
        mul3(make_vec(1.00f, 0.96f, 0.88f), 0.85f * sunAmount));

    setRadiancePayload(add3(sky, sunGlow));
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
    const float3 lightVector = sub3(params.lightPosition, hitPoint);
    const float lightDistance = sqrtf(dot3(lightVector, lightVector));
    const float3 lightDir = lightDistance > 0.0f ? mul3(lightVector, 1.0f / lightDistance) : make_vec(0.0f, 0.0f, 0.0f);

    const bool visible = traceShadow(
        params.handle,
        add3(hitPoint, mul3(normal, 0.002f)),
        lightDir,
        0.001f,
        lightDistance - 0.01f);

    const float ambient = 0.15f;
    const float diffuse = visible ? fmaxf(dot3(normal, lightDir), 0.0f) : 0.0f;
    const float3 viewDir = mul3(rayDirection, -1.0f);
    const float3 halfDir = normalize3(add3(lightDir, viewDir));
    const float specular = visible ? powf(fmaxf(dot3(normal, halfDir), 0.0f), 48.0f) : 0.0f;

    float3 localColor = add3(
        mul3(material.color, ambient + diffuse),
        mul3(make_vec(1.0f, 1.0f, 1.0f), 0.35f * specular));

    if (material.materialType == MaterialMirror && depth < static_cast<unsigned int>(params.maxDepth))
    {
        const float3 reflectedDir = normalize3(reflect3(rayDirection, normal));
        const float3 reflectedColor = traceRadiance(
            params.handle,
            add3(hitPoint, mul3(normal, 0.002f)),
            reflectedDir,
            0.001f,
            1e20f,
            depth + 1u);
        localColor = add3(mul3(localColor, 0.15f), mul3(reflectedColor, 0.85f));
    }

    setRadiancePayload(localColor);
}

extern "C" __global__ void __closesthit__shadow()
{
    setShadowPayload(false);
}
)";
}
