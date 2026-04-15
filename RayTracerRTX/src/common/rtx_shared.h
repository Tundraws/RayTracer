#pragma once

#include <cuda_runtime.h>
#include <optix.h>

enum MaterialType
{
    MaterialDiffuse = 0,
    MaterialMirror = 1
};

struct SphereMaterial
{
    float3 color;
    int materialType;
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
    int maxDepth;
};
