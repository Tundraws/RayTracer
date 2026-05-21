#pragma once

#include <cuda_runtime.h>
#include <optix.h>

enum MaterialType
{
    MaterialDiffuse = 0,
    MaterialMirror = 1,
    MaterialMetal = 2,
    MaterialDielectric = 3
};

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
