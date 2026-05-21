#pragma once

#include "../common/rtx_shared.h"

#include <cstdint>
#include <string>
#include <vector>

struct MeshVertex
{
    float3 position{};
    float3 normal{};
    float2 texcoord{};
    float3 tangent{};
    int hasTexcoord = 0;
};

struct MeshTriangle
{
    std::uint32_t i0 = 0;
    std::uint32_t i1 = 0;
    std::uint32_t i2 = 0;
    std::uint32_t materialIndex = 0;
};

struct MeshMaterial
{
    float3 color{};
    int materialType = MaterialDiffuse;
    std::string name;
    float3 specularColor{1.0f, 1.0f, 1.0f};
    float roughness = 0.35f;
    float ior = 1.5f;
    float alpha = 1.0f;
    std::string texturePath;
    int textureIndex = -1;
    std::string normalTexturePath;
    int normalTextureIndex = -1;
};

struct MeshTexture
{
    std::string path;
    unsigned int width = 0;
    unsigned int height = 0;
    std::vector<uchar4> pixels;
};

struct MeshData
{
    std::vector<MeshVertex> vertices;
    std::vector<MeshTriangle> triangles;
    std::vector<MeshMaterial> materials;
    std::vector<MeshTexture> textures;
};

bool isEmptyMesh(const MeshData& mesh);
bool hasValidMeshMaterialIndices(const MeshData& mesh);
