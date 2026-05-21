#pragma once

#include "../common/rtx_shared.h"

#include <cstdint>
#include <string>
#include <vector>

struct MeshVertex
{
    float3 position{};
    float3 normal{};
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
};

struct MeshData
{
    std::vector<MeshVertex> vertices;
    std::vector<MeshTriangle> triangles;
    std::vector<MeshMaterial> materials;
};

bool isEmptyMesh(const MeshData& mesh);
bool hasValidMeshMaterialIndices(const MeshData& mesh);
