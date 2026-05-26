#include "mesh.h"

namespace
{
MeshMaterial makePrimitiveMaterial(const char* name, const float3 color)
{
    MeshMaterial material;
    material.name = name;
    material.color = color;
    material.specularColor = make_float3(0.6f, 0.6f, 0.6f);
    material.roughness = 0.48f;
    material.ior = 1.5f;
    material.alpha = 1.0f;
    return material;
}

void addVertex(MeshData& mesh, const float3 position, const float3 normal, const float2 uv, const float3 tangent)
{
    MeshVertex vertex;
    vertex.position = position;
    vertex.normal = normal;
    vertex.texcoord = uv;
    vertex.tangent = tangent;
    vertex.hasTexcoord = 1;
    mesh.vertices.push_back(vertex);
}

void addQuad(MeshData& mesh, const float3 a, const float3 b, const float3 c, const float3 d, const float3 normal, const float3 tangent)
{
    const std::uint32_t base = static_cast<std::uint32_t>(mesh.vertices.size());
    addVertex(mesh, a, normal, make_float2(0.0f, 0.0f), tangent);
    addVertex(mesh, b, normal, make_float2(1.0f, 0.0f), tangent);
    addVertex(mesh, c, normal, make_float2(1.0f, 1.0f), tangent);
    addVertex(mesh, d, normal, make_float2(0.0f, 1.0f), tangent);
    mesh.triangles.push_back({base, base + 1u, base + 2u, 0u});
    mesh.triangles.push_back({base, base + 2u, base + 3u, 0u});
}

void addTriangle(MeshData& mesh, const float3 a, const float3 b, const float3 c, const float3 normal, const float3 tangent)
{
    const std::uint32_t base = static_cast<std::uint32_t>(mesh.vertices.size());
    addVertex(mesh, a, normal, make_float2(0.0f, 0.0f), tangent);
    addVertex(mesh, b, normal, make_float2(1.0f, 0.0f), tangent);
    addVertex(mesh, c, normal, make_float2(0.5f, 1.0f), tangent);
    mesh.triangles.push_back({base, base + 1u, base + 2u, 0u});
}
} // namespace

bool isEmptyMesh(const MeshData& mesh)
{
    return mesh.vertices.empty() || mesh.triangles.empty();
}

bool hasValidMeshMaterialIndices(const MeshData& mesh)
{
    if (mesh.materials.empty())
    {
        return mesh.triangles.empty();
    }

    for (const MeshTriangle& triangle : mesh.triangles)
    {
        if (triangle.materialIndex >= mesh.materials.size())
        {
            return false;
        }
    }

    return true;
}

MeshTextureMetadata getMeshTextureMetadata(const MeshData& mesh, const int textureIndex, const std::string& path)
{
    MeshTextureMetadata metadata;
    metadata.hasPath = !path.empty() || textureIndex >= 0;
    metadata.path = path;

    if (textureIndex < 0 || textureIndex >= static_cast<int>(mesh.textures.size()))
    {
        return metadata;
    }

    const MeshTexture& texture = mesh.textures[static_cast<size_t>(textureIndex)];
    metadata.loaded = !texture.pixels.empty() && texture.width > 0 && texture.height > 0;
    metadata.path = texture.path.empty() ? path : texture.path;
    metadata.type = texture.type;
    metadata.width = texture.width;
    metadata.height = texture.height;
    metadata.channels = texture.channels;
    return metadata;
}

MeshData createCubeMesh()
{
    MeshData mesh;
    mesh.materials.push_back(makePrimitiveMaterial("built_in_cube_matte", make_float3(0.72f, 0.78f, 0.86f)));

    const float s = 0.5f;
    addQuad(mesh, make_float3(-s, -s, s), make_float3(s, -s, s), make_float3(s, s, s), make_float3(-s, s, s), make_float3(0.0f, 0.0f, 1.0f), make_float3(1.0f, 0.0f, 0.0f));
    addQuad(mesh, make_float3(s, -s, -s), make_float3(-s, -s, -s), make_float3(-s, s, -s), make_float3(s, s, -s), make_float3(0.0f, 0.0f, -1.0f), make_float3(-1.0f, 0.0f, 0.0f));
    addQuad(mesh, make_float3(-s, -s, -s), make_float3(-s, -s, s), make_float3(-s, s, s), make_float3(-s, s, -s), make_float3(-1.0f, 0.0f, 0.0f), make_float3(0.0f, 0.0f, 1.0f));
    addQuad(mesh, make_float3(s, -s, s), make_float3(s, -s, -s), make_float3(s, s, -s), make_float3(s, s, s), make_float3(1.0f, 0.0f, 0.0f), make_float3(0.0f, 0.0f, -1.0f));
    addQuad(mesh, make_float3(-s, s, s), make_float3(s, s, s), make_float3(s, s, -s), make_float3(-s, s, -s), make_float3(0.0f, 1.0f, 0.0f), make_float3(1.0f, 0.0f, 0.0f));
    addQuad(mesh, make_float3(-s, -s, -s), make_float3(s, -s, -s), make_float3(s, -s, s), make_float3(-s, -s, s), make_float3(0.0f, -1.0f, 0.0f), make_float3(1.0f, 0.0f, 0.0f));

    return mesh;
}

MeshData createPyramidMesh()
{
    MeshData mesh;
    mesh.materials.push_back(makePrimitiveMaterial("built_in_pyramid_matte", make_float3(0.86f, 0.70f, 0.48f)));

    const float s = 0.6f;
    const float h = 1.0f;
    const float3 p0 = make_float3(-s, 0.0f, -s);
    const float3 p1 = make_float3(s, 0.0f, -s);
    const float3 p2 = make_float3(s, 0.0f, s);
    const float3 p3 = make_float3(-s, 0.0f, s);
    const float3 top = make_float3(0.0f, h, 0.0f);

    addQuad(mesh, p0, p1, p2, p3, make_float3(0.0f, -1.0f, 0.0f), make_float3(1.0f, 0.0f, 0.0f));
    addTriangle(mesh, p1, p0, top, make_float3(0.0f, 0.52f, -0.86f), make_float3(1.0f, 0.0f, 0.0f));
    addTriangle(mesh, p2, p1, top, make_float3(0.86f, 0.52f, 0.0f), make_float3(0.0f, 0.0f, 1.0f));
    addTriangle(mesh, p3, p2, top, make_float3(0.0f, 0.52f, 0.86f), make_float3(-1.0f, 0.0f, 0.0f));
    addTriangle(mesh, p0, p3, top, make_float3(-0.86f, 0.52f, 0.0f), make_float3(0.0f, 0.0f, -1.0f));

    return mesh;
}

MeshData createPlaneMesh()
{
    MeshData mesh;
    mesh.materials.push_back(makePrimitiveMaterial("built_in_panel_matte", make_float3(0.70f, 0.72f, 0.68f)));

    const float s = 0.5f;
    addQuad(mesh,
        make_float3(-s, 0.0f, -s),
        make_float3(s, 0.0f, -s),
        make_float3(s, 0.0f, s),
        make_float3(-s, 0.0f, s),
        make_float3(0.0f, 1.0f, 0.0f),
        make_float3(1.0f, 0.0f, 0.0f));

    return mesh;
}
