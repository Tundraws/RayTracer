#include "mesh.h"

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
