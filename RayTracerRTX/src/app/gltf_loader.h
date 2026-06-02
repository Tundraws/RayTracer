#pragma once

#include "mesh.h"
#include "scene.h"

#include <filesystem>
#include <string>
#include <vector>

struct GltfLoadResult
{
    bool ok = false;
    MeshData mesh;
    std::string error;
};

struct GltfSceneLoadResult
{
    bool ok = false;
    std::vector<MeshObject> meshObjects;
    std::string error;
};

GltfLoadResult loadGltfMesh(const std::filesystem::path& path);
GltfSceneLoadResult loadGltfMeshObjects(const std::filesystem::path& path);
