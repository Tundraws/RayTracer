#pragma once

#include "mesh.h"

#include <filesystem>
#include <string>

struct GltfLoadResult
{
    bool ok = false;
    MeshData mesh;
    std::string error;
};

GltfLoadResult loadGltfMesh(const std::filesystem::path& path);
