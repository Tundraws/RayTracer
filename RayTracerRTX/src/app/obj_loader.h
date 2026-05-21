#pragma once

#include "mesh.h"

#include <filesystem>
#include <string>

struct ObjLoadResult
{
    bool ok = false;
    MeshData mesh;
    std::string error;
};

ObjLoadResult loadObjMesh(const std::filesystem::path& path);
