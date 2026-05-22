#pragma once

#include "image_loader.h"
#include "mesh.h"
#include "obj_loader.h"

#include <filesystem>
#include <map>
#include <string>

class AssetCache
{
public:
    const ObjLoadResult& loadMesh(const std::filesystem::path& path);
    const MeshTexture* loadTexture(const std::filesystem::path& path, const std::string& type = {});

    int meshLoadCount(const std::filesystem::path& path) const;
    int textureLoadCount(const std::filesystem::path& path, const std::string& type = {}) const;
    void clear();

private:
    static std::string makePathKey(const std::filesystem::path& path);
    static std::string makeTextureKey(const std::filesystem::path& path, const std::string& type);
    static ObjLoadResult loadMeshByExtension(const std::filesystem::path& path);

    std::map<std::string, ObjLoadResult> meshCache;
    std::map<std::string, MeshTexture> textureCache;
    std::map<std::string, int> meshLoadCounters;
    std::map<std::string, int> textureLoadCounters;
};
