#include "asset_cache.h"

#include "gltf_loader.h"

#include <algorithm>
#include <cctype>

std::string AssetCache::makePathKey(const std::filesystem::path& path)
{
    std::error_code error;
    const std::filesystem::path absolutePath = std::filesystem::absolute(path, error);
    const std::filesystem::path normalized = error ? path.lexically_normal() : absolutePath.lexically_normal();
    return normalized.string();
}

std::string AssetCache::makeTextureKey(const std::filesystem::path& path, const std::string& type)
{
    return makePathKey(path) + "#" + type;
}

ObjLoadResult AssetCache::loadMeshByExtension(const std::filesystem::path& path)
{
    std::string extension = path.extension().string();
    std::transform(extension.begin(), extension.end(), extension.begin(), [](const unsigned char ch)
    {
        return static_cast<char>(std::tolower(ch));
    });

    if (extension == ".gltf" || extension == ".glb")
    {
        GltfLoadResult loaded = loadGltfMesh(path);
        return ObjLoadResult{loaded.ok, std::move(loaded.mesh), loaded.error};
    }
    return loadObjMesh(path);
}

const ObjLoadResult& AssetCache::loadMesh(const std::filesystem::path& path)
{
    const std::string key = makePathKey(path);
    const auto found = meshCache.find(key);
    if (found != meshCache.end())
    {
        return found->second;
    }

    meshLoadCounters[key] += 1;
    auto inserted = meshCache.emplace(key, loadMeshByExtension(path));
    return inserted.first->second;
}

const MeshTexture* AssetCache::loadTexture(const std::filesystem::path& path, const std::string& type)
{
    const std::string key = makeTextureKey(path, type);
    const auto found = textureCache.find(key);
    if (found != textureCache.end())
    {
        return &found->second;
    }

    MeshTexture texture;
    textureLoadCounters[key] += 1;
    if (!loadImageTexture(path, texture, type))
    {
        return nullptr;
    }

    auto inserted = textureCache.emplace(key, std::move(texture));
    return &inserted.first->second;
}

int AssetCache::meshLoadCount(const std::filesystem::path& path) const
{
    const auto found = meshLoadCounters.find(makePathKey(path));
    return found == meshLoadCounters.end() ? 0 : found->second;
}

int AssetCache::textureLoadCount(const std::filesystem::path& path, const std::string& type) const
{
    const auto found = textureLoadCounters.find(makeTextureKey(path, type));
    return found == textureLoadCounters.end() ? 0 : found->second;
}

void AssetCache::clear()
{
    meshCache.clear();
    textureCache.clear();
    meshLoadCounters.clear();
    textureLoadCounters.clear();
}
