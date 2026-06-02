#pragma once

#include "mesh.h"

#include <cstddef>
#include <filesystem>
#include <string>

bool loadImageTexture(const std::filesystem::path& path, MeshTexture& texture, const std::string& type = {});
bool loadImageTextureFromMemory(const unsigned char* data, size_t size, MeshTexture& texture, const std::string& type = {}, const std::string& label = {});
