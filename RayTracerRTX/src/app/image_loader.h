#pragma once

#include "mesh.h"

#include <filesystem>
#include <string>

bool loadImageTexture(const std::filesystem::path& path, MeshTexture& texture, const std::string& type = {});
