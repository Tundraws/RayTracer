#include "image_loader.h"

#include <algorithm>
#include <cctype>
#include <fstream>
#include <sstream>
#include <string>

#define STB_IMAGE_IMPLEMENTATION
#include "../../third_party/stb/stb_image.h"

namespace
{
bool readPpmToken(std::istream& input, std::string& token)
{
    token.clear();
    while (input >> token)
    {
        if (!token.empty() && token[0] == '#')
        {
            std::string ignored;
            std::getline(input, ignored);
            continue;
        }
        return true;
    }
    return false;
}

std::string lowerExtension(const std::filesystem::path& path)
{
    std::string extension = path.extension().string();
    std::transform(extension.begin(), extension.end(), extension.begin(), [](const unsigned char ch)
    {
        return static_cast<char>(std::tolower(ch));
    });
    return extension;
}

bool loadPpmTexture(const std::filesystem::path& path, MeshTexture& texture, const std::string& type)
{
    std::ifstream file(path, std::ios::binary);
    if (!file)
    {
        return false;
    }

    std::string token;
    std::string widthToken;
    std::string heightToken;
    std::string maxToken;
    if (!readPpmToken(file, token) || token != "P3" ||
        !readPpmToken(file, widthToken) || !readPpmToken(file, heightToken) || !readPpmToken(file, maxToken))
    {
        return false;
    }

    const int width = std::stoi(widthToken);
    const int height = std::stoi(heightToken);
    const int maxValue = std::stoi(maxToken);
    if (width <= 0 || height <= 0 || maxValue <= 0)
    {
        return false;
    }

    std::vector<uchar4> pixels;
    pixels.reserve(static_cast<size_t>(width) * static_cast<size_t>(height));
    for (int i = 0; i < width * height; ++i)
    {
        std::string rToken;
        std::string gToken;
        std::string bToken;
        if (!readPpmToken(file, rToken) || !readPpmToken(file, gToken) || !readPpmToken(file, bToken))
        {
            return false;
        }

        const auto toByte = [maxValue](const int value)
        {
            const int clamped = value < 0 ? 0 : (value > maxValue ? maxValue : value);
            return static_cast<unsigned char>((clamped * 255) / maxValue);
        };
        pixels.push_back(make_uchar4(
            toByte(std::stoi(rToken)),
            toByte(std::stoi(gToken)),
            toByte(std::stoi(bToken)),
            255));
    }

    texture.path = path.string();
    texture.type = type;
    texture.width = static_cast<unsigned int>(width);
    texture.height = static_cast<unsigned int>(height);
    texture.channels = 3u;
    texture.pixels = std::move(pixels);
    return true;
}
}

bool loadImageTexture(const std::filesystem::path& path, MeshTexture& texture, const std::string& type)
{
    if (lowerExtension(path) == ".ppm")
    {
        return loadPpmTexture(path, texture, type);
    }

    int width = 0;
    int height = 0;
    int channels = 0;
    stbi_uc* pixels = stbi_load(path.string().c_str(), &width, &height, &channels, 4);
    if (pixels == nullptr || width <= 0 || height <= 0)
    {
        if (pixels != nullptr)
        {
            stbi_image_free(pixels);
        }
        return false;
    }

    std::vector<uchar4> result;
    result.reserve(static_cast<size_t>(width) * static_cast<size_t>(height));
    for (int i = 0; i < width * height; ++i)
    {
        const int offset = i * 4;
        result.push_back(make_uchar4(
            pixels[offset],
            pixels[offset + 1],
            pixels[offset + 2],
            pixels[offset + 3]));
    }
    stbi_image_free(pixels);

    texture.path = path.string();
    texture.type = type;
    texture.width = static_cast<unsigned int>(width);
    texture.height = static_cast<unsigned int>(height);
    texture.channels = static_cast<unsigned int>(channels);
    texture.pixels = std::move(result);
    return true;
}
