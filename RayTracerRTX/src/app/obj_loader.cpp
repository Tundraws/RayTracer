#include "obj_loader.h"

#include <algorithm>
#include <charconv>
#include <cmath>
#include <fstream>
#include <sstream>
#include <string_view>
#include <unordered_map>

namespace
{
struct FaceVertex
{
    int positionIndex = -1;
    int texcoordIndex = -1;
    int normalIndex = -1;
};

float3 sub3(const float3 a, const float3 b)
{
    return make_float3(a.x - b.x, a.y - b.y, a.z - b.z);
}

float3 cross3(const float3 a, const float3 b)
{
    return make_float3(
        a.y * b.z - a.z * b.y,
        a.z * b.x - a.x * b.z,
        a.x * b.y - a.y * b.x);
}

float3 normalize3(const float3 v)
{
    const float length = std::sqrt(v.x * v.x + v.y * v.y + v.z * v.z);
    if (length <= 1e-8f)
    {
        return make_float3(0.0f, 1.0f, 0.0f);
    }

    return make_float3(v.x / length, v.y / length, v.z / length);
}

std::string trim(const std::string& value)
{
    const auto begin = std::find_if_not(value.begin(), value.end(), [](const unsigned char ch)
    {
        return std::isspace(ch) != 0;
    });
    const auto end = std::find_if_not(value.rbegin(), value.rend(), [](const unsigned char ch)
    {
        return std::isspace(ch) != 0;
    }).base();

    if (begin >= end)
    {
        return {};
    }

    return std::string(begin, end);
}

bool parseInt(std::string_view value, int& parsed)
{
    if (value.empty())
    {
        return false;
    }

    const char* begin = value.data();
    const char* end = begin + value.size();
    const auto result = std::from_chars(begin, end, parsed);
    return result.ec == std::errc{} && result.ptr == end;
}

bool parseFaceVertex(const std::string& token, FaceVertex& vertex)
{
    const std::string_view view(token);
    const size_t firstSlash = view.find('/');
    const std::string_view positionPart = firstSlash == std::string_view::npos
        ? view
        : view.substr(0, firstSlash);

    int objPositionIndex = 0;
    if (!parseInt(positionPart, objPositionIndex) || objPositionIndex <= 0)
    {
        return false;
    }

    vertex.positionIndex = objPositionIndex - 1;
    vertex.texcoordIndex = -1;
    vertex.normalIndex = -1;

    if (firstSlash == std::string_view::npos)
    {
        return true;
    }

    const size_t secondSlash = view.find('/', firstSlash + 1);
    const std::string_view texcoordPart = secondSlash == std::string_view::npos
        ? view.substr(firstSlash + 1)
        : view.substr(firstSlash + 1, secondSlash - firstSlash - 1);
    if (!texcoordPart.empty())
    {
        int objTexcoordIndex = 0;
        if (!parseInt(texcoordPart, objTexcoordIndex) || objTexcoordIndex <= 0)
        {
            return false;
        }
        vertex.texcoordIndex = objTexcoordIndex - 1;
    }

    if (secondSlash == std::string_view::npos)
    {
        return true;
    }

    const std::string_view normalPart = view.substr(secondSlash + 1);
    if (normalPart.empty())
    {
        return true;
    }

    int objNormalIndex = 0;
    if (!parseInt(normalPart, objNormalIndex) || objNormalIndex <= 0)
    {
        return false;
    }

    vertex.normalIndex = objNormalIndex - 1;
    return true;
}

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

bool loadPpmTexture(const std::filesystem::path& path, MeshTexture& texture)
{
    std::ifstream file(path, std::ios::binary);
    if (!file)
    {
        return false;
    }

    std::string token;
    if (!readPpmToken(file, token) || token != "P3")
    {
        return false;
    }

    std::string widthToken;
    std::string heightToken;
    std::string maxToken;
    if (!readPpmToken(file, widthToken) || !readPpmToken(file, heightToken) || !readPpmToken(file, maxToken))
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
    texture.width = static_cast<unsigned int>(width);
    texture.height = static_cast<unsigned int>(height);
    texture.pixels = std::move(pixels);
    return true;
}

float clampf(const float value, const float minValue, const float maxValue)
{
    return value < minValue ? minValue : (value > maxValue ? maxValue : value);
}

std::string toLower(std::string value)
{
    std::transform(value.begin(), value.end(), value.begin(), [](const unsigned char ch)
    {
        return static_cast<char>(std::tolower(ch));
    });
    return value;
}

int materialTypeFromName(const std::string& name)
{
    const std::string lowerName = toLower(name);
    if (lowerName.find("glass") != std::string::npos || lowerName.find("dielectric") != std::string::npos)
    {
        return MaterialDielectric;
    }
    if (lowerName.find("metal") != std::string::npos)
    {
        return MaterialMetal;
    }
    if (lowerName.find("mirror") != std::string::npos)
    {
        return MaterialMirror;
    }
    return MaterialDiffuse;
}

float roughnessFromNs(const float ns)
{
    const float clampedNs = clampf(ns, 1.0f, 1000.0f);
    return clampf(std::sqrt(2.0f / (clampedNs + 2.0f)), 0.02f, 1.0f);
}

MeshMaterial makeDefaultMaterial(const std::string& name = "default")
{
    MeshMaterial material;
    material.color = make_float3(0.80f, 0.80f, 0.78f);
    material.materialType = materialTypeFromName(name);
    material.name = name;
    material.specularColor = make_float3(1.0f, 1.0f, 1.0f);
    material.roughness = material.materialType == MaterialMirror ? 0.02f : 0.35f;
    material.ior = 1.5f;
    material.alpha = 1.0f;
    return material;
}

int ensureMaterial(
    MeshData& mesh,
    std::unordered_map<std::string, std::uint32_t>& materialIndices,
    const std::string& name)
{
    const auto existing = materialIndices.find(name);
    if (existing != materialIndices.end())
    {
        return static_cast<int>(existing->second);
    }

    const std::uint32_t index = static_cast<std::uint32_t>(mesh.materials.size());
    mesh.materials.push_back(makeDefaultMaterial(name));
    materialIndices[name] = index;
    return static_cast<int>(index);
}

void loadMtl(
    const std::filesystem::path& path,
    MeshData& mesh,
    std::unordered_map<std::string, std::uint32_t>& materialIndices)
{
    std::ifstream file(path);
    if (!file)
    {
        return;
    }

    int currentMaterial = -1;
    std::string line;
    while (std::getline(file, line))
    {
        line = trim(line);
        if (line.empty() || line[0] == '#')
        {
            continue;
        }

        std::istringstream input(line);
        std::string command;
        input >> command;

        if (command == "newmtl")
        {
            std::string name;
            input >> name;
            if (!name.empty())
            {
                currentMaterial = ensureMaterial(mesh, materialIndices, name);
            }
        }
        else if (command == "Kd" && currentMaterial >= 0)
        {
            float r = 0.8f;
            float g = 0.8f;
            float b = 0.78f;
            if (input >> r >> g >> b)
            {
                mesh.materials[static_cast<size_t>(currentMaterial)].color = make_float3(r, g, b);
            }
        }
        else if (command == "Ks" && currentMaterial >= 0)
        {
            float r = 1.0f;
            float g = 1.0f;
            float b = 1.0f;
            if (input >> r >> g >> b)
            {
                mesh.materials[static_cast<size_t>(currentMaterial)].specularColor = make_float3(r, g, b);
            }
        }
        else if (command == "Ns" && currentMaterial >= 0)
        {
            float ns = 64.0f;
            if (input >> ns)
            {
                mesh.materials[static_cast<size_t>(currentMaterial)].roughness = roughnessFromNs(ns);
            }
        }
        else if (command == "Ni" && currentMaterial >= 0)
        {
            float ior = 1.5f;
            if (input >> ior)
            {
                mesh.materials[static_cast<size_t>(currentMaterial)].ior = clampf(ior, 1.0f, 2.8f);
            }
        }
        else if (command == "d" && currentMaterial >= 0)
        {
            float alpha = 1.0f;
            if (input >> alpha)
            {
                mesh.materials[static_cast<size_t>(currentMaterial)].alpha = clampf(alpha, 0.0f, 1.0f);
            }
        }
        else if (command == "map_Kd" && currentMaterial >= 0)
        {
            std::string textureName;
            input >> textureName;
            if (!textureName.empty())
            {
                MeshMaterial& material = mesh.materials[static_cast<size_t>(currentMaterial)];
                material.texturePath = textureName;

                MeshTexture texture;
                if (loadPpmTexture(path.parent_path() / textureName, texture))
                {
                    material.textureIndex = static_cast<int>(mesh.textures.size());
                    mesh.textures.push_back(std::move(texture));
                }
            }
        }
    }
}

ObjLoadResult fail(std::string error)
{
    ObjLoadResult result;
    result.ok = false;
    result.error = std::move(error);
    return result;
}
} // namespace

ObjLoadResult loadObjMesh(const std::filesystem::path& path)
{
    std::ifstream file(path);
    if (!file)
    {
        return fail("OBJ file could not be opened: " + path.string());
    }

    ObjLoadResult result;
    result.ok = false;

    MeshData& mesh = result.mesh;
    std::vector<float3> positions;
    std::vector<float3> normals;
    std::vector<float2> texcoords;
    std::unordered_map<std::string, std::uint32_t> materialIndices;

    mesh.materials.push_back(makeDefaultMaterial());
    materialIndices["default"] = 0;
    int currentMaterial = 0;

    std::string line;
    int lineNumber = 0;
    while (std::getline(file, line))
    {
        ++lineNumber;
        line = trim(line);
        if (line.empty() || line[0] == '#')
        {
            continue;
        }

        std::istringstream input(line);
        std::string command;
        input >> command;

        if (command == "v")
        {
            float x = 0.0f;
            float y = 0.0f;
            float z = 0.0f;
            if (!(input >> x >> y >> z))
            {
                return fail("Invalid vertex at line " + std::to_string(lineNumber));
            }
            positions.push_back(make_float3(x, y, z));
        }
        else if (command == "vn")
        {
            float x = 0.0f;
            float y = 0.0f;
            float z = 0.0f;
            if (!(input >> x >> y >> z))
            {
                return fail("Invalid normal at line " + std::to_string(lineNumber));
            }
            normals.push_back(normalize3(make_float3(x, y, z)));
        }
        else if (command == "vt")
        {
            float u = 0.0f;
            float v = 0.0f;
            if (!(input >> u >> v))
            {
                return fail("Invalid texcoord at line " + std::to_string(lineNumber));
            }
            texcoords.push_back(make_float2(u, v));
        }
        else if (command == "mtllib")
        {
            std::string mtlName;
            input >> mtlName;
            if (!mtlName.empty())
            {
                loadMtl(path.parent_path() / mtlName, mesh, materialIndices);
            }
        }
        else if (command == "usemtl")
        {
            std::string name;
            input >> name;
            if (!name.empty())
            {
                currentMaterial = ensureMaterial(mesh, materialIndices, name);
            }
        }
        else if (command == "f")
        {
            std::vector<FaceVertex> faceVertices;
            std::string token;
            while (input >> token)
            {
                FaceVertex faceVertex;
                if (!parseFaceVertex(token, faceVertex))
                {
                    return fail("Invalid face vertex at line " + std::to_string(lineNumber));
                }
                if (faceVertex.positionIndex < 0 || static_cast<size_t>(faceVertex.positionIndex) >= positions.size())
                {
                    return fail("Face position index out of range at line " + std::to_string(lineNumber));
                }
                if (faceVertex.normalIndex >= 0 && static_cast<size_t>(faceVertex.normalIndex) >= normals.size())
                {
                    return fail("Face normal index out of range at line " + std::to_string(lineNumber));
                }
                if (faceVertex.texcoordIndex >= 0 && static_cast<size_t>(faceVertex.texcoordIndex) >= texcoords.size())
                {
                    return fail("Face texcoord index out of range at line " + std::to_string(lineNumber));
                }
                faceVertices.push_back(faceVertex);
            }

            if (faceVertices.size() != 3)
            {
                return fail("Only triangulated OBJ faces are supported at line " + std::to_string(lineNumber));
            }

            const float3 p0 = positions[static_cast<size_t>(faceVertices[0].positionIndex)];
            const float3 p1 = positions[static_cast<size_t>(faceVertices[1].positionIndex)];
            const float3 p2 = positions[static_cast<size_t>(faceVertices[2].positionIndex)];
            const float3 fallbackNormal = normalize3(cross3(sub3(p1, p0), sub3(p2, p0)));
            const std::uint32_t firstVertex = static_cast<std::uint32_t>(mesh.vertices.size());

            for (const FaceVertex& faceVertex : faceVertices)
            {
                const float3 position = positions[static_cast<size_t>(faceVertex.positionIndex)];
                const float3 normal = faceVertex.normalIndex >= 0
                    ? normals[static_cast<size_t>(faceVertex.normalIndex)]
                    : fallbackNormal;
                const float2 texcoord = faceVertex.texcoordIndex >= 0
                    ? texcoords[static_cast<size_t>(faceVertex.texcoordIndex)]
                    : make_float2(0.0f, 0.0f);
                mesh.vertices.push_back(MeshVertex{position, normal, texcoord});
            }

            mesh.triangles.push_back(MeshTriangle{
                firstVertex,
                firstVertex + 1u,
                firstVertex + 2u,
                static_cast<std::uint32_t>(currentMaterial)});
        }
    }

    if (mesh.triangles.empty())
    {
        return fail("OBJ file contains no triangles: " + path.string());
    }

    result.ok = true;
    return result;
}
