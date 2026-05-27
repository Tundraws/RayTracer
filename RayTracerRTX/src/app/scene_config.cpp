#include "scene_config.h"

#include "asset_cache.h"
#include "logger.h"
#include "obj_loader.h"

#include <algorithm>
#include <cmath>
#include <cctype>
#include <fstream>
#include <iomanip>
#include <map>
#include <sstream>
#include <stdexcept>
#include <string_view>
#include <utility>
#include <variant>

namespace
{
constexpr float kPi = 3.14159265358979323846f;

struct JsonValue;

using JsonObject = std::map<std::string, JsonValue>;
using JsonArray = std::vector<JsonValue>;

struct JsonValue
{
    std::variant<std::nullptr_t, bool, double, std::string, JsonArray, JsonObject> value;
};

class JsonParser
{
public:
    explicit JsonParser(std::string_view text) : text_(text) {}

    JsonValue parse()
    {
        JsonValue value = parseValue();
        skipWhitespace();
        if (!isAtEnd())
        {
            fail("Unexpected trailing characters");
        }
        return value;
    }

private:
    JsonValue parseValue()
    {
        skipWhitespace();
        if (isAtEnd())
        {
            fail("Unexpected end of JSON");
        }

        const char ch = peek();
        if (ch == '{')
        {
            return JsonValue{parseObject()};
        }
        if (ch == '[')
        {
            return JsonValue{parseArray()};
        }
        if (ch == '"')
        {
            return JsonValue{parseString()};
        }
        if (ch == '-' || std::isdigit(static_cast<unsigned char>(ch)) != 0)
        {
            return JsonValue{parseNumber()};
        }
        if (consumeLiteral("true"))
        {
            return JsonValue{true};
        }
        if (consumeLiteral("false"))
        {
            return JsonValue{false};
        }
        if (consumeLiteral("null"))
        {
            return JsonValue{nullptr};
        }

        fail("Unsupported JSON value");
        return {};
    }

    JsonObject parseObject()
    {
        expect('{');
        JsonObject object;
        skipWhitespace();
        if (tryConsume('}'))
        {
            return object;
        }

        while (true)
        {
            skipWhitespace();
            if (peek() != '"')
            {
                fail("Expected object key");
            }
            const std::string key = parseString();
            skipWhitespace();
            expect(':');
            object[key] = parseValue();
            skipWhitespace();
            if (tryConsume('}'))
            {
                return object;
            }
            expect(',');
        }
    }

    JsonArray parseArray()
    {
        expect('[');
        JsonArray array;
        skipWhitespace();
        if (tryConsume(']'))
        {
            return array;
        }

        while (true)
        {
            array.push_back(parseValue());
            skipWhitespace();
            if (tryConsume(']'))
            {
                return array;
            }
            expect(',');
        }
    }

    std::string parseString()
    {
        expect('"');
        std::string result;
        while (!isAtEnd())
        {
            const char ch = advance();
            if (ch == '"')
            {
                return result;
            }
            if (ch == '\\')
            {
                if (isAtEnd())
                {
                    fail("Unfinished string escape");
                }
                const char escaped = advance();
                switch (escaped)
                {
                case '"':
                case '\\':
                case '/':
                    result.push_back(escaped);
                    break;
                case 'b':
                    result.push_back('\b');
                    break;
                case 'f':
                    result.push_back('\f');
                    break;
                case 'n':
                    result.push_back('\n');
                    break;
                case 'r':
                    result.push_back('\r');
                    break;
                case 't':
                    result.push_back('\t');
                    break;
                default:
                    fail("Unsupported string escape");
                }
            }
            else
            {
                result.push_back(ch);
            }
        }

        fail("Unterminated string");
        return {};
    }

    double parseNumber()
    {
        const size_t begin = pos_;
        if (peek() == '-')
        {
            advance();
        }
        consumeDigits();
        if (!isAtEnd() && peek() == '.')
        {
            advance();
            consumeDigits();
        }
        if (!isAtEnd() && (peek() == 'e' || peek() == 'E'))
        {
            advance();
            if (!isAtEnd() && (peek() == '+' || peek() == '-'))
            {
                advance();
            }
            consumeDigits();
        }

        return std::stod(std::string(text_.substr(begin, pos_ - begin)));
    }

    void consumeDigits()
    {
        if (isAtEnd() || std::isdigit(static_cast<unsigned char>(peek())) == 0)
        {
            fail("Expected digit");
        }
        while (!isAtEnd() && std::isdigit(static_cast<unsigned char>(peek())) != 0)
        {
            advance();
        }
    }

    bool consumeLiteral(std::string_view literal)
    {
        if (text_.substr(pos_, literal.size()) == literal)
        {
            pos_ += literal.size();
            return true;
        }
        return false;
    }

    void expect(char expected)
    {
        skipWhitespace();
        if (isAtEnd() || advance() != expected)
        {
            std::string message = "Expected '";
            message.push_back(expected);
            message.push_back('\'');
            fail(message);
        }
    }

    bool tryConsume(char expected)
    {
        skipWhitespace();
        if (!isAtEnd() && peek() == expected)
        {
            advance();
            return true;
        }
        return false;
    }

    void skipWhitespace()
    {
        while (!isAtEnd() && std::isspace(static_cast<unsigned char>(peek())) != 0)
        {
            advance();
        }
    }

    char peek() const
    {
        return text_[pos_];
    }

    char advance()
    {
        return text_[pos_++];
    }

    bool isAtEnd() const
    {
        return pos_ >= text_.size();
    }

    [[noreturn]] void fail(const std::string& message) const
    {
        std::ostringstream out;
        out << message << " at byte " << pos_;
        throw std::runtime_error(out.str());
    }

    std::string_view text_;
    size_t pos_ = 0;
};

const JsonObject* asObject(const JsonValue& value)
{
    return std::get_if<JsonObject>(&value.value);
}

const JsonArray* asArray(const JsonValue& value)
{
    return std::get_if<JsonArray>(&value.value);
}

const std::string* asString(const JsonValue& value)
{
    return std::get_if<std::string>(&value.value);
}

const double* asNumber(const JsonValue& value)
{
    return std::get_if<double>(&value.value);
}

const JsonValue* findField(const JsonObject& object, const std::string& name)
{
    const auto it = object.find(name);
    return it == object.end() ? nullptr : &it->second;
}

bool readFloat3(const JsonObject& object, const std::string& name, float3& out, std::string& error)
{
    const JsonValue* field = findField(object, name);
    if (field == nullptr)
    {
        return true;
    }

    const JsonArray* array = asArray(*field);
    if (array == nullptr || array->size() != 3)
    {
        error = "'" + name + "' must be an array with 3 numbers";
        return false;
    }

    const double* x = asNumber((*array)[0]);
    const double* y = asNumber((*array)[1]);
    const double* z = asNumber((*array)[2]);
    if (x == nullptr || y == nullptr || z == nullptr)
    {
        error = "'" + name + "' must contain only numbers";
        return false;
    }

    out = make_float3(static_cast<float>(*x), static_cast<float>(*y), static_cast<float>(*z));
    return true;
}

bool readFloatField(const JsonObject& object, const std::string& name, float& out, std::string& error)
{
    const JsonValue* field = findField(object, name);
    if (field == nullptr)
    {
        return true;
    }

    const double* value = asNumber(*field);
    if (value == nullptr)
    {
        error = "'" + name + "' must be a number";
        return false;
    }

    out = static_cast<float>(*value);
    return true;
}

bool readStringField(const JsonObject& object, const std::string& name, std::string& out, std::string& error)
{
    const JsonValue* field = findField(object, name);
    if (field == nullptr)
    {
        return true;
    }

    const std::string* value = asString(*field);
    if (value == nullptr)
    {
        error = "'" + name + "' must be a string";
        return false;
    }

    out = *value;
    return true;
}

bool readSceneTuningFields(const JsonObject& object, SceneConfig& config, std::string& error)
{
    if (findField(object, "exposure") != nullptr)
    {
        if (!readFloatField(object, "exposure", config.exposure, error))
        {
            return false;
        }
        config.hasExposure = true;
    }
    if (findField(object, "skyIntensity") != nullptr)
    {
        if (!readFloatField(object, "skyIntensity", config.skyIntensity, error))
        {
            return false;
        }
        config.hasSkyIntensity = true;
    }
    if (findField(object, "skyHorizonColor") != nullptr)
    {
        if (!readFloat3(object, "skyHorizonColor", config.skyHorizonColor, error))
        {
            return false;
        }
        config.hasSkyHorizonColor = true;
    }
    if (findField(object, "skyZenithColor") != nullptr)
    {
        if (!readFloat3(object, "skyZenithColor", config.skyZenithColor, error))
        {
            return false;
        }
        config.hasSkyZenithColor = true;
    }
    if (findField(object, "skyGradientBlend") != nullptr)
    {
        if (!readFloatField(object, "skyGradientBlend", config.skyGradientBlend, error))
        {
            return false;
        }
        config.hasSkyGradientBlend = true;
    }
    if (findField(object, "lightIntensity") != nullptr)
    {
        if (!readFloatField(object, "lightIntensity", config.lightIntensity, error))
        {
            return false;
        }
        config.hasLightIntensity = true;
    }
    return true;
}

bool readPpmToken(std::istream& input, std::string& token)
{
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

bool loadPpmEnvironmentMap(const std::filesystem::path& path, MeshTexture& texture)
{
    std::ifstream file(path);
    if (!file)
    {
        return false;
    }

    std::string magic;
    std::string widthToken;
    std::string heightToken;
    std::string maxToken;
    if (!readPpmToken(file, magic) || magic != "P3" ||
        !readPpmToken(file, widthToken) ||
        !readPpmToken(file, heightToken) ||
        !readPpmToken(file, maxToken))
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
    pixels.reserve(static_cast<size_t>(width * height));
    for (int i = 0; i < width * height; ++i)
    {
        std::string rToken;
        std::string gToken;
        std::string bToken;
        if (!readPpmToken(file, rToken) || !readPpmToken(file, gToken) || !readPpmToken(file, bToken))
        {
            return false;
        }
        const auto toByte = [maxValue](const std::string& value)
        {
            const int parsed = std::stoi(value);
            const int clamped = std::clamp(parsed, 0, maxValue);
            return static_cast<unsigned char>((clamped * 255) / maxValue);
        };
        pixels.push_back(make_uchar4(toByte(rToken), toByte(gToken), toByte(bToken), 255));
    }

    texture.path = path.string();
    texture.width = static_cast<unsigned int>(width);
    texture.height = static_cast<unsigned int>(height);
    texture.pixels = std::move(pixels);
    return true;
}

std::string lowerCopy(std::string value)
{
    std::transform(value.begin(), value.end(), value.begin(), [](const unsigned char ch)
    {
        return static_cast<char>(std::tolower(ch));
    });
    return value;
}

float clampMaterial01(const float value)
{
    return value < 0.0f ? 0.0f : (value > 1.0f ? 1.0f : value);
}

float clampMaterialRoughness(const float value)
{
    return value < 0.02f ? 0.02f : (value > 1.0f ? 1.0f : value);
}

float clampMaterialIor(const float value)
{
    return value < 1.0f ? 1.0f : (value > 2.8f ? 2.8f : value);
}

int materialTypeFromConfigString(const std::string& type, bool& usedFallback)
{
    const std::string lower = lowerCopy(type);
    usedFallback = false;
    if (lower == "matte" || lower == "diffuse")
    {
        return MaterialDiffuse;
    }
    if (lower == "mirror")
    {
        return MaterialMirror;
    }
    if (lower == "metal")
    {
        return MaterialMetal;
    }
    if (lower == "glass" || lower == "dielectric")
    {
        return MaterialDielectric;
    }

    usedFallback = true;
    return MaterialDiffuse;
}

SceneMaterialConfig makeDefaultSceneMaterialConfig(const std::string& name)
{
    SceneMaterialConfig material;
    material.name = name;
    return material;
}

std::string jsonEscape(const std::string& value)
{
    std::ostringstream out;
    for (const char ch : value)
    {
        switch (ch)
        {
        case '\\':
            out << "\\\\";
            break;
        case '"':
            out << "\\\"";
            break;
        case '\n':
            out << "\\n";
            break;
        case '\r':
            out << "\\r";
            break;
        case '\t':
            out << "\\t";
            break;
        default:
            out << ch;
            break;
        }
    }
    return out.str();
}

const char* materialTypeToConfigName(const int materialType)
{
    switch (materialType)
    {
    case MaterialMirror:
        return "mirror";
    case MaterialMetal:
        return "metal";
    case MaterialDielectric:
        return "glass";
    default:
        return "matte";
    }
}

std::string meshPrimitiveName(const MeshObject& object)
{
    if (object.assetReference.find("cube") != std::string::npos)
    {
        return "cube";
    }
    if (object.assetReference.find("pyramid") != std::string::npos)
    {
        return "pyramid";
    }
    if (object.assetReference.find("panel") != std::string::npos ||
        object.assetReference.find("plane") != std::string::npos ||
        object.assetReference.find("environment:") == 0)
    {
        return "plane";
    }
    return {};
}

MeshData makePrimitiveMeshByName(const std::string& primitive)
{
    if (primitive == "cube")
    {
        return createCubeMesh();
    }
    if (primitive == "pyramid")
    {
        return createPyramidMesh();
    }
    if (primitive == "plane" || primitive == "panel")
    {
        return createPlaneMesh();
    }
    return {};
}

bool parseMaterialObject(
    const JsonObject& object,
    const std::string& fallbackName,
    SceneMaterialConfig& material,
    std::string& error)
{
    material = makeDefaultSceneMaterialConfig(fallbackName);
    if (!readStringField(object, "name", material.name, error))
    {
        return false;
    }

    std::string typeName = "matte";
    if (!readStringField(object, "type", typeName, error))
    {
        return false;
    }
    material.materialType = materialTypeFromConfigString(typeName, material.usedFallbackType);

    if (!readFloat3(object, "baseColor", material.baseColor, error))
    {
        return false;
    }
    if (findField(object, "color") != nullptr && !readFloat3(object, "color", material.baseColor, error))
    {
        return false;
    }
    if (!readFloat3(object, "specularColor", material.specularColor, error))
    {
        return false;
    }
    if (!readFloatField(object, "roughness", material.roughness, error) ||
        !readFloatField(object, "metallic", material.metallic, error) ||
        !readFloatField(object, "ior", material.ior, error) ||
        !readFloatField(object, "alpha", material.alpha, error))
    {
        return false;
    }
    if (!readStringField(object, "texture", material.texturePath, error) ||
        !readStringField(object, "map_Kd", material.texturePath, error) ||
        !readStringField(object, "normalMap", material.normalTexturePath, error) ||
        !readStringField(object, "normal", material.normalTexturePath, error) ||
        !readStringField(object, "metallicMap", material.metallicTexturePath, error) ||
        !readStringField(object, "metallicTexture", material.metallicTexturePath, error) ||
        !readStringField(object, "roughnessMap", material.roughnessTexturePath, error) ||
        !readStringField(object, "roughnessTexture", material.roughnessTexturePath, error))
    {
        return false;
    }

    material.baseColor = make_float3(
        clampMaterial01(material.baseColor.x),
        clampMaterial01(material.baseColor.y),
        clampMaterial01(material.baseColor.z));
    material.specularColor = make_float3(
        clampMaterial01(material.specularColor.x),
        clampMaterial01(material.specularColor.y),
        clampMaterial01(material.specularColor.z));
    material.roughness = clampMaterialRoughness(material.roughness);
    material.metallic = clampMaterial01(material.metallic);
    material.ior = clampMaterialIor(material.ior);
    material.alpha = clampMaterial01(material.alpha);
    if (material.metallic >= 0.5f && material.materialType == MaterialDiffuse)
    {
        material.materialType = MaterialMetal;
    }
    return true;
}

bool parseMaterialReference(
    const JsonValue& value,
    SceneConfig& config,
    const std::string& fallbackName,
    std::string& outName,
    std::string& error)
{
    if (const std::string* name = asString(value))
    {
        outName = *name;
        return true;
    }

    const JsonObject* object = asObject(value);
    if (object == nullptr)
    {
        error = "Material reference must be a string or object";
        return false;
    }

    SceneMaterialConfig material;
    if (!parseMaterialObject(*object, fallbackName, material, error))
    {
        return false;
    }
    outName = material.name;
    config.materials.push_back(std::move(material));
    return true;
}

float radians(float degrees)
{
    return degrees * kPi / 180.0f;
}

float3 rotateX(const float3 value, float angle)
{
    const float c = std::cos(angle);
    const float s = std::sin(angle);
    return make_float3(value.x, value.y * c - value.z * s, value.y * s + value.z * c);
}

float3 rotateY(const float3 value, float angle)
{
    const float c = std::cos(angle);
    const float s = std::sin(angle);
    return make_float3(value.x * c + value.z * s, value.y, -value.x * s + value.z * c);
}

float3 rotateZ(const float3 value, float angle)
{
    const float c = std::cos(angle);
    const float s = std::sin(angle);
    return make_float3(value.x * c - value.y * s, value.x * s + value.y * c, value.z);
}

float3 rotateEulerXyz(const float3 value, const float3 degrees)
{
    float3 result = rotateX(value, radians(degrees.x));
    result = rotateY(result, radians(degrees.y));
    result = rotateZ(result, radians(degrees.z));
    return result;
}

float3 normalize3(const float3 value)
{
    const float length = std::sqrt(value.x * value.x + value.y * value.y + value.z * value.z);
    if (length <= 1e-8f)
    {
        return make_float3(0.0f, 1.0f, 0.0f);
    }
    return make_float3(value.x / length, value.y / length, value.z / length);
}

MeshData transformMesh(const MeshData& source, const MeshTransformConfig& transform)
{
    MeshData mesh = source;
    for (MeshVertex& vertex : mesh.vertices)
    {
        float3 position = make_float3(
            vertex.position.x * transform.scale.x,
            vertex.position.y * transform.scale.y,
            vertex.position.z * transform.scale.z);
        position = rotateEulerXyz(position, transform.rotation);
        vertex.position = make_float3(
            position.x + transform.position.x,
            position.y + transform.position.y,
            position.z + transform.position.z);
        vertex.normal = normalize3(rotateEulerXyz(vertex.normal, transform.rotation));
        vertex.tangent = normalize3(rotateEulerXyz(vertex.tangent, transform.rotation));
    }
    return mesh;
}

std::array<float, 12> makeTransformMatrix(const MeshTransformConfig& transform)
{
    const float3 xAxis = rotateEulerXyz(make_float3(transform.scale.x, 0.0f, 0.0f), transform.rotation);
    const float3 yAxis = rotateEulerXyz(make_float3(0.0f, transform.scale.y, 0.0f), transform.rotation);
    const float3 zAxis = rotateEulerXyz(make_float3(0.0f, 0.0f, transform.scale.z), transform.rotation);
    return {
        xAxis.x, yAxis.x, zAxis.x, transform.position.x,
        xAxis.y, yAxis.y, zAxis.y, transform.position.y,
        xAxis.z, yAxis.z, zAxis.z, transform.position.z};
}

void appendMesh(MeshData& target, const MeshData& source)
{
    const std::uint32_t vertexOffset = static_cast<std::uint32_t>(target.vertices.size());
    const std::uint32_t materialOffset = static_cast<std::uint32_t>(target.materials.size());
    const int textureOffset = static_cast<int>(target.textures.size());

    target.vertices.insert(target.vertices.end(), source.vertices.begin(), source.vertices.end());
    target.textures.insert(target.textures.end(), source.textures.begin(), source.textures.end());
    for (MeshMaterial material : source.materials)
    {
        if (material.textureIndex >= 0)
        {
            material.textureIndex += textureOffset;
        }
        if (material.normalTextureIndex >= 0)
        {
            material.normalTextureIndex += textureOffset;
        }
        if (material.metallicTextureIndex >= 0)
        {
            material.metallicTextureIndex += textureOffset;
        }
        if (material.roughnessTextureIndex >= 0)
        {
            material.roughnessTextureIndex += textureOffset;
        }
        target.materials.push_back(std::move(material));
    }
    for (MeshTriangle triangle : source.triangles)
    {
        triangle.i0 += vertexOffset;
        triangle.i1 += vertexOffset;
        triangle.i2 += vertexOffset;
        triangle.materialIndex += materialOffset;
        target.triangles.push_back(triangle);
    }
}

std::filesystem::path resolvePath(const std::filesystem::path& path, const std::filesystem::path& baseDirectory)
{
    if (path.is_absolute() || baseDirectory.empty())
    {
        return path;
    }
    return baseDirectory / path;
}

std::map<std::string, SceneMaterialConfig> makeMaterialMap(const std::vector<SceneMaterialConfig>& materials)
{
    std::map<std::string, SceneMaterialConfig> materialMap;
    for (const SceneMaterialConfig& material : materials)
    {
        if (!material.name.empty())
        {
            materialMap[material.name] = material;
        }
    }
    return materialMap;
}

SphereMaterial toSphereMaterial(const SceneMaterialConfig& config)
{
    SphereMaterial material;
    material.color = config.baseColor;
    material.materialType = config.materialType;
    material.specularColor = config.specularColor;
    material.roughness = config.roughness;
    material.ior = config.ior;
    material.alpha = config.alpha;
    return material;
}

MeshMaterial toMeshMaterial(const SceneMaterialConfig& config, MeshMaterial base)
{
    base.name = config.name.empty() ? base.name : config.name;
    base.color = config.baseColor;
    base.materialType = config.materialType;
    base.specularColor = config.specularColor;
    base.roughness = config.roughness;
    base.ior = config.ior;
    base.alpha = config.alpha;
    base.texturePath = config.texturePath;
    base.textureIndex = -1;
    base.normalTexturePath = config.normalTexturePath;
    base.normalTextureIndex = -1;
    base.metallicTexturePath = config.metallicTexturePath;
    base.metallicTextureIndex = -1;
    base.roughnessTexturePath = config.roughnessTexturePath;
    base.roughnessTextureIndex = -1;
    return base;
}

void appendMaterialWarnings(const SceneConfig& config, SceneBuildResult& result)
{
    for (const SceneMaterialConfig& material : config.materials)
    {
        if (material.usedFallbackType)
        {
            const std::string warning = "Material '" + material.name + "' uses an unknown type; matte fallback was applied.";
            result.warnings.push_back(warning);
            logWarning(warning);
        }
    }
}

void applySphereMaterialConfig(
    const SceneConfig& config,
    const std::map<std::string, SceneMaterialConfig>& materialMap,
    SceneBuildResult& result)
{
    const size_t count = std::min(config.sphereMaterialRefs.size(), result.scene.materials.size());
    for (size_t i = 0; i < count; ++i)
    {
        const std::string& name = config.sphereMaterialRefs[i];
        const auto found = materialMap.find(name);
        if (found == materialMap.end())
        {
            const std::string warning = "Sphere material '" + name + "' was not found; default sphere material was kept.";
            result.warnings.push_back(warning);
            logWarning(warning);
            continue;
        }
        result.scene.materials[i] = toSphereMaterial(found->second);
    }
}

void applyMeshMaterialOverride(
    const MeshObjectConfig& object,
    const std::map<std::string, SceneMaterialConfig>& materialMap,
    MeshData& mesh,
    SceneBuildResult& result)
{
    if (object.materialOverride.empty())
    {
        return;
    }

    const auto found = materialMap.find(object.materialOverride);
    if (found == materialMap.end())
    {
        const std::string warning = "Mesh material override '" + object.materialOverride + "' was not found; source mesh materials were kept.";
        result.warnings.push_back(warning);
        logWarning(warning);
        return;
    }

    for (MeshMaterial& material : mesh.materials)
    {
        material = toMeshMaterial(found->second, material);
    }
}

bool equalFloat(const float a, const float b)
{
    return std::fabs(a - b) <= 0.0005f;
}

bool equalFloat3(const float3 a, const float3 b)
{
    return equalFloat(a.x, b.x) && equalFloat(a.y, b.y) && equalFloat(a.z, b.z);
}

bool equalMeshMaterialForSave(const MeshMaterial& a, const MeshMaterial& b)
{
    return a.materialType == b.materialType &&
        equalFloat3(a.color, b.color) &&
        equalFloat3(a.specularColor, b.specularColor) &&
        equalFloat(a.roughness, b.roughness) &&
        equalFloat(a.ior, b.ior) &&
        equalFloat(a.alpha, b.alpha) &&
        a.texturePath == b.texturePath &&
        a.textureIndex == b.textureIndex &&
        a.textureEnabled == b.textureEnabled &&
        a.normalTexturePath == b.normalTexturePath &&
        a.normalTextureIndex == b.normalTextureIndex &&
        a.metallicTexturePath == b.metallicTexturePath &&
        a.metallicTextureIndex == b.metallicTextureIndex &&
        a.roughnessTexturePath == b.roughnessTexturePath &&
        a.roughnessTextureIndex == b.roughnessTextureIndex;
}

bool meshObjectUsesSourceMaterials(const MeshObject& object)
{
    if (!meshPrimitiveName(object).empty())
    {
        return false;
    }
    if (object.sourceMaterials.empty() || object.sourceMaterials.size() != object.mesh.materials.size())
    {
        return false;
    }

    for (size_t i = 0; i < object.mesh.materials.size(); ++i)
    {
        if (!equalMeshMaterialForSave(object.mesh.materials[i], object.sourceMaterials[i]))
        {
            return false;
        }
    }
    return true;
}

SceneConfigResult parseSceneConfig(const JsonValue& root)
{
    SceneConfigResult result;
    const JsonObject* rootObject = asObject(root);
    if (rootObject == nullptr)
    {
        result.error = "Scene config root must be a JSON object";
        return result;
    }

    if (const JsonValue* materialsField = findField(*rootObject, "materials"))
    {
        const JsonArray* materials = asArray(*materialsField);
        if (materials == nullptr)
        {
            result.error = "'materials' must be an array";
            return result;
        }

        for (size_t i = 0; i < materials->size(); ++i)
        {
            const JsonObject* materialObject = asObject((*materials)[i]);
            if (materialObject == nullptr)
            {
                result.error = "Each materials item must be an object";
                return result;
            }

            SceneMaterialConfig material;
            std::string error;
            if (!parseMaterialObject(*materialObject, "material_" + std::to_string(i), material, error))
            {
                result.error = error;
                return result;
            }
            result.config.materials.push_back(std::move(material));
        }
    }

    if (const JsonValue* sphereMaterialsField = findField(*rootObject, "sphereMaterials"))
    {
        const JsonArray* sphereMaterials = asArray(*sphereMaterialsField);
        if (sphereMaterials == nullptr)
        {
            result.error = "'sphereMaterials' must be an array";
            return result;
        }

        for (size_t i = 0; i < sphereMaterials->size(); ++i)
        {
            std::string materialName;
            std::string error;
            if (!parseMaterialReference(
                    (*sphereMaterials)[i],
                    result.config,
                    "sphere_material_" + std::to_string(i),
                    materialName,
                    error))
            {
                result.error = error;
                return result;
            }
            result.config.sphereMaterialRefs.push_back(std::move(materialName));
        }
    }

    if (const JsonValue* spheresField = findField(*rootObject, "spheres"))
    {
        result.config.hasSpheres = true;
        const JsonArray* spheres = asArray(*spheresField);
        if (spheres == nullptr)
        {
            result.error = "'spheres' must be an array";
            return result;
        }

        for (const JsonValue& item : *spheres)
        {
            const JsonObject* sphereObject = asObject(item);
            if (sphereObject == nullptr)
            {
                result.error = "Each spheres item must be an object";
                return result;
            }

            SphereConfig sphere;
            std::string error;
            if (const JsonValue* nameField = findField(*sphereObject, "name"))
            {
                const std::string* name = asString(*nameField);
                if (name == nullptr)
                {
                    result.error = "'name' must be a string";
                    return result;
                }
                sphere.name = *name;
            }
            if (!readFloat3(*sphereObject, "position", sphere.position, error) ||
                !readFloatField(*sphereObject, "radius", sphere.radius, error))
            {
                result.error = error;
                return result;
            }
            if (const JsonValue* materialField = findField(*sphereObject, "material"))
            {
                if (!parseMaterialReference(
                        *materialField,
                        result.config,
                        "sphere_material_" + std::to_string(result.config.spheres.size()),
                        sphere.materialOverride,
                        error))
                {
                    result.error = error;
                    return result;
                }
            }
            result.config.spheres.push_back(std::move(sphere));
        }
    }

    if (const JsonValue* meshField = findField(*rootObject, "mesh"))
    {
        const std::string* meshPath = asString(*meshField);
        if (meshPath == nullptr)
        {
            result.error = "'mesh' must be a string";
            return result;
        }
        MeshObjectConfig meshObject;
        meshObject.meshPath = *meshPath;
        result.config.meshObjects.push_back(std::move(meshObject));
        result.config.hasMeshObjects = true;
    }

    if (const JsonValue* meshObjectsField = findField(*rootObject, "meshObjects"))
    {
        result.config.hasMeshObjects = true;
        const JsonArray* meshObjects = asArray(*meshObjectsField);
        if (meshObjects == nullptr)
        {
            result.error = "'meshObjects' must be an array";
            return result;
        }

        for (const JsonValue& item : *meshObjects)
        {
            const JsonObject* object = asObject(item);
            if (object == nullptr)
            {
                result.error = "Each meshObjects item must be an object";
                return result;
            }

            const JsonValue* pathField = findField(*object, "path");
            if (pathField == nullptr)
            {
                pathField = findField(*object, "mesh");
            }
            const JsonValue* primitiveField = findField(*object, "primitive");
            const std::string* path = pathField != nullptr ? asString(*pathField) : nullptr;
            const std::string* primitive = primitiveField != nullptr ? asString(*primitiveField) : nullptr;
            if (path == nullptr && primitive == nullptr)
            {
                result.error = "Each mesh object must define string 'path' or 'primitive'";
                return result;
            }

            MeshObjectConfig meshObject;
            if (path != nullptr)
            {
                meshObject.meshPath = *path;
            }
            if (primitive != nullptr)
            {
                meshObject.primitive = *primitive;
            }
            if (const JsonValue* nameField = findField(*object, "name"))
            {
                const std::string* name = asString(*nameField);
                if (name == nullptr)
                {
                    result.error = "'name' must be a string";
                    return result;
                }
                meshObject.name = *name;
            }
            std::string error;
            if (!readFloat3(*object, "position", meshObject.transform.position, error) ||
                !readFloat3(*object, "rotation", meshObject.transform.rotation, error) ||
                !readFloat3(*object, "scale", meshObject.transform.scale, error))
            {
                result.error = error;
                return result;
            }
            if (const JsonValue* materialField = findField(*object, "material"))
            {
                if (!parseMaterialReference(
                        *materialField,
                        result.config,
                        "mesh_material_" + std::to_string(result.config.meshObjects.size()),
                        meshObject.materialOverride,
                        error))
                {
                    result.error = error;
                    return result;
                }
            }
            if (const JsonValue* materialOverrideField = findField(*object, "materialOverride"))
            {
                if (!parseMaterialReference(
                        *materialOverrideField,
                        result.config,
                        "mesh_material_" + std::to_string(result.config.meshObjects.size()),
                        meshObject.materialOverride,
                        error))
                {
                    result.error = error;
                    return result;
                }
            }
            result.config.meshObjects.push_back(meshObject);
        }
    }

    if (const JsonValue* cameraField = findField(*rootObject, "camera"))
    {
        const JsonObject* camera = asObject(*cameraField);
        if (camera == nullptr)
        {
            result.error = "'camera' must be an object";
            return result;
        }

        result.config.hasCamera = true;
        std::string error;
        if (!readFloat3(*camera, "position", result.config.camera.position, error) ||
            !readFloatField(*camera, "yaw", result.config.camera.yaw, error) ||
            !readFloatField(*camera, "pitch", result.config.camera.pitch, error) ||
            !readFloatField(*camera, "fov", result.config.camera.fov, error))
        {
            result.error = error;
            return result;
        }
    }

    std::string tuningError;
    if (!readSceneTuningFields(*rootObject, result.config, tuningError))
    {
        result.error = tuningError;
        return result;
    }
    if (const JsonValue* renderField = findField(*rootObject, "render"))
    {
        const JsonObject* render = asObject(*renderField);
        if (render == nullptr)
        {
            result.error = "'render' must be an object";
            return result;
        }
        if (!readSceneTuningFields(*render, result.config, tuningError))
        {
            result.error = tuningError;
            return result;
        }
    }

    if (const JsonValue* lightField = findField(*rootObject, "light"))
    {
        const JsonObject* light = asObject(*lightField);
        if (light == nullptr)
        {
            result.error = "'light' must be an object";
            return result;
        }
        std::string error;
        if (!readFloat3(*light, "position", result.config.lightPosition, error))
        {
            result.error = error;
            return result;
        }
        if (findField(*light, "intensity") != nullptr)
        {
            if (!readFloatField(*light, "intensity", result.config.lightIntensity, error))
            {
                result.error = error;
                return result;
            }
            result.config.hasLightIntensity = true;
        }
        if (findField(*light, "size") != nullptr || findField(*light, "radius") != nullptr)
        {
            const char* fieldName = findField(*light, "size") != nullptr ? "size" : "radius";
            if (!readFloatField(*light, fieldName, result.config.areaLightRadius, error))
            {
                result.error = error;
                return result;
            }
            result.config.hasAreaLightRadius = true;
        }
        result.config.hasLightPosition = true;
    }

    if (const JsonValue* environmentField = findField(*rootObject, "environment"))
    {
        const JsonObject* environment = asObject(*environmentField);
        if (environment == nullptr)
        {
            result.error = "'environment' must be an object";
            return result;
        }

        std::string error;
        if (!readStringField(*environment, "type", result.config.environmentType, error))
        {
            result.error = error;
            return result;
        }
        std::string path;
        if (!readStringField(*environment, "path", path, error))
        {
            result.error = error;
            return result;
        }
        if (!path.empty())
        {
            result.config.environmentPath = path;
        }
        if (findField(*environment, "horizonColor") != nullptr)
        {
            if (!readFloat3(*environment, "horizonColor", result.config.skyHorizonColor, error))
            {
                result.error = error;
                return result;
            }
            result.config.hasSkyHorizonColor = true;
        }
        if (findField(*environment, "zenithColor") != nullptr)
        {
            if (!readFloat3(*environment, "zenithColor", result.config.skyZenithColor, error))
            {
                result.error = error;
                return result;
            }
            result.config.hasSkyZenithColor = true;
        }
        if (findField(*environment, "gradientBlend") != nullptr)
        {
            if (!readFloatField(*environment, "gradientBlend", result.config.skyGradientBlend, error))
            {
                result.error = error;
                return result;
            }
            result.config.hasSkyGradientBlend = true;
        }
        if (findField(*environment, "intensity") != nullptr)
        {
            if (!readFloatField(*environment, "intensity", result.config.environmentIntensity, error))
            {
                result.error = error;
                return result;
            }
            result.config.hasEnvironmentIntensity = true;
        }
    }

    result.ok = true;
    return result;
}
} // namespace

SceneConfigResult loadSceneConfigFile(const std::filesystem::path& path)
{
    std::ifstream file(path, std::ios::binary);
    if (!file)
    {
        logError("Scene config could not be opened: " + path.string());
        return SceneConfigResult{false, {}, "Scene config could not be opened: " + path.string()};
    }

    std::ostringstream buffer;
    buffer << file.rdbuf();
    const std::string text = buffer.str();
    try
    {
        JsonParser parser(text);
        return parseSceneConfig(parser.parse());
    }
    catch (const std::exception& ex)
    {
        logError("Invalid scene config: " + std::string(ex.what()));
        return SceneConfigResult{false, {}, "Invalid scene config: " + std::string(ex.what())};
    }
}

SceneBuildResult buildDefaultSceneInput()
{
    SceneBuildResult result;
    result.ok = true;
    result.scene = makeBaseEditorScene();
    result.camera = CameraState{};
    result.camera.position = make_float3(0.0f, 7.0f, -16.0f);
    result.camera.yaw = 90.0f;
    result.camera.pitch = -22.0f;
    result.camera.fov = 48.0f;
    return result;
}

SceneBuildResult buildSceneFromConfig(const SceneConfig& config, const std::filesystem::path& baseDirectory, AssetCache& assets)
{
    SceneBuildResult result = buildDefaultSceneInput();
    if (!result.ok)
    {
        return result;
    }

    if (config.hasCamera)
    {
        result.camera = config.camera;
    }
    if (config.hasLightPosition)
    {
        result.scene.lightPosition = config.lightPosition;
    }
    if (config.hasExposure)
    {
        result.scene.exposure = config.exposure;
    }
    if (config.hasSkyIntensity)
    {
        result.scene.skyIntensity = config.skyIntensity;
    }
    if (config.hasSkyHorizonColor)
    {
        result.scene.skyHorizonColor = config.skyHorizonColor;
    }
    if (config.hasSkyZenithColor)
    {
        result.scene.skyZenithColor = config.skyZenithColor;
    }
    if (config.hasSkyGradientBlend)
    {
        result.scene.skyGradientBlend = config.skyGradientBlend;
    }
    if (config.hasLightIntensity)
    {
        result.scene.lightIntensity = config.lightIntensity;
    }
    if (config.hasAreaLightRadius)
    {
        result.scene.areaLightRadius = config.areaLightRadius;
    }
    if (config.hasEnvironmentIntensity)
    {
        result.scene.environmentIntensity = config.environmentIntensity;
    }
    result.scene.environmentType = config.environmentType.empty() ? "gradient" : config.environmentType;
    result.scene.environmentPath = config.environmentPath.string();
    result.scene.areaLightRadius = clampSceneAreaLightRadius(result.scene.areaLightRadius);
    result.scene.environmentIntensity = clampSceneEnvironmentIntensity(result.scene.environmentIntensity);
    if (!config.environmentPath.empty())
    {
        const std::filesystem::path environmentPath = config.environmentPath.is_absolute()
            ? config.environmentPath
            : baseDirectory / config.environmentPath;
        if (!loadPpmEnvironmentMap(environmentPath, result.scene.environmentMap))
        {
            const std::string warning = "Environment map could not be loaded, using gradient sky: " + environmentPath.string();
            result.warnings.push_back(warning);
            logWarning(warning);
        }
        else
        {
            result.scene.environmentPath = environmentPath.string();
            result.scene.environmentType = "map";
        }
    }
    result.scene.exposure = clampSceneExposure(result.scene.exposure);
    result.scene.skyIntensity = clampSceneSkyIntensity(result.scene.skyIntensity);
    setSceneSkyHorizonColor(result.scene, result.scene.skyHorizonColor);
    setSceneSkyZenithColor(result.scene, result.scene.skyZenithColor);
    setSceneSkyGradientBlend(result.scene, result.scene.skyGradientBlend);
    result.scene.lightIntensity = clampSceneLightIntensity(result.scene.lightIntensity);
    appendMaterialWarnings(config, result);
    const std::map<std::string, SceneMaterialConfig> materialMap = makeMaterialMap(config.materials);
    if (config.hasSpheres)
    {
        result.scene.spheres.clear();
        result.scene.materials.clear();
        for (const SphereConfig& sphereConfig : config.spheres)
        {
            result.scene.spheres.push_back({sphereConfig.name, sphereConfig.position, sphereConfig.radius});
            SphereMaterial material{make_float3(0.72f, 0.76f, 0.72f), MaterialDiffuse, make_float3(0.72f, 0.72f, 0.72f), 0.52f, 1.5f, 1.0f};
            if (!sphereConfig.materialOverride.empty())
            {
                const auto found = materialMap.find(sphereConfig.materialOverride);
                if (found != materialMap.end())
                {
                    material = toSphereMaterial(found->second);
                }
                else
                {
                    const std::string warning = "Sphere material '" + sphereConfig.materialOverride + "' was not found; default sphere material was kept.";
                    result.warnings.push_back(warning);
                    logWarning(warning);
                }
            }
            result.scene.materials.push_back(material);
        }
    }
    else if (!config.sphereMaterialRefs.empty() && result.scene.spheres.empty())
    {
        result.scene.spheres.push_back({"Сфера", make_float3(0.0f, 1.25f, 0.0f), 1.25f});
        result.scene.materials.push_back({make_float3(0.72f, 0.76f, 0.72f), MaterialDiffuse, make_float3(0.72f, 0.72f, 0.72f), 0.52f, 1.5f, 1.0f});
    }
    applySphereMaterialConfig(config, materialMap, result);

    if (config.meshObjects.empty())
    {
        if (config.hasMeshObjects)
        {
            result.scene.meshObjects.clear();
            result.scene.mesh = {};
            clampScene(result.scene);
            return result;
        }

        const std::string warning = "Scene config has no mesh objects; using default editor scene.";
        result.warnings.push_back(warning);
        logWarning(warning);
        return result;
    }

    MeshData combinedMesh;
    std::vector<MeshObject> meshObjects;
    for (const MeshObjectConfig& object : config.meshObjects)
    {
        MeshData objectMesh;
        std::filesystem::path meshPath;
        if (!object.primitive.empty())
        {
            objectMesh = makePrimitiveMeshByName(object.primitive);
            if (isEmptyMesh(objectMesh))
            {
                result.ok = false;
                result.error = "Unknown built-in mesh primitive: " + object.primitive;
                logError(result.error);
                return result;
            }
        }
        else
        {
            meshPath = resolvePath(object.meshPath, baseDirectory);
            const ObjLoadResult& loaded = assets.loadMesh(meshPath);
            if (!loaded.ok)
            {
                result.ok = false;
                result.error = loaded.error;
                logError(result.error);
                return result;
            }
            objectMesh = loaded.mesh;
        }
        std::vector<MeshMaterial> sourceMaterials = objectMesh.materials;
        applyMeshMaterialOverride(object, materialMap, objectMesh, result);
        MeshObject meshObject;
        meshObject.assetReference = object.primitive.empty() ? meshPath.string() : "built-in " + object.primitive;
        meshObject.displayName = !object.name.empty()
            ? object.name
            : (object.primitive.empty()
                ? (meshPath.filename().string().empty() ? meshPath.string() : meshPath.filename().string())
                : object.primitive);
        meshObject.mesh = objectMesh;
        meshObject.sourceMaterials = std::move(sourceMaterials);
        meshObject.position = object.transform.position;
        meshObject.rotation = object.transform.rotation;
        meshObject.scale = object.transform.scale;
        meshObject.transform = makeTransformMatrix(object.transform);
        meshObjects.push_back(std::move(meshObject));
        appendMesh(combinedMesh, transformMesh(objectMesh, object.transform));
    }

    if (!isEmptyMesh(combinedMesh) && hasValidMeshMaterialIndices(combinedMesh) && !meshObjects.empty())
    {
        result.scene.mesh = std::move(combinedMesh);
        result.scene.meshObjects = std::move(meshObjects);
    }
    else
    {
        result.ok = false;
        result.error = "Scene config produced an invalid mesh.";
        logError(result.error);
    }

    return result;
}

SceneBuildResult buildSceneFromConfig(const SceneConfig& config, const std::filesystem::path& baseDirectory)
{
    AssetCache assets;
    return buildSceneFromConfig(config, baseDirectory, assets);
}

SceneBuildResult buildSceneFromMeshPath(const std::filesystem::path& meshPath, AssetCache& assets)
{
    SceneConfig config;
    MeshObjectConfig meshObject;
    meshObject.meshPath = meshPath;
    config.meshObjects.push_back(std::move(meshObject));
    return buildSceneFromConfig(config, {}, assets);
}

SceneBuildResult buildSceneFromMeshPath(const std::filesystem::path& meshPath)
{
    AssetCache assets;
    return buildSceneFromMeshPath(meshPath, assets);
}

bool reloadScenePresetFromConfig(
    const std::filesystem::path& configPath,
    SceneBuildResult& preset,
    SceneState& scene,
    CameraState& camera,
    AssetCache& assets,
    bool& accumulationResetRequested,
    std::string& error)
{
    accumulationResetRequested = false;
    error.clear();
    if (configPath.empty())
    {
        error = "Scene preset has no source config path.";
        return false;
    }

    const SceneConfigResult config = loadSceneConfigFile(configPath);
    if (!config.ok)
    {
        error = config.error;
        return false;
    }

    SceneBuildResult reloaded = buildSceneFromConfig(config.config, configPath.parent_path(), assets);
    if (!reloaded.ok)
    {
        error = reloaded.error;
        return false;
    }

    preset = reloaded;
    scene = reloaded.scene;
    camera = reloaded.camera;
    accumulationResetRequested = true;
    return true;
}

bool applyScenePresetByIndex(const std::vector<SceneBuildResult>& presets, const int index, SceneState& scene, CameraState& camera)
{
    if (index < 0 || index >= static_cast<int>(presets.size()) || !presets[static_cast<size_t>(index)].ok)
    {
        return false;
    }

    scene = presets[static_cast<size_t>(index)].scene;
    camera = presets[static_cast<size_t>(index)].camera;
    return true;
}

bool saveScenePresetByIndex(std::vector<SceneBuildResult>& presets, const int index, const SceneState& scene, const CameraState& camera)
{
    if (index < 0 || index >= static_cast<int>(presets.size()))
    {
        return false;
    }

    SceneBuildResult& preset = presets[static_cast<size_t>(index)];
    preset.ok = true;
    preset.scene = scene;
    preset.camera = camera;
    clampScene(preset.scene);
    return true;
}

bool saveSceneToConfigFile(const std::filesystem::path& path, const SceneState& scene, const CameraState& camera, std::string& error)
{
    error.clear();
    std::ofstream output(path, std::ios::binary);
    if (!output)
    {
        error = "Could not open scene file for writing: " + path.string();
        return false;
    }

    output << std::fixed << std::setprecision(3);
    const auto writeFloat3 = [&output](const float3 value)
    {
        output << "[" << value.x << ", " << value.y << ", " << value.z << "]";
    };
    const auto writeSphereMaterial = [&output, &writeFloat3](const SphereMaterial& material)
    {
        output << "{ \"type\": \"" << materialTypeToConfigName(material.materialType) << "\", \"baseColor\": ";
        writeFloat3(material.color);
        output << ", \"roughness\": " << material.roughness
               << ", \"ior\": " << material.ior
               << ", \"alpha\": " << material.alpha << " }";
    };
    const auto writeMeshMaterial = [&output, &writeFloat3](const MeshMaterial& material)
    {
        output << "{ \"type\": \"" << materialTypeToConfigName(material.materialType) << "\", \"baseColor\": ";
        writeFloat3(material.color);
        output << ", \"roughness\": " << material.roughness
               << ", \"ior\": " << material.ior
               << ", \"alpha\": " << material.alpha << " }";
    };

    output << "{\n";
    output << "  \"camera\": {\n";
    output << "    \"position\": ";
    writeFloat3(camera.position);
    output << ",\n    \"yaw\": " << camera.yaw
           << ",\n    \"pitch\": " << camera.pitch
           << ",\n    \"fov\": " << camera.fov << "\n";
    output << "  },\n";
    output << "  \"light\": {\n";
    output << "    \"position\": ";
    writeFloat3(scene.lightPosition);
    output << ",\n    \"intensity\": " << scene.lightIntensity
           << ",\n    \"size\": " << scene.areaLightRadius << "\n";
    output << "  },\n";
    output << "  \"render\": {\n";
    output << "    \"exposure\": " << scene.exposure
           << ",\n    \"skyIntensity\": " << scene.skyIntensity
           << ",\n    \"skyGradientBlend\": " << scene.skyGradientBlend << "\n";
    output << "  },\n";
    output << "  \"environment\": {\n";
    output << "    \"type\": \"" << jsonEscape(scene.environmentType.empty() ? "gradient" : scene.environmentType) << "\",\n";
    output << "    \"intensity\": " << scene.environmentIntensity << ",\n";
    output << "    \"horizonColor\": ";
    writeFloat3(scene.skyHorizonColor);
    output << ",\n    \"zenithColor\": ";
    writeFloat3(scene.skyZenithColor);
    output << ",\n    \"gradientBlend\": " << scene.skyGradientBlend;
    if (!scene.environmentPath.empty())
    {
        output << ",\n    \"path\": \"" << jsonEscape(scene.environmentPath) << "\"\n";
    }
    else
    {
        output << "\n";
    }
    output << "  },\n";

    output << "  \"spheres\": [\n";
    for (size_t i = 0; i < scene.spheres.size(); ++i)
    {
        const SphereGeometry& sphere = scene.spheres[i];
        output << "    { ";
        if (!sphere.displayName.empty())
        {
            output << "\"name\": \"" << jsonEscape(sphere.displayName) << "\", ";
        }
        output << "\"position\": ";
        writeFloat3(sphere.center);
        output << ", \"radius\": " << sphere.radius;
        if (i < scene.materials.size())
        {
            output << ", \"material\": ";
            writeSphereMaterial(scene.materials[i]);
        }
        output << " }" << (i + 1 < scene.spheres.size() ? "," : "") << "\n";
    }
    output << "  ],\n";

    output << "  \"meshObjects\": [\n";
    for (size_t i = 0; i < scene.meshObjects.size(); ++i)
    {
        const MeshObject& object = scene.meshObjects[i];
        const std::string primitive = meshPrimitiveName(object);
        output << "    {\n";
        output << "      \"name\": \"" << jsonEscape(object.displayName) << "\",\n";
        if (!primitive.empty())
        {
            output << "      \"primitive\": \"" << primitive << "\",\n";
        }
        else
        {
            output << "      \"path\": \"" << jsonEscape(object.assetReference) << "\",\n";
        }
        output << "      \"position\": ";
        writeFloat3(object.position);
        output << ",\n      \"rotation\": ";
        writeFloat3(object.rotation);
        output << ",\n      \"scale\": ";
        writeFloat3(object.scale);
        if (!object.mesh.materials.empty() && !meshObjectUsesSourceMaterials(object))
        {
            output << ",\n      \"material\": ";
            writeMeshMaterial(object.mesh.materials.front());
        }
        output << "\n    }" << (i + 1 < scene.meshObjects.size() ? "," : "") << "\n";
    }
    output << "  ]\n";
    output << "}\n";

    if (!output)
    {
        error = "Could not write complete scene file: " + path.string();
        return false;
    }
    return true;
}

bool resetSceneViewFromPreset(const SceneBuildResult& preset, SceneState& scene, CameraState& camera)
{
    if (!preset.ok)
    {
        return false;
    }

    camera = preset.camera;
    scene.lightPosition = preset.scene.lightPosition;
    return true;
}
