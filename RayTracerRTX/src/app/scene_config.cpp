#include "scene_config.h"

#include "gltf_loader.h"
#include "obj_loader.h"

#include <algorithm>
#include <cmath>
#include <cctype>
#include <fstream>
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
        !readStringField(object, "normal", material.normalTexturePath, error))
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

std::string lowerExtension(const std::filesystem::path& path)
{
    std::string extension = path.extension().string();
    std::transform(extension.begin(), extension.end(), extension.begin(), [](const unsigned char ch)
    {
        return static_cast<char>(std::tolower(ch));
    });
    return extension;
}

ObjLoadResult loadMeshByExtension(const std::filesystem::path& path)
{
    const std::string extension = lowerExtension(path);
    if (extension == ".gltf")
    {
        GltfLoadResult loaded = loadGltfMesh(path);
        return ObjLoadResult{loaded.ok, std::move(loaded.mesh), loaded.error};
    }
    return loadObjMesh(path);
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
    return base;
}

void appendMaterialWarnings(const SceneConfig& config, SceneBuildResult& result)
{
    for (const SceneMaterialConfig& material : config.materials)
    {
        if (material.usedFallbackType)
        {
            result.warnings.push_back("Material '" + material.name + "' uses an unknown type; matte fallback was applied.");
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
            result.warnings.push_back("Sphere material '" + name + "' was not found; default sphere material was kept.");
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
        result.warnings.push_back("Mesh material override '" + object.materialOverride + "' was not found; source mesh materials were kept.");
        return;
    }

    for (MeshMaterial& material : mesh.materials)
    {
        material = toMeshMaterial(found->second, material);
    }
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
    }

    if (const JsonValue* meshObjectsField = findField(*rootObject, "meshObjects"))
    {
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
            const std::string* path = pathField != nullptr ? asString(*pathField) : nullptr;
            if (path == nullptr)
            {
                result.error = "Each mesh object must define string 'path'";
                return result;
            }

            MeshObjectConfig meshObject;
            meshObject.meshPath = *path;
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
        result.config.hasLightPosition = true;
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
        return SceneConfigResult{false, {}, "Invalid scene config: " + std::string(ex.what())};
    }
}

SceneBuildResult buildDefaultSceneInput()
{
    SceneBuildResult result;
    result.ok = true;
    result.scene = makeDefaultScene();
    result.camera = CameraState{};
    return result;
}

SceneBuildResult buildSceneFromConfig(const SceneConfig& config, const std::filesystem::path& baseDirectory)
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
    if (config.hasLightIntensity)
    {
        result.scene.lightIntensity = config.lightIntensity;
    }
    result.scene.exposure = clampSceneExposure(result.scene.exposure);
    result.scene.skyIntensity = clampSceneSkyIntensity(result.scene.skyIntensity);
    result.scene.lightIntensity = clampSceneLightIntensity(result.scene.lightIntensity);
    appendMaterialWarnings(config, result);
    const std::map<std::string, SceneMaterialConfig> materialMap = makeMaterialMap(config.materials);
    applySphereMaterialConfig(config, materialMap, result);

    if (config.meshObjects.empty())
    {
        result.warnings.push_back("Scene config has no mesh objects; using default demo mesh.");
        return result;
    }

    MeshData combinedMesh;
    std::vector<MeshObject> meshObjects;
    for (const MeshObjectConfig& object : config.meshObjects)
    {
        const std::filesystem::path meshPath = resolvePath(object.meshPath, baseDirectory);
        const ObjLoadResult loaded = loadMeshByExtension(meshPath);
        if (!loaded.ok)
        {
            result.ok = false;
            result.error = loaded.error;
            return result;
        }
        MeshData objectMesh = loaded.mesh;
        applyMeshMaterialOverride(object, materialMap, objectMesh, result);
        MeshObject meshObject;
        meshObject.assetReference = meshPath.string();
        meshObject.mesh = objectMesh;
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
    }

    return result;
}

SceneBuildResult buildSceneFromMeshPath(const std::filesystem::path& meshPath)
{
    SceneConfig config;
    MeshObjectConfig meshObject;
    meshObject.meshPath = meshPath;
    config.meshObjects.push_back(std::move(meshObject));
    return buildSceneFromConfig(config, {});
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
