#include "gltf_loader.h"

#include <algorithm>
#include <cmath>
#include <cctype>
#include <cstdint>
#include <cstring>
#include <fstream>
#include <iterator>
#include <map>
#include <sstream>
#include <stdexcept>
#include <string_view>
#include <variant>

namespace
{
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

    char peek() const { return text_[pos_]; }
    char advance() { return text_[pos_++]; }
    bool isAtEnd() const { return pos_ >= text_.size(); }

    [[noreturn]] void fail(const std::string& message) const
    {
        std::ostringstream out;
        out << message << " at byte " << pos_;
        throw std::runtime_error(out.str());
    }

    std::string_view text_;
    size_t pos_ = 0;
};

const JsonObject* asObject(const JsonValue& value) { return std::get_if<JsonObject>(&value.value); }
const JsonArray* asArray(const JsonValue& value) { return std::get_if<JsonArray>(&value.value); }
const std::string* asString(const JsonValue& value) { return std::get_if<std::string>(&value.value); }
const double* asNumber(const JsonValue& value) { return std::get_if<double>(&value.value); }

const JsonValue* findField(const JsonObject& object, const std::string& name)
{
    const auto it = object.find(name);
    return it == object.end() ? nullptr : &it->second;
}

int intField(const JsonObject& object, const std::string& name, const int fallback = 0)
{
    const JsonValue* field = findField(object, name);
    const double* value = field != nullptr ? asNumber(*field) : nullptr;
    return value == nullptr ? fallback : static_cast<int>(*value);
}

std::string stringField(const JsonObject& object, const std::string& name)
{
    const JsonValue* field = findField(object, name);
    const std::string* value = field != nullptr ? asString(*field) : nullptr;
    return value == nullptr ? std::string{} : *value;
}

float numberField(const JsonObject& object, const std::string& name, const float fallback)
{
    const JsonValue* field = findField(object, name);
    const double* value = field != nullptr ? asNumber(*field) : nullptr;
    return value == nullptr ? fallback : static_cast<float>(*value);
}

float3 readFloat3Array(const JsonObject& object, const std::string& name, const float3 fallback)
{
    const JsonValue* field = findField(object, name);
    const JsonArray* array = field != nullptr ? asArray(*field) : nullptr;
    if (array == nullptr || array->size() < 3)
    {
        return fallback;
    }
    const double* x = asNumber((*array)[0]);
    const double* y = asNumber((*array)[1]);
    const double* z = asNumber((*array)[2]);
    return x != nullptr && y != nullptr && z != nullptr
        ? make_float3(static_cast<float>(*x), static_cast<float>(*y), static_cast<float>(*z))
        : fallback;
}

float4 readFloat4Array(const JsonObject& object, const std::string& name, const float4 fallback)
{
    const JsonValue* field = findField(object, name);
    const JsonArray* array = field != nullptr ? asArray(*field) : nullptr;
    if (array == nullptr || array->size() < 4)
    {
        return fallback;
    }
    const double* x = asNumber((*array)[0]);
    const double* y = asNumber((*array)[1]);
    const double* z = asNumber((*array)[2]);
    const double* w = asNumber((*array)[3]);
    return x != nullptr && y != nullptr && z != nullptr && w != nullptr
        ? make_float4(static_cast<float>(*x), static_cast<float>(*y), static_cast<float>(*z), static_cast<float>(*w))
        : fallback;
}

float3 add3(const float3 a, const float3 b) { return make_float3(a.x + b.x, a.y + b.y, a.z + b.z); }
float3 sub3(const float3 a, const float3 b) { return make_float3(a.x - b.x, a.y - b.y, a.z - b.z); }
float dot3(const float3 a, const float3 b) { return a.x * b.x + a.y * b.y + a.z * b.z; }

float3 normalize3(const float3 value)
{
    const float length = std::sqrt(dot3(value, value));
    if (length <= 1e-8f)
    {
        return make_float3(0.0f, 1.0f, 0.0f);
    }
    return make_float3(value.x / length, value.y / length, value.z / length);
}

float3 applyNodeTransform(const float3 value, const float3 translation, const float3 scale)
{
    return make_float3(value.x * scale.x + translation.x, value.y * scale.y + translation.y, value.z * scale.z + translation.z);
}

float3 computeTriangleTangent(const MeshVertex& a, const MeshVertex& b, const MeshVertex& c)
{
    if (a.hasTexcoord == 0 || b.hasTexcoord == 0 || c.hasTexcoord == 0)
    {
        return make_float3(0.0f, 0.0f, 0.0f);
    }
    const float3 edge1 = sub3(b.position, a.position);
    const float3 edge2 = sub3(c.position, a.position);
    const float du1 = b.texcoord.x - a.texcoord.x;
    const float dv1 = b.texcoord.y - a.texcoord.y;
    const float du2 = c.texcoord.x - a.texcoord.x;
    const float dv2 = c.texcoord.y - a.texcoord.y;
    const float determinant = du1 * dv2 - dv1 * du2;
    if (std::fabs(determinant) <= 1e-8f)
    {
        return make_float3(0.0f, 0.0f, 0.0f);
    }
    const float inv = 1.0f / determinant;
    return normalize3(make_float3(
        (edge1.x * dv2 - edge2.x * dv1) * inv,
        (edge1.y * dv2 - edge2.y * dv1) * inv,
        (edge1.z * dv2 - edge2.z * dv1) * inv));
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
        std::string r;
        std::string g;
        std::string b;
        if (!readPpmToken(file, r) || !readPpmToken(file, g) || !readPpmToken(file, b))
        {
            return false;
        }
        const auto convert = [maxValue](const std::string& value)
        {
            const int parsed = std::clamp(std::stoi(value), 0, maxValue);
            return static_cast<unsigned char>((parsed * 255) / maxValue);
        };
        pixels.push_back(make_uchar4(convert(r), convert(g), convert(b), 255u));
    }
    texture.path = path.string();
    texture.width = static_cast<unsigned int>(width);
    texture.height = static_cast<unsigned int>(height);
    texture.pixels = std::move(pixels);
    return true;
}

struct BufferView
{
    int buffer = 0;
    size_t byteOffset = 0;
    size_t byteLength = 0;
    size_t byteStride = 0;
};

struct Accessor
{
    int bufferView = -1;
    size_t byteOffset = 0;
    int componentType = 0;
    size_t count = 0;
    std::string type;
};

size_t componentSize(const int componentType)
{
    if (componentType == 5126 || componentType == 5125)
    {
        return 4;
    }
    if (componentType == 5123)
    {
        return 2;
    }
    if (componentType == 5121)
    {
        return 1;
    }
    throw std::runtime_error("Unsupported glTF accessor component type");
}

size_t componentCount(const std::string& type)
{
    if (type == "SCALAR") return 1;
    if (type == "VEC2") return 2;
    if (type == "VEC3") return 3;
    if (type == "VEC4") return 4;
    throw std::runtime_error("Unsupported glTF accessor type");
}

const unsigned char* accessorPointer(
    const std::vector<std::vector<unsigned char>>& buffers,
    const std::vector<BufferView>& views,
    const Accessor& accessor,
    const size_t index)
{
    if (accessor.bufferView < 0 || static_cast<size_t>(accessor.bufferView) >= views.size())
    {
        throw std::runtime_error("glTF accessor references invalid bufferView");
    }
    const BufferView& view = views[static_cast<size_t>(accessor.bufferView)];
    if (view.buffer < 0 || static_cast<size_t>(view.buffer) >= buffers.size())
    {
        throw std::runtime_error("glTF bufferView references invalid buffer");
    }
    const size_t packedStride = componentSize(accessor.componentType) * componentCount(accessor.type);
    const size_t stride = view.byteStride == 0 ? packedStride : view.byteStride;
    const size_t offset = view.byteOffset + accessor.byteOffset + index * stride;
    if (offset + packedStride > buffers[static_cast<size_t>(view.buffer)].size())
    {
        throw std::runtime_error("glTF accessor reads past buffer bounds");
    }
    return buffers[static_cast<size_t>(view.buffer)].data() + offset;
}

float readFloatComponent(const unsigned char* ptr)
{
    float value = 0.0f;
    std::memcpy(&value, ptr, sizeof(float));
    return value;
}

float3 readVec3(
    const std::vector<std::vector<unsigned char>>& buffers,
    const std::vector<BufferView>& views,
    const std::vector<Accessor>& accessors,
    const int accessorIndex,
    const size_t index)
{
    if (accessorIndex < 0 || static_cast<size_t>(accessorIndex) >= accessors.size())
    {
        return make_float3(0.0f, 1.0f, 0.0f);
    }
    const Accessor& accessor = accessors[static_cast<size_t>(accessorIndex)];
    if (accessor.componentType != 5126 || accessor.type != "VEC3")
    {
        throw std::runtime_error("glTF VEC3 accessor must use FLOAT components");
    }
    const unsigned char* ptr = accessorPointer(buffers, views, accessor, index);
    return make_float3(readFloatComponent(ptr), readFloatComponent(ptr + 4), readFloatComponent(ptr + 8));
}

float2 readVec2(
    const std::vector<std::vector<unsigned char>>& buffers,
    const std::vector<BufferView>& views,
    const std::vector<Accessor>& accessors,
    const int accessorIndex,
    const size_t index)
{
    if (accessorIndex < 0 || static_cast<size_t>(accessorIndex) >= accessors.size())
    {
        return make_float2(0.0f, 0.0f);
    }
    const Accessor& accessor = accessors[static_cast<size_t>(accessorIndex)];
    if (accessor.componentType != 5126 || accessor.type != "VEC2")
    {
        throw std::runtime_error("glTF VEC2 accessor must use FLOAT components");
    }
    const unsigned char* ptr = accessorPointer(buffers, views, accessor, index);
    return make_float2(readFloatComponent(ptr), readFloatComponent(ptr + 4));
}

std::uint32_t readIndex(
    const std::vector<std::vector<unsigned char>>& buffers,
    const std::vector<BufferView>& views,
    const std::vector<Accessor>& accessors,
    const int accessorIndex,
    const size_t index)
{
    const Accessor& accessor = accessors[static_cast<size_t>(accessorIndex)];
    const unsigned char* ptr = accessorPointer(buffers, views, accessor, index);
    if (accessor.componentType == 5125)
    {
        std::uint32_t value = 0;
        std::memcpy(&value, ptr, sizeof(value));
        return value;
    }
    if (accessor.componentType == 5123)
    {
        std::uint16_t value = 0;
        std::memcpy(&value, ptr, sizeof(value));
        return value;
    }
    if (accessor.componentType == 5121)
    {
        return *ptr;
    }
    throw std::runtime_error("glTF index accessor must use unsigned integer components");
}

std::vector<unsigned char> readBinaryFile(const std::filesystem::path& path)
{
    std::ifstream file(path, std::ios::binary);
    if (!file)
    {
        throw std::runtime_error("glTF binary buffer could not be opened: " + path.string());
    }
    return std::vector<unsigned char>(std::istreambuf_iterator<char>(file), {});
}

int attributeAccessor(const JsonObject& attributes, const std::string& name)
{
    const JsonValue* field = findField(attributes, name);
    const double* value = field != nullptr ? asNumber(*field) : nullptr;
    return value == nullptr ? -1 : static_cast<int>(*value);
}
} // namespace

GltfLoadResult loadGltfMesh(const std::filesystem::path& path)
{
    std::ifstream file(path, std::ios::binary);
    if (!file)
    {
        return GltfLoadResult{false, {}, "glTF file could not be opened: " + path.string()};
    }

    std::ostringstream buffer;
    buffer << file.rdbuf();

    try
    {
        const std::string text = buffer.str();
        JsonParser parser(text);
        const JsonValue rootValue = parser.parse();
        const JsonObject* root = asObject(rootValue);
        if (root == nullptr)
        {
            return GltfLoadResult{false, {}, "glTF root must be a JSON object"};
        }

        std::vector<std::vector<unsigned char>> buffers;
        const JsonValue* buffersField = findField(*root, "buffers");
        if (const JsonArray* jsonBuffers = buffersField != nullptr ? asArray(*buffersField) : nullptr)
        {
            for (const JsonValue& item : *jsonBuffers)
            {
                const JsonObject* object = asObject(item);
                const std::string uri = object != nullptr ? stringField(*object, "uri") : std::string{};
                if (uri.empty() || uri.find("data:") == 0)
                {
                    throw std::runtime_error("Only external .bin glTF buffers are supported");
                }
                buffers.push_back(readBinaryFile(path.parent_path() / uri));
            }
        }

        std::vector<BufferView> views;
        const JsonValue* viewsField = findField(*root, "bufferViews");
        if (const JsonArray* jsonViews = viewsField != nullptr ? asArray(*viewsField) : nullptr)
        {
            for (const JsonValue& item : *jsonViews)
            {
                const JsonObject* object = asObject(item);
                if (object == nullptr)
                {
                    throw std::runtime_error("glTF bufferView must be an object");
                }
                views.push_back(BufferView{
                    intField(*object, "buffer"),
                    static_cast<size_t>(intField(*object, "byteOffset")),
                    static_cast<size_t>(intField(*object, "byteLength")),
                    static_cast<size_t>(intField(*object, "byteStride"))});
            }
        }

        std::vector<Accessor> accessors;
        const JsonValue* accessorsField = findField(*root, "accessors");
        if (const JsonArray* jsonAccessors = accessorsField != nullptr ? asArray(*accessorsField) : nullptr)
        {
            for (const JsonValue& item : *jsonAccessors)
            {
                const JsonObject* object = asObject(item);
                if (object == nullptr)
                {
                    throw std::runtime_error("glTF accessor must be an object");
                }
                accessors.push_back(Accessor{
                    intField(*object, "bufferView", -1),
                    static_cast<size_t>(intField(*object, "byteOffset")),
                    intField(*object, "componentType"),
                    static_cast<size_t>(intField(*object, "count")),
                    stringField(*object, "type")});
            }
        }

        MeshData mesh;
        const JsonValue* materialsField = findField(*root, "materials");
        if (const JsonArray* jsonMaterials = materialsField != nullptr ? asArray(*materialsField) : nullptr)
        {
            for (const JsonValue& item : *jsonMaterials)
            {
                const JsonObject* object = asObject(item);
                MeshMaterial material;
                material.name = object != nullptr ? stringField(*object, "name") : "gltf_material";
                material.color = make_float3(0.8f, 0.8f, 0.8f);
                if (object != nullptr)
                {
                    const JsonValue* pbrField = findField(*object, "pbrMetallicRoughness");
                    if (const JsonObject* pbr = pbrField != nullptr ? asObject(*pbrField) : nullptr)
                    {
                        const float4 baseColor = readFloat4Array(*pbr, "baseColorFactor", make_float4(0.8f, 0.8f, 0.8f, 1.0f));
                        material.color = make_float3(baseColor.x, baseColor.y, baseColor.z);
                        material.alpha = baseColor.w;
                        const float metallic = numberField(*pbr, "metallicFactor", 0.0f);
                        material.roughness = std::clamp(numberField(*pbr, "roughnessFactor", 0.5f), 0.045f, 1.0f);
                        material.materialType = metallic >= 0.5f ? MaterialMetal : MaterialDiffuse;
                        const JsonValue* textureInfoField = findField(*pbr, "baseColorTexture");
                        if (const JsonObject* textureInfo = textureInfoField != nullptr ? asObject(*textureInfoField) : nullptr)
                        {
                            const int textureIndex = intField(*textureInfo, "index", -1);
                            const JsonValue* texturesField = findField(*root, "textures");
                            const JsonValue* imagesField = findField(*root, "images");
                            const JsonArray* textures = texturesField != nullptr ? asArray(*texturesField) : nullptr;
                            const JsonArray* images = imagesField != nullptr ? asArray(*imagesField) : nullptr;
                            if (textures != nullptr && images != nullptr && textureIndex >= 0 && static_cast<size_t>(textureIndex) < textures->size())
                            {
                                const JsonObject* texture = asObject((*textures)[static_cast<size_t>(textureIndex)]);
                                const int source = texture != nullptr ? intField(*texture, "source", -1) : -1;
                                if (source >= 0 && static_cast<size_t>(source) < images->size())
                                {
                                    const JsonObject* image = asObject((*images)[static_cast<size_t>(source)]);
                                    const std::string uri = image != nullptr ? stringField(*image, "uri") : std::string{};
                                    if (!uri.empty())
                                    {
                                        material.texturePath = uri;
                                        MeshTexture loadedTexture;
                                        if (loadPpmTexture(path.parent_path() / uri, loadedTexture))
                                        {
                                            material.textureIndex = static_cast<int>(mesh.textures.size());
                                            mesh.textures.push_back(std::move(loadedTexture));
                                        }
                                    }
                                }
                            }
                        }
                    }
                }
                mesh.materials.push_back(std::move(material));
            }
        }
        if (mesh.materials.empty())
        {
            MeshMaterial material;
            material.name = "gltf_default";
            material.color = make_float3(0.8f, 0.8f, 0.8f);
            mesh.materials.push_back(std::move(material));
        }

        const JsonValue* meshesField = findField(*root, "meshes");
        const JsonArray* meshes = meshesField != nullptr ? asArray(*meshesField) : nullptr;
        if (meshes == nullptr || meshes->empty())
        {
            return GltfLoadResult{false, {}, "glTF file contains no meshes"};
        }

        std::vector<std::pair<int, std::pair<float3, float3>>> nodeMeshes;
        const JsonValue* nodesField = findField(*root, "nodes");
        if (const JsonArray* nodes = nodesField != nullptr ? asArray(*nodesField) : nullptr)
        {
            for (const JsonValue& item : *nodes)
            {
                const JsonObject* node = asObject(item);
                if (node == nullptr || findField(*node, "mesh") == nullptr)
                {
                    continue;
                }
                nodeMeshes.push_back({
                    intField(*node, "mesh"),
                    {
                        readFloat3Array(*node, "translation", make_float3(0.0f, 0.0f, 0.0f)),
                        readFloat3Array(*node, "scale", make_float3(1.0f, 1.0f, 1.0f))
                    }});
            }
        }
        if (nodeMeshes.empty())
        {
            nodeMeshes.push_back({0, {make_float3(0.0f, 0.0f, 0.0f), make_float3(1.0f, 1.0f, 1.0f)}});
        }

        for (const auto& nodeMesh : nodeMeshes)
        {
            const int meshIndex = nodeMesh.first;
            if (meshIndex < 0 || static_cast<size_t>(meshIndex) >= meshes->size())
            {
                throw std::runtime_error("glTF node references invalid mesh");
            }
            const JsonObject* meshObject = asObject((*meshes)[static_cast<size_t>(meshIndex)]);
            const JsonValue* primitivesField = meshObject != nullptr ? findField(*meshObject, "primitives") : nullptr;
            const JsonArray* primitives = primitivesField != nullptr ? asArray(*primitivesField) : nullptr;
            if (primitives == nullptr)
            {
                continue;
            }
            const float3 translation = nodeMesh.second.first;
            const float3 scale = nodeMesh.second.second;
            for (const JsonValue& primitiveValue : *primitives)
            {
                const JsonObject* primitive = asObject(primitiveValue);
                const JsonValue* attributesField = primitive != nullptr ? findField(*primitive, "attributes") : nullptr;
                const JsonObject* attributes = attributesField != nullptr ? asObject(*attributesField) : nullptr;
                if (primitive == nullptr || attributes == nullptr)
                {
                    continue;
                }
                const int positionAccessor = attributeAccessor(*attributes, "POSITION");
                if (positionAccessor < 0 || static_cast<size_t>(positionAccessor) >= accessors.size())
                {
                    throw std::runtime_error("glTF primitive missing POSITION accessor");
                }
                const int normalAccessor = attributeAccessor(*attributes, "NORMAL");
                const int texcoordAccessor = attributeAccessor(*attributes, "TEXCOORD_0");
                const Accessor& positions = accessors[static_cast<size_t>(positionAccessor)];
                const std::uint32_t vertexOffset = static_cast<std::uint32_t>(mesh.vertices.size());
                for (size_t i = 0; i < positions.count; ++i)
                {
                    MeshVertex vertex;
                    vertex.position = applyNodeTransform(readVec3(buffers, views, accessors, positionAccessor, i), translation, scale);
                    vertex.normal = normalize3(readVec3(buffers, views, accessors, normalAccessor, i));
                    if (texcoordAccessor >= 0)
                    {
                        vertex.texcoord = readVec2(buffers, views, accessors, texcoordAccessor, i);
                        vertex.hasTexcoord = 1;
                    }
                    mesh.vertices.push_back(vertex);
                }

                const std::uint32_t materialIndex = static_cast<std::uint32_t>(std::clamp(
                    intField(*primitive, "material", 0),
                    0,
                    static_cast<int>(mesh.materials.size() - 1)));
                const int indicesAccessor = intField(*primitive, "indices", -1);
                std::vector<std::uint32_t> indices;
                if (indicesAccessor >= 0)
                {
                    const Accessor& accessor = accessors[static_cast<size_t>(indicesAccessor)];
                    indices.reserve(accessor.count);
                    for (size_t i = 0; i < accessor.count; ++i)
                    {
                        indices.push_back(vertexOffset + readIndex(buffers, views, accessors, indicesAccessor, i));
                    }
                }
                else
                {
                    for (size_t i = 0; i < positions.count; ++i)
                    {
                        indices.push_back(vertexOffset + static_cast<std::uint32_t>(i));
                    }
                }
                if (indices.size() % 3 != 0)
                {
                    throw std::runtime_error("glTF primitive indices must form triangles");
                }
                for (size_t i = 0; i < indices.size(); i += 3)
                {
                    MeshTriangle triangle{indices[i], indices[i + 1], indices[i + 2], materialIndex};
                    const float3 tangent = computeTriangleTangent(
                        mesh.vertices[triangle.i0],
                        mesh.vertices[triangle.i1],
                        mesh.vertices[triangle.i2]);
                    mesh.vertices[triangle.i0].tangent = add3(mesh.vertices[triangle.i0].tangent, tangent);
                    mesh.vertices[triangle.i1].tangent = add3(mesh.vertices[triangle.i1].tangent, tangent);
                    mesh.vertices[triangle.i2].tangent = add3(mesh.vertices[triangle.i2].tangent, tangent);
                    mesh.triangles.push_back(triangle);
                }
            }
        }

        for (MeshVertex& vertex : mesh.vertices)
        {
            vertex.normal = normalize3(vertex.normal);
            vertex.tangent = normalize3(vertex.tangent);
        }

        if (isEmptyMesh(mesh) || !hasValidMeshMaterialIndices(mesh))
        {
            return GltfLoadResult{false, {}, "glTF import produced an invalid mesh"};
        }
        return GltfLoadResult{true, std::move(mesh), {}};
    }
    catch (const std::exception& ex)
    {
        return GltfLoadResult{false, {}, "Invalid glTF file: " + std::string(ex.what())};
    }
}
