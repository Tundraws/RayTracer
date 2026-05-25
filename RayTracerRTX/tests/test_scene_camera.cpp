#include "../src/app/camera.h"
#include "../src/app/asset_cache.h"
#include "../src/app/gltf_loader.h"
#include "../src/app/image_loader.h"
#include "../src/app/logger.h"
#include "../src/app/material.h"
#include "../src/app/obj_loader.h"
#include "../src/app/scene.h"
#include "../src/app/scene_config.h"
#if defined(RAYTRACERRTX_ENABLE_GPU_TESTS)
#include "../src/gpu/optix_renderer.h"
#endif

#include "test_framework.h"

#include <chrono>
#include <cmath>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <string>
#include <vector>

#ifndef RAYTRACERRTX_SOURCE_DIR
#define RAYTRACERRTX_SOURCE_DIR ""
#endif

namespace
{
float length3(const float3& v)
{
    return std::sqrt(v.x * v.x + v.y * v.y + v.z * v.z);
}

float dot3(const float3& a, const float3& b)
{
    return a.x * b.x + a.y * b.y + a.z * b.z;
}

std::filesystem::path writeFixtureFile(const std::string& name, const std::string& content)
{
    const std::filesystem::path directory = std::filesystem::temp_directory_path() / "raytracerrtx_obj_loader_tests";
    std::filesystem::create_directories(directory);

    const std::filesystem::path path = directory / name;
    std::ofstream file(path, std::ios::binary);
    file << content;
    return path;
}

std::filesystem::path writeFixtureBinaryFile(const std::string& name, const std::vector<unsigned char>& content)
{
    const std::filesystem::path directory = std::filesystem::temp_directory_path() / "raytracerrtx_obj_loader_tests";
    std::filesystem::create_directories(directory);

    const std::filesystem::path path = directory / name;
    std::ofstream file(path, std::ios::binary);
    file.write(reinterpret_cast<const char*>(content.data()), static_cast<std::streamsize>(content.size()));
    return path;
}

std::string readTextFile(const std::filesystem::path& path)
{
    std::ifstream file(path, std::ios::binary);
    std::ostringstream buffer;
    buffer << file.rdbuf();
    return buffer.str();
}

std::vector<unsigned char> tinyPngBytes()
{
    return {
        0x89, 0x50, 0x4e, 0x47, 0x0d, 0x0a, 0x1a, 0x0a,
        0x00, 0x00, 0x00, 0x0d, 0x49, 0x48, 0x44, 0x52,
        0x00, 0x00, 0x00, 0x01, 0x00, 0x00, 0x00, 0x01,
        0x08, 0x04, 0x00, 0x00, 0x00, 0xb5, 0x1c, 0x0c,
        0x02, 0x00, 0x00, 0x00, 0x0b, 0x49, 0x44, 0x41,
        0x54, 0x78, 0xda, 0x63, 0xfc, 0xff, 0x1f, 0x00,
        0x03, 0x03, 0x02, 0x00, 0xef, 0xbf, 0xa7, 0xdb,
        0x00, 0x00, 0x00, 0x00, 0x49, 0x45, 0x4e, 0x44,
        0xae, 0x42, 0x60, 0x82
    };
}

void testLoggerWritesWarnings(TestContext& t)
{
    const std::filesystem::path logPath = writeFixtureFile("raytracerrtx_unit_test.log", "");
    setLogFilePath(logPath);
    clearLogFile();

    logWarning("test warning message");

    const std::string log = readTextFile(logPath);
    t.expect(log.find("[warning]") != std::string::npos, "Logger should include warning level.");
    t.expect(log.find("test warning message") != std::string::npos, "Logger should write message text.");
    setLogFilePath("RayTracerRTX.log");
}

std::filesystem::path findDemoObjAsset()
{
    const std::filesystem::path sourceDir = RAYTRACERRTX_SOURCE_DIR;
    const std::filesystem::path fromSource = sourceDir.empty()
        ? std::filesystem::path{}
        : sourceDir.parent_path() / "assets" / "meshes" / "demo.obj";

    const std::filesystem::path candidates[] = {
        fromSource,
        std::filesystem::path("RayTracerRTX") / "assets" / "meshes" / "demo.obj",
        std::filesystem::path("assets") / "meshes" / "demo.obj",
        std::filesystem::path("..") / "assets" / "meshes" / "demo.obj"
    };

    for (const std::filesystem::path& candidate : candidates)
    {
        if (!candidate.empty() && std::filesystem::exists(candidate))
        {
            return candidate;
        }
    }

    return {};
}

std::filesystem::path findAssetMesh(const std::string& fileName)
{
    const std::filesystem::path sourceDir = RAYTRACERRTX_SOURCE_DIR;
    const std::filesystem::path fromSource = sourceDir.empty()
        ? std::filesystem::path{}
        : sourceDir.parent_path() / "assets" / "meshes" / fileName;

    const std::filesystem::path candidates[] = {
        fromSource,
        std::filesystem::path("RayTracerRTX") / "assets" / "meshes" / fileName,
        std::filesystem::path("assets") / "meshes" / fileName,
        std::filesystem::path("..") / "assets" / "meshes" / fileName
    };

    for (const std::filesystem::path& candidate : candidates)
    {
        if (!candidate.empty() && std::filesystem::exists(candidate))
        {
            return candidate;
        }
    }

    return {};
}

std::filesystem::path findAssetScene(const std::string& fileName)
{
    const std::filesystem::path sourceDir = RAYTRACERRTX_SOURCE_DIR;
    const std::filesystem::path fromSource = sourceDir.empty()
        ? std::filesystem::path{}
        : sourceDir.parent_path() / "assets" / "scenes" / fileName;

    const std::filesystem::path candidates[] = {
        fromSource,
        std::filesystem::path("RayTracerRTX") / "assets" / "scenes" / fileName,
        std::filesystem::path("assets") / "scenes" / fileName,
        std::filesystem::path("..") / "assets" / "scenes" / fileName
    };

    for (const std::filesystem::path& candidate : candidates)
    {
        if (!candidate.empty() && std::filesystem::exists(candidate))
        {
            return candidate;
        }
    }

    return {};
}

void appendFloat(std::vector<unsigned char>& data, const float value)
{
    const auto* bytes = reinterpret_cast<const unsigned char*>(&value);
    data.insert(data.end(), bytes, bytes + sizeof(float));
}

void appendUint16(std::vector<unsigned char>& data, const std::uint16_t value)
{
    const auto* bytes = reinterpret_cast<const unsigned char*>(&value);
    data.insert(data.end(), bytes, bytes + sizeof(std::uint16_t));
}

void appendUint32(std::vector<unsigned char>& data, const std::uint32_t value)
{
    data.push_back(static_cast<unsigned char>(value & 0xffu));
    data.push_back(static_cast<unsigned char>((value >> 8u) & 0xffu));
    data.push_back(static_cast<unsigned char>((value >> 16u) & 0xffu));
    data.push_back(static_cast<unsigned char>((value >> 24u) & 0xffu));
}

void padToFourBytes(std::vector<unsigned char>& data, const unsigned char value)
{
    while (data.size() % 4 != 0)
    {
        data.push_back(value);
    }
}

std::vector<unsigned char> minimalGltfBin()
{
    std::vector<unsigned char> bin;
    const float positions[] = {0.0f, 0.0f, 0.0f, 1.0f, 0.0f, 0.0f, 0.0f, 1.0f, 0.0f};
    const float normals[] = {0.0f, 0.0f, 1.0f, 0.0f, 0.0f, 1.0f, 0.0f, 0.0f, 1.0f};
    const float texcoords[] = {0.0f, 0.0f, 1.0f, 0.0f, 0.0f, 1.0f};
    for (const float value : positions) { appendFloat(bin, value); }
    for (const float value : normals) { appendFloat(bin, value); }
    for (const float value : texcoords) { appendFloat(bin, value); }
    appendUint16(bin, 0);
    appendUint16(bin, 1);
    appendUint16(bin, 2);
    return bin;
}

std::filesystem::path writeMinimalGltfFixture()
{
    std::vector<unsigned char> bin = minimalGltfBin();
    writeFixtureBinaryFile("minimal_gltf.bin", bin);

    return writeFixtureFile(
        "minimal_gltf.gltf",
        "{\n"
        "  \"asset\": {\"version\": \"2.0\"},\n"
        "  \"buffers\": [{\"uri\": \"minimal_gltf.bin\", \"byteLength\": 102}],\n"
        "  \"bufferViews\": [\n"
        "    {\"buffer\": 0, \"byteOffset\": 0, \"byteLength\": 36},\n"
        "    {\"buffer\": 0, \"byteOffset\": 36, \"byteLength\": 36},\n"
        "    {\"buffer\": 0, \"byteOffset\": 72, \"byteLength\": 24},\n"
        "    {\"buffer\": 0, \"byteOffset\": 96, \"byteLength\": 6}\n"
        "  ],\n"
        "  \"accessors\": [\n"
        "    {\"bufferView\": 0, \"componentType\": 5126, \"count\": 3, \"type\": \"VEC3\"},\n"
        "    {\"bufferView\": 1, \"componentType\": 5126, \"count\": 3, \"type\": \"VEC3\"},\n"
        "    {\"bufferView\": 2, \"componentType\": 5126, \"count\": 3, \"type\": \"VEC2\"},\n"
        "    {\"bufferView\": 3, \"componentType\": 5123, \"count\": 3, \"type\": \"SCALAR\"}\n"
        "  ],\n"
        "  \"materials\": [{\n"
        "    \"name\": \"mat_metal_gltf\",\n"
        "    \"pbrMetallicRoughness\": {\n"
        "      \"baseColorFactor\": [0.2, 0.6, 0.9, 1.0],\n"
        "      \"metallicFactor\": 1.0,\n"
        "      \"roughnessFactor\": 0.25\n"
        "    }\n"
        "  }],\n"
        "  \"meshes\": [{\"primitives\": [{\"attributes\": {\"POSITION\": 0, \"NORMAL\": 1, \"TEXCOORD_0\": 2}, \"indices\": 3, \"material\": 0}]}],\n"
        "  \"nodes\": [{\"mesh\": 0, \"translation\": [1.0, 2.0, 3.0], \"scale\": [2.0, 2.0, 2.0]}],\n"
        "  \"scenes\": [{\"nodes\": [0]}],\n"
        "  \"scene\": 0\n"
        "}\n");
}

std::filesystem::path writeMinimalGlbFixture()
{
    std::vector<unsigned char> bin = minimalGltfBin();
    std::string json =
        "{"
        "\"asset\":{\"version\":\"2.0\"},"
        "\"buffers\":[{\"byteLength\":102}],"
        "\"bufferViews\":["
        "{\"buffer\":0,\"byteOffset\":0,\"byteLength\":36},"
        "{\"buffer\":0,\"byteOffset\":36,\"byteLength\":36},"
        "{\"buffer\":0,\"byteOffset\":72,\"byteLength\":24},"
        "{\"buffer\":0,\"byteOffset\":96,\"byteLength\":6}"
        "],"
        "\"accessors\":["
        "{\"bufferView\":0,\"componentType\":5126,\"count\":3,\"type\":\"VEC3\"},"
        "{\"bufferView\":1,\"componentType\":5126,\"count\":3,\"type\":\"VEC3\"},"
        "{\"bufferView\":2,\"componentType\":5126,\"count\":3,\"type\":\"VEC2\"},"
        "{\"bufferView\":3,\"componentType\":5123,\"count\":3,\"type\":\"SCALAR\"}"
        "],"
        "\"materials\":[{\"pbrMetallicRoughness\":{\"baseColorFactor\":[0.7,0.4,0.2,1.0],\"metallicFactor\":0.0,\"roughnessFactor\":0.4}}],"
        "\"meshes\":[{\"primitives\":[{\"attributes\":{\"POSITION\":0,\"NORMAL\":1,\"TEXCOORD_0\":2},\"indices\":3,\"material\":0}]}],"
        "\"nodes\":[{\"mesh\":0,\"translation\":[0.0,1.0,0.0],\"rotation\":[0.0,0.0,0.7071068,0.7071068],\"scale\":[2.0,2.0,2.0]}],"
        "\"scenes\":[{\"nodes\":[0]}],"
        "\"scene\":0"
        "}";

    std::vector<unsigned char> jsonChunk(json.begin(), json.end());
    padToFourBytes(jsonChunk, 0x20);
    padToFourBytes(bin, 0x00);

    std::vector<unsigned char> glb;
    appendUint32(glb, 0x46546c67u);
    appendUint32(glb, 2u);
    appendUint32(glb, static_cast<std::uint32_t>(12u + 8u + jsonChunk.size() + 8u + bin.size()));
    appendUint32(glb, static_cast<std::uint32_t>(jsonChunk.size()));
    appendUint32(glb, 0x4e4f534au);
    glb.insert(glb.end(), jsonChunk.begin(), jsonChunk.end());
    appendUint32(glb, static_cast<std::uint32_t>(bin.size()));
    appendUint32(glb, 0x004e4942u);
    glb.insert(glb.end(), bin.begin(), bin.end());
    return writeFixtureBinaryFile("minimal_mesh.glb", glb);
}

void testObjLoaderTriangleWithNormals(TestContext& t)
{
    const std::filesystem::path objPath = writeFixtureFile(
        "triangle_normals.obj",
        "mtllib triangle_normals.mtl\n"
        "v 0 0 0\n"
        "v 1 0 0\n"
        "v 0 1 0\n"
        "vn 0 0 1\n"
        "usemtl white\n"
        "f 1//1 2//1 3//1\n");

    writeFixtureFile(
        "triangle_normals.mtl",
        "newmtl white\n"
        "Kd 0.9 0.8 0.7\n");

    const ObjLoadResult result = loadObjMesh(objPath);
    t.expect(result.ok, "OBJ with vertices, normals, and one material should load.");
    t.expect(result.mesh.vertices.size() == 3, "Loaded triangle should create 3 packed vertices.");
    t.expect(result.mesh.triangles.size() == 1, "Loaded OBJ should contain one triangle.");
    t.expect(result.mesh.materials.size() == 2, "Default and MTL material should be available.");
    t.expect(almostEqual(result.mesh.vertices[0].normal.z, 1.0f), "OBJ normal should be assigned to vertices.");
    t.expect(almostEqual(result.mesh.materials[1].color.x, 0.9f), "MTL Kd color should be loaded.");
}

void testGgxMathHelpers(TestContext& t)
{
    const float dSmooth = ggxDistribution(1.0f, 0.08f);
    const float dRough = ggxDistribution(1.0f, 0.7f);
    t.expect(std::isfinite(dSmooth), "GGX distribution should stay finite for low roughness.");
    t.expect(std::isfinite(dRough), "GGX distribution should stay finite for rough material.");
    t.expect(dSmooth > dRough, "GGX distribution should sharpen as roughness decreases.");

    const float g = ggxGeometrySmith(0.7f, 0.6f, 0.4f);
    t.expect(std::isfinite(g), "GGX geometry term should stay finite.");
    t.expect(g >= 0.0f && g <= 1.0f, "GGX geometry term should stay in [0, 1].");

    const float f0 = 0.04f;
    const float fGrazing = ggxFresnelSchlick(0.0f, f0);
    const float fFacing = ggxFresnelSchlick(1.0f, f0);
    t.expect(almostEqual(fFacing, f0), "Schlick Fresnel should equal F0 at normal incidence.");
    t.expect(fGrazing > fFacing, "Schlick Fresnel should increase at grazing angles.");

    const float glassF0 = dielectricF0FromIor(1.5f);
    const float glassFacing = dielectricFresnelSchlick(1.0f, 1.5f);
    const float glassGrazing = dielectricFresnelSchlick(0.0f, 1.5f);
    t.expect(glassF0 > 0.03f && glassF0 < 0.05f, "Dielectric F0 should be plausible for IOR 1.5.");
    t.expect(almostEqual(glassFacing, glassF0), "Dielectric Fresnel should match F0 at normal incidence.");
    t.expect(glassGrazing > glassFacing && glassGrazing <= 1.0f, "Dielectric Fresnel should increase toward grazing angles.");

    float3 refracted{};
    const bool airToGlass = refractDirection(
        make_float3(0.0f, -1.0f, 0.0f),
        make_float3(0.0f, 1.0f, 0.0f),
        1.0f / 1.5f,
        refracted);
    t.expect(airToGlass, "Refraction should work at normal incidence.");
    t.expect(almostEqual(refracted.y, -1.0f), "Normal-incidence refraction should keep direction.");

    const bool totalInternalReflection = refractDirection(
        normalizeShared(make_float3(0.95f, -0.31f, 0.0f)),
        make_float3(0.0f, 1.0f, 0.0f),
        1.5f,
        refracted);
    t.expect(!totalInternalReflection, "Refraction helper should detect total internal reflection.");
    t.expect(almostEqual(clampMaterialRoughnessShared(-5.0f), 0.02f), "Shared roughness clamp should clamp low values.");
    t.expect(almostEqual(clampMaterialRoughnessShared(5.0f), 1.0f), "Shared roughness clamp should clamp high values.");
    t.expect(almostEqual(clampMaterialIorShared(0.1f), 1.01f), "Shared IOR clamp should clamp low values.");
    t.expect(almostEqual(clampMaterialIorShared(9.0f), 2.8f), "Shared IOR clamp should clamp high values.");
}

void testObjLoaderMultipleMaterials(TestContext& t)
{
    const std::filesystem::path objPath = writeFixtureFile(
        "multi_material.obj",
        "mtllib multi_material.mtl\n"
        "v 0 0 0\n"
        "v 1 0 0\n"
        "v 0 1 0\n"
        "v 1 1 0\n"
        "vn 0 0 1\n"
        "usemtl red\n"
        "f 1//1 2//1 3//1\n"
        "usemtl green\n"
        "f 2//1 4//1 3//1\n");

    writeFixtureFile(
        "multi_material.mtl",
        "newmtl red\n"
        "Kd 1 0 0\n"
        "newmtl green\n"
        "Kd 0 1 0\n");

    const ObjLoadResult result = loadObjMesh(objPath);
    t.expect(result.ok, "OBJ with multiple usemtl commands should load.");
    t.expect(result.mesh.triangles.size() == 2, "Multiple material OBJ should contain two triangles.");
    t.expect(result.mesh.materials.size() == 3, "Default plus two named materials should be loaded.");
    t.expect(result.mesh.triangles[0].materialIndex != result.mesh.triangles[1].materialIndex, "Triangles should reference different materials.");
    t.expect(hasValidMeshMaterialIndices(result.mesh), "Material indices should be valid.");
}

void testObjLoaderMirrorMaterialNameMapping(TestContext& t)
{
    const std::filesystem::path objPath = writeFixtureFile(
        "mirror_material.obj",
        "v 0 0 0\n"
        "v 1 0 0\n"
        "v 0 1 0\n"
        "usemtl mat_mirror\n"
        "f 1 2 3\n");

    const ObjLoadResult result = loadObjMesh(objPath);
    t.expect(result.ok, "OBJ with mirror material name should load.");
    t.expect(result.mesh.materials.size() == 2, "Mirror usemtl should create a named material.");
    t.expect(result.mesh.materials[1].materialType == MaterialMirror, "Material name containing mirror should map to MaterialMirror.");
}

void testObjLoaderExtendedMtlParameters(TestContext& t)
{
    const std::filesystem::path objPath = writeFixtureFile(
        "extended_mtl.obj",
        "mtllib extended_mtl.mtl\n"
        "v 0 0 0\n"
        "v 1 0 0\n"
        "v 0 1 0\n"
        "usemtl mat_metal\n"
        "f 1 2 3\n");

    writeFixtureFile(
        "extended_mtl.mtl",
        "newmtl mat_metal\n"
        "Kd 0.7 0.6 0.5\n"
        "Ks 0.9 0.8 0.7\n"
        "Ns 100\n"
        "Ni 1.7\n"
        "d 0.65\n");

    const ObjLoadResult result = loadObjMesh(objPath);
    t.expect(result.ok, "OBJ with extended MTL material should load.");
    t.expect(result.mesh.materials[1].materialType == MaterialMetal, "Material name containing metal should map to MaterialMetal.");
    t.expect(almostEqual(result.mesh.materials[1].color.x, 0.7f), "MTL Kd should set base color.");
    t.expect(almostEqual(result.mesh.materials[1].specularColor.x, 0.9f), "MTL Ks should set specular color.");
    t.expect(result.mesh.materials[1].roughness > 0.02f && result.mesh.materials[1].roughness < 0.2f, "MTL Ns should map to low roughness for glossy material.");
    t.expect(almostEqual(result.mesh.materials[1].ior, 1.7f), "MTL Ni should set index of refraction.");
    t.expect(almostEqual(result.mesh.materials[1].alpha, 0.65f), "MTL d should set alpha.");
}

void testObjLoaderDielectricMaterialNameMapping(TestContext& t)
{
    const std::filesystem::path objPath = writeFixtureFile(
        "glass_material.obj",
        "v 0 0 0\n"
        "v 1 0 0\n"
        "v 0 1 0\n"
        "usemtl mat_glass\n"
        "f 1 2 3\n");

    const ObjLoadResult result = loadObjMesh(objPath);
    t.expect(result.ok, "OBJ with glass material name should load.");
    t.expect(result.mesh.materials[1].materialType == MaterialDielectric, "Material name containing glass should map to MaterialDielectric.");
}

void testObjLoaderMaterialParameterClamping(TestContext& t)
{
    const std::filesystem::path objPath = writeFixtureFile(
        "clamped_mtl.obj",
        "mtllib clamped_mtl.mtl\n"
        "v 0 0 0\n"
        "v 1 0 0\n"
        "v 0 1 0\n"
        "usemtl mat_glass\n"
        "f 1 2 3\n");

    writeFixtureFile(
        "clamped_mtl.mtl",
        "newmtl mat_glass\n"
        "Ns 10000\n"
        "Ni 8\n"
        "d -2\n");

    const ObjLoadResult result = loadObjMesh(objPath);
    t.expect(result.ok, "OBJ with out-of-range material parameters should load.");
    t.expect(result.mesh.materials[1].roughness >= 0.02f && result.mesh.materials[1].roughness <= 1.0f, "Roughness should be clamped.");
    t.expect(almostEqual(result.mesh.materials[1].ior, 2.8f), "IOR should be clamped to supported upper bound.");
    t.expect(almostEqual(result.mesh.materials[1].alpha, 0.0f), "Alpha should be clamped to supported lower bound.");
    t.expect(hasValidMeshMaterialIndices(result.mesh), "Material indices should remain valid after extended MTL parsing.");
}

void testObjLoaderTextureCoordinates(TestContext& t)
{
    const std::filesystem::path objPath = writeFixtureFile(
        "textured_coords.obj",
        "v 0 0 0\n"
        "v 1 0 0\n"
        "v 0 1 0\n"
        "vt 0.25 0.75\n"
        "vt 1 0\n"
        "vt 0 1\n"
        "f 1/1 2/2 3/3\n");

    const ObjLoadResult result = loadObjMesh(objPath);
    t.expect(result.ok, "OBJ with v/vt face format should load.");
    t.expect(almostEqual(result.mesh.vertices[0].texcoord.x, 0.25f), "OBJ vt U coordinate should be assigned.");
    t.expect(almostEqual(result.mesh.vertices[0].texcoord.y, 0.75f), "OBJ vt V coordinate should be assigned.");
}

void testObjLoaderTexturedFaceWithNormals(TestContext& t)
{
    const std::filesystem::path objPath = writeFixtureFile(
        "textured_normals.obj",
        "v 0 0 0\n"
        "v 1 0 0\n"
        "v 0 1 0\n"
        "vt 0 0\n"
        "vt 1 0\n"
        "vt 0 1\n"
        "vn 0 0 1\n"
        "f 1/1/1 2/2/1 3/3/1\n");

    const ObjLoadResult result = loadObjMesh(objPath);
    t.expect(result.ok, "OBJ with v/vt/vn face format should load.");
    t.expect(almostEqual(result.mesh.vertices[2].texcoord.y, 1.0f), "OBJ v/vt/vn should preserve texcoords.");
    t.expect(almostEqual(result.mesh.vertices[2].normal.z, 1.0f), "OBJ v/vt/vn should preserve normals.");
}

void testObjLoaderComputesTangentsForTexturedTriangle(TestContext& t)
{
    const std::filesystem::path objPath = writeFixtureFile(
        "tangent_triangle.obj",
        "v 0 0 0\n"
        "v 1 0 0\n"
        "v 0 1 0\n"
        "vt 0 0\n"
        "vt 1 0\n"
        "vt 0 1\n"
        "vn 0 0 1\n"
        "f 1/1/1 2/2/1 3/3/1\n");

    const ObjLoadResult result = loadObjMesh(objPath);
    t.expect(result.ok, "OBJ with textured triangle should load.");
    t.expect(result.mesh.vertices[0].hasTexcoord == 1, "Textured triangle vertex should mark texcoord availability.");
    t.expect(almostEqual(result.mesh.vertices[0].tangent.x, 1.0f), "Textured triangle tangent should follow positive U.");
    t.expect(almostEqual(result.mesh.vertices[0].tangent.y, 0.0f), "Textured triangle tangent should be orthogonal to Y for this fixture.");
    t.expect(almostEqual(result.mesh.vertices[0].tangent.z, 0.0f), "Textured triangle tangent should be orthogonal to normal.");
}

void testObjLoaderMapKdTexture(TestContext& t)
{
    const std::filesystem::path objPath = writeFixtureFile(
        "map_kd.obj",
        "mtllib map_kd.mtl\n"
        "v 0 0 0\n"
        "v 1 0 0\n"
        "v 0 1 0\n"
        "vt 0 0\n"
        "vt 1 0\n"
        "vt 0 1\n"
        "usemtl textured\n"
        "f 1/1 2/2 3/3\n");

    writeFixtureFile(
        "map_kd.mtl",
        "newmtl textured\n"
        "Kd 1 1 1\n"
        "map_Kd tiny.ppm\n");
    writeFixtureFile(
        "tiny.ppm",
        "P3\n"
        "2 1\n"
        "255\n"
        "255 0 0  0 255 0\n");

    const ObjLoadResult result = loadObjMesh(objPath);
    t.expect(result.ok, "OBJ with map_Kd texture should load.");
    t.expect(result.mesh.materials[1].texturePath == "tiny.ppm", "MTL map_Kd path should be stored.");
    t.expect(result.mesh.materials[1].textureIndex == 0, "Loaded map_Kd texture should be assigned to material.");
    t.expect(result.mesh.textures.size() == 1, "Loaded map_Kd texture should be stored in mesh textures.");
    t.expect(result.mesh.textures[0].pixels.size() == 2, "PPM texture pixels should be loaded.");
    t.expect(hasValidMeshMaterialIndices(result.mesh), "Textured material indices should remain valid.");
}

void testImageLoaderPngTexture(TestContext& t)
{
    const std::filesystem::path pngPath = writeFixtureBinaryFile("tiny_png_texture.png", tinyPngBytes());

    MeshTexture texture;
    const bool loaded = loadImageTexture(pngPath, texture, "baseColor");

    t.expect(loaded, "PNG texture should load through stb_image.");
    t.expect(texture.width == 1 && texture.height == 1, "PNG texture metadata should include dimensions.");
    t.expect(texture.channels >= 1, "PNG texture metadata should include source channel count.");
    t.expect(texture.type == "baseColor", "PNG texture metadata should keep texture type.");
    t.expect(texture.pixels.size() == 1, "PNG texture should decode one pixel.");
}

void testImageLoaderInvalidPpmFallback(TestContext& t)
{
    const std::filesystem::path logPath = writeFixtureFile("invalid_ppm_loader.log", "");
    setLogFilePath(logPath);
    clearLogFile();
    const std::filesystem::path texturePath = writeFixtureFile(
        "invalid_texture.ppm",
        "P3\n"
        "bad 1\n"
        "255\n"
        "255 0 0\n");

    MeshTexture texture;
    const bool loaded = loadImageTexture(texturePath, texture, "baseColor");

    t.expect(!loaded, "Invalid PPM should fail cleanly.");
    t.expect(texture.pixels.empty(), "Invalid PPM should not leave partial pixels.");
    t.expect(readTextFile(logPath).find("Invalid PPM texture dimensions") != std::string::npos, "Invalid PPM should be written to the log.");
    setLogFilePath("RayTracerRTX.log");
}

void testObjLoaderCommonTextureMaps(TestContext& t)
{
    const std::filesystem::path objPath = writeFixtureFile(
        "common_texture_maps.obj",
        "mtllib common_texture_maps.mtl\n"
        "v 0 0 0\n"
        "v 1 0 0\n"
        "v 0 1 0\n"
        "vt 0 0\n"
        "vt 1 0\n"
        "vt 0 1\n"
        "usemtl textured\n"
        "f 1/1 2/2 3/3\n");

    writeFixtureFile(
        "common_texture_maps.mtl",
        "newmtl textured\n"
        "Kd 1 1 1\n"
        "map_Kd albedo.png\n"
        "norm normal.png\n"
        "map_Pm metallic.png\n"
        "map_Pr roughness.png\n");
    writeFixtureBinaryFile("albedo.png", tinyPngBytes());
    writeFixtureBinaryFile("normal.png", tinyPngBytes());
    writeFixtureBinaryFile("metallic.png", tinyPngBytes());
    writeFixtureBinaryFile("roughness.png", tinyPngBytes());

    const ObjLoadResult result = loadObjMesh(objPath);
    t.expect(result.ok, "OBJ with PNG texture maps should load.");
    t.expect(result.mesh.materials[1].textureIndex >= 0, "PNG base color texture should be assigned.");
    t.expect(result.mesh.materials[1].normalTextureIndex >= 0, "PNG normal texture should be assigned.");
    t.expect(result.mesh.materials[1].metallicTextureIndex >= 0, "PNG metallic texture should be assigned.");
    t.expect(result.mesh.materials[1].roughnessTextureIndex >= 0, "PNG roughness texture should be assigned.");
    t.expect(result.mesh.textures.size() == 4, "All PNG texture maps should be stored.");
    t.expect(result.mesh.textures[0].type == "baseColor", "Base color texture type metadata should be stored.");
    t.expect(result.mesh.textures[1].type == "normal", "Normal texture type metadata should be stored.");
    t.expect(result.mesh.textures[2].type == "metallic", "Metallic texture type metadata should be stored.");
    t.expect(result.mesh.textures[3].type == "roughness", "Roughness texture type metadata should be stored.");
    t.expect(hasValidMeshMaterialIndices(result.mesh), "Common texture map material indices should remain valid.");
}

void testObjLoaderUnsupportedTextureFallback(TestContext& t)
{
    const std::filesystem::path objPath = writeFixtureFile(
        "unsupported_texture.obj",
        "mtllib unsupported_texture.mtl\n"
        "v 0 0 0\n"
        "v 1 0 0\n"
        "v 0 1 0\n"
        "usemtl unsupported_tex\n"
        "f 1 2 3\n");

    writeFixtureFile(
        "unsupported_texture.mtl",
        "newmtl unsupported_tex\n"
        "Kd 0.4 0.5 0.6\n"
        "map_Kd unsupported.txt\n");
    writeFixtureFile("unsupported.txt", "not an image");

    const ObjLoadResult result = loadObjMesh(objPath);
    t.expect(result.ok, "Unsupported texture should not fail OBJ loading.");
    t.expect(result.mesh.materials[1].texturePath == "unsupported.txt", "Unsupported texture path should still be stored.");
    t.expect(result.mesh.materials[1].textureIndex < 0, "Unsupported texture should fall back to material color.");
    t.expect(result.mesh.textures.empty(), "Unsupported texture should not create texture pixels.");
    t.expect(hasValidMeshMaterialIndices(result.mesh), "Unsupported texture fallback should keep material indices valid.");
}

void testObjLoaderNormalMapTexture(TestContext& t)
{
    const std::filesystem::path objPath = writeFixtureFile(
        "normal_map.obj",
        "mtllib normal_map.mtl\n"
        "v 0 0 0\n"
        "v 1 0 0\n"
        "v 0 1 0\n"
        "vt 0 0\n"
        "vt 1 0\n"
        "vt 0 1\n"
        "usemtl normal_mapped\n"
        "f 1/1 2/2 3/3\n");

    writeFixtureFile(
        "normal_map.mtl",
        "newmtl normal_mapped\n"
        "Kd 1 1 1\n"
        "bump normal.ppm\n");
    writeFixtureFile(
        "normal.ppm",
        "P3\n"
        "1 1\n"
        "255\n"
        "128 128 255\n");

    const ObjLoadResult result = loadObjMesh(objPath);
    t.expect(result.ok, "OBJ with bump normal map should load.");
    t.expect(result.mesh.materials[1].normalTexturePath == "normal.ppm", "MTL bump normal map path should be stored.");
    t.expect(result.mesh.materials[1].normalTextureIndex == 0, "Loaded normal map should be assigned to material.");
    t.expect(result.mesh.textures.size() == 1, "Loaded normal map should be stored in mesh textures.");
}

void testObjLoaderMissingTextureFallback(TestContext& t)
{
    const std::filesystem::path objPath = writeFixtureFile(
        "missing_texture.obj",
        "mtllib missing_texture.mtl\n"
        "v 0 0 0\n"
        "v 1 0 0\n"
        "v 0 1 0\n"
        "usemtl missing_tex\n"
        "f 1 2 3\n");

    writeFixtureFile(
        "missing_texture.mtl",
        "newmtl missing_tex\n"
        "Kd 0.4 0.5 0.6\n"
        "map_Kd does_not_exist.ppm\n");

    const ObjLoadResult result = loadObjMesh(objPath);
    t.expect(result.ok, "Missing map_Kd texture should not fail OBJ loading.");
    t.expect(result.mesh.materials[1].texturePath == "does_not_exist.ppm", "Missing map_Kd path should still be stored.");
    t.expect(result.mesh.materials[1].textureIndex < 0, "Missing map_Kd texture should fall back to Kd.");
    t.expect(result.mesh.textures.empty(), "Missing map_Kd texture should not create texture pixels.");
    t.expect(hasValidMeshMaterialIndices(result.mesh), "Missing texture fallback material indices should remain valid.");
}

void testObjLoaderMissingUvDisablesNormalMap(TestContext& t)
{
    const std::filesystem::path objPath = writeFixtureFile(
        "normal_without_uv.obj",
        "mtllib normal_without_uv.mtl\n"
        "v 0 0 0\n"
        "v 1 0 0\n"
        "v 0 1 0\n"
        "usemtl normal_mapped\n"
        "f 1 2 3\n");

    writeFixtureFile(
        "normal_without_uv.mtl",
        "newmtl normal_mapped\n"
        "norm normal.ppm\n");
    writeFixtureFile(
        "normal.ppm",
        "P3\n"
        "1 1\n"
        "255\n"
        "128 128 255\n");

    const ObjLoadResult result = loadObjMesh(objPath);
    t.expect(result.ok, "OBJ without UV but with normal map should load.");
    t.expect(result.mesh.materials[1].normalTextureIndex == 0, "Normal texture can load even when geometry has no UV.");
    t.expect(result.mesh.vertices[0].hasTexcoord == 0, "Missing UV should disable normal map use for vertex.");
    t.expect(almostEqual(length3(result.mesh.vertices[0].tangent), 0.0f), "Missing UV should keep tangent empty.");
}

void testObjLoaderInvalidNormalMapFallback(TestContext& t)
{
    const std::filesystem::path objPath = writeFixtureFile(
        "invalid_normal_map.obj",
        "mtllib invalid_normal_map.mtl\n"
        "v 0 0 0\n"
        "v 1 0 0\n"
        "v 0 1 0\n"
        "vt 0 0\n"
        "vt 1 0\n"
        "vt 0 1\n"
        "usemtl normal_mapped\n"
        "f 1/1 2/2 3/3\n");

    writeFixtureFile(
        "invalid_normal_map.mtl",
        "newmtl normal_mapped\n"
        "map_Bump does_not_exist.ppm\n");

    const ObjLoadResult result = loadObjMesh(objPath);
    t.expect(result.ok, "Missing normal map texture should not fail OBJ loading.");
    t.expect(result.mesh.materials[1].normalTexturePath == "does_not_exist.ppm", "Missing normal map path should still be stored.");
    t.expect(result.mesh.materials[1].normalTextureIndex < 0, "Missing normal map should fall back to interpolated normal.");
    t.expect(result.mesh.textures.empty(), "Missing normal map should not create texture pixels.");
}

void testObjLoaderMissingNormalsFallback(TestContext& t)
{
    const std::filesystem::path objPath = writeFixtureFile(
        "missing_normals.obj",
        "v 0 0 0\n"
        "v 1 0 0\n"
        "v 0 1 0\n"
        "f 1 2 3\n");

    const ObjLoadResult result = loadObjMesh(objPath);
    t.expect(result.ok, "OBJ without normals should load with generated normals.");
    t.expect(result.mesh.vertices.size() == 3, "Fallback-normal OBJ should create vertices.");
    t.expect(almostEqual(result.mesh.vertices[0].normal.z, 1.0f), "Fallback normal should be computed from triangle winding.");
}

void testObjLoaderEmptyFile(TestContext& t)
{
    const std::filesystem::path objPath = writeFixtureFile("empty.obj", "");
    const ObjLoadResult result = loadObjMesh(objPath);
    t.expect(!result.ok, "Empty OBJ should fail validation.");
    t.expect(!result.error.empty(), "Empty OBJ failure should include an error message.");
}

void testObjLoaderInvalidFace(TestContext& t)
{
    const std::filesystem::path objPath = writeFixtureFile(
        "invalid_face.obj",
        "v 0 0 0\n"
        "v 1 0 0\n"
        "v 0 1 0\n"
        "v 1 1 0\n"
        "f 1 2 3 4\n");

    const ObjLoadResult result = loadObjMesh(objPath);
    t.expect(!result.ok, "Non-triangulated face should fail.");
    t.expect(result.error.find("triangulated") != std::string::npos, "Invalid face error should explain triangulated requirement.");
}

void testObjLoaderMissingFile(TestContext& t)
{
    const std::filesystem::path objPath = std::filesystem::temp_directory_path() / "raytracerrtx_missing_mesh.obj";
    const ObjLoadResult result = loadObjMesh(objPath);
    t.expect(!result.ok, "Missing OBJ file should fail.");
    t.expect(result.error.find("could not be opened") != std::string::npos, "Missing file error should explain open failure.");
}

void testObjLoaderUnknownLinesIgnored(TestContext& t)
{
    const std::filesystem::path objPath = writeFixtureFile(
        "unknown_lines.obj",
        "o DemoObject\n"
        "s off\n"
        "v 0 0 0\n"
        "v 1 0 0\n"
        "v 0 1 0\n"
        "g ignored_group\n"
        "f 1 2 3\n");

    const ObjLoadResult result = loadObjMesh(objPath);
    t.expect(result.ok, "Unknown OBJ lines should be ignored.");
    t.expect(result.mesh.triangles.size() == 1, "Unknown lines should not prevent triangle loading.");
}

void testObjLoaderMaterialFallbackWhenMtlMissing(TestContext& t)
{
    const std::filesystem::path objPath = writeFixtureFile(
        "missing_mtl.obj",
        "mtllib does_not_exist.mtl\n"
        "v 0 0 0\n"
        "v 1 0 0\n"
        "v 0 1 0\n"
        "usemtl fallback_name\n"
        "f 1 2 3\n");

    const ObjLoadResult result = loadObjMesh(objPath);
    t.expect(result.ok, "Missing MTL should fall back to generated material.");
    t.expect(result.mesh.materials.size() == 2, "Fallback named material should be created.");
    t.expect(result.mesh.materials[1].name == "fallback_name", "Fallback material should keep usemtl name.");
    t.expect(hasValidMeshMaterialIndices(result.mesh), "Fallback material index should be valid.");
}

void testObjLoaderInvalidMtlValuesFallback(TestContext& t)
{
    const std::filesystem::path logPath = writeFixtureFile("invalid_mtl_values.log", "");
    setLogFilePath(logPath);
    clearLogFile();

    const std::filesystem::path objPath = writeFixtureFile(
        "invalid_mtl_values.obj",
        "mtllib invalid_mtl_values.mtl\n"
        "v 0 0 0\n"
        "v 1 0 0\n"
        "v 0 1 0\n"
        "usemtl bad_values\n"
        "f 1 2 3\n");
    writeFixtureFile(
        "invalid_mtl_values.mtl",
        "newmtl bad_values\n"
        "Kd red green blue\n"
        "Ks shine shine shine\n"
        "Ns glossy\n"
        "Ni water\n"
        "d clear\n");

    const ObjLoadResult result = loadObjMesh(objPath);

    t.expect(result.ok, "Invalid MTL values should keep OBJ loading.");
    t.expect(result.mesh.materials.size() == 2, "Invalid MTL should still create the named material.");
    t.expect(almostEqual(result.mesh.materials[1].color.x, 0.8f), "Invalid Kd should keep fallback color.");
    t.expect(almostEqual(result.mesh.materials[1].roughness, 0.35f), "Invalid Ns should keep fallback roughness.");
    t.expect(readTextFile(logPath).find("Invalid MTL Kd value") != std::string::npos, "Invalid MTL values should be logged.");
    t.expect(hasValidMeshMaterialIndices(result.mesh), "Invalid MTL fallback should keep material indices valid.");
    setLogFilePath("RayTracerRTX.log");
}

void testObjLoaderSingleTriangleBoundary(TestContext& t)
{
    const std::filesystem::path objPath = writeFixtureFile(
        "single_triangle.obj",
        "v 0 0 0\n"
        "v 0 0 1\n"
        "v 0 1 0\n"
        "f 1 2 3\n");

    const ObjLoadResult result = loadObjMesh(objPath);
    t.expect(result.ok, "Single triangle OBJ should load.");
    t.expect(!isEmptyMesh(result.mesh), "Single triangle mesh should not be empty.");
    t.expect(result.mesh.triangles[0].i0 == 0, "Single triangle first index should be 0.");
    t.expect(result.mesh.triangles[0].i2 == 2, "Single triangle third index should be 2.");
}

void testGltfLoaderMinimalMesh(TestContext& t)
{
    const std::filesystem::path path = writeMinimalGltfFixture();
    const GltfLoadResult result = loadGltfMesh(path);

    t.expect(result.ok, "Minimal glTF should load.");
    if (!result.ok)
    {
        std::cout << "glTF load error: " << result.error << '\n';
        return;
    }
    t.expect(result.mesh.vertices.size() == 3, "Minimal glTF should produce three vertices.");
    t.expect(result.mesh.triangles.size() == 1, "Minimal glTF should produce one triangle.");
    t.expect(result.mesh.vertices[0].hasTexcoord == 1, "glTF TEXCOORD_0 should be stored.");
    t.expect(result.mesh.vertices[1].position.x > 2.9f, "glTF node translation/scale should affect positions.");
    t.expect(hasValidMeshMaterialIndices(result.mesh), "Minimal glTF material indices should be valid.");
}

void testGltfLoaderMissingFile(TestContext& t)
{
    const GltfLoadResult result = loadGltfMesh(std::filesystem::temp_directory_path() / "missing_raytracerrtx_mesh.gltf");

    t.expect(!result.ok, "Missing glTF should fail cleanly.");
    t.expect(!result.error.empty(), "Missing glTF should include an error message.");
}

void testGltfLoaderMissingBuffer(TestContext& t)
{
    const std::filesystem::path path = writeFixtureFile(
        "gltf_missing_buffer.gltf",
        "{\n"
        "  \"asset\": {\"version\": \"2.0\"},\n"
        "  \"buffers\": [{\"uri\": \"missing_gltf_buffer.bin\", \"byteLength\": 102}],\n"
        "  \"bufferViews\": [\n"
        "    {\"buffer\": 0, \"byteOffset\": 0, \"byteLength\": 36},\n"
        "    {\"buffer\": 0, \"byteOffset\": 36, \"byteLength\": 36},\n"
        "    {\"buffer\": 0, \"byteOffset\": 72, \"byteLength\": 24},\n"
        "    {\"buffer\": 0, \"byteOffset\": 96, \"byteLength\": 6}\n"
        "  ],\n"
        "  \"accessors\": [\n"
        "    {\"bufferView\": 0, \"componentType\": 5126, \"count\": 3, \"type\": \"VEC3\"},\n"
        "    {\"bufferView\": 1, \"componentType\": 5126, \"count\": 3, \"type\": \"VEC3\"},\n"
        "    {\"bufferView\": 2, \"componentType\": 5126, \"count\": 3, \"type\": \"VEC2\"},\n"
        "    {\"bufferView\": 3, \"componentType\": 5123, \"count\": 3, \"type\": \"SCALAR\"}\n"
        "  ],\n"
        "  \"meshes\": [{\"primitives\": [{\"attributes\": {\"POSITION\": 0, \"NORMAL\": 1, \"TEXCOORD_0\": 2}, \"indices\": 3}]}]\n"
        "}\n");

    const GltfLoadResult result = loadGltfMesh(path);
    t.expect(!result.ok, "glTF with missing external buffer should fail cleanly.");
    t.expect(!result.error.empty(), "Missing buffer failure should include an error message.");
}

void testGltfLoaderMaterialFactors(TestContext& t)
{
    const std::filesystem::path path = writeMinimalGltfFixture();
    const GltfLoadResult result = loadGltfMesh(path);

    t.expect(result.ok, "glTF material factor fixture should load.");
    if (!result.ok)
    {
        std::cout << "glTF material factor error: " << result.error << '\n';
        return;
    }
    t.expect(!result.mesh.materials.empty(), "glTF material list should not be empty.");
    t.expect(result.mesh.materials[0].materialType == MaterialMetal, "glTF metallicFactor should map to metal material.");
    t.expect(almostEqual(result.mesh.materials[0].roughness, 0.25f), "glTF roughnessFactor should be parsed.");
    t.expect(result.mesh.materials[0].color.z > result.mesh.materials[0].color.x, "glTF baseColorFactor should be parsed.");
}

void testGltfLoaderMinimalGlb(TestContext& t)
{
    const std::filesystem::path path = writeMinimalGlbFixture();
    const GltfLoadResult result = loadGltfMesh(path);

    t.expect(result.ok, "Minimal GLB should load.");
    if (!result.ok)
    {
        std::cout << "GLB load error: " << result.error << '\n';
        return;
    }
    t.expect(result.mesh.vertices.size() == 3, "Minimal GLB should produce three vertices.");
    t.expect(result.mesh.triangles.size() == 1, "Minimal GLB should produce one triangle.");
    t.expect(result.mesh.vertices[0].position.y > 0.9f, "GLB node translation should affect positions.");
    t.expect(result.mesh.vertices[1].position.y > 2.9f, "GLB quaternion rotation and scale should affect positions.");
    t.expect(hasValidMeshMaterialIndices(result.mesh), "Minimal GLB material indices should be valid.");
}

void testGltfLoaderTextureMaps(TestContext& t)
{
    std::vector<unsigned char> bin = minimalGltfBin();
    writeFixtureBinaryFile("gltf_textures.bin", bin);
    writeFixtureBinaryFile("gltf_albedo.png", tinyPngBytes());
    writeFixtureBinaryFile("gltf_normal.png", tinyPngBytes());
    writeFixtureBinaryFile("gltf_metal_rough.png", tinyPngBytes());
    const std::filesystem::path path = writeFixtureFile(
        "gltf_textures.gltf",
        "{\n"
        "  \"asset\": {\"version\": \"2.0\"},\n"
        "  \"buffers\": [{\"uri\": \"gltf_textures.bin\", \"byteLength\": 102}],\n"
        "  \"bufferViews\": [\n"
        "    {\"buffer\": 0, \"byteOffset\": 0, \"byteLength\": 36},\n"
        "    {\"buffer\": 0, \"byteOffset\": 36, \"byteLength\": 36},\n"
        "    {\"buffer\": 0, \"byteOffset\": 72, \"byteLength\": 24},\n"
        "    {\"buffer\": 0, \"byteOffset\": 96, \"byteLength\": 6}\n"
        "  ],\n"
        "  \"accessors\": [\n"
        "    {\"bufferView\": 0, \"componentType\": 5126, \"count\": 3, \"type\": \"VEC3\"},\n"
        "    {\"bufferView\": 1, \"componentType\": 5126, \"count\": 3, \"type\": \"VEC3\"},\n"
        "    {\"bufferView\": 2, \"componentType\": 5126, \"count\": 3, \"type\": \"VEC2\"},\n"
        "    {\"bufferView\": 3, \"componentType\": 5123, \"count\": 3, \"type\": \"SCALAR\"}\n"
        "  ],\n"
        "  \"images\": [\n"
        "    {\"uri\": \"gltf_albedo.png\"},\n"
        "    {\"uri\": \"gltf_normal.png\"},\n"
        "    {\"uri\": \"gltf_metal_rough.png\"}\n"
        "  ],\n"
        "  \"textures\": [{\"source\": 0}, {\"source\": 1}, {\"source\": 2}],\n"
        "  \"materials\": [{\n"
        "    \"name\": \"textured_gltf_material\",\n"
        "    \"normalTexture\": {\"index\": 1},\n"
        "    \"pbrMetallicRoughness\": {\n"
        "      \"baseColorTexture\": {\"index\": 0},\n"
        "      \"metallicRoughnessTexture\": {\"index\": 2},\n"
        "      \"metallicFactor\": 1.0,\n"
        "      \"roughnessFactor\": 0.5\n"
        "    }\n"
        "  }],\n"
        "  \"meshes\": [{\"primitives\": [{\"attributes\": {\"POSITION\": 0, \"NORMAL\": 1, \"TEXCOORD_0\": 2}, \"indices\": 3, \"material\": 0}]}]\n"
        "}\n");

    const GltfLoadResult result = loadGltfMesh(path);
    t.expect(result.ok, "glTF texture map fixture should load.");
    if (!result.ok)
    {
        std::cout << "glTF texture map error: " << result.error << '\n';
        return;
    }
    t.expect(result.mesh.materials[0].textureIndex >= 0, "glTF baseColorTexture should be stored.");
    t.expect(result.mesh.materials[0].normalTextureIndex >= 0, "glTF normalTexture should be stored.");
    t.expect(result.mesh.materials[0].metallicTextureIndex >= 0, "glTF metallicRoughnessTexture should store metallic map index.");
    t.expect(result.mesh.materials[0].roughnessTextureIndex >= 0, "glTF metallicRoughnessTexture should store roughness map index.");
    t.expect(result.mesh.materials[0].metallicTextureIndex == result.mesh.materials[0].roughnessTextureIndex, "glTF packed metallic/roughness should share one texture.");
    t.expect(result.mesh.textures.size() == 3, "glTF texture maps should load image data.");
}

void testGltfLoaderNodeHierarchyTransform(TestContext& t)
{
    std::vector<unsigned char> bin = minimalGltfBin();
    writeFixtureBinaryFile("gltf_nodes.bin", bin);
    const std::filesystem::path path = writeFixtureFile(
        "gltf_nodes.gltf",
        "{\n"
        "  \"asset\": {\"version\": \"2.0\"},\n"
        "  \"buffers\": [{\"uri\": \"gltf_nodes.bin\", \"byteLength\": 102}],\n"
        "  \"bufferViews\": [\n"
        "    {\"buffer\": 0, \"byteOffset\": 0, \"byteLength\": 36},\n"
        "    {\"buffer\": 0, \"byteOffset\": 36, \"byteLength\": 36},\n"
        "    {\"buffer\": 0, \"byteOffset\": 72, \"byteLength\": 24},\n"
        "    {\"buffer\": 0, \"byteOffset\": 96, \"byteLength\": 6}\n"
        "  ],\n"
        "  \"accessors\": [\n"
        "    {\"bufferView\": 0, \"componentType\": 5126, \"count\": 3, \"type\": \"VEC3\"},\n"
        "    {\"bufferView\": 1, \"componentType\": 5126, \"count\": 3, \"type\": \"VEC3\"},\n"
        "    {\"bufferView\": 2, \"componentType\": 5126, \"count\": 3, \"type\": \"VEC2\"},\n"
        "    {\"bufferView\": 3, \"componentType\": 5123, \"count\": 3, \"type\": \"SCALAR\"}\n"
        "  ],\n"
        "  \"meshes\": [{\"primitives\": [{\"attributes\": {\"POSITION\": 0, \"NORMAL\": 1, \"TEXCOORD_0\": 2}, \"indices\": 3}]}],\n"
        "  \"nodes\": [\n"
        "    {\"children\": [1], \"translation\": [2.0, 0.0, 0.0]},\n"
        "    {\"mesh\": 0, \"translation\": [0.0, 3.0, 0.0], \"scale\": [2.0, 2.0, 2.0]}\n"
        "  ],\n"
        "  \"scenes\": [{\"nodes\": [0]}],\n"
        "  \"scene\": 0\n"
        "}\n");

    const GltfLoadResult result = loadGltfMesh(path);
    t.expect(result.ok, "glTF node hierarchy fixture should load.");
    if (!result.ok)
    {
        std::cout << "glTF node hierarchy error: " << result.error << '\n';
        return;
    }
    t.expect(almostEqual(result.mesh.vertices[0].position.x, 2.0f), "Parent node translation should affect child mesh X.");
    t.expect(almostEqual(result.mesh.vertices[0].position.y, 3.0f), "Child node translation should affect mesh Y.");
    t.expect(almostEqual(result.mesh.vertices[1].position.x, 4.0f), "Child node scale should affect mesh positions.");
}

void testDemoGltfAssetLoads(TestContext& t)
{
    const std::filesystem::path gltfPath = findAssetMesh("minimal_gltf.gltf");
    t.expect(!gltfPath.empty(), "Demo glTF asset should exist.");
    if (gltfPath.empty())
    {
        return;
    }

    const GltfLoadResult result = loadGltfMesh(gltfPath);
    t.expect(result.ok, "Demo glTF asset should load.");
    t.expect(result.mesh.vertices.size() == 3, "Demo glTF should contain three vertices.");
    t.expect(result.mesh.materials[0].materialType == MaterialMetal, "Demo glTF material should map metallic factor.");
    t.expect(hasValidMeshMaterialIndices(result.mesh), "Demo glTF material indices should be valid.");
}

void testObjStillLoadsAfterGltfSupport(TestContext& t)
{
    const std::filesystem::path objPath = findDemoObjAsset();
    t.expect(!objPath.empty(), "Demo OBJ asset should exist.");
    const ObjLoadResult result = loadObjMesh(objPath);

    t.expect(result.ok, "OBJ loader should still work after adding glTF support.");
    t.expect(!isEmptyMesh(result.mesh), "OBJ loader should still produce geometry.");
}

void testDemoObjAssetLoads(TestContext& t)
{
    const std::filesystem::path objPath = findDemoObjAsset();
    t.expect(!objPath.empty(), "Demo OBJ asset must exist.");
    if (objPath.empty())
    {
        return;
    }

    const ObjLoadResult result = loadObjMesh(objPath);
    t.expect(result.ok, "Demo OBJ asset should load.");
    t.expect(result.mesh.vertices.size() > 0, "Demo OBJ should contain vertices.");
    t.expect(result.mesh.triangles.size() > 0, "Demo OBJ should contain triangles.");
    t.expect(result.mesh.materials.size() >= 4, "Demo OBJ should load default plus named MTL materials.");
    t.expect(hasValidMeshMaterialIndices(result.mesh), "Demo OBJ material indices should be valid.");
}

void testTexturedCubeAssetLoads(TestContext& t)
{
    const std::filesystem::path objPath = findAssetMesh("textured_cube.obj");
    t.expect(!objPath.empty(), "Textured cube OBJ asset must exist.");
    if (objPath.empty())
    {
        return;
    }

    const ObjLoadResult result = loadObjMesh(objPath);
    t.expect(result.ok, "Textured cube OBJ asset should load.");
    t.expect(result.mesh.vertices.size() == 36, "Textured cube should contain 12 triangulated faces.");
    t.expect(result.mesh.triangles.size() == 12, "Textured cube should contain 12 triangles.");
    t.expect(!result.mesh.textures.empty(), "Textured cube should load the shared checker texture.");
    t.expect(hasValidMeshMaterialIndices(result.mesh), "Textured cube material indices should be valid.");
}

void testDefaultScene(TestContext& t)
{
    const SceneState scene = makeDefaultScene();
    t.expect(scene.spheres.size() == 3, "Default scene must contain 3 spheres.");
    t.expect(scene.materials.size() == 3, "Default scene must contain 3 materials.");
    t.expect(!isEmptyMesh(scene.mesh), "Default scene must contain a triangle mesh.");
    t.expect(!scene.meshObjects.empty(), "Default scene must contain mesh objects.");
    t.expect(scene.mesh.materials.size() >= 3, "Default mesh should contain several materials.");
    t.expect(hasValidMeshMaterialIndices(scene.mesh), "Default mesh material indices should be valid.");
    bool hasMirrorMeshMaterial = false;
    for (const MeshMaterial& material : scene.mesh.materials)
    {
        hasMirrorMeshMaterial = hasMirrorMeshMaterial || material.materialType == MaterialMirror;
    }
    t.expect(hasMirrorMeshMaterial, "Default mesh should include a mirror material.");
    for (const SphereMaterial& material : scene.materials)
    {
        t.expect(material.materialType == MaterialDiffuse, "All default spheres should start as diffuse.");
    }
    t.expect(scene.selectedSphere == 0, "Default selected sphere should be index 0.");
}

void testDefaultSceneMeshGeometry(TestContext& t)
{
    const SceneState scene = makeDefaultScene();
    t.expect(scene.mesh.vertices.size() > 0, "Default mesh vertex count should be positive.");
    t.expect(scene.mesh.triangles.size() > 0, "Default mesh triangle count should be positive.");
    t.expect(!scene.mesh.materials.empty(), "Default mesh material count should be positive.");
    t.expect(hasValidMeshMaterialIndices(scene.mesh), "Default mesh material indices should stay in range.");
}

void testBuiltInCubeMesh(TestContext& t)
{
    const MeshData mesh = createCubeMesh();
    t.expect(mesh.vertices.size() == 24, "Built-in cube should use per-face vertices.");
    t.expect(mesh.triangles.size() == 12, "Built-in cube should contain 12 triangles.");
    t.expect(!isEmptyMesh(mesh), "Built-in cube should not be empty.");
    t.expect(hasValidMeshMaterialIndices(mesh), "Built-in cube material indices should be valid.");
}

void testBuiltInPyramidMesh(TestContext& t)
{
    const MeshData mesh = createPyramidMesh();
    t.expect(mesh.vertices.size() == 16, "Built-in pyramid should use face vertices.");
    t.expect(mesh.triangles.size() == 6, "Built-in pyramid should contain base and side triangles.");
    t.expect(!isEmptyMesh(mesh), "Built-in pyramid should not be empty.");
    t.expect(hasValidMeshMaterialIndices(mesh), "Built-in pyramid material indices should be valid.");
}

void testBuiltInPlaneMesh(TestContext& t)
{
    const MeshData mesh = createPlaneMesh();
    t.expect(mesh.vertices.size() == 4, "Built-in plane should contain 4 vertices.");
    t.expect(mesh.triangles.size() == 2, "Built-in plane should contain 2 triangles.");
    t.expect(!mesh.vertices.empty() && almostEqual(mesh.vertices[0].normal.y, 1.0f), "Built-in plane normal should point upward.");
    t.expect(hasValidMeshMaterialIndices(mesh), "Built-in plane material indices should be valid.");
}

void testSceneConfigLoadsValidScene(TestContext& t)
{
    const std::filesystem::path objPath = writeFixtureFile(
        "config_mesh.obj",
        "v 0 0 0\n"
        "v 1 0 0\n"
        "v 0 1 0\n"
        "f 1 2 3\n");

    const std::filesystem::path configPath = writeFixtureFile(
        "valid_scene.json",
        "{\n"
        "  \"camera\": {\"position\": [1, 2, 3], \"yaw\": 15, \"pitch\": -10, \"fov\": 55},\n"
        "  \"light\": {\"position\": [4, 5, 6]},\n"
        "  \"meshObjects\": [\n"
        "    {\"path\": \"config_mesh.obj\", \"position\": [2, 0, 0], \"rotation\": [0, 0, 0], \"scale\": [2, 2, 2]}\n"
        "  ]\n"
        "}\n");

    (void)objPath;
    const SceneConfigResult config = loadSceneConfigFile(configPath);
    t.expect(config.ok, "Valid scene config should load: " + config.error);
    const SceneBuildResult scene = buildSceneFromConfig(config.config, configPath.parent_path());
    t.expect(scene.ok, "Valid scene config should build a scene.");
    t.expect(almostEqual(scene.camera.position.x, 1.0f), "Scene config should apply camera position.");
    t.expect(almostEqual(scene.scene.lightPosition.y, 5.0f), "Scene config should apply light position.");
    t.expect(scene.scene.mesh.vertices.size() == 3, "Scene config mesh path should load OBJ vertices.");
    t.expect(almostEqual(scene.scene.mesh.vertices[1].position.x, 4.0f), "Scene config transform should affect mesh vertices.");
    t.expect(scene.scene.meshObjects.size() == 1, "Scene config should preserve mesh object list.");
    t.expect(almostEqual(scene.scene.meshObjects[0].scale.x, 2.0f), "Scene config should preserve mesh object scale.");
}

void testSceneConfigMultipleMeshes(TestContext& t)
{
    writeFixtureFile(
        "multi_a.obj",
        "v 0 0 0\n"
        "v 1 0 0\n"
        "v 0 1 0\n"
        "f 1 2 3\n");
    writeFixtureFile(
        "multi_b.obj",
        "v 0 0 0\n"
        "v 0 1 0\n"
        "v 0 0 1\n"
        "f 1 2 3\n");

    const std::filesystem::path configPath = writeFixtureFile(
        "multi_scene.json",
        "{\n"
        "  \"meshObjects\": [\n"
        "    {\"path\": \"multi_a.obj\", \"position\": [1, 0, 0], \"rotation\": [0, 0, 0], \"scale\": [1, 1, 1]},\n"
        "    {\"path\": \"multi_b.obj\", \"position\": [-1, 0, 0], \"rotation\": [0, 45, 0], \"scale\": [2, 1, 1]}\n"
        "  ]\n"
        "}\n");

    const SceneConfigResult config = loadSceneConfigFile(configPath);
    t.expect(config.ok, "Multiple-mesh scene config should parse: " + config.error);
    const SceneBuildResult scene = buildSceneFromConfig(config.config, configPath.parent_path());
    t.expect(scene.ok, "Multiple-mesh scene config should build.");
    t.expect(scene.scene.meshObjects.size() == 2, "SceneState should store two mesh objects.");
    t.expect(scene.scene.mesh.triangles.size() == 2, "Compatibility combined mesh should include both triangles.");
    t.expect(almostEqual(scene.scene.meshObjects[1].rotation.y, 45.0f), "Mesh object rotation should be preserved.");
    t.expect(almostEqual(scene.scene.meshObjects[1].scale.x, 2.0f), "Mesh object scale should be preserved.");
    t.expect(hasValidMeshMaterialIndices(scene.scene.meshObjects[0].mesh), "First mesh object material indices should be valid.");
    t.expect(hasValidMeshMaterialIndices(scene.scene.meshObjects[1].mesh), "Second mesh object material indices should be valid.");
}

void testSceneConfigMissingFile(TestContext& t)
{
    const std::filesystem::path configPath = std::filesystem::temp_directory_path() / "raytracerrtx_missing_scene_config.json";
    const SceneConfigResult config = loadSceneConfigFile(configPath);
    t.expect(!config.ok, "Missing scene config should fail cleanly.");
    t.expect(config.error.find("could not be opened") != std::string::npos, "Missing scene config should explain open failure.");
}

void testSceneConfigMeshPathApplied(TestContext& t)
{
    const std::filesystem::path objPath = writeFixtureFile(
        "direct_mesh.obj",
        "v 0 0 0\n"
        "v 0 2 0\n"
        "v 0 0 2\n"
        "f 1 2 3\n");

    const SceneBuildResult scene = buildSceneFromMeshPath(objPath);
    t.expect(scene.ok, "Direct mesh path should build a scene.");
    t.expect(scene.scene.mesh.vertices.size() == 3, "Direct mesh path should replace default mesh vertices.");
    t.expect(almostEqual(scene.scene.mesh.vertices[1].position.y, 2.0f), "Direct mesh path should use requested OBJ data.");
    t.expect(scene.scene.meshObjects.size() == 1, "Direct mesh path should create one mesh object.");
}

void testSceneConfigTuningFields(TestContext& t)
{
    const std::filesystem::path configPath = writeFixtureFile(
        "scene_tuning.json",
        "{\n"
        "  \"render\": {\"exposure\": 0.65, \"skyIntensity\": 0.55},\n"
        "  \"light\": {\"position\": [4, 7, -5], \"intensity\": 0.75}\n"
        "}\n");

    const SceneConfigResult config = loadSceneConfigFile(configPath);
    t.expect(config.ok, "Scene config with render tuning should load.");
    t.expect(config.config.hasExposure, "Scene config should mark exposure as present.");
    t.expect(config.config.hasSkyIntensity, "Scene config should mark sky intensity as present.");
    t.expect(config.config.hasLightIntensity, "Scene config should mark light intensity as present.");

    const SceneBuildResult scene = buildSceneFromConfig(config.config, configPath.parent_path());
    t.expect(scene.ok, "Scene config with tuning and fallback mesh should build.");
    t.expect(almostEqual(scene.scene.exposure, 0.65f), "Scene config should apply exposure.");
    t.expect(almostEqual(scene.scene.skyIntensity, 0.55f), "Scene config should apply sky intensity.");
    t.expect(almostEqual(scene.scene.lightIntensity, 0.75f), "Scene config should apply light intensity.");
}

void testSceneConfigTuningClamps(TestContext& t)
{
    const std::filesystem::path configPath = writeFixtureFile(
        "scene_tuning_clamp.json",
        "{\n"
        "  \"exposure\": -4.0,\n"
        "  \"skyIntensity\": 42.0,\n"
        "  \"light\": {\"intensity\": -2.0}\n"
        "}\n");

    const SceneConfigResult config = loadSceneConfigFile(configPath);
    t.expect(config.ok, "Out-of-range numeric tuning values should parse.");

    const SceneBuildResult scene = buildSceneFromConfig(config.config, configPath.parent_path());
    t.expect(scene.ok, "Out-of-range tuning values should build through clamp/fallback.");
    t.expect(almostEqual(scene.scene.exposure, 0.1f), "Exposure should clamp to minimum.");
    t.expect(almostEqual(scene.scene.skyIntensity, 3.0f), "Sky intensity should clamp to maximum.");
    t.expect(almostEqual(scene.scene.lightIntensity, 0.0f), "Light intensity should clamp to minimum.");
}

void testSceneConfigAreaLightAndEnvironment(TestContext& t)
{
    const std::filesystem::path envPath = writeFixtureFile(
        "raytracerrtx_env_map.ppm",
        "P3\n"
        "2 1\n"
        "255\n"
        "255 0 0   0 0 255\n");
    const std::filesystem::path configPath = writeFixtureFile(
        "raytracerrtx_area_environment_scene.json",
        "{\n"
        "  \"light\": {\"position\": [1, 4, -2], \"intensity\": 1.1, \"size\": 2.5},\n"
        "  \"environment\": {\"type\": \"map\", \"path\": \"raytracerrtx_env_map.ppm\", \"intensity\": 1.6}\n"
        "}\n");

    const SceneConfigResult config = loadSceneConfigFile(configPath);
    t.expect(config.ok, "Area light/environment config should parse: " + config.error);
    const SceneBuildResult scene = buildSceneFromConfig(config.config, configPath.parent_path());
    t.expect(scene.ok, "Area light/environment scene should build.");
    t.expect(almostEqual(scene.scene.areaLightRadius, 2.5f), "Area light size should parse.");
    t.expect(almostEqual(scene.scene.environmentIntensity, 1.6f), "Environment intensity should parse.");
    t.expect(scene.scene.environmentMap.width == 2u, "Environment PPM width should load.");
    t.expect(scene.scene.environmentMap.height == 1u, "Environment PPM height should load.");
    t.expect(scene.scene.environmentMap.pixels.size() == 2u, "Environment PPM pixels should load.");
}

void testSceneConfigAreaEnvironmentFallbacks(TestContext& t)
{
    const std::filesystem::path configPath = writeFixtureFile(
        "raytracerrtx_area_environment_fallback.json",
        "{\n"
        "  \"light\": {\"position\": [0, 5, 0], \"radius\": -10},\n"
        "  \"environment\": {\"type\": \"map\", \"path\": \"missing_environment.ppm\", \"intensity\": 99}\n"
        "}\n");

    const SceneConfigResult config = loadSceneConfigFile(configPath);
    t.expect(config.ok, "Fallback environment config should parse: " + config.error);
    const SceneBuildResult scene = buildSceneFromConfig(config.config, configPath.parent_path());
    t.expect(scene.ok, "Fallback environment scene should build.");
    t.expect(almostEqual(scene.scene.areaLightRadius, 0.0f), "Invalid area light size should clamp to zero.");
    t.expect(almostEqual(scene.scene.environmentIntensity, 4.0f), "Invalid environment intensity should clamp to maximum.");
    t.expect(scene.scene.environmentMap.pixels.empty(), "Missing environment map should fall back to gradient sky.");
    t.expect(!scene.warnings.empty(), "Missing environment map should add a warning.");
}

void testSceneConfigJsonMaterialParses(TestContext& t)
{
    const std::filesystem::path configPath = writeFixtureFile(
        "json_material_parse.json",
        "{\n"
        "  \"materials\": [{\n"
        "    \"name\": \"warm_metal\",\n"
        "    \"type\": \"metal\",\n"
        "    \"baseColor\": [0.8, 0.6, 0.3],\n"
        "    \"specularColor\": [0.9, 0.8, 0.7],\n"
        "    \"roughness\": 0.22,\n"
        "    \"metallic\": 1.0,\n"
        "    \"ior\": 1.6,\n"
        "    \"alpha\": 0.95,\n"
        "    \"texture\": \"albedo.ppm\",\n"
        "    \"normalMap\": \"normal.ppm\"\n"
        "  }]\n"
        "}\n");

    const SceneConfigResult config = loadSceneConfigFile(configPath);
    t.expect(config.ok, "JSON material config should parse: " + config.error);
    t.expect(config.config.materials.size() == 1, "One JSON material should be stored.");
    t.expect(config.config.materials[0].materialType == MaterialMetal, "JSON material type should map to metal.");
    t.expect(almostEqual(config.config.materials[0].baseColor.x, 0.8f), "JSON material baseColor should parse.");
    t.expect(almostEqual(config.config.materials[0].roughness, 0.22f), "JSON material roughness should parse.");
    t.expect(config.config.materials[0].texturePath == "albedo.ppm", "JSON material texture path should parse.");
    t.expect(config.config.materials[0].normalTexturePath == "normal.ppm", "JSON material normal map path should parse.");
}

void testSceneConfigMaterialAssignedToSphere(TestContext& t)
{
    const std::filesystem::path configPath = writeFixtureFile(
        "json_sphere_material.json",
        "{\n"
        "  \"materials\": [{\"name\": \"glass_blue\", \"type\": \"glass\", \"baseColor\": [0.45, 0.75, 1.0], \"alpha\": 0.4}],\n"
        "  \"sphereMaterials\": [\"glass_blue\"]\n"
        "}\n");

    const SceneConfigResult config = loadSceneConfigFile(configPath);
    t.expect(config.ok, "Sphere material config should parse.");
    const SceneBuildResult scene = buildSceneFromConfig(config.config, configPath.parent_path());
    t.expect(scene.ok, "Sphere material config should build.");
    t.expect(scene.scene.materials[0].materialType == MaterialDielectric, "JSON material should be assigned to first sphere.");
    t.expect(almostEqual(scene.scene.materials[0].alpha, 0.4f), "Sphere JSON material alpha should be applied.");
}

void testSceneConfigMaterialAssignedToMesh(TestContext& t)
{
    writeFixtureFile(
        "json_mesh_material.obj",
        "v 0 0 0\n"
        "v 1 0 0\n"
        "v 0 1 0\n"
        "f 1 2 3\n");
    const std::filesystem::path configPath = writeFixtureFile(
        "json_mesh_material.json",
        "{\n"
        "  \"materials\": [{\"name\": \"mirror_override\", \"type\": \"mirror\", \"roughness\": 0.03}],\n"
        "  \"meshObjects\": [{\"path\": \"json_mesh_material.obj\", \"material\": \"mirror_override\"}]\n"
        "}\n");

    const SceneConfigResult config = loadSceneConfigFile(configPath);
    t.expect(config.ok, "Mesh material override config should parse.");
    const SceneBuildResult scene = buildSceneFromConfig(config.config, configPath.parent_path());
    t.expect(scene.ok, "Mesh material override config should build.");
    t.expect(!scene.scene.meshObjects.empty(), "Mesh material override scene should keep mesh object.");
    t.expect(scene.scene.meshObjects[0].mesh.materials[0].materialType == MaterialMirror, "JSON material should override mesh material.");
    t.expect(scene.scene.mesh.materials[0].materialType == MaterialMirror, "Combined mesh should receive JSON material override.");
}

void testSceneConfigInvalidMaterialTypeFallback(TestContext& t)
{
    const std::filesystem::path configPath = writeFixtureFile(
        "json_invalid_material_type.json",
        "{\n"
        "  \"materials\": [{\"name\": \"odd\", \"type\": \"plasma\", \"baseColor\": [1, 0, 0]}],\n"
        "  \"sphereMaterials\": [\"odd\"]\n"
        "}\n");

    const SceneConfigResult config = loadSceneConfigFile(configPath);
    t.expect(config.ok, "Unknown material type should not fail config parsing.");
    t.expect(config.config.materials[0].materialType == MaterialDiffuse, "Unknown material type should fall back to matte/diffuse.");
    const SceneBuildResult scene = buildSceneFromConfig(config.config, configPath.parent_path());
    t.expect(scene.ok, "Unknown material type fallback should build.");
    t.expect(!scene.warnings.empty(), "Unknown material type should produce a build warning.");
}

void testSceneConfigInvalidJsonFailsCleanly(TestContext& t)
{
    const std::filesystem::path configPath = writeFixtureFile(
        "invalid_json_scene.json",
        "{ \"camera\": { \"position\": [0, 1, 2], }\n");

    const SceneConfigResult config = loadSceneConfigFile(configPath);

    t.expect(!config.ok, "Invalid JSON should fail cleanly.");
    t.expect(config.error.find("Invalid scene config") != std::string::npos, "Invalid JSON should explain scene config parsing.");
}

void testSceneConfigInvalidMaterialConfigFailsCleanly(TestContext& t)
{
    const std::filesystem::path configPath = writeFixtureFile(
        "invalid_material_config.json",
        "{\n"
        "  \"materials\": [{\"name\": \"bad\", \"type\": \"matte\", \"baseColor\": \"red\"}]\n"
        "}\n");

    const SceneConfigResult config = loadSceneConfigFile(configPath);

    t.expect(!config.ok, "Invalid material config should fail cleanly.");
    t.expect(config.error.find("baseColor") != std::string::npos, "Invalid material config should name the invalid field.");
}

void testSceneConfigInvalidTransformFailsCleanly(TestContext& t)
{
    const std::filesystem::path configPath = writeFixtureFile(
        "invalid_transform_config.json",
        "{\n"
        "  \"meshObjects\": [{\"path\": \"missing.obj\", \"position\": [1, 2]}]\n"
        "}\n");

    const SceneConfigResult config = loadSceneConfigFile(configPath);

    t.expect(!config.ok, "Invalid scene transform should fail cleanly.");
    t.expect(config.error.find("position") != std::string::npos, "Invalid transform error should name the invalid field.");
}

void testSceneConfigMaterialParameterClamps(TestContext& t)
{
    const std::filesystem::path configPath = writeFixtureFile(
        "json_material_clamps.json",
        "{\n"
        "  \"materials\": [{\"name\": \"clamped\", \"type\": \"matte\", \"roughness\": -4, \"metallic\": 8, \"alpha\": -2, \"ior\": 9}],\n"
        "  \"sphereMaterials\": [\"clamped\"]\n"
        "}\n");

    const SceneConfigResult config = loadSceneConfigFile(configPath);
    t.expect(config.ok, "Out-of-range material parameters should parse.");
    t.expect(almostEqual(config.config.materials[0].roughness, 0.02f), "JSON roughness should clamp to minimum.");
    t.expect(almostEqual(config.config.materials[0].metallic, 1.0f), "JSON metallic should clamp to maximum.");
    t.expect(almostEqual(config.config.materials[0].alpha, 0.0f), "JSON alpha should clamp to minimum.");
    t.expect(almostEqual(config.config.materials[0].ior, 2.8f), "JSON IOR should clamp to maximum.");
    t.expect(config.config.materials[0].materialType == MaterialMetal, "High metallic should map matte material to metal.");
}

void testSceneConfigOldSceneStillLoads(TestContext& t)
{
    const std::filesystem::path configPath = writeFixtureFile(
        "old_scene_without_materials.json",
        "{\n"
        "  \"camera\": {\"position\": [0, 4, 12], \"yaw\": -90, \"pitch\": -10, \"fov\": 60},\n"
        "  \"light\": {\"position\": [4, 8, -5]}\n"
        "}\n");

    const SceneConfigResult config = loadSceneConfigFile(configPath);
    t.expect(config.ok, "Old scene config without JSON materials should still parse.");
    const SceneBuildResult scene = buildSceneFromConfig(config.config, configPath.parent_path());
    t.expect(scene.ok, "Old scene config without JSON materials should still build.");
    t.expect(scene.scene.materials.size() == 3, "Old scene config should keep default sphere materials.");
    t.expect(!scene.scene.meshObjects.empty(), "Old scene config should keep fallback demo mesh.");
}

void testAllDemoSceneConfigsLoad(TestContext& t)
{
    const std::vector<std::string> scenes = {
        "demo_scene.json",
        "textured_cube_scene.json",
        "material_showcase_scene.json"
    };

    for (const std::string& fileName : scenes)
    {
        const std::filesystem::path scenePath = findAssetScene(fileName);
        t.expect(!scenePath.empty(), "Demo scene asset should exist: " + fileName);
        if (scenePath.empty())
        {
            continue;
        }

        const SceneConfigResult config = loadSceneConfigFile(scenePath);
        t.expect(config.ok, "Demo scene config should parse: " + fileName + " " + config.error);
        if (!config.ok)
        {
            continue;
        }

        const SceneBuildResult scene = buildSceneFromConfig(config.config, scenePath.parent_path());
        t.expect(scene.ok, "Demo scene config should build: " + fileName + " " + scene.error);
        t.expect(!scene.scene.meshObjects.empty(), "Demo scene should contain mesh objects: " + fileName);
        t.expect(hasValidMeshMaterialIndices(scene.scene.mesh), "Demo scene combined mesh material indices should be valid: " + fileName);
        t.expect(scene.scene.spheres.size() >= 3, "Public demo scene should keep visible default spheres: " + fileName);
    }
}

void testScenePresetResetCameraLight(TestContext& t)
{
    SceneBuildResult preset = buildDefaultSceneInput();
    t.expect(preset.ok, "Default preset should build for reset test.");
    SceneState scene = preset.scene;
    CameraState camera = preset.camera;
    scene.lightPosition = make_float3(-20.0f, 25.0f, 20.0f);
    camera.position = make_float3(7.0f, 8.0f, 9.0f);
    camera.yaw = -30.0f;
    camera.pitch = 20.0f;

    const bool reset = resetSceneViewFromPreset(preset, scene, camera);
    t.expect(reset, "Reset should succeed for a valid preset.");
    t.expect(almostEqual(camera.position.x, preset.camera.position.x), "Reset should restore camera position.");
    t.expect(almostEqual(camera.yaw, preset.camera.yaw), "Reset should restore camera yaw.");
    t.expect(almostEqual(scene.lightPosition.x, preset.scene.lightPosition.x), "Reset should restore light position.");
}

void testInvalidScenePresetIndexSafe(TestContext& t)
{
    std::vector<SceneBuildResult> presets;
    presets.push_back(buildDefaultSceneInput());
    SceneState scene = presets[0].scene;
    CameraState camera = presets[0].camera;
    scene.lightPosition = make_float3(1.0f, 2.0f, 3.0f);
    camera.yaw = 42.0f;

    const bool appliedNegative = applyScenePresetByIndex(presets, -1, scene, camera);
    const bool appliedHigh = applyScenePresetByIndex(presets, 99, scene, camera);
    t.expect(!appliedNegative, "Negative preset index should fail safely.");
    t.expect(!appliedHigh, "Out-of-range preset index should fail safely.");
    t.expect(almostEqual(scene.lightPosition.x, 1.0f), "Invalid preset should not change scene.");
    t.expect(almostEqual(camera.yaw, 42.0f), "Invalid preset should not change camera.");
}

void testScenePresetSessionEditsPersist(TestContext& t)
{
    std::vector<SceneBuildResult> presets;
    presets.push_back(buildDefaultSceneInput());
    presets.push_back(buildDefaultSceneInput());
    t.expect(presets[0].ok && presets[1].ok, "Default presets should build for session edit test.");

    SceneState scene = presets[0].scene;
    CameraState camera = presets[0].camera;
    const size_t originalSphereCount = scene.spheres.size();
    addSphere(scene);
    setSelectedSphereRadius(scene, 0.75f);
    camera.yaw = 12.0f;

    const bool saved = saveScenePresetByIndex(presets, 0, scene, camera);
    const bool switched = applyScenePresetByIndex(presets, 1, scene, camera);
    const bool returned = applyScenePresetByIndex(presets, 0, scene, camera);

    t.expect(saved, "Saving a valid scene preset index should succeed.");
    t.expect(switched, "Switching to another preset should succeed.");
    t.expect(returned, "Switching back to edited preset should succeed.");
    t.expect(scene.spheres.size() == originalSphereCount + 1, "Saved preset should keep added sphere.");
    t.expect(almostEqual(scene.spheres.back().radius, 0.75f), "Saved preset should keep edited sphere radius.");
    t.expect(almostEqual(camera.yaw, 12.0f), "Saved preset should keep edited camera.");
}

void testScenePresetSaveInvalidIndexSafe(TestContext& t)
{
    std::vector<SceneBuildResult> presets;
    presets.push_back(buildDefaultSceneInput());
    SceneState scene = presets[0].scene;
    CameraState camera = presets[0].camera;

    const bool savedNegative = saveScenePresetByIndex(presets, -1, scene, camera);
    const bool savedHigh = saveScenePresetByIndex(presets, 20, scene, camera);

    t.expect(!savedNegative, "Saving a negative preset index should fail safely.");
    t.expect(!savedHigh, "Saving an out-of-range preset index should fail safely.");
    t.expect(presets.size() == 1, "Invalid preset save should not resize preset list.");
}

void testSceneConfigFallbackDemoScene(TestContext& t)
{
    const SceneBuildResult scene = buildDefaultSceneInput();
    t.expect(scene.ok, "Default scene input should build.");
    t.expect(!isEmptyMesh(scene.scene.mesh), "Default scene input should keep fallback/demo mesh.");
    t.expect(!scene.scene.meshObjects.empty(), "Default scene input should keep mesh object list.");
    t.expect(hasValidMeshMaterialIndices(scene.scene.mesh), "Default scene input mesh material indices should be valid.");
    t.expect(scene.scene.exposure > 0.0f, "Default scene input should keep exposure.");
    t.expect(scene.scene.skyIntensity > 0.0f, "Default scene input should keep sky intensity.");
    t.expect(scene.scene.lightIntensity > 0.0f, "Default scene input should keep light intensity.");
}

void testAssetCacheReusesMeshPath(TestContext& t)
{
    const std::filesystem::path objPath = findDemoObjAsset();
    t.expect(!objPath.empty(), "Demo OBJ asset should exist for mesh cache test.");
    if (objPath.empty())
    {
        return;
    }

    AssetCache cache;
    const ObjLoadResult& first = cache.loadMesh(objPath);
    const ObjLoadResult& second = cache.loadMesh(objPath);

    t.expect(first.ok, "First cached mesh load should succeed.");
    t.expect(second.ok, "Second cached mesh load should succeed.");
    t.expect(cache.meshLoadCount(objPath) == 1, "Cache should load the same mesh path only once.");
    t.expect(first.mesh.vertices.size() == second.mesh.vertices.size(), "Cached mesh result should be reused.");
}

void testAssetCacheReusesTexturePath(TestContext& t)
{
    const std::filesystem::path texturePath = writeFixtureBinaryFile("asset_cache_texture.png", tinyPngBytes());
    AssetCache cache;

    const MeshTexture* first = cache.loadTexture(texturePath, "baseColor");
    const MeshTexture* second = cache.loadTexture(texturePath, "baseColor");

    t.expect(first != nullptr, "First cached texture load should succeed.");
    t.expect(second != nullptr, "Second cached texture load should succeed.");
    t.expect(first == second, "Texture cache should return the stored texture object.");
    t.expect(cache.textureLoadCount(texturePath, "baseColor") == 1, "Cache should load the same texture path only once.");
}

void testSceneReloadValidUpdatesScene(TestContext& t)
{
    writeFixtureFile(
        "reload_mesh.obj",
        "v 0 0 0\n"
        "v 1 0 0\n"
        "v 0 1 0\n"
        "vn 0 0 1\n"
        "f 1//1 2//1 3//1\n");

    const std::filesystem::path configPath = writeFixtureFile(
        "reload_valid_scene.json",
        "{\n"
        "  \"camera\": {\"position\": [1.0, 2.0, 3.0], \"yaw\": 25.0, \"pitch\": -5.0, \"fov\": 55.0},\n"
        "  \"light\": {\"position\": [4.0, 5.0, -6.0]},\n"
        "  \"meshObjects\": [{\"path\": \"reload_mesh.obj\"}]\n"
        "}\n");
    SceneBuildResult preset = buildDefaultSceneInput();
    SceneState scene = preset.scene;
    CameraState camera = preset.camera;
    AssetCache cache;
    bool reset = false;
    std::string error;

    const bool reloaded = reloadScenePresetFromConfig(configPath, preset, scene, camera, cache, reset, error);

    t.expect(reloaded, "Reloading a valid scene config should succeed.");
    t.expect(error.empty(), "Valid reload should not produce an error.");
    t.expect(reset, "Valid reload should request accumulation reset.");
    t.expect(almostEqual(scene.lightPosition.x, 4.0f), "Reload should update scene light.");
    t.expect(almostEqual(camera.position.z, 3.0f), "Reload should update camera.");
}

void testSceneReloadInvalidKeepsPreviousScene(TestContext& t)
{
    const std::filesystem::path configPath = writeFixtureFile("reload_invalid_scene.json", "{ invalid json\n");
    SceneBuildResult preset = buildDefaultSceneInput();
    SceneState scene = preset.scene;
    CameraState camera = preset.camera;
    scene.lightPosition = make_float3(9.0f, 8.0f, 7.0f);
    camera.yaw = 12.0f;
    AssetCache cache;
    bool reset = true;
    std::string error;

    const bool reloaded = reloadScenePresetFromConfig(configPath, preset, scene, camera, cache, reset, error);

    t.expect(!reloaded, "Reloading an invalid scene config should fail cleanly.");
    t.expect(!error.empty(), "Invalid reload should report an error.");
    t.expect(!reset, "Invalid reload should not request accumulation reset.");
    t.expect(almostEqual(scene.lightPosition.x, 9.0f), "Invalid reload should keep previous scene.");
    t.expect(almostEqual(camera.yaw, 12.0f), "Invalid reload should keep previous camera.");
}

void testToggleMaterial(TestContext& t)
{
    SceneState scene = makeDefaultScene();
    scene.selectedSphere = 1;
    const int initial = scene.materials[1].materialType;
    toggleSelectedMaterial(scene);
    t.expect(scene.materials[1].materialType != initial, "Material toggle should change type.");
    toggleSelectedMaterial(scene);
    t.expect(scene.materials[1].materialType == initial, "Material toggle should return to initial type.");
}

void testSphereMaterialPresetCycle(TestContext& t)
{
    SceneState scene = makeDefaultScene();
    scene.selectedSphere = 0;
    scene.materials[0].materialType = MaterialDiffuse;

    cycleSelectedSphereMaterialPreset(scene);
    t.expect(scene.materials[0].materialType == MaterialMirror, "Sphere material preset should cycle diffuse to mirror.");
    cycleSelectedSphereMaterialPreset(scene);
    t.expect(scene.materials[0].materialType == MaterialMetal, "Sphere material preset should cycle mirror to metal.");
    cycleSelectedSphereMaterialPreset(scene);
    t.expect(scene.materials[0].materialType == MaterialDielectric, "Sphere material preset should cycle metal to dielectric.");
}

void testSetSelectedSphereMaterialType(TestContext& t)
{
    SceneState scene = makeDefaultScene();
    scene.selectedSphere = 0;

    setSelectedSphereMaterialType(scene, MaterialDielectric);
    t.expect(scene.materials[0].materialType == MaterialDielectric, "Sphere material type setter should select glass.");
    t.expect(almostEqual(scene.materials[0].ior, 1.45f), "Glass preset should apply IOR default.");
    t.expect(almostEqual(scene.materials[0].alpha, 0.45f), "Glass preset should apply transparency default.");
    t.expect(almostEqual(scene.materials[0].roughness, 0.02f), "Glass preset should apply low roughness default.");

    setSelectedSphereMaterialType(scene, 999);
    t.expect(scene.materials[0].materialType == MaterialDiffuse, "Invalid sphere material type should fall back to matte.");
}

void testSetSelectedSphereMaterialTypeInvalidIndexSafe(TestContext& t)
{
    SceneState scene = makeDefaultScene();
    scene.selectedSphere = 100;
    const int unchanged = scene.materials[0].materialType;

    setSelectedSphereMaterialType(scene, MaterialMetal);

    t.expect(scene.materials[0].materialType == unchanged, "Invalid selected sphere should not modify materials.");
}

void testSelectedMeshChanges(TestContext& t)
{
    SceneState scene = makeDefaultScene();
    MeshObject copy = scene.meshObjects[0];
    copy.assetReference = "copy";
    scene.meshObjects.push_back(copy);
    scene.selectedMeshObject = 0;

    selectNextMeshObject(scene);
    t.expect(scene.selectedMeshObject == 1, "Selected mesh object should advance.");
    selectNextMeshObject(scene);
    t.expect(scene.selectedMeshObject == 0, "Selected mesh object should wrap.");
}

void testSelectedMeshMaterialPresetCycle(TestContext& t)
{
    SceneState scene = makeDefaultScene();
    scene.selectedMeshObject = 0;
    scene.selectedMeshMaterial = 0;
    scene.meshObjects[0].mesh.materials[0].materialType = MaterialDiffuse;

    cycleSelectedMeshMaterialPreset(scene);
    t.expect(scene.meshObjects[0].mesh.materials[0].materialType == MaterialMirror, "Mesh material preset should cycle diffuse to mirror.");
    cycleSelectedMeshMaterialPreset(scene);
    t.expect(scene.meshObjects[0].mesh.materials[0].materialType == MaterialMetal, "Mesh material preset should cycle mirror to metal.");
    t.expect(hasValidMeshMaterialIndices(scene.meshObjects[0].mesh), "Mesh material preset changes should keep material indices valid.");
}

void testSetSelectedMeshMaterialType(TestContext& t)
{
    SceneState scene = makeDefaultScene();
    scene.selectedMeshObject = 0;
    scene.selectedMeshMaterial = 0;
    scene.meshObjects[0].mesh.materials.push_back(scene.meshObjects[0].mesh.materials[0]);
    scene.mesh.materials.push_back(scene.mesh.materials[0]);

    setSelectedMeshMaterialType(scene, MaterialMetal);

    t.expect(scene.meshObjects[0].mesh.materials[0].materialType == MaterialMetal, "Mesh material type setter should select metal.");
    t.expect(scene.meshObjects[0].mesh.materials[1].materialType == MaterialMetal, "Mesh material type setter should apply to the whole selected mesh.");
    t.expect(scene.mesh.materials[0].materialType == MaterialMetal, "Combined mesh material should stay in sync.");
    t.expect(scene.mesh.materials[1].materialType == MaterialMetal, "Combined mesh material should sync every material slot of the selected mesh.");
    t.expect(almostEqual(scene.meshObjects[0].mesh.materials[0].roughness, 0.18f), "Metal preset should apply roughness default.");

    setSelectedMeshMaterialType(scene, 999);
    t.expect(scene.meshObjects[0].mesh.materials[0].materialType == MaterialDiffuse, "Invalid mesh material type should fall back to matte.");
    t.expect(scene.meshObjects[0].mesh.materials[1].materialType == MaterialDiffuse, "Invalid mesh material type fallback should apply to the whole selected mesh.");
    t.expect(scene.mesh.materials[0].materialType == MaterialDiffuse, "Combined mesh material should sync fallback type.");
    t.expect(scene.mesh.materials[1].materialType == MaterialDiffuse, "Combined mesh material should sync fallback type for every selected mesh material.");
}

void testSetSelectedMeshMaterialTypeInvalidSafe(TestContext& t)
{
    SceneState scene = makeDefaultScene();
    scene.selectedMeshObject = 100;
    scene.selectedMeshMaterial = 100;

    setSelectedMeshMaterialType(scene, MaterialDielectric);

    t.expect(scene.selectedMeshObject >= 0, "Invalid mesh type setter should clamp selected mesh object safely.");
    t.expect(scene.selectedMeshMaterial >= 0, "Invalid mesh type setter should clamp selected mesh material safely.");
    t.expect(hasValidMeshMaterialIndices(scene.mesh), "Invalid mesh type setter should keep combined material indices valid.");
}

void testSelectedMeshTransformControls(TestContext& t)
{
    SceneState scene = makeDefaultScene();
    scene.selectedMeshObject = 0;

    const bool moved = setSelectedMeshPosition(scene, make_float3(2.0f, 1.5f, -3.0f));
    const bool rotated = setSelectedMeshRotation(scene, make_float3(0.0f, 45.0f, 0.0f));
    const bool scaled = setSelectedMeshScale(scene, make_float3(1.5f, 2.0f, 0.75f));

    t.expect(moved, "Selected mesh position should be editable.");
    t.expect(rotated, "Selected mesh rotation should be editable.");
    t.expect(scaled, "Selected mesh scale should be editable.");
    t.expect(almostEqual(scene.meshObjects[0].position.x, 2.0f), "Mesh position should be stored.");
    t.expect(almostEqual(scene.meshObjects[0].rotation.y, 45.0f), "Mesh rotation should be stored.");
    t.expect(almostEqual(scene.meshObjects[0].scale.y, 2.0f), "Mesh scale should be stored.");
    t.expect(almostEqual(scene.meshObjects[0].transform[3], 2.0f), "Mesh transform matrix should include translation X.");
    t.expect(almostEqual(scene.meshObjects[0].transform[7], 1.5f), "Mesh transform matrix should include translation Y.");
    t.expect(almostEqual(scene.meshObjects[0].transform[11], -3.0f), "Mesh transform matrix should include translation Z.");
}

void testSelectedMeshTransformInvalidSafe(TestContext& t)
{
    SceneState scene = makeDefaultScene();
    scene.selectedMeshObject = 100;

    const bool moved = setSelectedMeshPosition(scene, make_float3(100.0f, -100.0f, 100.0f));
    const bool scaled = setSelectedMeshScale(scene, make_float3(-1.0f, 100.0f, 0.0f));

    t.expect(moved, "Invalid selected mesh should clamp before position edit.");
    t.expect(scaled, "Invalid selected mesh should clamp before scale edit.");
    t.expect(scene.selectedMeshObject == 0, "Invalid selected mesh transform edit should clamp selected index.");
    t.expect(almostEqual(scene.meshObjects[0].position.x, 50.0f), "Mesh position should clamp to max.");
    t.expect(almostEqual(scene.meshObjects[0].position.y, -10.0f), "Mesh position should clamp to min Y.");
    t.expect(almostEqual(scene.meshObjects[0].scale.x, 0.05f), "Mesh scale should clamp to min.");
    t.expect(almostEqual(scene.meshObjects[0].scale.y, 20.0f), "Mesh scale should clamp to max.");
}

void testInvalidMeshSelectionSafe(TestContext& t)
{
    SceneState scene = makeDefaultScene();
    scene.selectedMeshObject = 100;
    scene.selectedMeshMaterial = 100;

    cycleSelectedMeshMaterialPreset(scene);
    t.expect(scene.selectedMeshObject >= 0, "Invalid mesh object index should clamp safely.");
    t.expect(scene.selectedMeshMaterial >= 0, "Invalid mesh material index should clamp safely.");

    scene.meshObjects.clear();
    selectNextMeshObject(scene);
    cycleSelectedMeshMaterialPreset(scene);
    t.expect(scene.selectedMeshObject == 0, "Empty mesh object list should keep selected mesh at zero.");
}

void testAddSphereSelectsNewSphere(TestContext& t)
{
    SceneState scene = makeDefaultScene();
    const size_t sphereCount = scene.spheres.size();
    const size_t materialCount = scene.materials.size();
    scene.selectedSphere = 0;

    const bool added = addSphere(scene);

    t.expect(added, "Adding a sphere should succeed.");
    t.expect(scene.spheres.size() == sphereCount + 1, "Adding a sphere should append geometry.");
    t.expect(scene.materials.size() == materialCount + 1, "Adding a sphere should append material.");
    t.expect(scene.selectedSphere == static_cast<int>(scene.spheres.size()) - 1, "New sphere should become selected.");
}

void testRemoveSelectedSphereKeepsSceneValid(TestContext& t)
{
    SceneState scene = makeDefaultScene();
    const size_t sphereCount = scene.spheres.size();
    scene.selectedSphere = 1;

    const bool removed = removeSelectedSphere(scene);

    t.expect(removed, "Removing selected sphere should succeed when more than one sphere exists.");
    t.expect(scene.spheres.size() == sphereCount - 1, "Removing a sphere should erase geometry.");
    t.expect(scene.materials.size() == scene.spheres.size(), "Removing a sphere should keep materials aligned.");
    t.expect(scene.selectedSphere >= 0 && scene.selectedSphere < static_cast<int>(scene.spheres.size()), "Selected sphere should remain valid after removal.");
}

void testRemoveLastSphereSafe(TestContext& t)
{
    SceneState scene = makeDefaultScene();
    scene.spheres.resize(1);
    scene.materials.resize(1);
    scene.selectedSphere = 0;

    const bool removed = removeSelectedSphere(scene);

    t.expect(!removed, "Removing the final sphere should fail safely.");
    t.expect(scene.spheres.size() == 1, "Final sphere should stay in the scene.");
    t.expect(scene.materials.size() == 1, "Final sphere material should stay in the scene.");
    t.expect(scene.selectedSphere == 0, "Selected sphere should stay valid when removal is blocked.");
}

void testRemovePenultimateSphereKeepsSelectionValid(TestContext& t)
{
    SceneState scene = makeDefaultScene();
    scene.spheres.resize(2);
    scene.materials.resize(2);
    scene.selectedSphere = 1;

    const bool removed = removeSelectedSphere(scene);

    t.expect(removed, "Removing the penultimate sphere should succeed.");
    t.expect(scene.spheres.size() == 1, "One sphere should remain after removing from two.");
    t.expect(scene.materials.size() == 1, "Material list should match remaining sphere count.");
    t.expect(scene.selectedSphere == 0, "Selection should clamp to the remaining sphere.");
}

void testSelectedSphereRadiusClamp(TestContext& t)
{
    SceneState scene = makeDefaultScene();
    scene.selectedSphere = 0;

    setSelectedSphereRadius(scene, -10.0f);
    t.expect(almostEqual(scene.spheres[0].radius, 0.25f), "Sphere radius should clamp to supported minimum.");
    t.expect(scene.spheres[0].center.y >= scene.spheres[0].radius, "Sphere should stay above floor after radius clamp.");

    setSelectedSphereRadius(scene, 100.0f);
    t.expect(almostEqual(scene.spheres[0].radius, 5.0f), "Sphere radius should clamp to supported maximum.");
}

void testSelectedSphereRadiusKeepsBottomFixed(TestContext& t)
{
    SceneState scene = makeDefaultScene();
    scene.selectedSphere = 0;
    const float bottomBefore = scene.spheres[0].center.y - scene.spheres[0].radius;

    setSelectedSphereRadius(scene, 2.0f);

    const float bottomAfter = scene.spheres[0].center.y - scene.spheres[0].radius;
    t.expect(almostEqual(bottomAfter, bottomBefore), "Changing sphere radius should keep the bottom point fixed.");
    t.expect(almostEqual(scene.spheres[0].center.x, 0.0f), "Changing sphere radius should not move sphere X.");
    t.expect(almostEqual(scene.spheres[0].center.z, 0.0f), "Changing sphere radius should not move sphere Z.");
}

void testSelectedSphereColorClamp(TestContext& t)
{
    SceneState scene = makeDefaultScene();
    scene.selectedSphere = 0;

    setSelectedSphereColor(scene, make_float3(-1.0f, 0.4f, 3.0f));

    t.expect(almostEqual(scene.materials[0].color.x, 0.0f), "Sphere color red channel should clamp to minimum.");
    t.expect(almostEqual(scene.materials[0].color.y, 0.4f), "Sphere color green channel should keep valid value.");
    t.expect(almostEqual(scene.materials[0].color.z, 1.0f), "Sphere color blue channel should clamp to maximum.");
}

void testMoveSphereClamp(TestContext& t)
{
    SceneState scene = makeDefaultScene();
    scene.selectedSphere = 0;
    const float radius = scene.spheres[0].radius;
    moveSelectedSphere(scene, make_float3(200.0f, -200.0f, 200.0f));
    const float3 center = scene.spheres[0].center;

    t.expect(center.x <= 24.0f, "Sphere X should be clamped by scene bounds.");
    t.expect(center.z <= 24.0f, "Sphere Z should be clamped by scene bounds.");
    t.expect(center.y >= radius, "Sphere Y should stay above floor with radius offset.");
}

void testMoveLightClamp(TestContext& t)
{
    SceneState scene = makeDefaultScene();
    moveLight(scene, make_float3(500.0f, -500.0f, -500.0f));
    t.expect(scene.lightPosition.x <= 40.0f, "Light X should be clamped.");
    t.expect(scene.lightPosition.y >= 6.0f, "Light Y should be clamped.");
    t.expect(scene.lightPosition.z >= -40.0f, "Light Z should be clamped.");
}

void testSceneExposureChangesSafely(TestContext& t)
{
    SceneState scene = makeDefaultScene();
    setSceneExposure(scene, 0.8f);
    adjustSceneExposure(scene, 0.1f);
    t.expect(almostEqual(scene.exposure, 0.9f), "Exposure control should increase scene exposure.");
    adjustSceneExposure(scene, -0.2f);
    t.expect(almostEqual(scene.exposure, 0.7f), "Exposure control should decrease scene exposure.");
}

void testSceneSkyIntensityChangesSafely(TestContext& t)
{
    SceneState scene = makeDefaultScene();
    setSceneSkyIntensity(scene, 0.75f);
    adjustSceneSkyIntensity(scene, 0.1f);
    t.expect(almostEqual(scene.skyIntensity, 0.85f), "Sky intensity control should increase sky contribution.");
    adjustSceneSkyIntensity(scene, -0.25f);
    t.expect(almostEqual(scene.skyIntensity, 0.6f), "Sky intensity control should decrease sky contribution.");
}

void testSceneLightIntensityChangesSafely(TestContext& t)
{
    SceneState scene = makeDefaultScene();
    setSceneLightIntensity(scene, 1.0f);
    adjustSceneLightIntensity(scene, 0.25f);
    t.expect(almostEqual(scene.lightIntensity, 1.25f), "Light intensity control should increase light contribution.");
    adjustSceneLightIntensity(scene, -0.5f);
    t.expect(almostEqual(scene.lightIntensity, 0.75f), "Light intensity control should decrease light contribution.");
}

void testSceneTuningInvalidValuesClamp(TestContext& t)
{
    SceneState scene = makeDefaultScene();
    setSceneExposure(scene, -100.0f);
    setSceneSkyIntensity(scene, 100.0f);
    setSceneLightIntensity(scene, -50.0f);
    t.expect(almostEqual(scene.exposure, 0.1f), "Invalid exposure should clamp to supported minimum.");
    t.expect(almostEqual(scene.skyIntensity, 3.0f), "Invalid sky intensity should clamp to supported maximum.");
    t.expect(almostEqual(scene.lightIntensity, 0.0f), "Invalid light intensity should clamp to supported minimum.");
}

void testClampSceneSelectedSphereBounds(TestContext& t)
{
    SceneState scene = makeDefaultScene();
    scene.selectedSphere = -100;
    clampScene(scene);
    t.expect(scene.selectedSphere == 0, "Selected sphere must clamp to lower bound.");

    scene.selectedSphere = 999;
    clampScene(scene);
    t.expect(scene.selectedSphere == static_cast<int>(scene.spheres.size()) - 1, "Selected sphere must clamp to upper bound.");
}

void testInvalidSelectedSphereOps(TestContext& t)
{
    SceneState scene = makeDefaultScene();
    const float3 centerBefore = scene.spheres[0].center;
    const int materialBefore = scene.materials[0].materialType;

    scene.selectedSphere = -1;
    moveSelectedSphere(scene, make_float3(1.0f, 1.0f, 1.0f));
    toggleSelectedMaterial(scene);
    t.expect(almostEqual(scene.spheres[0].center.x, centerBefore.x), "Invalid selected sphere must not move geometry.");
    t.expect(scene.materials[0].materialType == materialBefore, "Invalid selected sphere must not toggle material.");

    scene.selectedSphere = 100;
    moveSelectedSphere(scene, make_float3(1.0f, 1.0f, 1.0f));
    toggleSelectedMaterial(scene);
    t.expect(almostEqual(scene.spheres[0].center.x, centerBefore.x), "Out-of-range sphere index must not move geometry.");
    t.expect(scene.materials[0].materialType == materialBefore, "Out-of-range sphere index must not toggle material.");
}

void testAllSpheresStayAboveFloorAfterClamp(TestContext& t)
{
    SceneState scene = makeDefaultScene();
    for (size_t i = 0; i < scene.spheres.size(); ++i)
    {
        scene.spheres[i].center = make_float3(0.0f, -100.0f, 0.0f);
    }

    clampScene(scene);

    for (size_t i = 0; i < scene.spheres.size(); ++i)
    {
        const SphereGeometry& sphere = scene.spheres[i];
        t.expect(sphere.center.y >= sphere.radius, "Sphere center Y must stay above floor + radius.");
    }
}

void testCameraBasis(TestContext& t)
{
    CameraState camera;
    float3 forward{};
    float3 right{};
    float3 up{};
    float scale = 0.0f;
    float aspect = 0.0f;

    updateCameraBasis(camera, 1280, 720, forward, right, up, scale, aspect);

    t.expect(almostEqual(length3(forward), 1.0f, 1e-3f), "Forward vector must be normalized.");
    t.expect(almostEqual(length3(right), 1.0f, 1e-3f), "Right vector must be normalized.");
    t.expect(almostEqual(length3(up), 1.0f, 1e-3f), "Up vector must be normalized.");

    t.expect(std::fabs(dot3(forward, right)) < 1e-3f, "Forward and right must be orthogonal.");
    t.expect(std::fabs(dot3(forward, up)) < 1e-3f, "Forward and up must be orthogonal.");
    t.expect(std::fabs(dot3(right, up)) < 1e-3f, "Right and up must be orthogonal.");

    t.expect(almostEqual(aspect, 1280.0f / 720.0f, 1e-6f), "Aspect ratio should match viewport.");
    t.expect(scale > 0.0f, "Camera scale must be positive.");
}

void testCameraAspectFallback(TestContext& t)
{
    CameraState camera;
    float3 forward{};
    float3 right{};
    float3 up{};
    float scale = 0.0f;
    float aspect = 0.0f;

    updateCameraBasis(camera, 1920, 0, forward, right, up, scale, aspect);
    t.expect(almostEqual(aspect, 1.0f, 1e-6f), "Aspect ratio should fallback to 1.0 when height is zero.");
    t.expect(scale > 0.0f, "Camera scale must remain valid for zero-height fallback.");
}

void testCameraScaleIncreasesWithFov(TestContext& t)
{
    CameraState cameraNarrow;
    cameraNarrow.fov = 30.0f;
    CameraState cameraWide;
    cameraWide.fov = 90.0f;

    float3 forward{};
    float3 right{};
    float3 up{};
    float scaleNarrow = 0.0f;
    float aspect = 0.0f;
    updateCameraBasis(cameraNarrow, 1280, 720, forward, right, up, scaleNarrow, aspect);

    float scaleWide = 0.0f;
    updateCameraBasis(cameraWide, 1280, 720, forward, right, up, scaleWide, aspect);

    t.expect(scaleWide > scaleNarrow, "Camera scale should increase with larger FOV.");
}

void testToneMappingAndGammaCorrection(TestContext& t)
{
    const float3 mapped = toneMapAndGammaCorrect(make_float3(4.0f, 1.0f, 0.25f));
    t.expect(mapped.x > mapped.y, "Tone mapping should preserve channel ordering.");
    t.expect(mapped.x <= 1.0f && mapped.y <= 1.0f && mapped.z <= 1.0f, "Tone mapping output should stay displayable.");
    t.expect(mapped.z > 0.0f, "Gamma correction should keep positive low-intensity color visible.");

    const float3 clamped = toneMapAndGammaCorrect(make_float3(-1.0f, 0.0f, 0.0f));
    t.expect(almostEqual(clamped.x, 0.0f), "Tone mapping should clamp negative output before gamma.");

    const float3 dimmed = toneMapAndGammaCorrect(make_float3(2.0f, 2.0f, 2.0f), 0.5f);
    const float3 full = toneMapAndGammaCorrect(make_float3(2.0f, 2.0f, 2.0f), 1.0f);
    t.expect(dimmed.x < full.x, "Lower exposure should reduce tone-mapped brightness.");
}

void testDefaultRenderQuality(TestContext& t)
{
    t.expect(clampRenderQuality(RenderQualityHigh) == RenderQualityHigh, "High quality should be valid.");
    t.expect(renderQualityMaxDepth(RenderQualityHigh) > renderQualityMaxDepth(RenderQualityLow), "High quality should increase max depth.");
    t.expect(renderQualitySamplesPerPixel(RenderQualityHigh) > renderQualitySamplesPerPixel(RenderQualityLow), "High quality should increase samples per pixel.");
}

void testRenderQualityCycles(TestContext& t)
{
    int quality = RenderQualityLow;
    quality = nextRenderQuality(quality);
    t.expect(quality == RenderQualityMedium, "Quality should cycle Low to Medium.");
    quality = nextRenderQuality(quality);
    t.expect(quality == RenderQualityHigh, "Quality should cycle Medium to High.");
    quality = nextRenderQuality(quality);
    t.expect(quality == RenderQualityPathTracing, "Quality should cycle High to PathTracing.");
    quality = nextRenderQuality(quality);
    t.expect(quality == RenderQualityLow, "Quality should wrap PathTracing to Low.");
}

void testRenderQualityDepthAndFallback(TestContext& t)
{
    t.expect(clampRenderQuality(-100) == RenderQualityMedium, "Invalid low quality should fall back to Medium.");
    t.expect(clampRenderQuality(100) == RenderQualityMedium, "Invalid high quality should fall back to Medium.");
    t.expect(renderQualityMaxDepth(-100) == renderQualityMaxDepth(RenderQualityMedium), "Invalid quality should use Medium max depth.");
    t.expect(renderQualitySamplesPerPixel(100) == renderQualitySamplesPerPixel(RenderQualityMedium), "Invalid quality should use Medium sample count.");
    t.expect(!renderQualityShadowsEnabled(RenderQualityLow), "Low quality should disable direct shadow rays.");
    t.expect(renderQualityShadowsEnabled(RenderQualityMedium), "Medium quality should enable direct shadow rays.");
    t.expect(renderQualityUsesPathTracing(RenderQualityPathTracing), "PathTracing quality should request progressive rendering.");
    t.expect(renderQualityUsesDenoiser(RenderQualityPathTracing), "PathTracing quality should request denoiser.");
}

bool runGpuSmokeTest(TestContext& t)
{
#if !defined(RAYTRACERRTX_ENABLE_GPU_TESTS)
    (void)t;
    std::cout << "[SKIP] GPU smoke test skipped: RAYTRACERRTX_ENABLE_GPU_TESTS is not enabled.\n";
    return false;
#else
    try
    {
        OptixRenderer renderer;
        renderer.setRenderSize(64, 64);
        renderer.initialize();

        SceneState scene = makeDefaultScene();
        CameraState camera;
        std::vector<uchar4> pixels(64u * 64u);
        float gpuTimeMs = -1.0f;

        renderer.renderFrame(scene, camera, pixels, &gpuTimeMs);
        renderer.destroy();

        bool hasNonZeroPixel = false;
        for (const uchar4 px : pixels)
        {
            if (px.x != 0u || px.y != 0u || px.z != 0u || px.w != 0u)
            {
                hasNonZeroPixel = true;
                break;
            }
        }

        t.expect(hasNonZeroPixel, "GPU smoke: rendered frame must contain non-zero pixels.");
        t.expect(gpuTimeMs >= 0.0f, "GPU smoke: GPU time must be non-negative.");
        return true;
    }
    catch (const std::exception& ex)
    {
        std::cout << "[SKIP] GPU smoke test skipped: " << ex.what() << '\n';
        return false;
    }
#endif
}

bool runProgressiveGpuSmokeTest(TestContext& t)
{
#if !defined(RAYTRACERRTX_ENABLE_GPU_TESTS)
    (void)t;
    std::cout << "[SKIP] Progressive GPU smoke test skipped: RAYTRACERRTX_ENABLE_GPU_TESTS is not enabled.\n";
    return false;
#else
    try
    {
        OptixRenderer renderer;
        renderer.setRenderSize(64, 64);
        renderer.initialize();
        renderer.setRenderMode(RenderModeProgressive);

        SceneState scene = makeDefaultScene();
        CameraState camera;
        std::vector<uchar4> pixels(64u * 64u);
        float gpuTimeMs = -1.0f;

        renderer.renderFrame(scene, camera, pixels, &gpuTimeMs);
        t.expect(renderer.getAccumulationSampleCount() == 1u, "Progressive mode should increment sample count after first frame.");
        renderer.renderFrame(scene, camera, pixels, &gpuTimeMs);
        t.expect(renderer.getAccumulationSampleCount() == 2u, "Progressive mode should keep accumulating stable frames.");

        camera.yaw += 2.0f;
        renderer.renderFrame(scene, camera, pixels, &gpuTimeMs);
        t.expect(renderer.getAccumulationSampleCount() == 1u, "Progressive accumulation should reset on camera change.");

        renderer.renderFrame(scene, camera, pixels, &gpuTimeMs);
        t.expect(renderer.getAccumulationSampleCount() == 2u, "Progressive mode should continue after camera reset.");

        scene.exposure += 0.1f;
        renderer.renderFrame(scene, camera, pixels, &gpuTimeMs);
        t.expect(renderer.getAccumulationSampleCount() == 1u, "Progressive accumulation should reset on exposure change.");

        scene.lightPosition.x += 0.5f;
        renderer.renderFrame(scene, camera, pixels, &gpuTimeMs);
        t.expect(renderer.getAccumulationSampleCount() == 1u, "Progressive accumulation should reset on light change.");

        bool hasNonZeroPixel = false;
        for (const uchar4 px : pixels)
        {
            if (px.x != 0u || px.y != 0u || px.z != 0u || px.w != 0u)
            {
                hasNonZeroPixel = true;
                break;
            }
        }

        t.expect(hasNonZeroPixel, "Progressive GPU smoke: rendered frame must contain non-zero pixels.");
        t.expect(gpuTimeMs >= 0.0f, "Progressive GPU smoke: GPU time must be non-negative.");
        renderer.destroy();
        return true;
    }
    catch (const std::exception& ex)
    {
        std::cout << "[SKIP] Progressive GPU smoke test skipped: " << ex.what() << '\n';
        return false;
    }
#endif
}

bool runDynamicSphereGpuSmokeTest(TestContext& t)
{
#if !defined(RAYTRACERRTX_ENABLE_GPU_TESTS)
    (void)t;
    std::cout << "[SKIP] Dynamic sphere GPU smoke test skipped: RAYTRACERRTX_ENABLE_GPU_TESTS is not enabled.\n";
    return false;
#else
    try
    {
        OptixRenderer renderer;
        renderer.setRenderSize(64, 64);
        renderer.initialize();

        SceneState scene = makeDefaultScene();
        CameraState camera;
        std::vector<uchar4> pixels(64u * 64u);
        float gpuTimeMs = -1.0f;

        renderer.renderFrame(scene, camera, pixels, &gpuTimeMs);
        const bool added = addSphere(scene);
        renderer.renderFrame(scene, camera, pixels, &gpuTimeMs);
        setSelectedSphereRadius(scene, 0.75f);
        setSelectedSphereColor(scene, make_float3(0.95f, 0.25f, 0.25f));
        renderer.renderFrame(scene, camera, pixels, &gpuTimeMs);
        const bool removed = removeSelectedSphere(scene);
        renderer.renderFrame(scene, camera, pixels, &gpuTimeMs);

        bool hasNonZeroPixel = false;
        for (const uchar4 px : pixels)
        {
            if (px.x != 0u || px.y != 0u || px.z != 0u || px.w != 0u)
            {
                hasNonZeroPixel = true;
                break;
            }
        }

        t.expect(added, "Dynamic sphere GPU smoke: addSphere should succeed.");
        t.expect(removed, "Dynamic sphere GPU smoke: removeSelectedSphere should succeed.");
        t.expect(hasNonZeroPixel, "Dynamic sphere GPU smoke: rendered frame after sphere changes must contain non-zero pixels.");
        t.expect(gpuTimeMs >= 0.0f, "Dynamic sphere GPU smoke: GPU time must be non-negative.");
        renderer.destroy();
        return true;
    }
    catch (const std::exception& ex)
    {
        std::cout << "[SKIP] Dynamic sphere GPU smoke test skipped: " << ex.what() << '\n';
        return false;
    }
#endif
}

bool runDenoiserGpuSmokeTest(TestContext& t)
{
#if !defined(RAYTRACERRTX_ENABLE_GPU_TESTS)
    (void)t;
    std::cout << "[SKIP] OptiX denoiser smoke test skipped: RAYTRACERRTX_ENABLE_GPU_TESTS is not enabled.\n";
    return false;
#else
    try
    {
        OptixRenderer renderer;
        renderer.setRenderSize(64, 64);
        renderer.initialize();
        renderer.setRenderMode(RenderModeProgressive);
        renderer.setDenoiserEnabled(false);

        SceneState scene = makeDefaultScene();
        CameraState camera;
        std::vector<uchar4> pixels(64u * 64u);
        float gpuTimeMs = -1.0f;

        renderer.renderFrame(scene, camera, pixels, &gpuTimeMs);
        t.expect(renderer.getAccumulationSampleCount() == 1u, "Denoiser disabled path should still accumulate samples.");

        renderer.setDenoiserEnabled(true);
        t.expect(renderer.isDenoiserEnabled(), "Denoiser request flag should be stored.");
        renderer.renderFrame(scene, camera, pixels, &gpuTimeMs);
        t.expect(renderer.getAccumulationSampleCount() == 1u, "Enabling denoiser should reset progressive accumulation.");

        bool hasNonZeroPixel = false;
        for (const uchar4 px : pixels)
        {
            if (px.x != 0u || px.y != 0u || px.z != 0u || px.w != 0u)
            {
                hasNonZeroPixel = true;
                break;
            }
        }
        t.expect(hasNonZeroPixel, "Denoiser enabled or graceful fallback should render non-zero pixels.");
        t.expect(gpuTimeMs >= 0.0f, "Denoiser smoke: GPU time must be non-negative.");

        renderer.destroy();
        return true;
    }
    catch (const std::exception& ex)
    {
        std::cout << "[SKIP] OptiX denoiser smoke test skipped: " << ex.what() << '\n';
        return false;
    }
#endif
}

int runGpuBenchmark()
{
#if !defined(RAYTRACERRTX_ENABLE_GPU_TESTS)
    std::cout << "GPU benchmark is not available: RAYTRACERRTX_ENABLE_GPU_TESTS is not enabled.\n";
    return 1;
#else
    struct Scenario
    {
        const char* name;
        int width;
        int height;
        int frames;
    };

    const Scenario scenarios[] = {
        {"Low", 640, 360, 30},
        {"HD", 1280, 720, 30},
        {"Full HD", 1920, 1080, 20},
    };

    std::cout << "| Scenario | Resolution | FPS | Avg frame ms | Avg GPU ms |\n";
    std::cout << "|---|---:|---:|---:|---:|\n";

    const auto runBenchmarkRow = [](const char* name, const int width, const int height, const int frames, const int quality, const bool denoiser)
    {
        OptixRenderer renderer;
        renderer.setRenderSize(width, height);
        renderer.initialize();
        renderer.setRenderQuality(quality);
        renderer.setRenderMode(renderQualityUsesPathTracing(quality) ? RenderModeProgressive : RenderModeRealtime);
        renderer.setDenoiserEnabled(denoiser);

        SceneState scene = makeDefaultScene();
        CameraState camera;
        std::vector<uchar4> pixels(static_cast<size_t>(width) * static_cast<size_t>(height));

        float warmupGpuTimeMs = 0.0f;
        renderer.renderFrame(scene, camera, pixels, &warmupGpuTimeMs);

        double hostTotalMs = 0.0;
        double gpuTotalMs = 0.0;
        for (int frame = 0; frame < frames; ++frame)
        {
            float gpuTimeMs = 0.0f;
            const auto start = std::chrono::steady_clock::now();
            renderer.renderFrame(scene, camera, pixels, &gpuTimeMs);
            const auto stop = std::chrono::steady_clock::now();

            hostTotalMs += std::chrono::duration<double, std::milli>(stop - start).count();
            gpuTotalMs += static_cast<double>(gpuTimeMs);
        }

        renderer.destroy();

        const double avgHostMs = hostTotalMs / static_cast<double>(frames);
        const double avgGpuMs = gpuTotalMs / static_cast<double>(frames);
        const double fps = avgHostMs > 0.0 ? 1000.0 / avgHostMs : 0.0;

        std::cout << "| " << name
                  << " | " << width << "x" << height
                  << " | " << std::fixed << std::setprecision(2) << fps
                  << " | " << avgHostMs
                  << " | " << avgGpuMs
                  << " |\n";
    };

    for (const Scenario& scenario : scenarios)
    {
        std::string name = std::string(scenario.name) + " resolution / High quality";
        runBenchmarkRow(name.c_str(), scenario.width, scenario.height, scenario.frames, RenderQualityHigh, false);
    }

    runBenchmarkRow("Low quality", 640, 360, 30, RenderQualityLow, false);
    runBenchmarkRow("Medium quality", 640, 360, 30, RenderQualityMedium, false);
    runBenchmarkRow("High quality", 640, 360, 30, RenderQualityHigh, false);
    runBenchmarkRow("PathTracing quality", 640, 360, 12, RenderQualityPathTracing, false);
    runBenchmarkRow("PathTracing quality + denoiser", 640, 360, 12, RenderQualityPathTracing, true);

    return 0;
#endif
}
} // namespace

int main(int argc, char** argv)
{
    if (argc > 1 && std::string(argv[1]) == "--benchmark")
    {
        return runGpuBenchmark();
    }

    TestContext t;
    int testsRun = 0;
    int testsFailed = 0;
    int testsSkipped = 0;

    const auto runTest = [&](const char* name, void (*fn)(TestContext&))
    {
        const int failuresBefore = t.failures;
        const int checksBefore = t.checks;
        ++testsRun;
        fn(t);
        const int checksDelta = t.checks - checksBefore;
        if (t.failures == failuresBefore)
        {
            std::cout << "[PASS] " << name << " (checks: " << checksDelta << ")\n";
        }
        else
        {
            ++testsFailed;
            std::cout << "[FAIL] " << name << " (new failures: " << (t.failures - failuresBefore) << ")\n";
        }
    };

    runTest("Logger writes warnings", testLoggerWritesWarnings);
    runTest("Default scene", testDefaultScene);
    runTest("Toggle material", testToggleMaterial);
    runTest("Sphere material preset cycle", testSphereMaterialPresetCycle);
    runTest("Set selected sphere material type", testSetSelectedSphereMaterialType);
    runTest("Set selected sphere material type invalid index", testSetSelectedSphereMaterialTypeInvalidIndexSafe);
    runTest("Selected mesh changes", testSelectedMeshChanges);
    runTest("Selected mesh material preset cycle", testSelectedMeshMaterialPresetCycle);
    runTest("Set selected mesh material type", testSetSelectedMeshMaterialType);
    runTest("Set selected mesh material type invalid safe", testSetSelectedMeshMaterialTypeInvalidSafe);
    runTest("Selected mesh transform controls", testSelectedMeshTransformControls);
    runTest("Selected mesh transform invalid safe", testSelectedMeshTransformInvalidSafe);
    runTest("Invalid mesh selection safe", testInvalidMeshSelectionSafe);
    runTest("Add sphere selects new sphere", testAddSphereSelectsNewSphere);
    runTest("Remove selected sphere keeps scene valid", testRemoveSelectedSphereKeepsSceneValid);
    runTest("Remove last sphere safe", testRemoveLastSphereSafe);
    runTest("Remove penultimate sphere keeps selection valid", testRemovePenultimateSphereKeepsSelectionValid);
    runTest("Selected sphere radius clamp", testSelectedSphereRadiusClamp);
    runTest("Selected sphere radius keeps bottom fixed", testSelectedSphereRadiusKeepsBottomFixed);
    runTest("Selected sphere color clamp", testSelectedSphereColorClamp);
    runTest("Move sphere clamp", testMoveSphereClamp);
    runTest("Move light clamp", testMoveLightClamp);
    runTest("Scene exposure changes safely", testSceneExposureChangesSafely);
    runTest("Scene sky intensity changes safely", testSceneSkyIntensityChangesSafely);
    runTest("Scene light intensity changes safely", testSceneLightIntensityChangesSafely);
    runTest("Scene tuning invalid values clamp", testSceneTuningInvalidValuesClamp);
    runTest("Clamp selected sphere index", testClampSceneSelectedSphereBounds);
    runTest("Invalid selected sphere operations", testInvalidSelectedSphereOps);
    runTest("All spheres above floor after clamp", testAllSpheresStayAboveFloorAfterClamp);
    runTest("Camera basis", testCameraBasis);
    runTest("Camera aspect fallback", testCameraAspectFallback);
    runTest("Camera scale vs FOV", testCameraScaleIncreasesWithFov);
    runTest("Tone mapping and gamma correction", testToneMappingAndGammaCorrection);
    runTest("Default render quality", testDefaultRenderQuality);
    runTest("Render quality cycles", testRenderQualityCycles);
    runTest("Render quality depth and fallback", testRenderQualityDepthAndFallback);
    runTest("GGX math helpers", testGgxMathHelpers);
    runTest("OBJ loader triangle with normals", testObjLoaderTriangleWithNormals);
    runTest("OBJ loader multiple materials", testObjLoaderMultipleMaterials);
    runTest("OBJ loader mirror material name mapping", testObjLoaderMirrorMaterialNameMapping);
    runTest("OBJ loader extended MTL parameters", testObjLoaderExtendedMtlParameters);
    runTest("OBJ loader dielectric material name mapping", testObjLoaderDielectricMaterialNameMapping);
    runTest("OBJ loader material parameter clamping", testObjLoaderMaterialParameterClamping);
    runTest("OBJ loader texture coordinates", testObjLoaderTextureCoordinates);
    runTest("OBJ loader textured face with normals", testObjLoaderTexturedFaceWithNormals);
    runTest("OBJ loader computes tangents for textured triangle", testObjLoaderComputesTangentsForTexturedTriangle);
    runTest("OBJ loader map_Kd texture", testObjLoaderMapKdTexture);
    runTest("Image loader PNG texture", testImageLoaderPngTexture);
    runTest("Image loader invalid PPM fallback", testImageLoaderInvalidPpmFallback);
    runTest("OBJ loader common texture maps", testObjLoaderCommonTextureMaps);
    runTest("OBJ loader normal map texture", testObjLoaderNormalMapTexture);
    runTest("OBJ loader missing texture fallback", testObjLoaderMissingTextureFallback);
    runTest("OBJ loader unsupported texture fallback", testObjLoaderUnsupportedTextureFallback);
    runTest("OBJ loader missing UV disables normal map", testObjLoaderMissingUvDisablesNormalMap);
    runTest("OBJ loader invalid normal map fallback", testObjLoaderInvalidNormalMapFallback);
    runTest("OBJ loader missing normals fallback", testObjLoaderMissingNormalsFallback);
    runTest("OBJ loader empty file", testObjLoaderEmptyFile);
    runTest("OBJ loader invalid face", testObjLoaderInvalidFace);
    runTest("OBJ loader missing file", testObjLoaderMissingFile);
    runTest("OBJ loader unknown lines ignored", testObjLoaderUnknownLinesIgnored);
    runTest("OBJ loader material fallback when MTL missing", testObjLoaderMaterialFallbackWhenMtlMissing);
    runTest("OBJ loader invalid MTL values fallback", testObjLoaderInvalidMtlValuesFallback);
    runTest("OBJ loader single triangle boundary", testObjLoaderSingleTriangleBoundary);
    runTest("glTF loader minimal mesh", testGltfLoaderMinimalMesh);
    runTest("glTF loader missing file", testGltfLoaderMissingFile);
    runTest("glTF loader missing buffer", testGltfLoaderMissingBuffer);
    runTest("glTF loader material factors", testGltfLoaderMaterialFactors);
    runTest("glTF loader minimal GLB", testGltfLoaderMinimalGlb);
    runTest("glTF loader texture maps", testGltfLoaderTextureMaps);
    runTest("glTF loader node hierarchy transform", testGltfLoaderNodeHierarchyTransform);
    runTest("Demo glTF asset loads", testDemoGltfAssetLoads);
    runTest("OBJ loader still loads after glTF support", testObjStillLoadsAfterGltfSupport);
    runTest("Demo OBJ asset loads", testDemoObjAssetLoads);
    runTest("Textured cube asset loads", testTexturedCubeAssetLoads);
    runTest("Default scene mesh geometry", testDefaultSceneMeshGeometry);
    runTest("Built-in cube mesh", testBuiltInCubeMesh);
    runTest("Built-in pyramid mesh", testBuiltInPyramidMesh);
    runTest("Built-in plane mesh", testBuiltInPlaneMesh);
    runTest("Scene config loads valid scene", testSceneConfigLoadsValidScene);
    runTest("Scene config with multiple meshes loads", testSceneConfigMultipleMeshes);
    runTest("Scene config missing file", testSceneConfigMissingFile);
    runTest("Scene config mesh path applied", testSceneConfigMeshPathApplied);
    runTest("Scene config tuning fields", testSceneConfigTuningFields);
    runTest("Scene config tuning clamps", testSceneConfigTuningClamps);
    runTest("Scene config area light and environment", testSceneConfigAreaLightAndEnvironment);
    runTest("Scene config area environment fallbacks", testSceneConfigAreaEnvironmentFallbacks);
    runTest("Scene config JSON material parses", testSceneConfigJsonMaterialParses);
    runTest("Scene config material assigned to sphere", testSceneConfigMaterialAssignedToSphere);
    runTest("Scene config material assigned to mesh", testSceneConfigMaterialAssignedToMesh);
    runTest("Scene config invalid material type fallback", testSceneConfigInvalidMaterialTypeFallback);
    runTest("Scene config invalid JSON fails cleanly", testSceneConfigInvalidJsonFailsCleanly);
    runTest("Scene config invalid material config fails cleanly", testSceneConfigInvalidMaterialConfigFailsCleanly);
    runTest("Scene config invalid transform fails cleanly", testSceneConfigInvalidTransformFailsCleanly);
    runTest("Scene config material parameter clamps", testSceneConfigMaterialParameterClamps);
    runTest("Scene config old scene still loads", testSceneConfigOldSceneStillLoads);
    runTest("All demo scene configs load", testAllDemoSceneConfigsLoad);
    runTest("Scene preset reset camera and light", testScenePresetResetCameraLight);
    runTest("Invalid scene preset index safe", testInvalidScenePresetIndexSafe);
    runTest("Scene preset session edits persist", testScenePresetSessionEditsPersist);
    runTest("Scene preset save invalid index safe", testScenePresetSaveInvalidIndexSafe);
    runTest("Scene config fallback demo scene", testSceneConfigFallbackDemoScene);
    runTest("Asset cache reuses mesh path", testAssetCacheReusesMeshPath);
    runTest("Asset cache reuses texture path", testAssetCacheReusesTexturePath);
    runTest("Scene reload valid updates scene", testSceneReloadValidUpdatesScene);
    runTest("Scene reload invalid keeps previous scene", testSceneReloadInvalidKeepsPreviousScene);

    ++testsRun;
    if (runGpuSmokeTest(t))
    {
        std::cout << "[PASS] GPU smoke test (checks: 2)\n";
    }
    else
    {
        ++testsSkipped;
    }
    ++testsRun;
    if (runProgressiveGpuSmokeTest(t))
    {
        std::cout << "[PASS] Progressive GPU smoke test (checks: 8)\n";
    }
    else
    {
        ++testsSkipped;
    }
    ++testsRun;
    if (runDynamicSphereGpuSmokeTest(t))
    {
        std::cout << "[PASS] Dynamic sphere GPU smoke test (checks: 4)\n";
    }
    else
    {
        ++testsSkipped;
    }
    ++testsRun;
    if (runDenoiserGpuSmokeTest(t))
    {
        std::cout << "[PASS] OptiX denoiser smoke test (checks: 5)\n";
    }
    else
    {
        ++testsSkipped;
    }

    if (t.failures == 0)
    {
        std::cout << "\nAll tests passed. Tests: " << testsRun
                  << ", skipped: " << testsSkipped
                  << ", checks: " << t.checks << '\n';
        return 0;
    }

    std::cerr << "\nTests failed. Failed tests: " << testsFailed
              << ", failed checks: " << t.failures
              << " / total checks: " << t.checks << '\n';
    return 1;
}
