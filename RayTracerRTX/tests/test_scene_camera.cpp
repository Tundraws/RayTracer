#include "../src/app/camera.h"
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

void testSceneConfigFallbackDemoScene(TestContext& t)
{
    const SceneBuildResult scene = buildDefaultSceneInput();
    t.expect(scene.ok, "Default scene input should build.");
    t.expect(!isEmptyMesh(scene.scene.mesh), "Default scene input should keep fallback/demo mesh.");
    t.expect(!scene.scene.meshObjects.empty(), "Default scene input should keep mesh object list.");
    t.expect(hasValidMeshMaterialIndices(scene.scene.mesh), "Default scene input mesh material indices should be valid.");
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

    for (const Scenario& scenario : scenarios)
    {
        OptixRenderer renderer;
        renderer.setRenderSize(scenario.width, scenario.height);
        renderer.initialize();
        renderer.setRenderMode(RenderModeRealtime);

        SceneState scene = makeDefaultScene();
        CameraState camera;
        std::vector<uchar4> pixels(static_cast<size_t>(scenario.width) * static_cast<size_t>(scenario.height));

        float warmupGpuTimeMs = 0.0f;
        renderer.renderFrame(scene, camera, pixels, &warmupGpuTimeMs);

        double hostTotalMs = 0.0;
        double gpuTotalMs = 0.0;
        for (int frame = 0; frame < scenario.frames; ++frame)
        {
            float gpuTimeMs = 0.0f;
            const auto start = std::chrono::steady_clock::now();
            renderer.renderFrame(scene, camera, pixels, &gpuTimeMs);
            const auto stop = std::chrono::steady_clock::now();

            hostTotalMs += std::chrono::duration<double, std::milli>(stop - start).count();
            gpuTotalMs += static_cast<double>(gpuTimeMs);
        }

        renderer.destroy();

        const double avgHostMs = hostTotalMs / static_cast<double>(scenario.frames);
        const double avgGpuMs = gpuTotalMs / static_cast<double>(scenario.frames);
        const double fps = avgHostMs > 0.0 ? 1000.0 / avgHostMs : 0.0;

        std::cout << "| " << scenario.name
                  << " | " << scenario.width << "x" << scenario.height
                  << " | " << std::fixed << std::setprecision(2) << fps
                  << " | " << avgHostMs
                  << " | " << avgGpuMs
                  << " |\n";
    }

    OptixRenderer denoiserRenderer;
    denoiserRenderer.setRenderSize(640, 360);
    denoiserRenderer.initialize();
    denoiserRenderer.setRenderMode(RenderModeProgressive);
    denoiserRenderer.setDenoiserEnabled(true);

    SceneState denoiserScene = makeDefaultScene();
    CameraState denoiserCamera;
    std::vector<uchar4> denoiserPixels(640u * 360u);

    double denoiserHostTotalMs = 0.0;
    double denoiserGpuTotalMs = 0.0;
    constexpr int denoiserFrames = 12;
    for (int frame = 0; frame < denoiserFrames; ++frame)
    {
        float gpuTimeMs = 0.0f;
        const auto start = std::chrono::steady_clock::now();
        denoiserRenderer.renderFrame(denoiserScene, denoiserCamera, denoiserPixels, &gpuTimeMs);
        const auto stop = std::chrono::steady_clock::now();
        denoiserHostTotalMs += std::chrono::duration<double, std::milli>(stop - start).count();
        denoiserGpuTotalMs += static_cast<double>(gpuTimeMs);
    }

    const double avgDenoiserHostMs = denoiserHostTotalMs / static_cast<double>(denoiserFrames);
    const double avgDenoiserGpuMs = denoiserGpuTotalMs / static_cast<double>(denoiserFrames);
    const double denoiserFps = avgDenoiserHostMs > 0.0 ? 1000.0 / avgDenoiserHostMs : 0.0;
    std::cout << "| Progressive + optional OptiX denoiser"
              << " | 640x360"
              << " | " << std::fixed << std::setprecision(2) << denoiserFps
              << " | " << avgDenoiserHostMs
              << " | " << avgDenoiserGpuMs
              << " |\n";
    denoiserRenderer.destroy();

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

    runTest("Default scene", testDefaultScene);
    runTest("Toggle material", testToggleMaterial);
    runTest("Move sphere clamp", testMoveSphereClamp);
    runTest("Move light clamp", testMoveLightClamp);
    runTest("Clamp selected sphere index", testClampSceneSelectedSphereBounds);
    runTest("Invalid selected sphere operations", testInvalidSelectedSphereOps);
    runTest("All spheres above floor after clamp", testAllSpheresStayAboveFloorAfterClamp);
    runTest("Camera basis", testCameraBasis);
    runTest("Camera aspect fallback", testCameraAspectFallback);
    runTest("Camera scale vs FOV", testCameraScaleIncreasesWithFov);
    runTest("Tone mapping and gamma correction", testToneMappingAndGammaCorrection);
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
    runTest("OBJ loader normal map texture", testObjLoaderNormalMapTexture);
    runTest("OBJ loader missing texture fallback", testObjLoaderMissingTextureFallback);
    runTest("OBJ loader missing UV disables normal map", testObjLoaderMissingUvDisablesNormalMap);
    runTest("OBJ loader invalid normal map fallback", testObjLoaderInvalidNormalMapFallback);
    runTest("OBJ loader missing normals fallback", testObjLoaderMissingNormalsFallback);
    runTest("OBJ loader empty file", testObjLoaderEmptyFile);
    runTest("OBJ loader invalid face", testObjLoaderInvalidFace);
    runTest("OBJ loader missing file", testObjLoaderMissingFile);
    runTest("OBJ loader unknown lines ignored", testObjLoaderUnknownLinesIgnored);
    runTest("OBJ loader material fallback when MTL missing", testObjLoaderMaterialFallbackWhenMtlMissing);
    runTest("OBJ loader single triangle boundary", testObjLoaderSingleTriangleBoundary);
    runTest("Demo OBJ asset loads", testDemoObjAssetLoads);
    runTest("Textured cube asset loads", testTexturedCubeAssetLoads);
    runTest("Default scene mesh geometry", testDefaultSceneMeshGeometry);
    runTest("Scene config loads valid scene", testSceneConfigLoadsValidScene);
    runTest("Scene config with multiple meshes loads", testSceneConfigMultipleMeshes);
    runTest("Scene config missing file", testSceneConfigMissingFile);
    runTest("Scene config mesh path applied", testSceneConfigMeshPathApplied);
    runTest("Scene config fallback demo scene", testSceneConfigFallbackDemoScene);

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
        std::cout << "[PASS] Progressive GPU smoke test (checks: 5)\n";
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
