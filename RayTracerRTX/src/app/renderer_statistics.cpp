#include "renderer_statistics.h"

RendererStatistics computeRendererStatistics(const SceneState& scene, const unsigned int accumulationSamples)
{
    RendererStatistics statistics;
    statistics.sphereCount = scene.spheres.size();
    statistics.meshObjectCount = scene.meshObjects.size();
    statistics.materialCount = scene.materials.size();
    statistics.accumulationSamples = accumulationSamples;

    for (const MeshObject& object : scene.meshObjects)
    {
        statistics.triangleCount += object.mesh.triangles.size();
        statistics.materialCount += object.mesh.materials.size();
    }

    return statistics;
}

