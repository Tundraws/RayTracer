#pragma once

#include "app_state.h"

#include <cstddef>

struct RendererStatistics
{
    std::size_t sphereCount = 0;
    std::size_t meshObjectCount = 0;
    std::size_t triangleCount = 0;
    std::size_t materialCount = 0;
    unsigned int accumulationSamples = 0;
};

RendererStatistics computeRendererStatistics(const SceneState& scene, unsigned int accumulationSamples);

