#pragma once

#include <filesystem>

struct ApplicationOptions
{
    std::filesystem::path meshPath;
    std::filesystem::path sceneConfigPath;
    bool showHelp = false;
};

void run_optix_app(const ApplicationOptions& options = {});
