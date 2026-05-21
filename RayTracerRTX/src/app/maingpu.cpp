#include "application.h"

#include <exception>
#include <iostream>
#include <string>

namespace
{
ApplicationOptions parseOptions(int argc, char** argv)
{
    ApplicationOptions options;
    for (int i = 1; i < argc; ++i)
    {
        const std::string arg = argv[i];
        if (arg == "--mesh" && i + 1 < argc)
        {
            options.meshPath = argv[++i];
        }
        else if (arg == "--scene" && i + 1 < argc)
        {
            options.sceneConfigPath = argv[++i];
        }
        else if (arg == "--help" || arg == "-h")
        {
            options.showHelp = true;
        }
        else
        {
            std::cerr << "Unknown or incomplete argument: " << arg << '\n';
        }
    }
    return options;
}
} // namespace

int main(int argc, char** argv)
{
    try
    {
        const ApplicationOptions options = parseOptions(argc, argv);
        if (options.showHelp)
        {
            std::cout << "Usage: RayTracerRTX.exe [--mesh path/to/model.obj] [--scene path/to/scene.json]\n";
            return 0;
        }
        run_optix_app(options);
        return 0;
    }
    catch (const std::exception& ex)
    {
        std::cerr << ex.what() << '\n';
        return 1;
    }
}
