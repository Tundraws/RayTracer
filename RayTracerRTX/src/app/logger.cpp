#include "logger.h"

#include <fstream>
#include <iostream>
#include <mutex>

namespace
{
std::filesystem::path gLogFilePath = "RayTracerRTX.log";
std::mutex gLogMutex;

const char* levelName(const LogLevel level)
{
    switch (level)
    {
    case LogLevel::Info:
        return "info";
    case LogLevel::Warning:
        return "warning";
    case LogLevel::Error:
        return "error";
    }
    return "info";
}
}

void setLogFilePath(const std::filesystem::path& path)
{
    std::lock_guard<std::mutex> lock(gLogMutex);
    gLogFilePath = path.empty() ? std::filesystem::path{"RayTracerRTX.log"} : path;
}

std::filesystem::path getLogFilePath()
{
    std::lock_guard<std::mutex> lock(gLogMutex);
    return gLogFilePath;
}

void clearLogFile()
{
    std::lock_guard<std::mutex> lock(gLogMutex);
    std::ofstream file(gLogFilePath, std::ios::trunc);
}

void logMessage(const LogLevel level, const std::string& message)
{
    std::lock_guard<std::mutex> lock(gLogMutex);
    const std::string line = "[" + std::string(levelName(level)) + "] " + message;
    if (level == LogLevel::Info)
    {
        std::cout << line << '\n';
    }
    else
    {
        std::cerr << line << '\n';
    }

    std::ofstream file(gLogFilePath, std::ios::app);
    if (file)
    {
        file << line << '\n';
    }
}

void logInfo(const std::string& message)
{
    logMessage(LogLevel::Info, message);
}

void logWarning(const std::string& message)
{
    logMessage(LogLevel::Warning, message);
}

void logError(const std::string& message)
{
    logMessage(LogLevel::Error, message);
}
