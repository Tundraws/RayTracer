#pragma once

#include <filesystem>
#include <string>

enum class LogLevel
{
    Info,
    Warning,
    Error
};

void setLogFilePath(const std::filesystem::path& path);
std::filesystem::path getLogFilePath();
void clearLogFile();
void logMessage(LogLevel level, const std::string& message);
void logInfo(const std::string& message);
void logWarning(const std::string& message);
void logError(const std::string& message);
