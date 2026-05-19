#pragma once

#include <cmath>
#include <iostream>
#include <string>

struct TestContext
{
    int checks = 0;
    int failures = 0;

    void expect(bool condition, const std::string& message)
    {
        ++checks;
        if (!condition)
        {
            ++failures;
            std::cerr << "[FAIL] " << message << '\n';
        }
    }
};

inline bool almostEqual(float a, float b, float eps = 1e-4f)
{
    return std::fabs(a - b) <= eps;
}
