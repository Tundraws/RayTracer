#include "camera.h"

#include <cmath>

namespace
{
constexpr float kPi = 3.14159265358979323846f;

float3 add3(const float3 a, const float3 b)
{
    return make_float3(a.x + b.x, a.y + b.y, a.z + b.z);
}

float3 sub3(const float3 a, const float3 b)
{
    return make_float3(a.x - b.x, a.y - b.y, a.z - b.z);
}

float3 mul3(const float3 a, const float value)
{
    return make_float3(a.x * value, a.y * value, a.z * value);
}

float dot3(const float3 a, const float3 b)
{
    return a.x * b.x + a.y * b.y + a.z * b.z;
}

float3 cross3(const float3 a, const float3 b)
{
    return make_float3(
        a.y * b.z - a.z * b.y,
        a.z * b.x - a.x * b.z,
        a.x * b.y - a.y * b.x);
}

float3 normalize3(const float3 v)
{
    const float length = std::sqrt(dot3(v, v));
    if (length <= 0.0f)
    {
        return make_float3(0.0f, 0.0f, 0.0f);
    }
    return mul3(v, 1.0f / length);
}
} // namespace

void updateCameraBasis(
    const CameraState& camera,
    unsigned int viewportWidth,
    unsigned int viewportHeight,
    float3& forward,
    float3& right,
    float3& up,
    float& scale,
    float& aspect)
{
    const float yaw = camera.yaw * kPi / 180.0f;
    const float pitch = camera.pitch * kPi / 180.0f;

    forward = normalize3(make_float3(
        std::cos(yaw) * std::cos(pitch),
        std::sin(pitch),
        std::sin(yaw) * std::cos(pitch)));

    const float3 worldUp = make_float3(0.0f, 1.0f, 0.0f);
    right = normalize3(cross3(forward, worldUp));
    up = normalize3(cross3(right, forward));
    aspect = viewportHeight == 0u ? 1.0f : static_cast<float>(viewportWidth) / static_cast<float>(viewportHeight);
    scale = std::tan((camera.fov * 0.5f) * kPi / 180.0f);
}
