#pragma once

#include "../common/rtx_shared.h"

struct CameraState
{
    float3 position = make_float3(0.0f, 8.0f, -15.0f);
    float yaw = 90.0f;
    float pitch = -25.0f;
    float fov = 45.0f;
};

struct InputState
{
    double lastX = 400.0;
    double lastY = 300.0;
    bool firstMouse = true;
};

void updateCameraBasis(
    const CameraState& camera,
    unsigned int viewportWidth,
    unsigned int viewportHeight,
    float3& forward,
    float3& right,
    float3& up,
    float& scale,
    float& aspect);
