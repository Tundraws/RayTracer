#pragma once

struct float2
{
    float x;
    float y;
};

struct float3
{
    float x;
    float y;
    float z;
};

struct uchar4
{
    unsigned char x;
    unsigned char y;
    unsigned char z;
    unsigned char w;
};

inline float2 make_float2(float x, float y)
{
    return float2{x, y};
}

inline float3 make_float3(float x, float y, float z)
{
    return float3{x, y, z};
}

inline uchar4 make_uchar4(unsigned char x, unsigned char y, unsigned char z, unsigned char w)
{
    return uchar4{x, y, z, w};
}
