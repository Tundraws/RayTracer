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

struct float4
{
    float x;
    float y;
    float z;
    float w;
};

struct uchar4
{
    unsigned char x;
    unsigned char y;
    unsigned char z;
    unsigned char w;
};

struct uint3
{
    unsigned int x;
    unsigned int y;
    unsigned int z;
};

inline float2 make_float2(float x, float y)
{
    return float2{x, y};
}

inline float3 make_float3(float x, float y, float z)
{
    return float3{x, y, z};
}

inline float4 make_float4(float x, float y, float z, float w)
{
    return float4{x, y, z, w};
}

inline uchar4 make_uchar4(unsigned char x, unsigned char y, unsigned char z, unsigned char w)
{
    return uchar4{x, y, z, w};
}

inline uint3 make_uint3(unsigned int x, unsigned int y, unsigned int z)
{
    return uint3{x, y, z};
}
