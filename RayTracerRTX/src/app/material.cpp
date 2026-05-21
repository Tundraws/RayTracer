#include "material.h"

const char* materialName(int materialType)
{
    switch (materialType)
    {
    case MaterialMirror:
        return "mirror";
    case MaterialMetal:
        return "metal";
    case MaterialDielectric:
        return "dielectric";
    default:
        return "diffuse";
    }
}

const wchar_t* materialNameW(int materialType)
{
    switch (materialType)
    {
    case MaterialMirror:
        return L"\u0417\u0415\u0420\u041A\u0410\u041B\u041E";
    case MaterialMetal:
        return L"\u041C\u0415\u0422\u0410\u041B\u041B";
    case MaterialDielectric:
        return L"\u0421\u0422\u0415\u041A\u041B\u041E";
    default:
        return L"\u041C\u0410\u0422\u041E\u0412\u042B\u0419";
    }
}
