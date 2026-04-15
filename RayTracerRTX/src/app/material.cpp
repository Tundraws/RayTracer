#include "material.h"

const char* materialName(int materialType)
{
    return materialType == MaterialMirror ? "mirror" : "diffuse";
}

const wchar_t* materialNameW(int materialType)
{
    return materialType == MaterialMirror ? L"\u0417\u0415\u0420\u041A\u0410\u041B\u041E" : L"\u041C\u0410\u0422\u041E\u0412\u042B\u0419";
}
