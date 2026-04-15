#include "application.h"

#include <exception>
#include <iostream>

int main()
{
    try
    {
        run_optix_app();
        return 0;
    }
    catch (const std::exception& ex)
    {
        std::cerr << ex.what() << '\n';
        return 1;
    }
}
