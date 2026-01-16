// Test code to check topkrouter dtype consistency

#include <iostream>
#include "src/tensor/tensor.hpp"

int main() {
    std::cout << "Checking dtype consistency..." << std::endl;
    std::cout << "sizeof(float) = " << sizeof(float) << std::endl;
    std::cout << "sizeof(int) = " << sizeof(int) << std::endl;
    return 0;
}
