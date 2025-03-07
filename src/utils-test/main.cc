#include "utils_test.h"

int main(int argc, char *argv[]) {
    int failed = 0;
    failed += test_rearrange();

    return failed;
}
