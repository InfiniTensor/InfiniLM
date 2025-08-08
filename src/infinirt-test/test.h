#ifndef __INFINIRT_TEST_H__
#define __INFINIRT_TEST_H__
#include "../utils.h"

bool testSetDevice(infiniDevice_t device, int deviceId);
bool testMemcpy(infiniDevice_t device, int deviceId, size_t dataSize);

#endif
