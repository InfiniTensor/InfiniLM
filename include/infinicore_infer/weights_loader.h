#ifndef WEIGHTS_LOADER_H
#define WEIGHTS_LOADER_H

#include <infinirt.h>

#ifndef __INFINI_C
// Compat: older InfiniCore headers use `__C` instead of `__INFINI_C`.
#define __INFINI_C __C
#endif

struct ModelWeights;

__INFINI_C __export void
loadModelWeight(struct ModelWeights *weights, const char *name, void *data);

__INFINI_C __export void
loadModelWeightDistributed(struct ModelWeights *weights, const char *name, void *data, int *ranks, int nrank);

#endif // WEIGHTS_LOADER_H
