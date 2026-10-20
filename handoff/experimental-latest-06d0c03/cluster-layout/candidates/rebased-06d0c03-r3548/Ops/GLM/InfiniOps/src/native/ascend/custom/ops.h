/*
 * Copyright (c) 2024 Huawei Technologies Co., Ltd.
 * Copyright (c) 2025 InfiniTensor.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 *
 * Adapted from https://github.com/vllm-project/vllm-ascend/blob/main/csrc/ops.h
 */

#ifndef OPS_H
#define OPS_H

namespace ascend::detail {

at::Tensor RmsNorm(const at::Tensor& input, const at::Tensor& weight,
                   double eps);

}  // namespace ascend::detail

#endif  // OPS_H
