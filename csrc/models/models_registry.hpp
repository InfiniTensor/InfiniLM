#pragma once

#include "infinicore/device.hpp"
#include "infinilm_model.hpp"
#include <functional>
#include <map>
#include <memory>
#include <string>

namespace infinilm::models {

using ModelCreator = std::function<std::shared_ptr<InfinilmModel>(std::shared_ptr<config::ModelConfig>,
                                                                  const infinicore::Device &)>;

using ConfigCreator = std::function<std::shared_ptr<config::ModelConfig>(std::shared_ptr<config::ModelConfig>)>;

/// Register one causal LM model.
void register_causal_lm_model(const std::string &model_type, ModelCreator creator);

/// Register one model config post-processor.
void register_model_config(const std::string &model_type, ConfigCreator creator);

/// Map: model_type -> causal LM constructor.
const std::map<std::string, ModelCreator> &get_causal_lm_model_map();

/// Map: model_type -> config post-processor.
const std::map<std::string, ConfigCreator> &get_model_config_map();

} // namespace infinilm::models
