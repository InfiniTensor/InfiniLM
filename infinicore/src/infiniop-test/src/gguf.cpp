#include "gguf.hpp"
#include <cstring>
#include <fstream>
#include <iostream>
#include <sstream>
#include <stdexcept>

#ifdef _WIN32
#include <windows.h>
#else
#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>

#endif

std::string GGUFKeyValue::toString() const {
    std::ostringstream oss;
    oss << "Key: " << key << ", Type: " << GGUF_TYPE_NAME[gguf_type] << ", Value: ";
    if (gguf_type == GGUF_TYPE_STRING) {
        std::string str(value.begin(), value.end());
        oss << str;
    } else if (value.size() > GGUF_TYPE_SIZE[gguf_type]) {
        oss << "[";
        for (size_t i = 0; i < value.size() / GGUF_TYPE_SIZE[gguf_type]; ++i) {
            oss << ggufDataToString(value.data() + i * GGUF_TYPE_SIZE[gguf_type], gguf_type);
            if (i < value.size() / GGUF_TYPE_SIZE[gguf_type] - 1) {
                oss << ", ";
            }
        }
        oss << "]";
    } else {
        oss << ggufDataToString(value.data(), gguf_type);
    }

    return oss.str();
}

std::string GGUFTensorInfo::toString() const {
    std::ostringstream oss;
    oss << "Name: " << name << ", NDims: " << ndim << ", Shape: [";
    for (size_t i = 0; i < shape.size(); ++i) {
        oss << shape[i];
        if (i < shape.size() - 1) {
            oss << ", ";
        }
    }
    oss << "], DataType: " << GGML_TYPE_NAME[ggml_type] << ", DataOffset: " << data_offset;
    return oss.str();
}

GGUFFileReader::GGUFFileReader(const std::string &filepath) {
    try {
        _file = std::make_shared<FileMapping>(filepath);
    } catch (const std::exception &e) {
        throw e;
    }
    _data = _file->ptr();
    _cursor = reinterpret_cast<uint8_t *>(_data);
    readHeader();
    readMetaKVs();
    readTensorInfos();
    size_t padding = (size_t)(32 - ((char *)_cursor - (char *)_data) % 32) % 32;
    _cursor += padding;
}

const std::unordered_map<std::string, std::shared_ptr<GGUFKeyValue>> &
GGUFFileReader::getAttributeMap() const {
    return _attributes_map;
}

const std::unordered_map<std::string, std::shared_ptr<GGUFTensorInfo>> &
GGUFFileReader::getTensorInfoMap() const {
    return _tensors_info_map;
}

void GGUFFileReader::readHeader() {
    if (std::memcmp(_cursor, "GGUF", 4) != 0) {
        throw std::runtime_error("Invalid GGUF magic");
    }
    _cursor += 4;

    _version = read<uint32_t>();
    _num_tensors = read<int64_t>();
    _num_meta_kvs = read<int64_t>();
    _attributes_map = std::unordered_map<std::string, std::shared_ptr<GGUFKeyValue>>();
    _tensors_info_map = std::unordered_map<std::string, std::shared_ptr<GGUFTensorInfo>>();
}

void GGUFFileReader::readMetaKVs() {
    for (int64_t i = 0; i < _num_meta_kvs; ++i) {
        auto kv = std::make_shared<GGUFKeyValue>();
        kv->key = readString();
        kv->gguf_type = read<GGUF_TYPE>();

        if (kv->gguf_type == GGUF_TYPE_ARRAY) {
            GGUF_TYPE array_type = read<GGUF_TYPE>();
            uint64_t array_size = read<uint64_t>();
            kv->value.resize(array_size * GGUF_TYPE_SIZE[array_type]);
            kv->gguf_type = array_type;
            std::memcpy(kv->value.data(), _cursor, kv->value.size());
            _cursor += kv->value.size();
        } else if (kv->gguf_type == GGUF_TYPE_STRING) {
            uint64_t str_size = read<uint64_t>();
            kv->value.resize(str_size);
            std::memcpy(kv->value.data(), _cursor, str_size);
            _cursor += str_size;
        } else {
            kv->value.resize(GGUF_TYPE_SIZE[kv->gguf_type]);
            std::memcpy(kv->value.data(), _cursor, kv->value.size());
            _cursor += kv->value.size();
        }

        _meta_kvs.push_back(kv);
        _attributes_map.emplace(kv->key, kv);
    }
}

void GGUFFileReader::readTensorInfos() {
    for (int64_t i = 0; i < _num_tensors; ++i) {
        auto tensor_info = std::make_shared<GGUFTensorInfo>();
        tensor_info->name = readString();
        tensor_info->ndim = read<uint32_t>();
        tensor_info->shape.resize(tensor_info->ndim);
        for (size_t j = 0; j < tensor_info->ndim; ++j) {
            tensor_info->shape[j] = read<int64_t>();
        }
        tensor_info->ggml_type = read<GGML_TYPE>();
        tensor_info->data_offset = read<uint64_t>();
        _tensor_infos.push_back(tensor_info);
        _tensors_info_map.emplace(tensor_info->name, tensor_info);
    }
}

std::string GGUFFileReader::readString() {
    uint64_t length = read<uint64_t>();
    std::string str(reinterpret_cast<const char *>(_cursor), length);
    _cursor += length;
    return str;
}

template <typename T>
T GGUFFileReader::read() {
    T value;
    std::memcpy(&value, _cursor, sizeof(T));
    _cursor += sizeof(T);
    return value;
}

std::string GGUFFileReader::toString() const {
    std::ostringstream oss;
    oss << "GGUF File Contents: " << std::endl;
    oss << "Version: " << _version << std::endl;
    oss << "Number of Meta KVs: " << _num_meta_kvs << std::endl;
    oss << "Number of Tensors: " << _num_tensors << std::endl
        << std::endl;
    oss << "Meta KVs: " << std::endl;
    for (const auto &kv : _meta_kvs) {
        oss << kv->toString() << std::endl;
    }
    oss << std::endl;
    oss << "Tensor INFOs: " << std::endl;
    for (const auto &info : _tensor_infos) {
        oss << info->toString() << std::endl;
    }
    return oss.str();
}
