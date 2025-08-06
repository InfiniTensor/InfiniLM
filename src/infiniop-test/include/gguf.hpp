#ifndef __INFINIOPTEST_GGUF_HPP__
#define __INFINIOPTEST_GGUF_HPP__
#include "file_mapping.hpp"
#include <cstdint>
#include <memory>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <vector>

#ifdef _WIN32
#include <windows.h>
#endif

typedef enum {
    GGUF_TYPE_UINT8 = 0,
    GGUF_TYPE_INT8 = 1,
    GGUF_TYPE_UINT16 = 2,
    GGUF_TYPE_INT16 = 3,
    GGUF_TYPE_UINT32 = 4,
    GGUF_TYPE_INT32 = 5,
    GGUF_TYPE_FLOAT32 = 6,
    GGUF_TYPE_BOOL = 7,
    GGUF_TYPE_STRING = 8,
    GGUF_TYPE_ARRAY = 9,
    GGUF_TYPE_UINT64 = 10,
    GGUF_TYPE_INT64 = 11,
    GGUF_TYPE_FLOAT64 = 12,
    GGUF_TYPE_COUNT, // marks the end of the enum
} GGUF_TYPE;

constexpr const char *GGUF_TYPE_NAME[GGUF_TYPE_COUNT] = {
    "GGUF_TYPE_UINT8",
    "GGUF_TYPE_INT8",
    "GGUF_TYPE_UINT16",
    "GGUF_TYPE_INT16",
    "GGUF_TYPE_UINT32",
    "GGUF_TYPE_INT32",
    "GGUF_TYPE_FLOAT32",
    "GGUF_TYPE_BOOL",
    "GGUF_TYPE_STRING",
    "GGUF_TYPE_ARRAY",
    "GGUF_TYPE_UINT64",
    "GGUF_TYPE_INT64",
    "GGUF_TYPE_FLOAT64",
};

struct gguf_str {
    uint64_t n;
    char *data;
};

static const size_t GGUF_TYPE_SIZE[GGUF_TYPE_COUNT] = {
    sizeof(uint8_t),  // GGUF_TYPE_UINT8
    sizeof(int8_t),   // GGUF_TYPE_INT8
    sizeof(uint16_t), // GGUF_TYPE_UINT16
    sizeof(int16_t),  // GGUF_TYPE_INT16
    sizeof(uint32_t), // GGUF_TYPE_UINT32
    sizeof(int32_t),  // GGUF_TYPE_INT32
    sizeof(float),    // GGUF_TYPE_FLOAT32
    sizeof(bool),     // GGUF_TYPE_BOOL
    sizeof(gguf_str), // GGUF_TYPE_STRING
    0,                // GGUF_TYPE_ARRAY (undefined)
    sizeof(uint64_t), // GGUF_TYPE_UINT64
    sizeof(int64_t),  // GGUF_TYPE_INT64
    sizeof(double),   // GGUF_TYPE_FLOAT64
};

inline std::string ggufDataToString(const uint8_t *data, GGUF_TYPE gguf_type) {
    switch (gguf_type) {

#define RETURN_GGUF_DATA(CASE, CTYPE) \
    case CASE:                        \
        return std::to_string(*reinterpret_cast<const CTYPE *>(data));

        RETURN_GGUF_DATA(GGUF_TYPE_UINT8, uint8_t)
        RETURN_GGUF_DATA(GGUF_TYPE_INT8, int8_t)
        RETURN_GGUF_DATA(GGUF_TYPE_UINT16, uint16_t)
        RETURN_GGUF_DATA(GGUF_TYPE_INT16, int16_t)
        RETURN_GGUF_DATA(GGUF_TYPE_UINT32, uint32_t)
        RETURN_GGUF_DATA(GGUF_TYPE_INT32, int32_t)
        RETURN_GGUF_DATA(GGUF_TYPE_FLOAT32, float)
        RETURN_GGUF_DATA(GGUF_TYPE_BOOL, bool)
        RETURN_GGUF_DATA(GGUF_TYPE_UINT64, uint64_t)
        RETURN_GGUF_DATA(GGUF_TYPE_INT64, int64_t)
        RETURN_GGUF_DATA(GGUF_TYPE_FLOAT64, double)
        RETURN_GGUF_DATA(GGUF_TYPE_STRING, char)

    case GGUF_TYPE_ARRAY:
        throw std::runtime_error("GGUF_TYPE_ARRAY should be processed element by element");

    default:
        return "GGUF_TYPE_UNKNOWN";
    }

#undef RETURN_GGUF_DATA
}

struct GGUFKeyValue {
    std::string key;
    GGUF_TYPE gguf_type; // gguf_type
    std::vector<uint8_t> value;

    std::string toString() const;
};

typedef enum {
    GGML_TYPE_F32 = 0,
    GGML_TYPE_F16 = 1,
    GGML_TYPE_Q4_0 = 2,
    GGML_TYPE_Q4_1 = 3,
    GGML_TYPE_Q5_0 = 6,
    GGML_TYPE_Q5_1 = 7,
    GGML_TYPE_Q8_0 = 8,
    GGML_TYPE_Q8_1 = 9,
    GGML_TYPE_Q2_K = 10,
    GGML_TYPE_Q3_K = 11,
    GGML_TYPE_Q4_K = 12,
    GGML_TYPE_Q5_K = 13,
    GGML_TYPE_Q6_K = 14,
    GGML_TYPE_Q8_K = 15,
    GGML_TYPE_IQ2_XXS = 16,
    GGML_TYPE_IQ2_XS = 17,
    GGML_TYPE_IQ3_XXS = 18,
    GGML_TYPE_IQ1_S = 19,
    GGML_TYPE_IQ4_NL = 20,
    GGML_TYPE_IQ3_S = 21,
    GGML_TYPE_IQ2_S = 22,
    GGML_TYPE_IQ4_XS = 23,
    GGML_TYPE_I8 = 24,
    GGML_TYPE_I16 = 25,
    GGML_TYPE_I32 = 26,
    GGML_TYPE_I64 = 27,
    GGML_TYPE_F64 = 28,
    GGML_TYPE_IQ1_M = 29,
    GGML_TYPE_BF16 = 30,
    GGML_TYPE_TQ1_0 = 34,
    GGML_TYPE_TQ2_0 = 35,
    GGML_TYPE_COUNT = 36,
} GGML_TYPE;

inline size_t ggmlTypeSize(GGML_TYPE ggml_type) {
    switch (ggml_type) {
    case GGML_TYPE_Q8_K:
        return 1;
    case GGML_TYPE_I8:
        return 1;
    case GGML_TYPE_I16:
        return 2;
    case GGML_TYPE_I32:
        return 4;
    case GGML_TYPE_I64:
        return 8;
    case GGML_TYPE_BF16:
        return 2;
    case GGML_TYPE_F16:
        return 2;
    case GGML_TYPE_F32:
        return 4;
    case GGML_TYPE_F64:
        return 8;
    default:
        throw std::runtime_error("GGML_TYPE_SIZE: Unsupported GGML_TYPE");
    }
    return 0;
}

constexpr const char *GGML_TYPE_NAME[GGML_TYPE_COUNT] = {
    "F32",
    "F16",
    "Q4_0",
    "Q4_1",
    nullptr, // 4 (gap)
    nullptr, // 5 (gap)
    "Q5_0",
    "Q5_1",
    "Q8_0",
    "Q8_1",
    "Q2_K",
    "Q3_K",
    "Q4_K",
    "Q5_K",
    "Q6_K",
    "Q8_K",
    "IQ2_XXS",
    "IQ2_XS",
    "IQ3_XXS",
    "IQ1_S",
    "IQ4_NL",
    "IQ3_S",
    "IQ2_S",
    "IQ4_XS",
    "I8",
    "I16",
    "I32",
    "I64",
    "F64",
    "IQ1_M",
    "BF16",
    nullptr, // 31 (gap)
    nullptr, // 32 (gap)
    nullptr, // 33 (gap)
    "TQ1_0",
    "TQ2_0",
};

struct GGUFTensorInfo {
    std::string name;
    uint32_t ndim;
    std::vector<int64_t> shape;
    GGML_TYPE ggml_type;
    uint64_t data_offset;

    std::string toString() const;
};

class GGUFFileReader {
public:
    GGUFFileReader(const std::string &filepath);
    ~GGUFFileReader() = default;
    std::string toString() const;

    const std::unordered_map<std::string, std::shared_ptr<GGUFKeyValue>> &getAttributeMap() const;
    const std::unordered_map<std::string, std::shared_ptr<GGUFTensorInfo>> &getTensorInfoMap() const;

    std::shared_ptr<FileMapping> getFileMapping() const { return _file; }
    void *getGgmlStart() const { return _cursor; }

private:
    void readHeader();
    void readMetaKVs();
    void readTensorInfos();
    std::string readString();
    template <typename T>
    T read();

    std::shared_ptr<FileMapping> _file;
    void *_data = nullptr;
    uint8_t *_cursor = nullptr;
    uint32_t _version;
    int64_t _num_tensors;
    int64_t _num_meta_kvs;
    std::vector<std::shared_ptr<GGUFKeyValue>> _meta_kvs;
    std::vector<std::shared_ptr<GGUFTensorInfo>> _tensor_infos;

    std::unordered_map<std::string, std::shared_ptr<GGUFKeyValue>> _attributes_map;
    std::unordered_map<std::string, std::shared_ptr<GGUFTensorInfo>> _tensors_info_map;
};

#endif
