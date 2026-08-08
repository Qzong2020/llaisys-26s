#pragma once

#include "llaisys/models/qwen2.h"

#include "../../tensor/tensor.hpp"

#include <vector>

namespace llaisys::models::qwen2 {

struct Weights {
    tensor_t in_embed;
    tensor_t out_embed;
    tensor_t out_norm_w;

    std::vector<tensor_t> attn_norm_w;
    std::vector<tensor_t> attn_q_w;
    std::vector<tensor_t> attn_q_b;
    std::vector<tensor_t> attn_k_w;
    std::vector<tensor_t> attn_k_b;
    std::vector<tensor_t> attn_v_w;
    std::vector<tensor_t> attn_v_b;
    std::vector<tensor_t> attn_o_w;

    std::vector<tensor_t> mlp_norm_w;
    std::vector<tensor_t> mlp_gate_w;
    std::vector<tensor_t> mlp_up_w;
    std::vector<tensor_t> mlp_down_w;
};

struct Buffers {
    tensor_t token_ids;
    tensor_t position_ids;
    tensor_t hidden;
    tensor_t residual;
    tensor_t normalized;
    tensor_t query;
    tensor_t key;
    tensor_t value;
    tensor_t attention;
    tensor_t gate;
    tensor_t up;
    tensor_t logits;
    tensor_t max_index;
    tensor_t max_value;
};

struct KVCache {
    std::vector<tensor_t> keys;
    std::vector<tensor_t> values;
    size_t length;
    size_t capacity;
};

class Model {
private:
    LlaisysQwen2Meta _meta;
    llaisysDeviceType_t _device_type;
    int _device_id;
    Weights _weights;
    Buffers _buffers;
    KVCache _cache;

    void validateMeta() const;
    tensor_t createTensor(const std::vector<size_t> &shape) const;
    tensor_t createTensor(const std::vector<size_t> &shape, llaisysDataType_t dtype) const;
    void initializeWeights();
    void ensureCacheCapacity(size_t required);

public:
    Model(const LlaisysQwen2Meta &meta, llaisysDeviceType_t device_type, int device_id);
    ~Model() = default;

    Model(const Model &) = delete;
    Model &operator=(const Model &) = delete;
    Model(Model &&) = delete;
    Model &operator=(Model &&) = delete;

    const LlaisysQwen2Meta &meta() const;
    llaisysDeviceType_t deviceType() const;
    int deviceId() const;

    Weights &weights();
    const Weights &weights() const;

    size_t cacheLength() const;
    size_t cacheCapacity() const;
    int64_t infer(const int64_t *token_ids, size_t ntoken);
    void reset();
};

} // namespace llaisys::models::qwen2
