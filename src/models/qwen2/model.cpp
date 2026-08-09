#include "model.hpp"

#include "../../ops/add/op.hpp"
#include "../../ops/argmax/op.hpp"
#include "../../ops/embedding/op.hpp"
#include "../../ops/linear/op.hpp"
#include "../../ops/rms_norm/op.hpp"
#include "../../ops/rope/op.hpp"
#include "../../ops/self_attention/op.hpp"
#include "../../ops/swiglu/op.hpp"
#include "../../utils.hpp"

#include <cmath>
#include <cstring>
#include <vector>

namespace llaisys::models::qwen2 {

Model::Model(const LlaisysQwen2Meta &meta, llaisysDeviceType_t device_type, int device_id)
    : _meta(meta), _device_type(device_type), _device_id(device_id),
      _cache{{}, {}, 0, 0} {
    validateMeta();
    initializeWeights();
    _cache.keys.resize(_meta.nlayer);
    _cache.values.resize(_meta.nlayer);
}

void Model::validateMeta() const {
    CHECK_ARGUMENT(_meta.nlayer > 0, "Qwen2: nlayer must be positive");
    CHECK_ARGUMENT(_meta.hs > 0, "Qwen2: hidden size must be positive");
    CHECK_ARGUMENT(_meta.nh > 0, "Qwen2: attention head count must be positive");
    CHECK_ARGUMENT(_meta.nkvh > 0, "Qwen2: KV head count must be positive");
    CHECK_ARGUMENT(_meta.dh > 0, "Qwen2: head dimension must be positive");
    CHECK_ARGUMENT(_meta.di > 0, "Qwen2: intermediate size must be positive");
    CHECK_ARGUMENT(_meta.maxseq > 0, "Qwen2: maximum sequence length must be positive");
    CHECK_ARGUMENT(_meta.voc > 0, "Qwen2: vocabulary size must be positive");
    CHECK_ARGUMENT(_meta.hs == _meta.nh * _meta.dh,
                   "Qwen2: hidden size must equal attention heads times head dimension");
    CHECK_ARGUMENT(_meta.nh % _meta.nkvh == 0,
                   "Qwen2: attention head count must be divisible by KV head count");
    CHECK_ARGUMENT(_meta.dtype == LLAISYS_DTYPE_F32
                       || _meta.dtype == LLAISYS_DTYPE_F16
                       || _meta.dtype == LLAISYS_DTYPE_BF16,
                   "Qwen2: unsupported model data type");
}

tensor_t Model::createTensor(const std::vector<size_t> &shape) const {
    return createTensor(shape, _meta.dtype);
}

tensor_t Model::createTensor(const std::vector<size_t> &shape, llaisysDataType_t dtype) const {
    return Tensor::create(shape, dtype, _device_type, _device_id);
}

void Model::initializeWeights() {
    const size_t q_size = _meta.nh * _meta.dh;
    const size_t kv_size = _meta.nkvh * _meta.dh;

    _weights.in_embed = createTensor({_meta.voc, _meta.hs});
    _weights.out_embed = createTensor({_meta.voc, _meta.hs});
    _weights.out_norm_w = createTensor({_meta.hs});

    _weights.attn_norm_w.reserve(_meta.nlayer);
    _weights.attn_q_w.reserve(_meta.nlayer);
    _weights.attn_q_b.reserve(_meta.nlayer);
    _weights.attn_k_w.reserve(_meta.nlayer);
    _weights.attn_k_b.reserve(_meta.nlayer);
    _weights.attn_v_w.reserve(_meta.nlayer);
    _weights.attn_v_b.reserve(_meta.nlayer);
    _weights.attn_o_w.reserve(_meta.nlayer);
    _weights.mlp_norm_w.reserve(_meta.nlayer);
    _weights.mlp_gate_w.reserve(_meta.nlayer);
    _weights.mlp_up_w.reserve(_meta.nlayer);
    _weights.mlp_down_w.reserve(_meta.nlayer);

    for (size_t layer = 0; layer < _meta.nlayer; ++layer) {
        _weights.attn_norm_w.emplace_back(createTensor({_meta.hs}));
        _weights.attn_q_w.emplace_back(createTensor({q_size, _meta.hs}));
        _weights.attn_q_b.emplace_back(createTensor({q_size}));
        _weights.attn_k_w.emplace_back(createTensor({kv_size, _meta.hs}));
        _weights.attn_k_b.emplace_back(createTensor({kv_size}));
        _weights.attn_v_w.emplace_back(createTensor({kv_size, _meta.hs}));
        _weights.attn_v_b.emplace_back(createTensor({kv_size}));
        _weights.attn_o_w.emplace_back(createTensor({_meta.hs, q_size}));

        _weights.mlp_norm_w.emplace_back(createTensor({_meta.hs}));
        _weights.mlp_gate_w.emplace_back(createTensor({_meta.di, _meta.hs}));
        _weights.mlp_up_w.emplace_back(createTensor({_meta.di, _meta.hs}));
        _weights.mlp_down_w.emplace_back(createTensor({_meta.hs, _meta.di}));
    }
}

void Model::ensureCacheCapacity(size_t required) {
    if (required <= _cache.capacity) {
        return;
    }

    size_t new_capacity = _cache.capacity == 0 ? 1 : _cache.capacity;
    while (new_capacity < required) {
        if (new_capacity > _meta.maxseq / 2) {
            new_capacity = _meta.maxseq;
            break;
        }
        new_capacity *= 2;
    }

    std::vector<tensor_t> new_keys;
    std::vector<tensor_t> new_values;
    new_keys.reserve(_meta.nlayer);
    new_values.reserve(_meta.nlayer);

    for (size_t layer = 0; layer < _meta.nlayer; ++layer) {
        new_keys.emplace_back(
            createTensor({new_capacity, _meta.nkvh, _meta.dh}));
        new_values.emplace_back(
            createTensor({new_capacity, _meta.nkvh, _meta.dh}));
    }

    if (_cache.length > 0) {
        const size_t layer_prefix_bytes =
            _cache.length * _meta.nkvh * _meta.dh
          * _weights.in_embed->elementSize();
        if (_device_type == LLAISYS_DEVICE_CPU) {
            for (size_t layer = 0; layer < _meta.nlayer; ++layer) {
                std::memcpy(new_keys[layer]->data(), _cache.keys[layer]->data(),
                            layer_prefix_bytes);
                std::memcpy(new_values[layer]->data(), _cache.values[layer]->data(),
                            layer_prefix_bytes);
            }
        } else {
            core::context().setDevice(_device_type, _device_id);
            auto &runtime = core::context().runtime();
            // Keep the old cache storage alive until every prefix copy completes.
            for (size_t layer = 0; layer < _meta.nlayer; ++layer) {
                runtime.api()->memcpy_sync(
                    new_keys[layer]->data(), _cache.keys[layer]->data(),
                    layer_prefix_bytes, LLAISYS_MEMCPY_D2D);
                runtime.api()->memcpy_sync(
                    new_values[layer]->data(), _cache.values[layer]->data(),
                    layer_prefix_bytes, LLAISYS_MEMCPY_D2D);
            }
        }
    }

    _cache.keys = std::move(new_keys);
    _cache.values = std::move(new_values);
    _cache.capacity = new_capacity;
}

const LlaisysQwen2Meta &Model::meta() const {
    return _meta;
}

llaisysDeviceType_t Model::deviceType() const {
    return _device_type;
}

int Model::deviceId() const {
    return _device_id;
}

Weights &Model::weights() {
    return _weights;
}

const Weights &Model::weights() const {
    return _weights;
}

size_t Model::cacheLength() const {
    return _cache.length;
}

size_t Model::cacheCapacity() const {
    return _cache.capacity;
}

int64_t Model::infer(const int64_t *token_ids, size_t ntoken) {
    CHECK_ARGUMENT(token_ids != nullptr, "Qwen2: token_ids must not be null");
    CHECK_ARGUMENT(ntoken > 0, "Qwen2: ntoken must be positive");
    CHECK_ARGUMENT(_cache.length == 0 || ntoken == 1,
                   "Qwen2: incremental decode expects exactly one new token");
    CHECK_ARGUMENT(ntoken <= _meta.maxseq - _cache.length,
                   "Qwen2: token count exceeds remaining sequence capacity");
    for (size_t i = 0; i < ntoken; ++i) {
        CHECK_ARGUMENT(token_ids[i] >= 0 && static_cast<size_t>(token_ids[i]) < _meta.voc,
                       "Qwen2: token id is outside the vocabulary");
    }

    const size_t cache_start = _cache.length;
    const size_t cache_end = cache_start + ntoken;
    ensureCacheCapacity(cache_end);

    _buffers = {};
    _buffers.token_ids = createTensor({ntoken}, LLAISYS_DTYPE_I64);
    _buffers.position_ids = createTensor({ntoken}, LLAISYS_DTYPE_I64);
    _buffers.hidden = createTensor({ntoken, _meta.hs});
    _buffers.residual = createTensor({ntoken, _meta.hs});
    _buffers.normalized = createTensor({ntoken, _meta.hs});
    _buffers.query = createTensor({ntoken, _meta.nh * _meta.dh});
    _buffers.key = createTensor({ntoken, _meta.nkvh * _meta.dh});
    _buffers.value = createTensor({ntoken, _meta.nkvh * _meta.dh});
    _buffers.attention = createTensor({ntoken, _meta.nh, _meta.dh});
    _buffers.gate = createTensor({ntoken, _meta.di});
    _buffers.up = createTensor({ntoken, _meta.di});
    _buffers.logits = createTensor({1, _meta.voc});
    _buffers.max_index = createTensor({1}, LLAISYS_DTYPE_I64);
    _buffers.max_value = createTensor({1});

    _buffers.token_ids->load(token_ids);
    std::vector<int64_t> position_ids(ntoken);
    for (size_t i = 0; i < ntoken; ++i) {
        position_ids[i] = static_cast<int64_t>(cache_start + i);
    }
    _buffers.position_ids->load(position_ids.data());

    ops::embedding(_buffers.hidden, _buffers.token_ids, _weights.in_embed);
    const float attention_scale = 1.0f / std::sqrt(static_cast<float>(_meta.dh));

    for (size_t layer = 0; layer < _meta.nlayer; ++layer) {
        ops::rms_norm(_buffers.normalized, _buffers.hidden,
                      _weights.attn_norm_w[layer], _meta.epsilon);
        ops::linear(_buffers.query, _buffers.normalized,
                    _weights.attn_q_w[layer], _weights.attn_q_b[layer]);
        ops::linear(_buffers.key, _buffers.normalized,
                    _weights.attn_k_w[layer], _weights.attn_k_b[layer]);
        ops::linear(_buffers.value, _buffers.normalized,
                    _weights.attn_v_w[layer], _weights.attn_v_b[layer]);

        auto query_heads = _buffers.query->view({ntoken, _meta.nh, _meta.dh});
        auto key_heads = _buffers.key->view({ntoken, _meta.nkvh, _meta.dh});
        auto value_heads = _buffers.value->view({ntoken, _meta.nkvh, _meta.dh});
        ops::rope(query_heads, query_heads, _buffers.position_ids, _meta.theta);
        ops::rope(key_heads, key_heads, _buffers.position_ids, _meta.theta);

        auto key_write = _cache.keys[layer]->slice(0, cache_start, cache_end);
        auto value_write = _cache.values[layer]->slice(0, cache_start, cache_end);
        const size_t new_kv_bytes = ntoken * _meta.nkvh * _meta.dh
                                  * _buffers.key->elementSize();
        if (_device_type == LLAISYS_DEVICE_CPU) {
            std::memcpy(key_write->data(), key_heads->data(), new_kv_bytes);
            std::memcpy(value_write->data(), value_heads->data(), new_kv_bytes);
        } else {
            core::context().setDevice(_device_type, _device_id);
            auto &runtime = core::context().runtime();
            runtime.api()->memcpy_async(
                key_write->data(), key_heads->data(), new_kv_bytes,
                LLAISYS_MEMCPY_D2D, runtime.stream());
            runtime.api()->memcpy_async(
                value_write->data(), value_heads->data(), new_kv_bytes,
                LLAISYS_MEMCPY_D2D, runtime.stream());
        }

        auto cached_keys = _cache.keys[layer]->slice(0, 0, cache_end);
        auto cached_values = _cache.values[layer]->slice(0, 0, cache_end);
        ops::self_attention(_buffers.attention, query_heads, cached_keys, cached_values,
                            attention_scale);

        auto attention_flat = _buffers.attention->view({ntoken, _meta.hs});
        ops::linear(_buffers.residual, attention_flat,
                    _weights.attn_o_w[layer], nullptr);
        ops::add(_buffers.hidden, _buffers.hidden, _buffers.residual);

        ops::rms_norm(_buffers.normalized, _buffers.hidden,
                      _weights.mlp_norm_w[layer], _meta.epsilon);
        ops::linear(_buffers.gate, _buffers.normalized,
                    _weights.mlp_gate_w[layer], nullptr);
        ops::linear(_buffers.up, _buffers.normalized,
                    _weights.mlp_up_w[layer], nullptr);
        ops::swiglu(_buffers.gate, _buffers.gate, _buffers.up);
        ops::linear(_buffers.residual, _buffers.gate,
                    _weights.mlp_down_w[layer], nullptr);
        ops::add(_buffers.hidden, _buffers.hidden, _buffers.residual);
    }

    auto hidden_last = _buffers.hidden->slice(0, ntoken - 1, ntoken);
    auto normalized_last = _buffers.normalized->slice(0, ntoken - 1, ntoken);
    ops::rms_norm(normalized_last, hidden_last, _weights.out_norm_w, _meta.epsilon);
    ops::linear(_buffers.logits, normalized_last, _weights.out_embed, nullptr);
    ops::argmax(_buffers.max_index, _buffers.max_value, _buffers.logits);

    int64_t max_index;
    if (_device_type == LLAISYS_DEVICE_CPU) {
        std::memcpy(&max_index, _buffers.max_index->data(), sizeof(max_index));
    } else {
        core::context().setDevice(_device_type, _device_id);
        auto &runtime = core::context().runtime();
        auto host_index = runtime.allocateHostStorage(sizeof(max_index));
        runtime.api()->memcpy_async(
            host_index->memory(), _buffers.max_index->data(), sizeof(max_index),
            LLAISYS_MEMCPY_D2H, runtime.stream());
        runtime.synchronize();
        std::memcpy(&max_index, host_index->memory(), sizeof(max_index));
    }

    _cache.length = cache_end;
    return max_index;
}

void Model::reset() {
    _buffers = {};
    _cache.length = 0;
}

} // namespace llaisys::models::qwen2
