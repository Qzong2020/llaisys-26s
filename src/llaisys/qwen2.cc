#include "llaisys/models/qwen2.h"
#include "llaisys/runtime.h"

#include "llaisys_tensor.hpp"

#include "../models/qwen2/model.hpp"
#include "../utils.hpp"

#include <memory>
#include <vector>

struct LlaisysQwen2Model {
    std::unique_ptr<llaisys::models::qwen2::Model> model;
    std::vector<std::unique_ptr<LlaisysTensor>> tensor_handles;

    std::vector<llaisysTensor_t> attn_norm_w;
    std::vector<llaisysTensor_t> attn_q_w;
    std::vector<llaisysTensor_t> attn_q_b;
    std::vector<llaisysTensor_t> attn_k_w;
    std::vector<llaisysTensor_t> attn_k_b;
    std::vector<llaisysTensor_t> attn_v_w;
    std::vector<llaisysTensor_t> attn_v_b;
    std::vector<llaisysTensor_t> attn_o_w;
    std::vector<llaisysTensor_t> mlp_norm_w;
    std::vector<llaisysTensor_t> mlp_gate_w;
    std::vector<llaisysTensor_t> mlp_up_w;
    std::vector<llaisysTensor_t> mlp_down_w;

    LlaisysQwen2Weights weights;

    LlaisysQwen2Model(const LlaisysQwen2Meta &meta, llaisysDeviceType_t device_type, int device_id)
        : model(std::make_unique<llaisys::models::qwen2::Model>(meta, device_type, device_id)),
          weights{} {
        tensor_handles.reserve(3 + 12 * meta.nlayer);
        auto &model_weights = model->weights();

        bind(attn_norm_w, model_weights.attn_norm_w);
        bind(attn_q_w, model_weights.attn_q_w);
        bind(attn_q_b, model_weights.attn_q_b);
        bind(attn_k_w, model_weights.attn_k_w);
        bind(attn_k_b, model_weights.attn_k_b);
        bind(attn_v_w, model_weights.attn_v_w);
        bind(attn_v_b, model_weights.attn_v_b);
        bind(attn_o_w, model_weights.attn_o_w);
        bind(mlp_norm_w, model_weights.mlp_norm_w);
        bind(mlp_gate_w, model_weights.mlp_gate_w);
        bind(mlp_up_w, model_weights.mlp_up_w);
        bind(mlp_down_w, model_weights.mlp_down_w);

        weights = {
            wrap(model_weights.in_embed),
            wrap(model_weights.out_embed),
            wrap(model_weights.out_norm_w),
            attn_norm_w.data(),
            attn_q_w.data(),
            attn_q_b.data(),
            attn_k_w.data(),
            attn_k_b.data(),
            attn_v_w.data(),
            attn_v_b.data(),
            attn_o_w.data(),
            mlp_norm_w.data(),
            mlp_gate_w.data(),
            mlp_up_w.data(),
            mlp_down_w.data(),
        };
    }

private:
    llaisysTensor_t wrap(const llaisys::tensor_t &tensor) {
        tensor_handles.emplace_back(std::make_unique<LlaisysTensor>(LlaisysTensor{tensor}));
        return tensor_handles.back().get();
    }

    void bind(std::vector<llaisysTensor_t> &handles, const std::vector<llaisys::tensor_t> &tensors) {
        handles.reserve(tensors.size());
        for (const auto &tensor : tensors) {
            handles.emplace_back(wrap(tensor));
        }
    }
};

__C {
    struct LlaisysQwen2Model *llaisysQwen2ModelCreate(
        const LlaisysQwen2Meta *meta,
        llaisysDeviceType_t device,
        int *device_ids,
        int ndevice) noexcept(false) {
        CHECK_ARGUMENT(meta != nullptr, "Qwen2: meta must not be null");
        CHECK_ARGUMENT(device_ids != nullptr, "Qwen2: device_ids must not be null");
        CHECK_ARGUMENT(ndevice == 1, "Qwen2: exactly one device is required");
#ifdef ENABLE_NVIDIA_API
        CHECK_ARGUMENT(device == LLAISYS_DEVICE_CPU || device == LLAISYS_DEVICE_NVIDIA,
                       "Qwen2: unsupported device type");
#else
        CHECK_ARGUMENT(device == LLAISYS_DEVICE_CPU,
                       "Qwen2: requested device is not available in this build");
#endif
        CHECK_ARGUMENT(device_ids[0] >= 0, "Qwen2: device id must not be negative");
        const auto *runtime_api = llaisysGetRuntimeAPI(device);
        CHECK_ARGUMENT(device_ids[0] < runtime_api->get_device_count(),
                       "Qwen2: device id is out of range");
        return new LlaisysQwen2Model(*meta, device, device_ids[0]);
    }

    void llaisysQwen2ModelDestroy(struct LlaisysQwen2Model *model) {
        delete model;
    }

    struct LlaisysQwen2Weights *llaisysQwen2ModelWeights(struct LlaisysQwen2Model *model) noexcept(false) {
        CHECK_ARGUMENT(model != nullptr, "Qwen2: model must not be null");
        return &model->weights;
    }

    void llaisysQwen2ModelReset(struct LlaisysQwen2Model *model) noexcept(false) {
        CHECK_ARGUMENT(model != nullptr, "Qwen2: model must not be null");
        model->model->reset();
    }

    int64_t llaisysQwen2ModelInfer(
        struct LlaisysQwen2Model *model,
        int64_t *token_ids,
        size_t ntoken) noexcept(false) {
        CHECK_ARGUMENT(model != nullptr, "Qwen2: model must not be null");
        CHECK_ARGUMENT(token_ids != nullptr, "Qwen2: token_ids must not be null");
        CHECK_ARGUMENT(ntoken > 0, "Qwen2: ntoken must be positive");
        return model->model->infer(token_ids, ntoken);
    }
}
