#include "context.hpp"
#include "../../utils.hpp"
#include <thread>

namespace llaisys::core {

Context::Context() {
    // All device types, put CPU at the end
    std::vector<llaisysDeviceType_t> device_typs;
    for (int i = 1; i < LLAISYS_DEVICE_TYPE_COUNT; i++) {
        device_typs.push_back(static_cast<llaisysDeviceType_t>(i));
    }
    device_typs.push_back(LLAISYS_DEVICE_CPU);

    // Create runtimes for each device type.
    // Activate the first available device. If no other device is available, activate CPU runtime.
    for (auto device_type : device_typs) {
        const LlaisysRuntimeAPI *api_ = llaisysGetRuntimeAPI(device_type);
        int device_count = api_->get_device_count();
        const auto insert_result = _runtime_map.emplace(
            device_type,
            std::vector<Runtime *>(device_count, nullptr));
        ASSERT(insert_result.second, "Runtime map already contains this device type.");

        if (_current_runtime == nullptr && device_count > 0) {
            setDevice(device_type, 0);
        }
    }
}

Context::~Context() {
    for (auto &runtime_entry : _runtime_map) {
        for (auto &runtime : runtime_entry.second) {
            if (runtime != nullptr) {
                if (_current_runtime != nullptr && _current_runtime != runtime) {
                    _current_runtime->_deactivate();
                }
                runtime->_activate();
                _current_runtime = runtime;
                delete runtime;
                runtime = nullptr;
                _current_runtime = nullptr;
            }
        }
    }
    _runtime_map.clear();
}

void Context::setDevice(llaisysDeviceType_t device_type, int device_id) {
    auto runtime_entry = _runtime_map.find(device_type);
    CHECK_ARGUMENT(runtime_entry != _runtime_map.end(), "invalid device type");

    auto &runtimes = runtime_entry->second;
    CHECK_ARGUMENT(device_id >= 0 && static_cast<size_t>(device_id) < runtimes.size(), "invalid device id");

    Runtime *target_runtime = runtimes[device_id];
    if (_current_runtime == target_runtime && target_runtime != nullptr) {
        return;
    }

    Runtime *previous_runtime = _current_runtime;
    if (previous_runtime != nullptr) {
        previous_runtime->_deactivate();
    }

    try {
        if (target_runtime == nullptr) {
            target_runtime = new Runtime(device_type, device_id);
            runtimes[device_id] = target_runtime;
        }
        target_runtime->_activate();
        _current_runtime = target_runtime;
    } catch (...) {
        if (previous_runtime != nullptr) {
            previous_runtime->_activate();
        }
        _current_runtime = previous_runtime;
        throw;
    }
}

Runtime &Context::runtime() {
    ASSERT(_current_runtime != nullptr, "No runtime is activated, please call setDevice() first.");
    return *_current_runtime;
}

// Global API to get thread-local context.
Context &context() {
    thread_local Context thread_context;
    return thread_context;
}

} // namespace llaisys::core
