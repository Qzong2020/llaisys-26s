#include "runtime.hpp"

#include "../../device/runtime_api.hpp"
#include "../allocator/naive_allocator.hpp"
#include "../context/context.hpp"

namespace llaisys::core {
Runtime::Runtime(llaisysDeviceType_t device_type, int device_id)
    : _device_type(device_type),
      _device_id(device_id),
      _api(llaisys::device::getRuntimeAPI(device_type)),
      _allocator(nullptr),
      _resource(nullptr),
      _is_active(false),
      _stream(nullptr) {
    _api->set_device(_device_id);
    _stream = _api->create_stream();
    try {
        _resource = llaisys::device::createDeviceResource(
            _device_type, _device_id, _stream);
        _allocator = new allocators::NaiveAllocator(_api);
    } catch (...) {
        delete _allocator;
        _allocator = nullptr;
        delete _resource;
        _resource = nullptr;
        _api->destroy_stream(_stream);
        throw;
    }
}

Runtime::~Runtime() {
    if (!_is_active) {
        std::cerr << "Mallicious destruction of inactive runtime." << std::endl;
    }
    delete _resource;
    _resource = nullptr;
    delete _allocator;
    _allocator = nullptr;
    _api->destroy_stream(_stream);
    _api = nullptr;
}

void Runtime::_activate() {
    _api->set_device(_device_id);
    _is_active = true;
}

void Runtime::_deactivate() {
    _is_active = false;
}

bool Runtime::isActive() const {
    return _is_active;
}

llaisysDeviceType_t Runtime::deviceType() const {
    return _device_type;
}

int Runtime::deviceId() const {
    return _device_id;
}

const LlaisysRuntimeAPI *Runtime::api() const {
    return _api;
}

llaisys::device::DeviceResource &Runtime::resource() const {
    ASSERT(_resource != nullptr, "Runtime device resource is not initialized.");
    return *_resource;
}

storage_t Runtime::allocateDeviceStorage(size_t size) {
    return std::shared_ptr<Storage>(new Storage(_allocator->allocate(size), size, *this, false));
}

storage_t Runtime::allocateHostStorage(size_t size) {
    return std::shared_ptr<Storage>(new Storage((std::byte *)_api->malloc_host(size), size, *this, true));
}

void Runtime::freeStorage(Storage *storage) {
    auto &active_runtime = context().runtime();
    const auto previous_device_type = active_runtime.deviceType();
    const auto previous_device_id = active_runtime.deviceId();
    const bool switch_runtime = previous_device_type != deviceType()
                             || previous_device_id != deviceId();

    if (switch_runtime) {
        context().setDevice(deviceType(), deviceId());
    }

    try {
        if (storage->isHost()) {
            _api->free_host(storage->memory());
        } else {
            _allocator->release(storage->memory());
        }
    } catch (...) {
        if (switch_runtime) {
            context().setDevice(previous_device_type, previous_device_id);
        }
        throw;
    }

    if (switch_runtime) {
        context().setDevice(previous_device_type, previous_device_id);
    }
}

llaisysStream_t Runtime::stream() const {
    return _stream;
}

void Runtime::synchronize() const {
    _api->stream_synchronize(_stream);
}

} // namespace llaisys::core
