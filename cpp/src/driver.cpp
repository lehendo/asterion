#include "asterion/driver.h"
#include <cuda.h>
#include <algorithm>
#include <bit>
#include <cstdio>
#include <cstring>
#include <memory>
#include <mutex>
#include <unordered_map>
#include <array>

namespace {
struct Module {
    CUcontext context{};
    CUmodule module{};
    CUfunction function{};
    unsigned shared{}, threads{}, arguments{};
    bool captured{};
    std::mutex mutex;
    std::unordered_map<CUstream, CUevent> completions;
};
struct BorrowedContext {
    bool pushed{};
    ~BorrowedContext() {
        if (pushed) {
            CUcontext popped{};
            cuCtxPopCurrent(&popped);
        }
    }
};
int message(const char* text, char* error, std::size_t size) {
    if (error && size) std::snprintf(error, size, "%s", text);
    return 1;
}
int status(CUresult result, char* error, std::size_t size) {
    if (result == CUDA_SUCCESS) return 0;
    const char* name = nullptr;
    const char* description = nullptr;
    cuGetErrorName(result, &name);
    cuGetErrorString(result, &description);
    if (error && size) std::snprintf(error, size, "%s: %s",
        name ? name : "CUDA error", description ? description : "unknown error");
    return static_cast<int>(result);
}
}
extern "C" uint32_t asterion_driver_abi_version(void) { return 1; }

extern "C" int asterion_driver_open(const unsigned char* cubin, std::size_t size,
    const char* entry, uint32_t shared, uint32_t threads, uint32_t arguments,
    void** output, char* error, std::size_t error_size) {
    static_assert(std::endian::native == std::endian::little);
    if (!cubin || size < 64 || std::memcmp(cubin, "\177ELF", 4) != 0 || !entry ||
        !output || !threads || threads > 1024 || !arguments || arguments > 8)
        return message("invalid trusted-CUBIN launch description", error, error_size);
    *output = nullptr;
    try {
        auto module = std::make_unique<Module>();
        auto result = cuCtxGetCurrent(&module->context);
        if (result != CUDA_SUCCESS) return status(result, error, error_size);
        if (!module->context) return message("a current borrowed CUDA context is required", error, error_size);
        result = cuModuleLoadDataEx(&module->module, cubin, 0, nullptr, nullptr);
        if (result != CUDA_SUCCESS) return status(result, error, error_size);
        result = cuModuleGetFunction(&module->function, module->module, entry);
        if (result != CUDA_SUCCESS) {
            cuModuleUnload(module->module);
            return status(result, error, error_size);
        }
        if (shared > 49152) {
            result = cuFuncSetAttribute(module->function, CU_FUNC_ATTRIBUTE_MAX_DYNAMIC_SHARED_SIZE_BYTES,
                                       static_cast<int>(shared));
            if (result != CUDA_SUCCESS) {
                cuModuleUnload(module->module);
                return status(result, error, error_size);
            }
        }
        module->shared = shared;
        module->threads = threads;
        module->arguments = arguments;
        *output = module.release();
        return 0;
    } catch (...) { return message("native allocation failure", error, error_size); }
}

extern "C" int asterion_driver_launch(void* handle, const uint64_t* values, std::size_t count,
    uint32_t grid_x, uint32_t grid_y, void* stream_pointer, char* error, std::size_t error_size) {
    if (!handle || !values || !grid_x || !grid_y)
        return message("invalid driver launch", error, error_size);
    auto& module = *static_cast<Module*>(handle);
    std::lock_guard lock(module.mutex);
    if (count != module.arguments) return message("kernel argument count mismatch", error, error_size);
    CUcontext current{};
    auto result = cuCtxGetCurrent(&current);
    if (result != CUDA_SUCCESS) return status(result, error, error_size);
    BorrowedContext borrowed;
    if (!current) {
        // PyTorch may leave a new host thread without a current driver context.
        // Borrow the module's existing context for this call; never create one.
        result = cuCtxPushCurrent(module.context);
        if (result != CUDA_SUCCESS) return status(result, error, error_size);
        borrowed.pushed = true;
    } else if (current != module.context) {
        return message("current CUDA context differs from module context", error, error_size);
    }
    try {
        auto stream = reinterpret_cast<CUstream>(stream_pointer);
        CUstreamCaptureStatus capture{};
        result = cuStreamIsCapturing(stream, &capture);
        if (result != CUDA_SUCCESS) return status(result, error, error_size);
        if (capture == CU_STREAM_CAPTURE_STATUS_INVALIDATED)
            return message("stream capture is invalidated", error, error_size);
        const bool capturing = capture == CU_STREAM_CAPTURE_STATUS_ACTIVE;
        module.captured = module.captured || capturing;
        auto event = module.completions.find(stream);
        if (!capturing && event == module.completions.end()) {
            CUevent completion{};
            result = cuEventCreate(&completion, CU_EVENT_DISABLE_TIMING);
            if (result != CUDA_SUCCESS) return status(result, error, error_size);
            event = module.completions.emplace(stream, completion).first;
        }
        // The verified adapter supports only pointer/u32 scalars and one null scratch pointer.
        std::array<uint64_t, 8> storage{};
        std::copy_n(values, count, storage.begin());
        std::array<void*, 8> arguments{};
        for (std::size_t i = 0; i < count; ++i) arguments[i] = &storage[i];
        result = cuLaunchKernel(module.function, grid_x, grid_y, 1, module.threads, 1, 1,
                                module.shared, stream, arguments.data(), nullptr);
        if (result != CUDA_SUCCESS) return status(result, error, error_size);
        // Internal capture events cannot later serve as host completion fences.
        // Graph replays contain only the kernel; close performs a context barrier.
        if (capturing) return 0;
        return status(cuEventRecord(event->second, stream), error, error_size);
    } catch (...) { return message("native launch allocation failure", error, error_size); }
}

extern "C" int asterion_driver_close(void* handle, char* error, std::size_t error_size) {
    if (!handle) return 0;
    std::unique_ptr<Module> module(static_cast<Module*>(handle));
    std::lock_guard lock(module->mutex);
    CUcontext previous{};
    auto result = cuCtxGetCurrent(&previous);
    if (result != CUDA_SUCCESS) return status(result, error, error_size);
    bool pushed = previous != module->context;
    if (pushed) {
        result = cuCtxPushCurrent(module->context);
        if (result != CUDA_SUCCESS) return status(result, error, error_size);
    }
    // CUDA Graphs borrow this module. No later replay is permitted after close.
    // A context-wide barrier is required because replays can use other streams.
    CUresult first = module->captured ? cuCtxSynchronize() : CUDA_SUCCESS;
    for (const auto& [stream, event] : module->completions) {
        (void)stream;
        result = cuEventSynchronize(event);
        if (first == CUDA_SUCCESS && result != CUDA_SUCCESS) first = result;
        result = cuEventDestroy(event);
        if (first == CUDA_SUCCESS && result != CUDA_SUCCESS) first = result;
    }
    result = cuModuleUnload(module->module);
    if (first == CUDA_SUCCESS && result != CUDA_SUCCESS) first = result;
    if (pushed) {
        CUcontext popped{};
        result = cuCtxPopCurrent(&popped);
        if (first == CUDA_SUCCESS && result != CUDA_SUCCESS) first = result;
    }
    return status(first, error, error_size);
}
