#pragma once
#include <cuda_runtime.h>
#include <utility>

template <typename T>
class CudaBuffer
{
    T* devicePtr = nullptr;
    size_t sizeBytes = 0;
    cudaError_t status = cudaSuccess;

public:
    explicit CudaBuffer(size_t count) noexcept
        : sizeBytes(count * sizeof(T))
    {
        status = cudaMalloc(reinterpret_cast<void**>(&devicePtr), sizeBytes);
    }

    // Non-copyable (avoid double free)
    CudaBuffer(const CudaBuffer&) = delete;
    CudaBuffer& operator=(const CudaBuffer&) = delete;

    // Movable
    CudaBuffer(CudaBuffer&& other) noexcept
        : devicePtr(std::exchange(other.devicePtr, nullptr)),
        sizeBytes(std::exchange(other.sizeBytes, 0)),
        status(std::exchange(other.status, cudaSuccess))
    {
    }

    CudaBuffer& operator=(CudaBuffer&& other) noexcept
    {
        if (this != &other)
        {
            if (devicePtr)
                cudaFree(devicePtr);
            devicePtr = std::exchange(other.devicePtr, nullptr);
            sizeBytes = std::exchange(other.sizeBytes, 0);
            status = std::exchange(other.status, cudaSuccess);
        }
        return *this;
    }

    ~CudaBuffer()
    {
        if (devicePtr)
            cudaFree(devicePtr);
    }

    // Accessors
    T* get() noexcept { return devicePtr; }
    const T* get() const noexcept { return devicePtr; }
    size_t size() const noexcept { return sizeBytes / sizeof(T); }
    bool valid() const noexcept { return status == cudaSuccess && devicePtr != nullptr; }
    cudaError_t error() const noexcept { return status; }
};