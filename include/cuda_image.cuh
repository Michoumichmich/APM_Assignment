#pragma once

#include "internal/cuda_kernel_launchers.cuh"
#include "pixel.cuh"

/**
 * @class cuda_image
 */
struct cuda_image {
    struct image_view {
        friend cuda_image;

    protected:
        uint32_t width_;
        uint32_t height_;
        mutable pixel* __restrict data_;   // size is width * height without pitch
    public:
        __host__ __device__ inline auto width() const { return width_; }
        __host__ __device__ inline auto height() const { return height_; }
        __host__ __device__ inline pixel& at(int width, int height) const { return data_[width_ * height + width]; }
        __host__ __device__ inline pixel& linear(int n) const { return data_[n]; }
        auto raw_data() const { return data_; };
    };

    image_view view;

    /**
     * Constructor to be used. Allocates memory and eventually a stream.
     * @param width_
     * @param height_
     * @param use_separate_stream
     */
    cuda_image(uint32_t width_, uint32_t height_, bool use_separate_stream = false) {
        view.width_ = width_;
        view.height_ = height_;
        if (use_separate_stream) { cudaStreamCreate(&stream); }
        cudaMalloc((void**) (&view.data_), sizeof(pixel) * view.width_ * view.height_);
    }

    auto width() const { return view.width_; }

    auto height() const { return view.height_; }

    operator image_view() { return view; }

    ~cuda_image() {
        cudaStreamSynchronize(stream);
        if (stream) { cudaStreamDestroy(stream); }
        cudaFree(view.data_);
    }

    /**
     * Not copy constructor
     */
    cuda_image(cuda_image const&) = delete;

    /**
     * Not assignment operator
     * @return
     */
    cuda_image& operator=(cuda_image const&) = delete;

    /**
     * Move assignment allowed.
     * @param other
     * @return
     */
    cuda_image& operator=(cuda_image&& other) noexcept {
        std::swap(view, other.view);
        return *this;
    };

    /**
     * Move copy constructor allowed
     * @param other
     */
    cuda_image(cuda_image&& other) noexcept { view = std::exchange(other.view, {}); };

    cudaStream_t get_stream() { return stream; }

    // DEPRECATED :(
    template<typename func> void apply_1D_filter(func&& kernel) { dispatch_1D_kernel(width() * height(), get_stream(), kernel); }


private:
    cudaStream_t stream = nullptr;
};
