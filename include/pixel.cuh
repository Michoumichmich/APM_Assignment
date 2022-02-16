#pragma once

//#include<cuda_stdint.h>
#include <cstdint>
#include <utility>

/**
 * @class pixel
 */
class pixel {

public:
    enum composant { red, green, blue, alpha };

    template<typename T> static __host__ __device__ inline uint8_t clamp_byte(T val) {
        if constexpr (std::is_signed_v<T> || std::is_floating_point_v<T>) {   // Avoid "pointless" warnings if T is unsigned
            if (val < 0) return 0;
        }
        if (val > 255) return 255U;
        return val;
    }

private:
    static_assert(sizeof(uint8_t) == 1U);

    // Using bitfields so the pixels will get packed into a single word yay
    // union { struct {
    uint32_t r_ : 8;
    uint32_t g_ : 8;
    uint32_t b_ : 8;
    uint32_t a_ : 8;   // nice for padding!
    // }; uint32_t word_;};


public:
    __host__ __device__ inline pixel() : pixel(0U, 0U, 0U) {}

    __host__ __device__ inline pixel(uint8_t r, uint8_t g, uint8_t b, uint8_t a = 0U) : r_(r), g_(g), b_(b), a_(a) {}

    __host__ __device__ inline explicit pixel(uint8_t val) : pixel(val, val, val, val) {}

    __host__ __device__ inline explicit pixel(uint3 val) : pixel(clamp_byte(val.x), clamp_byte(val.y), clamp_byte(val.z)) {}

    __host__ __device__ inline explicit pixel(int3 val) : pixel(clamp_byte(val.x), clamp_byte(val.y), clamp_byte(val.z)) {}

    __host__ __device__ inline explicit pixel(uint32_t val) : pixel(clamp_byte(val)) {}

    __host__ __device__ static inline pixel from_yuv(int yValue, int uValue, int vValue) {
        float rTmp = (float) yValue + 1.13983 * (float) vValue;
        float gTmp = (float) yValue - 0.39465 * (float) uValue - 0.58060 * (float) vValue;
        float bTmp = (float) yValue + 2.03211 * (float) uValue;

        //        int rTmp = yValue + ((351 * (vValue - 128)) >> 8);
        //        int gTmp = yValue - ((179 * (vValue - 128) + 86 * (uValue - 128)) >> 8);
        //        int bTmp = yValue + ((443 * (uValue - 128)) >> 8);
        return pixel(make_int3(rTmp, gTmp, bTmp));
    }

    template<composant comp> [[nodiscard]] __host__ __device__ inline uint8_t get() const noexcept {
        if constexpr (comp == red) {
            return r_;
        } else if constexpr (comp == green) {
            return g_;
        } else if constexpr (comp == blue) {
            return b_;
        } else {
            return a_;
        }
        // unreachable;
        return 0;
    }

    template<composant comp> [[nodiscard]] __host__ __device__ inline float getf() const noexcept { return static_cast<float>(get<comp>()); }

    template<composant comp> __host__ __device__ inline void set(uint8_t val) noexcept {
        if constexpr (comp == red) {
            r_ = val;
        } else if constexpr (comp == green) {
            g_ = val;
        } else if constexpr (comp == blue) {
            b_ = val;
        } else {
            a_ = val;
        }
    }

    __host__ __device__ inline uint3 as_uint3() const { return make_uint3(r_, g_, b_); }
    __host__ __device__ inline int3 as_int3() const { return make_int3(r_, g_, b_); }
};


__host__ __device__ inline void operator+=(uint3& a, uint3 b) {
    a.x += b.x;
    a.y += b.y;
    a.z += b.z;
}
__host__ __device__ inline void operator+=(int3& a, int3 b) {
    a.x += b.x;
    a.y += b.y;
    a.z += b.z;
}
__host__ __device__ inline int3 operator*(int3 a, int b) { return make_int3(a.x * b, a.y * b, a.z * b); }

__host__ __device__ inline int3 operator/(int3 a, int b) { return make_int3(a.x / b, a.y / b, a.z / b); }
