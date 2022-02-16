#include <cuda_filters.cuh>
#include <internal/static_sort.cuh>
#include <stdexcept>

size_t saturate(cuda_image& image, pixel::composant composant) {
    switch (composant) {
        case pixel::red: image.apply_1D_filter([v = image.view] __device__(int x) { v.linear(x).set<pixel::red>(255U); }); break;
        case pixel::green: image.apply_1D_filter([v = image.view] __device__(int x) { v.linear(x).set<pixel::green>(255U); }); break;
        case pixel::blue: image.apply_1D_filter([v = image.view] __device__(int x) { v.linear(x).set<pixel::blue>(255U); }); break;
        case pixel::alpha: image.apply_1D_filter([v = image.view] __device__(int x) { v.linear(x).set<pixel::alpha>(255U); }); break;
    }
    return 2 * image.width() * image.height();
}

size_t negative(cuda_image& image) {
    image.apply_1D_filter([image = image.view] __device__(int x) {
        image.linear(x).set<pixel::red>(255U - image.linear(x).get<pixel::red>());
        image.linear(x).set<pixel::green>(255U - image.linear(x).get<pixel::green>());
        image.linear(x).set<pixel::blue>(255U - image.linear(x).get<pixel::blue>());
    });
    return 2 * image.width() * image.height();
}


size_t mirror_vertical(cuda_image& image) {
    dispatch_2D_stencil_kernel(image.width() / 2, image.height(), image.get_stream(), [image = image.view] __device__(int x, int y) {
        auto tmp = image.at(x, y);
        image.at(x, y) = image.at(image.width() - x, y);
        image.at(image.width() - x, y) = tmp;
    });
    return 2 * image.width() * image.height();
}

size_t mirror_horitonzal(cuda_image& image) {
    dispatch_2D_stencil_kernel(image.width(), image.height() / 2, image.get_stream(), [image = image.view] __device__(int x, int y) {
        auto tmp = image.at(x, y);
        image.at(x, y) = image.at(x, image.height() - y);
        image.at(x, image.height() - y) = tmp;
    });
    return 2 * image.width() * image.height();
}


template<bool avoid_fp_use> __device__ inline uint8_t convert_pixel_bw(const pixel& px) {
    if constexpr (avoid_fp_use) {
        const auto red_factor = static_cast<uint32_t>(0.229F * 2048.F);
        const auto green_factor = static_cast<uint32_t>(0.587F * 2048.F);
        const auto blue_factor = static_cast<uint32_t>(0.114F * 2048.F);
        const uint32_t avg = red_factor * px.get<pixel::red>() + green_factor * px.get<pixel::green>() + blue_factor * px.get<pixel::blue>();
        return avg / 2048U;
    } else {
        const float avg = 0.229F * px.getf<pixel::red>() + 0.587F * px.getf<pixel::green>() + 0.114F * px.getf<pixel::blue>();
        return static_cast<uint8_t>(avg);
    }
    // unreachable;
    return 0;
}


/**
 * @see https://godbolt.org/z/vj44KPjYh
 * @param img
 * @param avoid_fp_use
 * @return
 */
size_t bw_filter(cuda_image& img, bool avoid_fp_use) {
    if (avoid_fp_use) {
        img.apply_1D_filter([img = img.view] __device__(int x) { img.linear(x) = pixel(convert_pixel_bw<true>(img.linear(x))); });
    } else {
        img.apply_1D_filter([img = img.view] __device__(int x) { img.linear(x) = pixel(convert_pixel_bw<false>(img.linear(x))); });
    }

    return 2 * img.width() * img.height();
}


size_t blur_filter(cuda_image& image, int passes) {
    auto tmp_image = cuda_image(image.width(), image.height(), true);

    for (int i = 0; i < passes; ++i) {
        dispatch_2D_stencil_kernel(image.width(), image.height(), tmp_image.get_stream(), [src = image.view, tmp = tmp_image.view] __device__(int x, int y) {
            int mean_count = 1;
            uint3 sum = src.at(x, y).as_uint3();

            if (x + 1 < src.width()) {
                ++mean_count;
                sum += src.at(x + 1, y).as_uint3();
            }

            if (x >= 1) {
                ++mean_count;
                sum += src.at(x - 1, y).as_uint3();
            }

            if (y + 1 < src.height()) {
                ++mean_count;
                sum += src.at(x, y + 1).as_uint3();
            }

            if (y >= 1) {
                ++mean_count;
                sum += src.at(x, y - 1).as_uint3();
            }
            sum.x /= mean_count;
            sum.y /= mean_count;
            sum.z /= mean_count;
            tmp.at(x, y) = pixel{sum};
        });
        std::swap(tmp_image, image);
    }

    return 6 * image.width() * image.height() * passes;
}

template<bool vertical> inline uint8_t __device__ compute_sobel_convolution(const cuda_image::image_view& src, int x, int y) {
    if constexpr (vertical) {
        int3 vertical_pass{};
        vertical_pass += src.at(x - 1, y - 1).as_int3() * -1;
        vertical_pass += src.at(x - 1, y - 0).as_int3() * -2;
        vertical_pass += src.at(x - 1, y + 1).as_int3() * -1;
        vertical_pass += src.at(x + 1, y - 1).as_int3() * 1;
        vertical_pass += src.at(x + 1, y - 0).as_int3() * 2;
        vertical_pass += src.at(x + 1, y + 1).as_int3() * 1;
        return convert_pixel_bw<true>(pixel(vertical_pass));
    } else {
        int3 horizontal_pass{};
        horizontal_pass += src.at(x - 1, y - 1).as_int3() * -1;
        horizontal_pass += src.at(x - 0, y - 1).as_int3() * -2;
        horizontal_pass += src.at(x + 1, y - 1).as_int3() * -1;
        horizontal_pass += src.at(x - 1, y + 1).as_int3() * 1;
        horizontal_pass += src.at(x + 0, y + 1).as_int3() * 2;
        horizontal_pass += src.at(x - 1, y + 1).as_int3() * 1;
        return convert_pixel_bw<true>(pixel(horizontal_pass));
    }
    // unreachable;
    return 0;
}


inline uint8_t __device__ corner_convolution(const cuda_image::image_view& src, int x, int y) {
    int3 pass{};
    pass += src.at(x - 1, y - 1).as_int3() * 4;
    pass += src.at(x + 1, y + 1).as_int3() * 4;
    pass += src.at(x + 1, y - 1).as_int3() * 4;
    pass += src.at(x - 1, y + 1).as_int3() * 4;
    pass += src.at(x, y).as_int3() * 4;

    pass += src.at(x, y - 1).as_int3() * -5;
    pass += src.at(x, y + 1).as_int3() * -5;
    pass += src.at(x + 1, y).as_int3() * -5;
    pass += src.at(x - 1, y).as_int3() * -5;
    return convert_pixel_bw<true>(pixel(pass));
}

size_t sobel_filter(cuda_image& image, bool use_custom_algo) {
    if (!use_custom_algo) {
        auto tmp_image = cuda_image(image.width(), image.height(), true);
        dispatch_2D_stencil_kernel(image.width(), image.height(), image.get_stream(), [src = image.view, tmp = tmp_image.view] __device__(int x, int y) {
            int vertical_intensity = 0, horizontal_intensity = 0;
            if (x + 1 < src.width() && y + 1 < src.height() && x >= 1 && y >= 1) {
                vertical_intensity = compute_sobel_convolution<true>(src, x, y);
                horizontal_intensity = compute_sobel_convolution<false>(src, x, y);
            }
            tmp.at(x, y) = pixel(pixel::clamp_byte(horizontal_intensity + vertical_intensity));
        });
        std::swap(tmp_image, image);

        return 9 * image.width() * image.height();
    } else {
        size_t preprocessed = blur_filter(image, 3);
        preprocessed += median_filter(image, 5);
        auto tmp_image = cuda_image(image.width(), image.height(), true);
        dispatch_2D_stencil_kernel(image.width(), image.height(), image.get_stream(), [src = image.view, tmp = tmp_image.view] __device__(int x, int y) {
            int vertical_intensity = 0, horizontal_intensity = 0, corner = 0;
            if (x + 1 < src.width() && y + 1 < src.height() && x >= 1 && y >= 1) {
                vertical_intensity = compute_sobel_convolution<true>(src, x, y);
                horizontal_intensity = compute_sobel_convolution<false>(src, x, y);
                corner = corner_convolution(src, x, y);
            }
            const auto gold = make_int3(218, 155, 32);
            const auto teal = make_int3(0, 128, 128);
            const auto dark_red = make_int3(139, 0, 0);

            const auto weight_teal = horizontal_intensity;
            const auto weight_gold = vertical_intensity;
            const int weight_red = corner;

            auto res = gold * weight_gold;
            res += teal * weight_teal;
            res += dark_red * weight_red;

            tmp.at(x, y) = pixel(res / (64));   //pixel::from_yuv(sqrt(intensity * intensity), (vertical_intensity + 128) % 256 - 128, (horizontal_intensity + 128) % 256 - 128);
        });
        std::swap(tmp_image, image);

        return 10 * image.width() * image.height() + preprocessed;
    }
}

template<int w_border> static inline __device__ pixel run_median_filter_kernel(cuda_image::image_view src, int x, int y) {
    constexpr int w_width = w_border * 2 + 1;
    StaticSort<w_width * w_width> boseNelsonSort;
    uint8_t reds[w_width * w_width];
    uint8_t greens[w_width * w_width];
    uint8_t blues[w_width * w_width];
#pragma unroll
    for (int i = -w_border; i <= w_border; ++i) {
#pragma unroll
        for (int j = -w_border; j <= w_border; ++j) {
            reds[(i + w_border) * w_width + j + w_border] = src.at(i + x, j + y).get<pixel::red>();
            greens[(i + w_border) * w_width + j + w_border] = src.at(i + x, j + y).get<pixel::green>();
            blues[(i + w_border) * w_width + j + w_border] = src.at(i + x, j + y).get<pixel::blue>();
        }
    }
    boseNelsonSort(reds);
    boseNelsonSort(greens);
    boseNelsonSort(blues);
    return pixel(reds[w_width * w_width / 2], greens[w_width * w_width / 2], blues[w_width * w_width / 2], 0);
}


size_t median_filter(cuda_image& image, int width) {
    auto tmp_image = cuda_image(image.width(), image.height(), true);
    if (width == 3) {
        dispatch_2D_stencil_kernel(image.width(), image.height(), image.get_stream(), [src = image.view, tmp = tmp_image.view] __device__(int x, int y) {
            constexpr int w_border = 1;
            if (x + w_border < src.width() && y + w_border < src.height() && x >= w_border && y >= w_border) { tmp.at(x, y) = run_median_filter_kernel<w_border>(src, x, y); }
        });
    } else if (width == 5) {
        dispatch_2D_stencil_kernel(image.width(), image.height(), image.get_stream(), [src = image.view, tmp = tmp_image.view] __device__(int x, int y) {
            constexpr int w_border = 2;
            if (x + w_border < src.width() && y + w_border < src.height() && x >= w_border && y >= w_border) { tmp.at(x, y) = run_median_filter_kernel<w_border>(src, x, y); }
        });
    } else if (width == 7) {
        dispatch_2D_stencil_kernel(image.width(), image.height(), image.get_stream(), [src = image.view, tmp = tmp_image.view] __device__(int x, int y) {
            constexpr int w_border = 3;
            if (x + w_border < src.width() && y + w_border < src.height() && x >= w_border && y >= w_border) { tmp.at(x, y) = run_median_filter_kernel<w_border>(src, x, y); }
        });
    } else {
        throw std::runtime_error("Supports only widths of 3, 5 or 7.");
    }

    std::swap(tmp_image, image);

    return (width * width + 1) * image.width() * image.height();
}


size_t popart_filter(cuda_image& image) {
    dispatch_2D_stencil_kernel(image.width(), image.height(), image.get_stream(), [image = image.view] __device__(int x, int y) {
        int quadrant_id = 0;                                // 2D linear id kind of
        if (x < image.width() / 2) { quadrant_id += 1; }    // first dim
        if (y < image.height() / 2) { quadrant_id += 2; }   // second dim
        switch (quadrant_id) {
            case 0: image.at(x, y).set<pixel::red>(255U); break;
            case 1: image.at(x, y).set<pixel::green>(255U); break;
            case 2: image.at(x, y).set<pixel::blue>(255U); break;
            case 3: image.at(x, y) = pixel(convert_pixel_bw<true>(image.at(x, y))); break;
        }
    });
    return 2 * image.width() * image.height();
}
