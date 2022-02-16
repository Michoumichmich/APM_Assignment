#include <cuFilters>

static void run_bench_suite(const std::string& in) {
    auto image = load_cuda_image(in);
    auto split_images = load_cuda_image_into_quadrants(in);
    //size_t size = 2048;
    //auto image = cuda_image(2 * size, 2 * size);
    //auto split_images = std::array{cuda_image(size, size, true), cuda_image(size, size, true), cuda_image(size, size, true), cuda_image(size, size, true)};


    // clang-format off
    benchmark_filter_bundle("BW avoid FP"s, [&](auto &counter) { counter += bw_filter(image, true); });
    benchmark_filter_bundle("BW using FP"s, [&](auto &counter) { counter += bw_filter(image, false); });
    benchmark_filter_bundle("Saturate   "s, [&](auto &counter) { counter += saturate(image, pixel::red); });
    benchmark_filter_bundle("Negative   "s, [&](auto &counter) { counter += negative(image); });
    benchmark_filter_bundle("Mirror V   "s, [&](auto &counter) { counter += mirror_vertical(image); });
    benchmark_filter_bundle("Mirror H   "s, [&](auto &counter) { counter += mirror_horitonzal(image); });
    benchmark_filter_bundle("Median 3   "s, [&](auto &counter) { counter += median_filter(image, 3); });
    benchmark_filter_bundle("Median 5   "s, [&](auto &counter) { counter += median_filter(image, 5); }, 400);
    benchmark_filter_bundle("Median 7   "s, [&](auto &counter) { counter += median_filter(image, 7); }, 100);
    benchmark_filter_bundle("Blur  1    "s, [&](auto &counter) { counter += blur_filter(image, 1); });
    benchmark_filter_bundle("Blur  10   "s, [&](auto &counter) { counter += blur_filter(image, 10); });
    benchmark_filter_bundle("Sobel orig "s, [&](auto &counter) { counter += sobel_filter(image, false); });
    benchmark_filter_bundle("Sobel cust "s, [&](auto &counter) { counter += sobel_filter(image, true); }, 100);
    benchmark_filter_bundle("Popart     "s, [&](auto &counter) { counter += popart_filter(image); });
    // clang-format on

    benchmark_filter_bundle("Popart strm"s, [&](auto& c) {
        c += bw_filter(split_images[0]);
        c += saturate(split_images[1], pixel::red);
        c += saturate(split_images[3], pixel::blue);
        c += saturate(split_images[2], pixel::green);
    });
}


static inline void run_hardcoded_demo(const std::string& filename) {
    /* Question 6 */
    apply_filter_bundle(filename, filename + "_saturate_red.jpg", [](auto& im) {   //
        saturate(im, pixel::red);
    });

    /* Question 7 */
    apply_filter_bundle(filename, filename + "_mirror_horiz.jpg", [](auto& im) {   //
        mirror_horitonzal(im);
    });

    /* Question 8 */
    apply_filter_bundle(filename, filename + "_blur3.jpg", [](auto& im) {   //
        blur_filter(im, 3);
    });

    /* Question 9 */
    apply_filter_bundle(filename, filename + "_bw.jpg", [](auto& im) {   //
        bw_filter(im, true);
    });

    /* Question 10 */
    apply_filter_bundle(filename, filename + "_sobel.jpg", [](auto& im) {   //
        sobel_filter(im);
    });

    /* Question 11 -- Extra filters */
    apply_filter_bundle(filename, filename + "_custom.jpg", [](auto& im) {   //
        negative(im);
        median_filter(im, 3);
        sobel_filter(im, true);
    });

    /* Question 12 */
    apply_filter_bundle(filename, filename + "_popart.jpg", [](auto& im) {   //
        popart_filter(im);
    });

    /* Question 14 */
    apply_filter_bundle_per_quadrant(filename, filename + "_popart_separate_streams.jpg", [](auto& top_left, auto& top_right, auto& bottom_left, auto& bottom_right) {
        saturate(top_right, pixel::red);
        saturate(bottom_right, pixel::blue);
        saturate(top_left, pixel::green);
        bw_filter(bottom_left);
    });

    /* Misc */
    apply_filter_bundle_per_quadrant(filename, filename + "_misc_quadrants.jpg", [](auto& top_left, auto& top_right, auto& bottom_left, auto& bottom_right) {
        saturate(top_right, pixel::red);
        saturate(bottom_right, pixel::blue);
        saturate(top_left, pixel::green);
        bw_filter(bottom_left);

        popart_filter(top_right);
        popart_filter(bottom_right);
        popart_filter(top_left);
        popart_filter(bottom_left);

        mirror_vertical(top_right);
        mirror_horitonzal(bottom_right);
        sobel_filter(top_left, true);
        median_filter(bottom_left, 7);
    });
}


int main(int argc, char** argv) {
    if (argc < 2) {
        std::cerr << "Usage: " << argv[0] << " image.truc  [-benchmark]" << std::endl;
        return 1;
    } else if (argc >= 3) {
        if ("-benchmark"s != argv[2]) { std::cerr << "Usage: ." << argv[0] << " image.truc  [-benchmark]" << std::endl; }
        run_bench_suite(std::string(argv[1]));
        return 0;
    } else {
        run_hardcoded_demo(std::string(argv[1]));
    }
}
