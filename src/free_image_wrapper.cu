#include <FreeImage.h>
#include <array>
#include <cuda_image.cuh>
#include <free_image_wrapper.cuh>
#include <internal/cuda_err_check.cuh>
#include <iostream>
#include <stdexcept>

cuda_image load_cuda_image(const std::string& filename) {
    // load and decode a regular file
    FREE_IMAGE_FORMAT fif = FreeImage_GetFileType(filename.c_str());
    FIBITMAP* bitmap = FreeImage_Load(fif, filename.c_str(), RAW_DISPLAY);
    if (!bitmap) { throw(std::runtime_error("Cannot allocate image or image not found")); }
    unsigned width = FreeImage_GetWidth(bitmap);
    unsigned height = FreeImage_GetHeight(bitmap);
    unsigned pitch = FreeImage_GetPitch(bitmap);
    std::cerr << "Processing Image of size: " << width << " x " << height << std::endl;
    auto img = new pixel[width * height];
    BYTE* bits = (BYTE*) FreeImage_GetBits(bitmap);
    for (int y = 0; y < height; y++) {
        BYTE* pixel = (BYTE*) bits;
        for (int x = 0; x < width; x++) {
            int idx = ((y * width) + x);
            img[idx].set<pixel::red>(pixel[FI_RGBA_RED]);
            img[idx].set<pixel::green>(pixel[FI_RGBA_GREEN]);
            img[idx].set<pixel::blue>(pixel[FI_RGBA_BLUE]);
            pixel += 3;
        }
        // next line
        bits += pitch;
    }

    auto image = cuda_image(width, height);
    CUDA_CHECK(cudaMemcpy(image.view.raw_data(), img, sizeof(pixel) * width * height, cudaMemcpyHostToDevice))
    cudaStreamSynchronize(0);
    delete[](img);
    FreeImage_Unload(bitmap);
    CHECK_LAST_ERROR_ASYNC
    return image;
}


void store_cuda_image(cuda_image::image_view im, const std::string& filename) {
    auto img = new pixel[im.width() * im.height()];
    CUDA_CHECK(cudaMemcpy(img, im.raw_data(), sizeof(pixel) * im.width() * im.height(), cudaMemcpyDeviceToHost))
    auto bitmap = FreeImage_Allocate(im.width(), im.height(), 24);
    CUDA_CHECK(cudaStreamSynchronize(0))

    for (int y = 0; y < im.height(); y++) {
        for (int x = 0; x < im.width(); x++) {
            RGBQUAD newcolor;
            int idx = ((y * im.width()) + x);
            newcolor.rgbRed = img[idx].get<pixel::red>();
            newcolor.rgbGreen = img[idx].get<pixel::green>();
            newcolor.rgbBlue = img[idx].get<pixel::blue>();
            newcolor.rgbReserved = img[idx].get<pixel::alpha>();
            if (!FreeImage_SetPixelColor(bitmap, x, y, &newcolor)) { fprintf(stderr, "(%d, %d) Fail...\n", x, y); }
        }
        // next line
    }

    if (FreeImage_Save(FreeImage_GetFIFFromFilename(filename.c_str()), bitmap, filename.c_str(), 0)) {
        std::cout << "Image successfully saved ! " << std::endl;
    } else {
        std::cerr << "Cannot write output image to " << filename << std::endl;
    }
    FreeImage_Unload(bitmap);   //Cleanup !
    delete[](img);
    CHECK_LAST_ERROR_ASYNC
}

std::array<cuda_image, 4> load_cuda_image_into_quadrants(const std::string& filename) {
    // load and decode a regular file
    FREE_IMAGE_FORMAT fif = FreeImage_GetFileType(filename.c_str());
    FIBITMAP* bitmap = FreeImage_Load(fif, filename.c_str(), 0);
    if (!bitmap) { throw(std::runtime_error("Cannot allocate image or image not found")); }
    unsigned width = FreeImage_GetWidth(bitmap);
    unsigned height = FreeImage_GetHeight(bitmap);
    unsigned pitch = FreeImage_GetPitch(bitmap);
    std::cerr << "Processing Image of size: " << width << " x " << height << std::endl;

    std::array<cuda_image, 4> quadrants{
            cuda_image(width / 2, height / 2, true),               // Quadrant top left
            cuda_image((width + 1) / 2, height / 2, true),         // Quadrant top right
            cuda_image(width / 2, (height + 1) / 2, true),         // Quadrant bottom left
            cuda_image((width + 1) / 2, (height + 1) / 2, true),   // Quadrant bottom right
    };

    std::array<pixel*, 4> img_bufs;
    for (int dim = 0; dim < 4; ++dim) { img_bufs[dim] = new pixel[quadrants[dim].width() * quadrants[dim].height()]; }


    BYTE* bits = (BYTE*) FreeImage_GetBits(bitmap);
    std::array<int, 4> img_idxs{};
    for (int y = 0; y < height; y++) {
        BYTE* pixel = (BYTE*) bits;
        for (int x = 0; x < width; x++) {
            int quadrant_id = 0;                        // 2D linear id kind of
            if (x < width / 2) { quadrant_id += 1; }    // first dim
            if (y < height / 2) { quadrant_id += 2; }   // second dim
            img_bufs[quadrant_id][img_idxs[quadrant_id]].set<pixel::red>(pixel[FI_RGBA_RED]);
            img_bufs[quadrant_id][img_idxs[quadrant_id]].set<pixel::green>(pixel[FI_RGBA_GREEN]);
            img_bufs[quadrant_id][img_idxs[quadrant_id]].set<pixel::blue>(pixel[FI_RGBA_BLUE]);
            img_idxs[quadrant_id]++;
            pixel += 3;
        }
        bits += pitch;
    }

    for (int dim = 0; dim < 4; ++dim) {
        CUDA_CHECK(cudaMemcpyAsync(quadrants[dim].view.raw_data(), img_bufs[dim], sizeof(pixel) * quadrants[dim].width() * quadrants[dim].height(),   //
                                   cudaMemcpyHostToDevice, quadrants[dim].get_stream()))
    }

    for (int dim = 0; dim < 4; ++dim) {
        CUDA_CHECK(cudaStreamSynchronize(quadrants[dim].get_stream()))
        delete[](img_bufs[dim]);
    }

    FreeImage_Unload(bitmap);
    return quadrants;
}

void store_cuda_image_quadrants(std::array<cuda_image, 4>& quadrants, const std::string& filename) {
    int width = quadrants[0].width() + quadrants[1].width();
    int height = quadrants[0].height() + quadrants[2].height();
    std::array<pixel*, 4> img_bufs;
    for (int dim = 0; dim < 4; ++dim) { img_bufs[dim] = new pixel[quadrants[dim].width() * quadrants[dim].height()]; }

    for (int dim = 0; dim < 4; ++dim) {
        CUDA_CHECK(cudaMemcpyAsync(img_bufs[dim], quadrants[dim].view.raw_data(), sizeof(pixel) * quadrants[dim].width() * quadrants[dim].height(),   //
                                   cudaMemcpyDeviceToHost, quadrants[dim].get_stream()))
    }
    auto bitmap = FreeImage_Allocate(width, height, 24);
    for (int dim = 0; dim < 4; ++dim) { CUDA_CHECK(cudaStreamSynchronize(quadrants[dim].get_stream())) }
    std::array<int, 4> linear_ids{};

    for (int y = 0; y < height; y++) {
        for (int x = 0; x < width; x++) {
            int quadrant_id = 0;                        // 2D linear id kind of
            if (x < width / 2) { quadrant_id += 1; }    // first dim
            if (y < height / 2) { quadrant_id += 2; }   // second dim
            RGBQUAD newcolor;
            auto px = img_bufs[quadrant_id][linear_ids[quadrant_id]++];
            newcolor.rgbRed = px.get<pixel::red>();
            newcolor.rgbGreen = px.get<pixel::green>();
            newcolor.rgbBlue = px.get<pixel::blue>();
            newcolor.rgbReserved = px.get<pixel::alpha>();
            if (!FreeImage_SetPixelColor(bitmap, x, y, &newcolor)) { fprintf(stderr, "(%d, %d) Fail...\n", x, y); }
        }
    }

    if (FreeImage_Save(FreeImage_GetFIFFromFilename(filename.c_str()), bitmap, filename.c_str(), 0)) {
        std::cout << "Image successfully saved ! " << std::endl;
    } else {
        std::cerr << "Cannot write output image to " << filename << std::endl;
    }

    FreeImage_Unload(bitmap);   //Cleanup !
    for (auto img_buf: img_bufs) { delete[](img_buf); }
    CHECK_LAST_ERROR_ASYNC
}