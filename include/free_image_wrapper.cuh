#pragma once

#include "cuda_image.cuh"
#include <array>
#include <string>

/**
 * Loads the image into a cuda_image.
 * @param filename
 * @return
 */
cuda_image load_cuda_image(const std::string& filename);

/**
 * Saves an image into a file.
 * @param im
 * @param filename
 */
void store_cuda_image(cuda_image::image_view im, const std::string& filename);

/**
 * Loads and splits the image into four quadrants/cuda_image, each with its own stream.
 * @param filename
 * @return
 */
std::array<cuda_image, 4> load_cuda_image_into_quadrants(const std::string& filename);

/**
 * Assembled four quadrants and saves the image. Dimensions must match.
 * @param quadrants
 * @param filename
 */
void store_cuda_image_quadrants(std::array<cuda_image, 4>& quadrants, const std::string& filename);
