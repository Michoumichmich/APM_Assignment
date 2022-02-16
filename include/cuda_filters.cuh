#pragma once

#include "cuda_image.cuh"

/**
 * Saturaes a component of the image.
 * @param image
 * @param composant
 * @return
 */
size_t saturate(cuda_image& image, pixel::composant composant);

/**
 * Diapositive.
 * @param image
 * @return
 */
size_t negative(cuda_image& image);

/**
 * Flips the image around the vertical axis.
 * @param image
 * @return
 */
size_t mirror_vertical(cuda_image& image);

/**
 * Flips the image around the horizontal axis.
 * @param image
 * @return
 */
size_t mirror_horitonzal(cuda_image& image);

/**
 * Median filter
 * @param image might point to reallocated memory.
 * @param width width of the box.
 * @return
 */
size_t median_filter(cuda_image& image, int width);

/**
 * Black and white filter
 * @param image
 * @param avoid_fp_use algorithm that do not use floating point numbers.
 * @return
 */
size_t bw_filter(cuda_image& image, bool avoid_fp_use = true);

/**
 * Blur filter
 * @param image might point to reallocated memory.
 * @param passes number of times to run the filter
 * @return
 */
size_t blur_filter(cuda_image& image, int passes = 1);

/**
 * Sobel filter
 * @param image might point to reallocated memory.
 * @param use_custom_algo if set to true, then runs a modified version that maps the angles to the colors + other filters. Default false.
 * @return
 */
size_t sobel_filter(cuda_image& image, bool use_custom_algo = false);

/**
 * Pop-art?
 * @param image
 * @return
 */
size_t popart_filter(cuda_image& image);
