#ifndef _MERIAN_SHADERS_IMAGEBUFFER_H_
#define _MERIAN_SHADERS_IMAGEBUFFER_H_

// Defines to store a image in a buffer using a z-Curve.

// power of two
#define image_to_buffer_block_size_power 3 // 2^3 = 8
#define image_to_buffer_block_size (1 << image_to_buffer_block_size_power)
#define image_to_buffer_block_size_minus_one (image_to_buffer_block_size - 1)
// increases the number such that it is divisible by the block size
#define image_to_buffer_dimension_for_block_size(number) (((number) + image_to_buffer_block_size_minus_one) & ~image_to_buffer_block_size_minus_one)
// computes the buffer size (element count) needed to store the image
#define image_to_buffer_size(width, height) (image_to_buffer_dimension_for_block_size(width) * image_to_buffer_dimension_for_block_size(height))
// only valid in C
#define image_to_buffer_size_bytes(width, height, Type) (image_to_buffer_size(width, height) * sizeof(Type))

// z-Curve for better memory locality
#define image_to_buffer_block_index(ipos, resolution) (((ipos).x >> image_to_buffer_block_size_power) + (((resolution).x + image_to_buffer_block_size_minus_one) >> image_to_buffer_block_size_power) * ((ipos).y >> image_to_buffer_block_size_power))
#define image_to_buffer_inner_index(ipos) (((ipos).x & image_to_buffer_block_size_minus_one) + image_to_buffer_block_size * ((ipos).y & image_to_buffer_block_size_minus_one))
#define image_to_buffer_index(ipos, resolution) (image_to_buffer_inner_index(ipos) + image_to_buffer_block_index(ipos, resolution) * image_to_buffer_block_size * image_to_buffer_block_size)

#endif
