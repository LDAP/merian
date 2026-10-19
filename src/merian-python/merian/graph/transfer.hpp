#pragma once

// Python.h must precede the standard headers
#include <nanobind/nanobind.h>
#include <nanobind/ndarray.h>

#include "merian-graph/nodes/host_source/host_buffer_source.hpp"
#include "merian-graph/nodes/host_source/host_image_source.hpp"
#include "merian/vk/utils/host_copy.hpp"

namespace merian::python {

namespace nb = nanobind;

using HostArray = nb::ndarray<nb::device::cpu, nb::c_contig>;

// A numpy view of the copy's texels in the order of its format (e.g. BGRA), (height, width,
// channels) or (depth, height, width, channels), (size,) bytes for a buffer.
nb::ndarray<nb::numpy> host_copy_array(const HostCopy& copy);

// Uploads a (height, width) or (height, width, channels) array; three channels get an opaque
// alpha. uint8 data is sRGB-encoded if srgb is set.
void upload_image(HostImageSource& source, const HostArray& array, const bool srgb);

void upload_bytes(HostBufferSource& source, const HostArray& array);

} // namespace merian::python
