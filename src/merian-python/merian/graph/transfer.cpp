#include "transfer.hpp"

#include <array>
#include <cstring>
#include <string_view>
#include <vulkan/vulkan_format_traits.hpp>

namespace merian::python {

namespace {

using DtypeCode = nb::dlpack::dtype_code;

nb::dlpack::dtype make_dtype(const DtypeCode code, const uint8_t bits) {
    return {static_cast<uint8_t>(code), bits, 1};
}

nb::dlpack::dtype texel_dtype(const vk::Format format) {
    if (vk::componentsAreCompressed(format) || vk::packed(format) != 0) {
        throw std::invalid_argument(
            fmt::format("{} has no array representation", vk::to_string(format)));
    }
    const uint8_t bits = vk::componentBits(format, 0);
    for (uint8_t component = 1; component < vk::componentCount(format); component++) {
        if (vk::componentBits(format, component) != bits) {
            throw std::invalid_argument(
                fmt::format("{} has no array representation", vk::to_string(format)));
        }
    }
    const std::string_view numeric = vk::componentNumericFormat(format, 0);
    if (numeric == "SFLOAT") {
        return make_dtype(DtypeCode::Float, bits);
    }
    if (numeric == "SNORM" || numeric == "SINT" || numeric == "SSCALED") {
        return make_dtype(DtypeCode::Int, bits);
    }
    return make_dtype(DtypeCode::UInt, bits);
}

// R, RG, RGB (unused, padded to RGBA) and RGBA formats per array element type.
struct UploadFormats {
    nb::dlpack::dtype dtype;
    std::array<vk::Format, 4> formats;
    std::array<vk::Format, 4> srgb_formats;
};

const std::array<UploadFormats, 6>& upload_formats() {
    using F = vk::Format;
    static const std::array<UploadFormats, 6> formats = {{
        {make_dtype(DtypeCode::Float, 32),
         {F::eR32Sfloat, F::eR32G32Sfloat, F::eUndefined, F::eR32G32B32A32Sfloat},
         {}},
        {make_dtype(DtypeCode::Float, 16),
         {F::eR16Sfloat, F::eR16G16Sfloat, F::eUndefined, F::eR16G16B16A16Sfloat},
         {}},
        {make_dtype(DtypeCode::UInt, 8),
         {F::eR8Unorm, F::eR8G8Unorm, F::eUndefined, F::eR8G8B8A8Unorm},
         {F::eR8Srgb, F::eR8G8Srgb, F::eUndefined, F::eR8G8B8A8Srgb}},
        {make_dtype(DtypeCode::UInt, 16),
         {F::eR16Unorm, F::eR16G16Unorm, F::eUndefined, F::eR16G16B16A16Unorm},
         {}},
        {make_dtype(DtypeCode::UInt, 32),
         {F::eR32Uint, F::eR32G32Uint, F::eUndefined, F::eR32G32B32A32Uint},
         {}},
        {make_dtype(DtypeCode::Int, 32),
         {F::eR32Sint, F::eR32G32Sint, F::eUndefined, F::eR32G32B32A32Sint},
         {}},
    }};
    return formats;
}

template <typename T>
void pad_to_rgba(const void* rgb, void* rgba, const size_t texel_count, T alpha) {
    const auto* src = static_cast<const T*>(rgb);
    auto* dst = static_cast<T*>(rgba);
    for (size_t texel = 0; texel < texel_count; texel++, src += 3, dst += 4) {
        dst[0] = src[0];
        dst[1] = src[1];
        dst[2] = src[2];
        dst[3] = alpha;
    }
}

void pad_to_rgba(const void* rgb,
                 void* rgba,
                 const size_t texel_count,
                 const nb::dlpack::dtype dtype) {
    if (dtype == make_dtype(DtypeCode::Float, 32)) {
        pad_to_rgba<float>(rgb, rgba, texel_count, 1.f);
    } else if (dtype == make_dtype(DtypeCode::Float, 16)) {
        pad_to_rgba<uint16_t>(rgb, rgba, texel_count, 0x3C00);
    } else if (dtype == make_dtype(DtypeCode::UInt, 8)) {
        pad_to_rgba<uint8_t>(rgb, rgba, texel_count, 0xFF);
    } else {
        pad_to_rgba<uint16_t>(rgb, rgba, texel_count, 0xFFFF);
    }
}

} // namespace

nb::ndarray<nb::numpy> host_copy_array(const HostCopy& copy) {
    const bool bytes = copy.format == vk::Format::eUndefined;
    const nb::dlpack::dtype dtype =
        bytes ? make_dtype(DtypeCode::UInt, 8) : texel_dtype(copy.format);

    void* const data = copy.memory->map();
    copy.memory->invalidate();
    auto* const owner_copy = new HostCopy(copy);
    const nb::capsule owner(owner_copy, [](void* pointer) noexcept {
        auto* const owned = static_cast<HostCopy*>(pointer);
        owned->memory->unmap();
        delete owned;
    });

    if (bytes) {
        const std::array<size_t, 1> shape = {copy.size};
        return {data, shape.size(), shape.data(), owner, nullptr, dtype};
    }
    const size_t channels = vk::componentCount(copy.format);
    const vk::Extent3D& extent = copy.extent;
    if (extent.depth > 1) {
        const std::array<size_t, 4> shape = {extent.depth, extent.height, extent.width, channels};
        return {data, shape.size(), shape.data(), owner, nullptr, dtype};
    }
    const std::array<size_t, 3> shape = {extent.height, extent.width, channels};
    return {data, shape.size(), shape.data(), owner, nullptr, dtype};
}

void upload_image(HostImageSource& source, const HostArray& array, const bool srgb) {
    if (array.ndim() != 2 && array.ndim() != 3) {
        throw std::invalid_argument("expected an array of shape (height, width[, channels])");
    }
    const size_t channels = array.ndim() == 3 ? array.shape(2) : 1;
    if (channels < 1 || channels > 4) {
        throw std::invalid_argument("expected 1 to 4 channels");
    }
    const auto entry = std::ranges::find(upload_formats(), array.dtype(), &UploadFormats::dtype);
    if (entry == upload_formats().end()) {
        throw std::invalid_argument(
            "expected float32, float16, uint8, uint16, uint32 or int32 elements");
    }
    const size_t padded_channels = channels == 3 ? 4 : channels;
    const bool use_srgb = srgb && entry->srgb_formats[0] != vk::Format::eUndefined;
    if (srgb && !use_srgb) {
        throw std::invalid_argument("srgb needs uint8 elements");
    }
    if (channels == 3 && array.dtype().code != static_cast<uint8_t>(DtypeCode::Float) &&
        array.dtype().bits > 16) {
        throw std::invalid_argument("three channels need float or normalized elements");
    }
    const vk::Format format =
        (use_srgb ? entry->srgb_formats : entry->formats)[padded_channels - 1];
    const vk::Extent3D extent{static_cast<uint32_t>(array.shape(1)),
                              static_cast<uint32_t>(array.shape(0)), 1};

    const nb::gil_scoped_release release;
    source.upload(format, extent, [&](void* texels) {
        if (channels == 3) {
            pad_to_rgba(array.data(), texels, static_cast<size_t>(extent.width) * extent.height,
                        array.dtype());
        } else {
            std::memcpy(texels, array.data(), array.nbytes());
        }
    });
}

void upload_bytes(HostBufferSource& source, const HostArray& array) {
    const nb::gil_scoped_release release;
    source.upload(array.nbytes(),
                  [&](void* data) { std::memcpy(data, array.data(), array.nbytes()); });
}

} // namespace merian::python
