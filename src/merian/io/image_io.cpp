#include "merian/io/image_io.hpp"

#include "merian/io/dds.hpp"

#include <stb_image.h>
#include <stb_image_write.h>

#include <algorithm>
#include <cstring>
#include <fstream>
#include <stdexcept>
#include <string>

namespace merian {

namespace {

class StbiBlob : public Blob {
  public:
    StbiBlob(void* data, const std::size_t size_bytes) : data_(data), size_(size_bytes) {}
    StbiBlob(const StbiBlob&) = delete;
    StbiBlob& operator=(const StbiBlob&) = delete;
    StbiBlob(StbiBlob&&) = delete;
    StbiBlob& operator=(StbiBlob&&) = delete;
    ~StbiBlob() override {
        stbi_image_free(data_);
    }

    void* get_data() override {
        return data_;
    }
    std::size_t get_size() override {
        return size_;
    }

  private:
    void* data_;
    std::size_t size_;
};

ImageFormat resolve_format(const std::filesystem::path& path, const ImageFormat requested) {
    if (requested != ImageFormat::AUTO) {
        return requested;
    }
    const ImageFormat ext = image_format_from_extension(path);
    if (ext == ImageFormat::AUTO) {
        throw std::runtime_error{"image_io: cannot infer format from " + path.string()};
    }
    return ext;
}

// PFM: ASCII header, then raw little/big-endian floats, scanlines bottom-up.
BlobHandle load_pfm(const std::filesystem::path& path, ImageInfo& info, int desired_channels) {
    std::ifstream file(path, std::ios::binary);
    if (!file) {
        throw std::runtime_error{"image_io: cannot open " + path.string()};
    }

    std::string magic;
    file >> magic;
    int src_channels = 0;
    if (magic == "PF") {
        src_channels = 3;
    } else if (magic == "Pf") {
        src_channels = 1;
    } else {
        throw std::runtime_error{"image_io: bad PFM magic in " + path.string()};
    }

    int width = 0;
    int height = 0;
    float scale = 0;
    file >> width >> height >> scale;
    file.get(); // skip the single whitespace separating header from pixels
    if (width <= 0 || height <= 0 || !file) {
        throw std::runtime_error{"image_io: malformed PFM header in " + path.string()};
    }
    const bool little_endian = scale < 0;
    const int out_channels = desired_channels == 0 ? src_channels : desired_channels;
    info = {.width = width, .height = height, .channels = out_channels};

    const std::size_t pixel_count = static_cast<std::size_t>(width) * height;
    const std::size_t src_row_floats = static_cast<std::size_t>(width) * src_channels;
    const std::size_t dst_row_floats = static_cast<std::size_t>(width) * out_channels;
    std::vector<float> out(pixel_count * out_channels, 0.f);
    std::vector<float> row(src_row_floats);

    // Read one source row at a time and place it in the flipped destination row, expanding
    // channels in-line. Avoids a full second pass for the bottom-up → top-down flip.
    for (int y_file = 0; y_file < height; ++y_file) {
        file.read(reinterpret_cast<char*>(row.data()),
                  static_cast<std::streamsize>(src_row_floats * sizeof(float)));
        if (!file) {
            throw std::runtime_error{"image_io: truncated PFM in " + path.string()};
        }
        if (!little_endian) {
            for (float& f : row) {
                uint32_t bits = 0;
                std::memcpy(&bits, &f, sizeof(bits));
                bits = ((bits & 0x000000FFu) << 24) | ((bits & 0x0000FF00u) << 8) |
                       ((bits & 0x00FF0000u) >> 8) | ((bits & 0xFF000000u) >> 24);
                std::memcpy(&f, &bits, sizeof(bits));
            }
        }

        const std::size_t dst_y = static_cast<std::size_t>(height - 1 - y_file);
        float* dst = out.data() + (dst_y * dst_row_floats);
        if (out_channels == src_channels) {
            std::memcpy(dst, row.data(), src_row_floats * sizeof(float));
        } else {
            for (int x = 0; x < width; ++x) {
                const float r = row[static_cast<std::size_t>(x) * src_channels];
                const float g = src_channels >= 2
                                    ? row[(static_cast<std::size_t>(x) * src_channels) + 1]
                                    : r;
                const float b = src_channels >= 3
                                    ? row[(static_cast<std::size_t>(x) * src_channels) + 2]
                                    : r;
                const float a = src_channels >= 4
                                    ? row[(static_cast<std::size_t>(x) * src_channels) + 3]
                                    : 1.f;
                float* px = dst + (static_cast<std::size_t>(x) * out_channels);
                if (out_channels >= 1)
                    px[0] = r;
                if (out_channels >= 2)
                    px[1] = g;
                if (out_channels >= 3)
                    px[2] = b;
                if (out_channels >= 4)
                    px[3] = a;
            }
        }
    }
    return std::make_shared<VectorBlob<float>>(std::move(out));
}

void save_pfm(const std::filesystem::path& path,
              const float* data,
              const int width,
              const int height,
              const int channels) {
    if (channels != 1 && channels != 3) {
        throw std::runtime_error{"image_io: PFM supports only 1 or 3 channels"};
    }
    std::ofstream file(path, std::ios::binary);
    if (!file) {
        throw std::runtime_error{"image_io: cannot open " + path.string() + " for write"};
    }
    file << (channels == 3 ? "PF\n" : "Pf\n") << width << ' ' << height << '\n' << -1.0f << '\n';
    // PFM is bottom-up.
    const std::size_t row_floats = static_cast<std::size_t>(width) * channels;
    for (int y = height - 1; y >= 0; --y) {
        file.write(reinterpret_cast<const char*>(data + (static_cast<std::size_t>(y) * row_floats)),
                   static_cast<std::streamsize>(row_floats * sizeof(float)));
    }
}

} // namespace

ImageFormat image_format_from_extension(const std::filesystem::path& path) noexcept {
    std::string ext = path.extension().string();
    std::ranges::transform(ext, ext.begin(),
                           [](const unsigned char c) { return std::tolower(c); });
    if (ext == ".png")
        return ImageFormat::PNG;
    if (ext == ".jpg" || ext == ".jpeg")
        return ImageFormat::JPG;
    if (ext == ".bmp")
        return ImageFormat::BMP;
    if (ext == ".tga")
        return ImageFormat::TGA;
    if (ext == ".hdr")
        return ImageFormat::HDR;
    if (ext == ".pfm")
        return ImageFormat::PFM;
    return ImageFormat::AUTO;
}

BlobHandle image_load_u8(const std::filesystem::path& path,
                         ImageInfo& info,
                         const int desired_channels) {
    // BCn-compressed DDS files are not handled by stb; decode them to RGBA8 here so every consumer
    // (e.g. the Image Read node) can load them. srgb-ness is irrelevant for the raw decoded bytes.
    if (is_dds(path)) {
        return dds_decode_rgba8(dds_load(path, false), info);
    }

    int w = 0;
    int h = 0;
    int native = 0;
    stbi_uc* pixels = stbi_load(path.string().c_str(), &w, &h, &native, desired_channels);
    if (pixels == nullptr) {
        throw std::runtime_error{"image_io: stbi_load failed for " + path.string() + ": " +
                                 stbi_failure_reason()};
    }
    const int out_channels = desired_channels == 0 ? native : desired_channels;
    info = {.width = w, .height = h, .channels = out_channels, .source_channels = native};
    return std::make_shared<StbiBlob>(pixels,
                                      static_cast<std::size_t>(w) * h * out_channels);
}

BlobHandle image_load_f32(const std::filesystem::path& path,
                          ImageInfo& info,
                          const int desired_channels) {
    if (image_format_from_extension(path) == ImageFormat::PFM) {
        return load_pfm(path, info, desired_channels);
    }
    int w = 0;
    int h = 0;
    int native = 0;
    float* pixels = stbi_loadf(path.string().c_str(), &w, &h, &native, desired_channels);
    if (pixels == nullptr) {
        throw std::runtime_error{"image_io: stbi_loadf failed for " + path.string() + ": " +
                                 stbi_failure_reason()};
    }
    const int out_channels = desired_channels == 0 ? native : desired_channels;
    info = {.width = w, .height = h, .channels = out_channels};
    return std::make_shared<StbiBlob>(pixels, static_cast<std::size_t>(w) * h * out_channels *
                                                  sizeof(float));
}

void image_save_u8(const std::filesystem::path& path,
                   const uint8_t* data,
                   const int width,
                   const int height,
                   const int channels,
                   const ImageFormat format) {
    const std::string p = path.string();
    int ok = 0;
    switch (resolve_format(path, format)) {
    case ImageFormat::PNG:
        ok = stbi_write_png(p.c_str(), width, height, channels, data, width * channels);
        break;
    case ImageFormat::JPG:
        ok = stbi_write_jpg(p.c_str(), width, height, channels, data, 95);
        break;
    case ImageFormat::BMP:
        ok = stbi_write_bmp(p.c_str(), width, height, channels, data);
        break;
    case ImageFormat::TGA:
        ok = stbi_write_tga(p.c_str(), width, height, channels, data);
        break;
    case ImageFormat::HDR:
    case ImageFormat::PFM:
        throw std::runtime_error{"image_io: image_save_u8 cannot write HDR/PFM"};
    case ImageFormat::AUTO:
        break;
    }
    if (ok == 0) {
        throw std::runtime_error{"image_io: write failed for " + p};
    }
}

void image_save_f32(const std::filesystem::path& path,
                    const float* data,
                    const int width,
                    const int height,
                    const int channels,
                    const ImageFormat format) {
    switch (resolve_format(path, format)) {
    case ImageFormat::HDR:
        if (stbi_write_hdr(path.string().c_str(), width, height, channels, data) == 0) {
            throw std::runtime_error{"image_io: stbi_write_hdr failed for " + path.string()};
        }
        return;
    case ImageFormat::PFM:
        save_pfm(path, data, width, height, channels);
        return;
    default:
        throw std::runtime_error{"image_io: image_save_f32 only supports HDR/PFM"};
    }
}

} // namespace merian
