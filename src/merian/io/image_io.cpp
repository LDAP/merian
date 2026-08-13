#include "merian/io/image_io.hpp"

#include "merian/io/dds.hpp"

#include <fmt/format.h>
#include <spdlog/spdlog.h>
#include <stb_image.h>
#include <stb_image_write.h>

#include <algorithm>
#include <cstring>
#include <fstream>
#include <stdexcept>
#include <string>
#include <vector>

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
                const float g =
                    src_channels >= 2 ? row[(static_cast<std::size_t>(x) * src_channels) + 1] : r;
                const float b =
                    src_channels >= 3 ? row[(static_cast<std::size_t>(x) * src_channels) + 2] : r;
                const float a =
                    src_channels >= 4 ? row[(static_cast<std::size_t>(x) * src_channels) + 3] : 1.f;
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

void append_be32(std::vector<uint8_t>& out, const uint32_t v) {
    out.push_back(static_cast<uint8_t>(v >> 24));
    out.push_back(static_cast<uint8_t>(v >> 16));
    out.push_back(static_cast<uint8_t>(v >> 8));
    out.push_back(static_cast<uint8_t>(v));
}

uint32_t crc32(const uint8_t* data, const std::size_t size) {
    uint32_t crc = 0xFFFFFFFFu;
    for (std::size_t i = 0; i < size; i++) {
        crc ^= data[i];
        for (int k = 0; k < 8; k++) {
            crc = (crc >> 1) ^ (0xEDB88320u & (0u - (crc & 1u)));
        }
    }
    return ~crc;
}

// A tEXt keyword is 1 to 79 printable latin-1 bytes without leading, trailing or repeated spaces.
std::string png_keyword(const std::string& key) {
    std::string keyword;
    for (const unsigned char c : key) {
        const bool printable = (c >= 32 && c <= 126) || (c >= 161 && c <= 255);
        if (!printable || (c == ' ' && (keyword.empty() || keyword.back() == ' '))) {
            continue;
        }
        keyword += static_cast<char>(c);
        if (keyword.size() == 79) {
            break;
        }
    }
    while (!keyword.empty() && keyword.back() == ' ') {
        keyword.pop_back();
    }
    return keyword;
}

// tEXt chunks inserted directly after IHDR (always the fixed 25-byte chunk after the signature)
void png_insert_metadata(std::vector<uint8_t>& file, const ImageMetadata& metadata) {
    std::vector<uint8_t> chunks;
    for (const auto& [key, value] : metadata) {
        const std::string keyword = png_keyword(key);
        if (keyword.empty()) {
            SPDLOG_WARN("image_io: dropping metadata with an unusable keyword '{}'", key);
            continue;
        }
        append_be32(chunks, static_cast<uint32_t>(keyword.size() + 1 + value.size()));
        const std::size_t crc_begin = chunks.size();
        chunks.insert(chunks.end(), {'t', 'E', 'X', 't'});
        chunks.insert(chunks.end(), keyword.begin(), keyword.end());
        chunks.push_back(0);
        chunks.insert(chunks.end(), value.begin(), value.end());
        append_be32(chunks, crc32(chunks.data() + crc_begin, chunks.size() - crc_begin));
    }
    file.insert(file.begin() + 8 + 25, chunks.begin(), chunks.end());
}

// COM markers after SOI and the APPn segments (JFIF requires APP0 directly after SOI);
// oversized values split across markers (payload limit 65533 bytes)
void jpg_insert_metadata(std::vector<uint8_t>& file, const ImageMetadata& metadata) {
    constexpr std::size_t MAX_PAYLOAD = 65533;
    std::vector<uint8_t> markers;
    for (const auto& [key, value] : metadata) {
        // every segment repeats the key, so a value longer than one marker stays attributable
        const std::size_t segments = 1 + (key.size() + 2 + value.size()) / MAX_PAYLOAD;
        std::size_t segment = 0;
        for (std::size_t at = 0; at < value.size() || segment == 0; segment++) {
            std::string payload = key;
            if (segments > 1) {
                payload += fmt::format("({}/{})", segment + 1, segments);
            }
            payload += ": ";
            const std::size_t take = std::min(MAX_PAYLOAD - payload.size(), value.size() - at);
            payload += value.substr(at, take);
            at += take;
            const std::size_t n = payload.size();
            markers.push_back(0xFF);
            markers.push_back(0xFE);
            markers.push_back(static_cast<uint8_t>((n + 2) >> 8));
            markers.push_back(static_cast<uint8_t>((n + 2) & 0xFF));
            markers.insert(markers.end(), payload.begin(), payload.end());
        }
    }
    std::size_t pos = 2;
    while (pos + 4 <= file.size() && file[pos] == 0xFF && (file[pos + 1] & 0xF0) == 0xE0) {
        pos += 2 + ((static_cast<std::size_t>(file[pos + 2]) << 8) | file[pos + 3]);
    }
    file.insert(file.begin() + static_cast<std::ptrdiff_t>(pos), markers.begin(), markers.end());
}

// KEY=value lines inserted before the blank line that terminates the Radiance header
void hdr_insert_metadata(std::vector<uint8_t>& file, const ImageMetadata& metadata) {
    std::string vars;
    for (const auto& [key, value] : metadata) {
        std::string line = key + "=" + value;
        std::ranges::replace(line, '\n', ' ');
        vars += line + '\n';
    }
    for (std::size_t i = 0; i + 1 < std::min<std::size_t>(file.size(), 512); i++) {
        if (file[i] == '\n' && file[i + 1] == '\n') {
            file.insert(file.begin() + static_cast<std::ptrdiff_t>(i + 1), vars.begin(),
                        vars.end());
            return;
        }
    }
}

void stbi_vector_write(void* context, void* data, int size) {
    auto* const out = static_cast<std::vector<uint8_t>*>(context);
    const uint8_t* const bytes = static_cast<const uint8_t*>(data);
    out->insert(out->end(), bytes, bytes + size);
}

void write_file(const std::filesystem::path& path, const std::vector<uint8_t>& data) {
    std::ofstream file(path, std::ios::binary);
    if (!file) {
        throw std::runtime_error{"image_io: cannot open " + path.string() + " for write"};
    }
    file.write(reinterpret_cast<const char*>(data.data()),
               static_cast<std::streamsize>(data.size()));
    if (!file) {
        throw std::runtime_error{"image_io: write failed for " + path.string()};
    }
}

void save_pfm(const std::filesystem::path& path,
              const float* data,
              const int width,
              const int height,
              const int channels) {
    if (channels != 1 && channels != 3 && channels != 4) {
        throw std::runtime_error{"image_io: PFM supports only 1, 3 or 4 channels"};
    }
    std::ofstream file(path, std::ios::binary);
    if (!file) {
        throw std::runtime_error{"image_io: cannot open " + path.string() + " for write"};
    }
    // PFM has no alpha channel, RGBA is written as RGB.
    const int out_channels = channels == 1 ? 1 : 3;
    file << (out_channels == 3 ? "PF\n" : "Pf\n") << width << ' ' << height << '\n'
         << -1.0f << '\n';
    // PFM is bottom-up.
    const std::size_t row_floats = static_cast<std::size_t>(width) * channels;
    if (channels == out_channels) {
        for (int y = height - 1; y >= 0; --y) {
            file.write(
                reinterpret_cast<const char*>(data + (static_cast<std::size_t>(y) * row_floats)),
                static_cast<std::streamsize>(row_floats * sizeof(float)));
        }
        return;
    }
    std::vector<float> row(static_cast<std::size_t>(width) * out_channels);
    for (int y = height - 1; y >= 0; --y) {
        const float* src = data + (static_cast<std::size_t>(y) * row_floats);
        for (int x = 0; x < width; ++x) {
            std::copy_n(src + (static_cast<std::size_t>(x) * channels), out_channels,
                        row.data() + (static_cast<std::size_t>(x) * out_channels));
        }
        file.write(reinterpret_cast<const char*>(row.data()),
                   static_cast<std::streamsize>(row.size() * sizeof(float)));
    }
}

} // namespace

const char* image_format_extension(const ImageFormat format) noexcept {
    switch (format) {
    case ImageFormat::PNG:
        return ".png";
    case ImageFormat::JPG:
        return ".jpg";
    case ImageFormat::BMP:
        return ".bmp";
    case ImageFormat::TGA:
        return ".tga";
    case ImageFormat::HDR:
        return ".hdr";
    case ImageFormat::PFM:
        return ".pfm";
    case ImageFormat::AUTO:
        break;
    }
    return "";
}

ImageFormat image_format_from_extension(const std::filesystem::path& path) noexcept {
    std::string ext = path.extension().string();
    std::ranges::transform(ext, ext.begin(), [](const unsigned char c) { return std::tolower(c); });
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

BlobHandle
image_load_u8(const std::filesystem::path& path, ImageInfo& info, const int desired_channels) {
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
    return std::make_shared<StbiBlob>(pixels, static_cast<std::size_t>(w) * h * out_channels);
}

BlobHandle
image_load_f32(const std::filesystem::path& path, ImageInfo& info, const int desired_channels) {
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
                   const ImageFormat format,
                   const ImageMetadata& metadata) {
    const ImageFormat resolved = resolve_format(path, format);
    std::vector<uint8_t> file;
    int ok = 0;
    switch (resolved) {
    case ImageFormat::PNG:
        ok = stbi_write_png_to_func(stbi_vector_write, &file, width, height, channels, data,
                                    width * channels);
        break;
    case ImageFormat::JPG:
        ok = stbi_write_jpg_to_func(stbi_vector_write, &file, width, height, channels, data, 95);
        break;
    case ImageFormat::BMP:
        ok = stbi_write_bmp_to_func(stbi_vector_write, &file, width, height, channels, data);
        break;
    case ImageFormat::TGA:
        ok = stbi_write_tga_to_func(stbi_vector_write, &file, width, height, channels, data);
        break;
    case ImageFormat::HDR:
    case ImageFormat::PFM:
        throw std::runtime_error{"image_io: image_save_u8 cannot write HDR/PFM"};
    case ImageFormat::AUTO:
        break;
    }
    if (ok == 0) {
        throw std::runtime_error{"image_io: encode failed for " + path.string()};
    }
    if (!metadata.empty()) {
        if (resolved == ImageFormat::PNG) {
            png_insert_metadata(file, metadata);
        } else if (resolved == ImageFormat::JPG) {
            jpg_insert_metadata(file, metadata);
        }
    }
    write_file(path, file);
}

void image_save_f32(const std::filesystem::path& path,
                    const float* data,
                    const int width,
                    const int height,
                    const int channels,
                    const ImageFormat format,
                    const ImageMetadata& metadata) {
    switch (resolve_format(path, format)) {
    case ImageFormat::HDR: {
        std::vector<uint8_t> file;
        if (stbi_write_hdr_to_func(stbi_vector_write, &file, width, height, channels, data) == 0) {
            throw std::runtime_error{"image_io: stbi_write_hdr failed for " + path.string()};
        }
        if (!metadata.empty()) {
            hdr_insert_metadata(file, metadata);
        }
        write_file(path, file);
        return;
    }
    case ImageFormat::PFM:
        save_pfm(path, data, width, height, channels);
        return;
    default:
        throw std::runtime_error{"image_io: image_save_f32 only supports HDR/PFM"};
    }
}

} // namespace merian
