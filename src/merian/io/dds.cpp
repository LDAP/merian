#include "merian/io/dds.hpp"

#include <algorithm>
#include <array>
#include <cctype>
#include <cstring>
#include <fstream>

namespace merian {

namespace {

constexpr uint32_t make_fourcc(const char a, const char b, const char c, const char d) {
    return static_cast<uint32_t>(static_cast<uint8_t>(a)) |
           (static_cast<uint32_t>(static_cast<uint8_t>(b)) << 8) |
           (static_cast<uint32_t>(static_cast<uint8_t>(c)) << 16) |
           (static_cast<uint32_t>(static_cast<uint8_t>(d)) << 24);
}

enum DxgiFormat : uint32_t {
    DXGI_BC1_UNORM = 71,
    DXGI_BC1_UNORM_SRGB = 72,
    DXGI_BC2_UNORM = 74,
    DXGI_BC2_UNORM_SRGB = 75,
    DXGI_BC3_UNORM = 77,
    DXGI_BC3_UNORM_SRGB = 78,
    DXGI_BC4_UNORM = 80,
    DXGI_BC5_UNORM = 83,
    DXGI_BC7_UNORM = 98,
    DXGI_BC7_UNORM_SRGB = 99,
};

uint32_t block_bytes_for(const vk::Format format) {
    switch (format) {
    case vk::Format::eBc1RgbaUnormBlock:
    case vk::Format::eBc1RgbaSrgbBlock:
    case vk::Format::eBc4UnormBlock:
        return 8;
    default:
        return 16;
    }
}

// --- BCn block decoders (one 4x4 block -> 16 RGBA8 texels) ---

void decode_565(const uint16_t c, std::array<uint8_t, 4>& out) {
    const uint8_t r = (c >> 11) & 0x1f;
    const uint8_t g = (c >> 5) & 0x3f;
    const uint8_t b = c & 0x1f;
    out = {static_cast<uint8_t>((r << 3) | (r >> 2)), static_cast<uint8_t>((g << 2) | (g >> 4)),
           static_cast<uint8_t>((b << 3) | (b >> 2)), 255};
}

// Decodes the BC1 color portion (8 bytes). `opaque` forces the 4-colour interpolation mode (used by
// BC2/BC3 where the 1-bit-alpha mode does not apply).
void decode_color_block(const uint8_t* src,
                        const bool opaque,
                        std::array<std::array<uint8_t, 4>, 16>& out) {
    const uint16_t c0 = static_cast<uint16_t>(src[0] | (src[1] << 8));
    const uint16_t c1 = static_cast<uint16_t>(src[2] | (src[3] << 8));
    std::array<std::array<uint8_t, 4>, 4> color;
    decode_565(c0, color[0]);
    decode_565(c1, color[1]);
    if (opaque || c0 > c1) {
        for (int i = 0; i < 3; i++) {
            color[2][i] = static_cast<uint8_t>((2 * color[0][i] + color[1][i]) / 3);
            color[3][i] = static_cast<uint8_t>((color[0][i] + 2 * color[1][i]) / 3);
        }
        color[2][3] = color[3][3] = 255;
    } else {
        for (int i = 0; i < 3; i++) {
            color[2][i] = static_cast<uint8_t>((color[0][i] + color[1][i]) / 2);
        }
        color[2][3] = 255;
        color[3] = {0, 0, 0, 0};
    }
    const uint32_t bits =
        src[4] | (src[5] << 8) | (src[6] << 16) | (static_cast<uint32_t>(src[7]) << 24);
    for (int i = 0; i < 16; i++) {
        out[i] = color[(bits >> (2 * i)) & 0x3];
    }
}

// True for formats that carry an alpha channel. DXT1/BC1 is treated as opaque (the common
// convention — cutouts ship as DXT5/BC3/BC7); BC4/BC5 are data, not color.
bool format_has_alpha(const vk::Format format) {
    switch (format) {
    case vk::Format::eBc2UnormBlock:
    case vk::Format::eBc2SrgbBlock:
    case vk::Format::eBc3UnormBlock:
    case vk::Format::eBc3SrgbBlock:
    case vk::Format::eBc7UnormBlock:
    case vk::Format::eBc7SrgbBlock:
        return true;
    default:
        return false;
    }
}

// Decodes a BC4-style 8-byte block to 16 scalar values.
void decode_alpha_block(const uint8_t* src, std::array<uint8_t, 16>& out) {
    std::array<uint8_t, 8> a{};
    a[0] = src[0];
    a[1] = src[1];
    if (a[0] > a[1]) {
        for (int i = 1; i <= 6; i++) {
            a[i + 1] = static_cast<uint8_t>(((7 - i) * a[0] + i * a[1]) / 7);
        }
    } else {
        for (int i = 1; i <= 4; i++) {
            a[i + 1] = static_cast<uint8_t>(((5 - i) * a[0] + i * a[1]) / 5);
        }
        a[6] = 0;
        a[7] = 255;
    }
    uint64_t bits = 0;
    for (int i = 0; i < 6; i++) {
        bits |= static_cast<uint64_t>(src[2 + i]) << (8 * i);
    }
    for (int i = 0; i < 16; i++) {
        out[i] = a[(bits >> (3 * i)) & 0x7];
    }
}

} // namespace

bool is_dds(const std::filesystem::path& path) {
    std::string ext = path.extension().string();
    std::transform(ext.begin(), ext.end(), ext.begin(),
                   [](const unsigned char c) { return static_cast<char>(std::tolower(c)); });
    return ext == ".dds";
}

DdsImage dds_load(const std::filesystem::path& path, const bool srgb) {
    std::ifstream file(path, std::ios::binary | std::ios::ate);
    if (!file) {
        throw std::runtime_error{"dds: cannot open " + path.string()};
    }
    const auto file_size = static_cast<size_t>(file.tellg());
    file.seekg(0);

    // Magic (4) + DDS_HEADER (124) = 128; a DX10 header adds 20.
    uint8_t header[148];
    const size_t header_size = std::min(file_size, sizeof(header));
    file.read(reinterpret_cast<char*>(header), static_cast<std::streamsize>(header_size));

    const auto u32 = [&](const size_t off) {
        uint32_t v = 0;
        std::memcpy(&v, header + off, 4);
        return v;
    };
    if (header_size < 128 || u32(0) != make_fourcc('D', 'D', 'S', ' ')) {
        throw std::runtime_error{"dds: not a DDS file " + path.string()};
    }

    DdsImage dds;
    dds.height = u32(12);
    dds.width = u32(16);
    dds.mip_levels = std::max(1u, u32(28));
    const uint32_t four_cc = u32(84);
    size_t data_offset = 128;

    if (four_cc == make_fourcc('D', 'X', 'T', '1')) {
        dds.format = srgb ? vk::Format::eBc1RgbaSrgbBlock : vk::Format::eBc1RgbaUnormBlock;
    } else if (four_cc == make_fourcc('D', 'X', 'T', '3')) {
        dds.format = srgb ? vk::Format::eBc2SrgbBlock : vk::Format::eBc2UnormBlock;
    } else if (four_cc == make_fourcc('D', 'X', 'T', '5')) {
        dds.format = srgb ? vk::Format::eBc3SrgbBlock : vk::Format::eBc3UnormBlock;
    } else if (four_cc == make_fourcc('A', 'T', 'I', '1') ||
               four_cc == make_fourcc('B', 'C', '4', 'U')) {
        dds.format = vk::Format::eBc4UnormBlock;
    } else if (four_cc == make_fourcc('A', 'T', 'I', '2') ||
               four_cc == make_fourcc('B', 'C', '5', 'U')) {
        dds.format = vk::Format::eBc5UnormBlock;
    } else if (four_cc == make_fourcc('D', 'X', '1', '0')) {
        if (header_size < 148) {
            throw std::runtime_error{"dds: truncated DX10 header in " + path.string()};
        }
        data_offset = 148;
        switch (u32(128)) {
        case DXGI_BC1_UNORM:
            dds.format = vk::Format::eBc1RgbaUnormBlock;
            break;
        case DXGI_BC1_UNORM_SRGB:
            dds.format = vk::Format::eBc1RgbaSrgbBlock;
            break;
        case DXGI_BC2_UNORM:
            dds.format = vk::Format::eBc2UnormBlock;
            break;
        case DXGI_BC2_UNORM_SRGB:
            dds.format = vk::Format::eBc2SrgbBlock;
            break;
        case DXGI_BC3_UNORM:
            dds.format = vk::Format::eBc3UnormBlock;
            break;
        case DXGI_BC3_UNORM_SRGB:
            dds.format = vk::Format::eBc3SrgbBlock;
            break;
        case DXGI_BC4_UNORM:
            dds.format = vk::Format::eBc4UnormBlock;
            break;
        case DXGI_BC5_UNORM:
            dds.format = vk::Format::eBc5UnormBlock;
            break;
        case DXGI_BC7_UNORM:
            dds.format = vk::Format::eBc7UnormBlock;
            break;
        case DXGI_BC7_UNORM_SRGB:
            dds.format = vk::Format::eBc7SrgbBlock;
            break;
        default:
            throw std::runtime_error{"dds: unsupported DXGI format in " + path.string()};
        }
    } else {
        throw std::runtime_error{"dds: unsupported FourCC in " + path.string()};
    }

    // Stream the compressed payload straight into its final buffer.
    dds.data.resize(file_size - data_offset);
    file.seekg(static_cast<std::streamoff>(data_offset));
    file.read(reinterpret_cast<char*>(dds.data.data()),
              static_cast<std::streamsize>(dds.data.size()));
    dds.has_alpha = format_has_alpha(dds.format);
    return dds;
}

BlobHandle dds_decode_rgba8(const DdsImage& dds, ImageInfo& info) {
    const bool is_bc1 =
        dds.format == vk::Format::eBc1RgbaUnormBlock || dds.format == vk::Format::eBc1RgbaSrgbBlock;
    const bool is_bc2 =
        dds.format == vk::Format::eBc2UnormBlock || dds.format == vk::Format::eBc2SrgbBlock;
    const bool is_bc3 =
        dds.format == vk::Format::eBc3UnormBlock || dds.format == vk::Format::eBc3SrgbBlock;
    const bool is_bc4 = dds.format == vk::Format::eBc4UnormBlock;
    const bool is_bc5 = dds.format == vk::Format::eBc5UnormBlock;
    if (!is_bc1 && !is_bc2 && !is_bc3 && !is_bc4 && !is_bc5) {
        throw std::runtime_error{"dds: CPU decode supports BC1-BC5 only"};
    }

    const uint32_t w = dds.width;
    const uint32_t h = dds.height;
    const uint32_t block_bytes = block_bytes_for(dds.format);
    std::vector<uint8_t> rgba(static_cast<size_t>(w) * h * 4, 0);

    const uint8_t* src = dds.data.data();
    const uint32_t blocks_x = std::max(1u, (w + 3) / 4);
    const uint32_t blocks_y = std::max(1u, (h + 3) / 4);

    for (uint32_t by = 0; by < blocks_y; by++) {
        for (uint32_t bx = 0; bx < blocks_x; bx++) {
            std::array<std::array<uint8_t, 4>, 16> texels{};

            if (is_bc1) {
                decode_color_block(src, false, texels);
            } else { // BC2/BC3 colour is in the second 8 bytes (opaque mode); BC4/BC5 are scalar.
                std::array<uint8_t, 16> r{};
                std::array<uint8_t, 16> g{};
                if (is_bc4 || is_bc5) {
                    decode_alpha_block(src, r);
                    if (is_bc5) {
                        decode_alpha_block(src + 8, g);
                    }
                    for (int i = 0; i < 16; i++) {
                        texels[i] = {r[i], is_bc5 ? g[i] : r[i], is_bc5 ? uint8_t(0) : r[i], 255};
                    }
                } else { // BC2 / BC3
                    decode_color_block(src + 8, true, texels);
                    if (is_bc3) {
                        std::array<uint8_t, 16> alpha{};
                        decode_alpha_block(src, alpha);
                        for (int i = 0; i < 16; i++) {
                            texels[i][3] = alpha[i];
                        }
                    } else { // BC2: explicit 4-bit alpha
                        for (int i = 0; i < 16; i++) {
                            const uint8_t nib = (src[i / 2] >> ((i & 1) * 4)) & 0xf;
                            texels[i][3] = static_cast<uint8_t>((nib << 4) | nib);
                        }
                    }
                }
            }

            for (int row = 0; row < 4; row++) {
                for (int col = 0; col < 4; col++) {
                    const uint32_t x = bx * 4 + col;
                    const uint32_t y = by * 4 + row;
                    if (x < w && y < h) {
                        std::memcpy(rgba.data() + (static_cast<size_t>(y) * w + x) * 4,
                                    texels[row * 4 + col].data(), 4);
                    }
                }
            }
            src += block_bytes;
        }
    }

    info = {.width = static_cast<int>(w),
            .height = static_cast<int>(h),
            .channels = 4,
            .source_channels = dds.has_alpha ? 4 : 3};
    return std::make_shared<VectorBlob<uint8_t>>(std::move(rgba));
}

} // namespace merian
