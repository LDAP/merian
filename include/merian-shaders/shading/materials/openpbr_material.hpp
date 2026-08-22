#pragma once

#include "merian-shaders/shading/materials/material_system.hpp"

#include <cstdint>
#include <cstring>
#include <fmt/format.h>
#include <optional>
#include <string>

namespace merian {

inline float ggx_roughness_to_alpha(const float roughness) {
    return roughness * roughness;
}

// What a sampled roughness texel holds. The material itself stores the GGX alpha = roughness^2.
enum class RoughnessEncoding : uint32_t {
    Roughness,
    Alpha,
    AlphaSquared,
};

inline const char* slang_name(const RoughnessEncoding encoding) {
    switch (encoding) {
    case RoughnessEncoding::Alpha:
        return "merian::RoughnessEncoding::Alpha";
    case RoughnessEncoding::AlphaSquared:
        return "merian::RoughnessEncoding::AlphaSquared";
    case RoughnessEncoding::Roughness:
        break;
    }
    return "merian::RoughnessEncoding::Roughness";
}

struct OpenPBRTransmissionData {
    float weight{0.0f};
    float3 color{1, 1, 1};
};
struct OpenPBRClearcoatData {
    float weight{0.0f};
    float alpha{0.0f};
    float ior{1.6f};
};
struct OpenPBRSheenData {
    float weight{0.0f};
    float3 color{1, 1, 1};
    float alpha{0.09f};
};
struct OpenPBRVolumeData {
    float3 absorption{0, 0, 0};
};

struct OpenPBRMaterial : Material {
    float3 base_color{1, 1, 1};
    float opacity{1.0f};
    float metalness{0.0f};
    float2 specular_alpha{1.0f, 1.0f}; // per-axis (tangent, bitangent)
    float specular_weight{1.0f};
    float specular_ior{1.5f};
    float3 emission{0, 0, 0};
    float normal_scale{1.0f};
    TextureID metalness_texture{TextureID(-1)};
    TextureID roughness_texture{TextureID(-1)};
    TextureID emission_texture{TextureID(-1)};
    TextureID normal_texture{TextureID(-1)};
    RoughnessEncoding roughness_encoding{RoughnessEncoding::Roughness};

    std::optional<OpenPBRTransmissionData> transmission;
    std::optional<OpenPBRClearcoatData> clearcoat;
    std::optional<OpenPBRSheenData> sheen;
    std::optional<OpenPBRVolumeData> volume;

    OpenPBRMaterial() {
        header.alpha_texture_id = TextureID(-1);
    }

    std::string variant_type_name() const {
        const auto b = [](bool v) { return v ? "true" : "false"; };
        return fmt::format("merian::OpenPBRMaterial<{}, {}, {}, {}, {}>",
                           b(transmission.has_value()), b(clearcoat.has_value()),
                           b(sheen.has_value()), b(volume.has_value()),
                           slang_name(roughness_encoding));
    }

    uint32_t get_payload_size() const override {
        return serialize(nullptr);
    }

    void write_payload(void* dest) const override {
        serialize(static_cast<std::byte*>(dest));
    }

  private:
    uint32_t serialize(std::byte* dst) const {
        uint32_t off = 0;
        const auto put = [&](const auto& v) {
            if (dst != nullptr) {
                std::memcpy(dst + off, &v, sizeof(v));
            }
            off += static_cast<uint32_t>(sizeof(v));
        };

        put(base_color);
        put(opacity);
        put(metalness);
        put(specular_alpha);
        put(specular_weight);
        put(specular_ior);
        put(emission);
        put(normal_scale);
        put(metalness_texture);
        put(roughness_texture);
        put(emission_texture);
        put(normal_texture);

        if (transmission) {
            put(transmission->weight);
            put(transmission->color);
        }
        if (clearcoat) {
            put(clearcoat->weight);
            put(clearcoat->alpha);
            put(clearcoat->ior);
        }
        if (sheen) {
            put(sheen->weight);
            put(sheen->color);
            put(sheen->alpha);
        }
        if (volume) {
            put(volume->absorption);
        }
        return off;
    }
};

inline constexpr const char* OPENPBR_MATERIAL_SLANG_MODULE_PATH =
    "merian-shaders/shading/materials/openpbr-material.slang";

} // namespace merian
