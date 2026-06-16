#pragma once

#include "merian-scene/material_system.hpp"

#include <cstdint>
#include <cstring>
#include <fmt/format.h>
#include <optional>
#include <string>

namespace merian {

struct PBRTTransmissionData {
    float weight{0.0f};
    float3 color{1, 1, 1};
};
struct PBRTClearcoatData {
    float weight{0.0f};
    float roughness{0.0f};
    float ior{1.6f};
};
struct PBRTSheenData {
    float weight{0.0f};
    float3 color{1, 1, 1};
    float roughness{0.3f};
};

struct PBRTMaterial : Material {
    float3 base_color{1, 1, 1};
    float opacity{1.0f};
    float metalness{0.0f};
    float roughness{1.0f};
    float specular_weight{1.0f};
    float specular_ior{1.5f};
    float3 emission{0, 0, 0};
    float normal_scale{1.0f};
    TextureID metalness_texture{TextureID(-1)};
    TextureID roughness_texture{TextureID(-1)};
    TextureID emission_texture{TextureID(-1)};
    TextureID normal_texture{TextureID(-1)};

    std::optional<PBRTTransmissionData> transmission;
    std::optional<PBRTClearcoatData> clearcoat;
    std::optional<PBRTSheenData> sheen;

    PBRTMaterial() {
        header.alpha_texture_id = TextureID(-1);
    }

    std::string variant_type_name() const {
        const auto b = [](bool v) { return v ? "true" : "false"; };
        return fmt::format("merian::PBRTMaterial<{}, {}, {}>", b(transmission.has_value()),
                           b(clearcoat.has_value()), b(sheen.has_value()));
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
        put(roughness);
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
            put(clearcoat->roughness);
            put(clearcoat->ior);
        }
        if (sheen) {
            put(sheen->weight);
            put(sheen->color);
            put(sheen->roughness);
        }
        return off;
    }
};

inline constexpr const char* PBRT_MATERIAL_SLANG_MODULE_PATH =
    "merian-shaders/shading/materials/pbrt-material.slang";

} // namespace merian
