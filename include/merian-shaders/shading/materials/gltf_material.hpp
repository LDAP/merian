#pragma once

#include "merian-shaders/shading/materials/material_system.hpp"

#include <cstdint>
#include <cstring>
#include <fmt/format.h>
#include <optional>
#include <string>

namespace merian {

// Optional-extension payloads, mirroring the Slang Gltf* structs. Present features are appended to
// the payload in declaration order, matching the Conditional<T, FLAG> members on the Slang side.
struct GltfTransmissionData {
    float weight{0.0f};
};
struct GltfVolumeData {
    float3 absorption{0, 0, 0};
};
struct GltfClearcoatData {
    float weight{0.0f};
    float roughness{0.0f};
    TextureID texture{TextureID(-1)};
    TextureID roughness_texture{TextureID(-1)};
};
struct GltfSheenData {
    float3 color{0, 0, 0};
    float roughness{0.0f};
    TextureID color_texture{TextureID(-1)};
    TextureID roughness_texture{TextureID(-1)};
};
struct GltfIridescenceData {
    float factor{0.0f};
    float ior{1.3f};
    float thickness_min{100.0f};
    float thickness_max{400.0f};
    TextureID texture{TextureID(-1)};
    TextureID thickness_texture{TextureID(-1)};
};
struct GltfAnisotropyData {
    float strength{0.0f};
    float rotation{0.0f};
    TextureID texture{TextureID(-1)};
};

// glTF material instance. The variant (which extensions are present) selects a specialized
// GltfMaterial<...> on the Slang side; the payload carries the core fields plus the present
// features, packed scalar (float3 = 12 B, TextureID = 2 B) to match the Slang layout.
struct GltfMaterial : Material {
    float4 base_color_factor{1, 1, 1, 1};
    float3 emissive_factor{0, 0, 0};
    float metallic_factor{1.0f};
    float roughness_factor{1.0f};
    float normal_scale{1.0f};
    float occlusion_strength{1.0f};
    float ior{1.5f};
    TextureID metallic_roughness_texture{TextureID(-1)};
    TextureID normal_texture{TextureID(-1)};
    TextureID occlusion_texture{TextureID(-1)};
    TextureID emissive_texture{TextureID(-1)};

    std::optional<GltfTransmissionData> transmission;
    std::optional<GltfVolumeData> volume;
    std::optional<GltfClearcoatData> clearcoat;
    std::optional<GltfSheenData> sheen;
    std::optional<GltfIridescenceData> iridescence;
    std::optional<GltfAnisotropyData> anisotropy;

    GltfMaterial() {
        header.alpha_texture_id = TextureID(-1);
    }

    // Specialized Slang type for this feature combination; order matches the bool generic params.
    std::string variant_type_name() const {
        const auto b = [](bool v) { return v ? "true" : "false"; };
        return fmt::format("merian::GltfMaterial<{}, {}, {}, {}, {}, {}>",
                           b(transmission.has_value()), b(volume.has_value()),
                           b(clearcoat.has_value()), b(sheen.has_value()),
                           b(iridescence.has_value()), b(anisotropy.has_value()));
    }

    uint32_t get_payload_size() const override {
        return serialize(nullptr);
    }

    void write_payload(void* dest) const override {
        serialize(static_cast<std::byte*>(dest));
    }

  private:
    // Single source of truth for the layout: a null cursor counts, a real one writes.
    uint32_t serialize(std::byte* dst) const {
        uint32_t off = 0;
        const auto put = [&](const auto& v) {
            if (dst != nullptr) {
                std::memcpy(dst + off, &v, sizeof(v));
            }
            off += static_cast<uint32_t>(sizeof(v));
        };

        put(base_color_factor);
        put(emissive_factor);
        put(metallic_factor);
        put(roughness_factor);
        put(normal_scale);
        put(occlusion_strength);
        put(ior);
        put(metallic_roughness_texture);
        put(normal_texture);
        put(occlusion_texture);
        put(emissive_texture);

        if (transmission) {
            put(transmission->weight);
        }
        if (volume) {
            put(volume->absorption);
        }
        if (clearcoat) {
            put(clearcoat->weight);
            put(clearcoat->roughness);
            put(clearcoat->texture);
            put(clearcoat->roughness_texture);
        }
        if (sheen) {
            put(sheen->color);
            put(sheen->roughness);
            put(sheen->color_texture);
            put(sheen->roughness_texture);
        }
        if (iridescence) {
            put(iridescence->factor);
            put(iridescence->ior);
            put(iridescence->thickness_min);
            put(iridescence->thickness_max);
            put(iridescence->texture);
            put(iridescence->thickness_texture);
        }
        if (anisotropy) {
            put(anisotropy->strength);
            put(anisotropy->rotation);
            put(anisotropy->texture);
        }
        return off;
    }
};

inline constexpr const char* GLTF_MATERIAL_SLANG_MODULE_PATH =
    "merian-shaders/shading/materials/gltf-material.slang";

} // namespace merian
