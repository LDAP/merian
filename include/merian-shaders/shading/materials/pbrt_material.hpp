#pragma once

#include "merian-shaders/shading/materials/material_system.hpp"

#include <cstdint>
#include <cstring>

namespace merian {

// Mirrors the Slang-side struct byte-for-byte; order, sizes and packing must stay in sync.
struct PBRTMaterialPayload {
    float3 base_color{1, 1, 1};
    float opacity{1.0f};
    float metalness{0.0f};
    float roughness{1.0f};
    float specular_weight{1.0f};
    float specular_ior{1.5f};
    float3 emission{0, 0, 0};
    float normal_scale{1.0f};
    float coat_weight{0.0f};
    float coat_roughness{0.0f};
    float coat_ior{1.6f};
    float sheen_weight{0.0f};
    float3 sheen_color{0, 0, 0};
    float sheen_roughness{0.3f};
    float transmission_weight{0.0f};
    float3 transmission_color{1, 1, 1};
    TextureID metalness_texture{TextureID(-1)};
    TextureID roughness_texture{TextureID(-1)};
    TextureID emission_texture{TextureID(-1)};
    TextureID normal_texture{TextureID(-1)};
};
static_assert(sizeof(PBRTMaterialPayload) == 104,
              "PBRTMaterialPayload layout must match Slang PBRTMaterial");

struct PBRTMaterial : Material {
    PBRTMaterialPayload payload;

    PBRTMaterial() {
        header.alpha_texture_id = TextureID(-1);
    }

    uint32_t get_payload_size() const override {
        return static_cast<uint32_t>(sizeof(PBRTMaterialPayload));
    }

    void write_payload(void* dest) const override {
        std::memcpy(dest, &payload, sizeof(PBRTMaterialPayload));
    }
};

inline constexpr const char* PBRT_MATERIAL_SLANG_TYPE_NAME = "merian::PBRTMaterial";
inline constexpr const char* PBRT_MATERIAL_SLANG_MODULE_PATH =
    "merian-shaders/shading/materials/pbrt-material.slang";

} // namespace merian
