#pragma once

#include "merian-shaders/shading/materials/material_system.hpp"

#include <cstdint>
#include <cstring>

namespace merian {

// Mirrors the Slang-side struct byte-for-byte; order, sizes and packing must stay in sync.
struct GltfMaterialPayload {
    float4 base_color_factor{1, 1, 1, 1};
    float3 emissive_factor{0, 0, 0};
    float metallic_factor{1.0f};
    float roughness_factor{1.0f};
    float normal_scale{1.0f};
    float occlusion_strength{1.0f};
    float transmission_weight{0.0f}; // KHR_materials_transmission
    float ior{1.5f};                 // KHR_materials_ior
    float clearcoat_weight{0.0f};    // KHR_materials_clearcoat
    float clearcoat_roughness{0.0f};
    float3 absorption{0, 0, 0};      // KHR_materials_volume: precomputed sigma_a (Beer-Lambert)
    TextureID metallic_roughness_texture{TextureID(-1)};
    TextureID normal_texture{TextureID(-1)};
    TextureID occlusion_texture{TextureID(-1)};
    TextureID emissive_texture{TextureID(-1)};
    TextureID clearcoat_texture{TextureID(-1)};
    TextureID clearcoat_roughness_texture{TextureID(-1)};
};
static_assert(sizeof(GltfMaterialPayload) == 84,
              "GltfMaterialPayload layout must match Slang GltfMaterial");

struct GltfMaterial : Material {
    GltfMaterialPayload payload;

    GltfMaterial() {
        header.alpha_texture_id = TextureID(-1);
    }

    uint32_t get_payload_size() const override {
        return static_cast<uint32_t>(sizeof(GltfMaterialPayload));
    }

    void write_payload(void* dest) const override {
        std::memcpy(dest, &payload, sizeof(GltfMaterialPayload));
    }
};

inline constexpr const char* GLTF_MATERIAL_SLANG_TYPE_NAME = "merian::GltfMaterial";
inline constexpr const char* GLTF_MATERIAL_SLANG_MODULE_PATH =
    "merian-shaders/shading/materials/gltf-material.slang";

} // namespace merian
