// Parsing logic derived from pbrt-v4 (https://github.com/mmp/pbrt-v4),
// Copyright (c) 1998-2020 Matt Pharr, Wenzel Jakob, and Greg Humphreys. Apache-2.0.
#pragma once

#include "pbrt_params.hpp"

#include <filesystem>
#include <memory>
#include <optional>
#include <string>
#include <unordered_map>
#include <vector>

namespace merian::pbrt {

struct AreaLightDesc {
    float3 radiance{1, 1, 1}; // L * scale
    bool twosided = false;
};

struct TextureDesc {
    std::string name;
    bool is_float = false; // declared "float" vs "spectrum"
    std::string cls;
    ParamDict params;
};

struct MaterialDesc {
    std::string name; // empty for inline materials
    std::string type;
    ParamDict params;
};

struct ShapeDesc {
    enum MaterialRef : int32_t { MATERIAL_DEFAULT = -1, MATERIAL_INTERFACE = -2 };

    std::string type;
    ParamDict params;
    float4x4 ctm;
    bool reverse_orientation = false;
    int32_t material = MATERIAL_DEFAULT; // index into PBRTSceneDesc::materials
    std::optional<AreaLightDesc> area_light;
    int32_t object = -1; // index into PBRTSceneDesc::objects, -1 = placed directly
};

struct ObjectDesc {
    std::string name;
    float4x4 begin_ctm;
    std::vector<int32_t> shape_indices;
};

struct InstanceDesc {
    int32_t object;
    float4x4 ctm;
};

struct InfiniteLightDesc {
    std::string filename; // empty -> constant radiance
    std::optional<float3> radiance;
    float scale = 1.0f;
    float4x4 ctm;
};

struct CameraDesc {
    float4x4 world_to_camera;
    float fov = 90.0f; // degrees, applies to the shorter image axis
};

struct PBRTSceneDesc {
    std::filesystem::path base_dir;

    std::vector<TextureDesc> textures;
    std::unordered_map<std::string, int32_t> texture_index;
    std::vector<MaterialDesc> materials;
    std::vector<ShapeDesc> shapes;
    std::vector<ObjectDesc> objects;
    std::vector<InstanceDesc> instances;
    std::vector<InfiniteLightDesc> infinite_lights;
    std::optional<CameraDesc> camera;
    int32_t film_width = 1280;
    int32_t film_height = 720;

    const TextureDesc* find_texture(const std::string& name) const {
        const auto it = texture_index.find(name);
        return it != texture_index.end() ? &textures[it->second] : nullptr;
    }
};

// Throws PBRTParseError on unrecoverable structure errors; anything else logs and continues.
std::unique_ptr<PBRTSceneDesc> parse_pbrt_file(const std::filesystem::path& path);

} // namespace merian::pbrt
