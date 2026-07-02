#pragma once

#include "merian/shader/shader_cursor.hpp"
#include "merian/shader/slang_composition.hpp"
#include "merian/utils/properties.hpp"
#include "merian/utils/vector_matrix.hpp"
#include "merian/vk/memory/resource_allocations.hpp"

#include <array>
#include <memory>
#include <string>
#include <utility>

namespace merian {

class EnvMap {
  public:
    virtual ~EnvMap() = default;

    virtual SlangComposition::SlangModule get_slang_module() const = 0;

    virtual std::string get_type_name() const = 0;

    virtual void write_to(ShaderCursor cursor) const = 0;

    // Yaw/pitch UI; subclasses can extend to expose more state.
    virtual void properties(Properties& props) {
        bool changed = false;
        changed |= props.config_angle("Yaw", yaw_rad);
        changed |= props.config_angle("Pitch", pitch_rad, "", -90, 90);
        if (changed) {
            rebuild_transform();
        }
        props.config_float("Intensity", intensity, "", 0.01f, 0.f);
    }

    // Seeded by the host once (e.g. Scene::get_up alignment); yaw/pitch compose on top.
    void set_base_transform(const float3x3& m) {
        base_to_local = float4x4(m);
        rebuild_transform();
    }

    void set_yaw_pitch(const float yaw, const float pitch) {
        yaw_rad = yaw;
        pitch_rad = pitch;
        rebuild_transform();
    }

    float get_yaw_rad() const {
        return yaw_rad;
    }
    float get_pitch_rad() const {
        return pitch_rad;
    }

    void set_intensity(const float i) {
        intensity = i;
    }

    float get_intensity() const {
        return intensity;
    }

    void set_transform(const float4x4& m) {
        to_local = m;
    }

    const float4x4& get_transform() const {
        return to_local;
    }

  protected:
    void rebuild_transform() {
        const float4x4 yaw = rotation(float3(0, 1, 0), yaw_rad);
        const float4x4 pitch = rotation(float3(1, 0, 0), pitch_rad);
        to_local = mul(pitch, mul(yaw, base_to_local));
    }

    float4x4 to_local = identity<float4x4>();
    float4x4 base_to_local = identity<float4x4>();
    float yaw_rad = 0.f;
    float pitch_rad = 0.f;
    float intensity = 1.f;
};
using EnvMapHandle = std::shared_ptr<EnvMap>;

class EmptyEnvMap : public EnvMap {
  public:
    SlangComposition::SlangModule get_slang_module() const override {
        return SlangComposition::SlangModule::from_path(
            "merian-shaders/scene/environment-map.slang", false);
    }

    std::string get_type_name() const override {
        return "merian::EmptyEnv";
    }

    void write_to([[maybe_unused]] ShaderCursor cursor) const override {}
};

class LatLongEnvMap : public EnvMap {
  public:
    explicit LatLongEnvMap(TextureHandle texture) : texture(std::move(texture)) {}

    SlangComposition::SlangModule get_slang_module() const override {
        return SlangComposition::SlangModule::from_path(
            "merian-shaders/scene/environment-map.slang", false);
    }

    std::string get_type_name() const override {
        return "merian::LatLongMap";
    }

    void write_to(ShaderCursor cursor) const override {
        cursor["env_map"] = texture;
        cursor["to_local"] = to_local;
        cursor["intensity"] = intensity;
    }

    const TextureHandle& get_texture() const {
        return texture;
    }

    void set_texture(TextureHandle t) {
        texture = std::move(t);
    }

  private:
    TextureHandle texture;
};

class CubeMapEnvMap : public EnvMap {
  public:
    explicit CubeMapEnvMap(std::array<TextureHandle, 6> faces) : faces(std::move(faces)) {}

    SlangComposition::SlangModule get_slang_module() const override {
        return SlangComposition::SlangModule::from_path(
            "merian-shaders/scene/environment-map.slang", false);
    }

    std::string get_type_name() const override {
        return "merian::CubeMap";
    }

    void write_to(ShaderCursor cursor) const override {
        cursor["rt"] = faces[0];
        cursor["bk"] = faces[1];
        cursor["lf"] = faces[2];
        cursor["ft"] = faces[3];
        cursor["up"] = faces[4];
        cursor["dn"] = faces[5];
        cursor["to_local"] = to_local;
        cursor["intensity"] = intensity;
    }

    const std::array<TextureHandle, 6>& get_faces() const {
        return faces;
    }

    void set_faces(std::array<TextureHandle, 6> f) {
        faces = std::move(f);
    }

  private:
    std::array<TextureHandle, 6> faces;
};

} // namespace merian
