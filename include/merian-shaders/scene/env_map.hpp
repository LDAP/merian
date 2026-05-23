#pragma once

#include "merian/shader/shader_cursor.hpp"
#include "merian/shader/slang_composition.hpp"
#include "merian/utils/vector_matrix.hpp"
#include "merian/vk/memory/resource_allocations.hpp"

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

    // Rotation from world space into env-map space; applied to the sample direction.
    void set_transform(const float3x3& m) {
        to_local = m;
    }

    const float3x3& get_transform() const {
        return to_local;
    }

  protected:
    float3x3 to_local = identity<float3x3>();
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

} // namespace merian
