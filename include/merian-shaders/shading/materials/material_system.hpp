#pragma once

#include "merian-shaders/shading/materials/material-system-data.slangh"
#include "merian-shaders/utils/texture_manager.hpp"
#include "merian/shader/shader_object.hpp"
#include "merian/shader/slang_composition.hpp"
#include "merian/shader/slang_program.hpp"

#include <string>
#include <unordered_map>
#include <vector>

namespace merian {

class Properties;

// CPU-side base for all material definitions.
// Mirrors the Slang MaterialModel layout: MaterialHeader first, then material-specific payload.
struct Material {
    MaterialHeader header;

    virtual ~Material() = default;
    // Size of the payload AFTER the header, in bytes.
    virtual uint32_t get_payload_size() const = 0;
    // Write the payload AFTER the header to dest.
    virtual void write_payload(void* dest) const = 0;
};

struct DiffuseMaterial : Material {
    float4 base_color_factor{1, 1, 1, 1};

    DiffuseMaterial() {
        header.alpha_texture_id = TextureID(-1);
    }

    DiffuseMaterial(float4 color, TextureID texture_id) : base_color_factor(color) {
        header.alpha_texture_id = texture_id;
    }

    uint32_t get_payload_size() const override {
        return sizeof(float4);
    }

    void write_payload(void* dest) const override {
        std::memcpy(dest, &base_color_factor, sizeof(float4));
    }
};

class MaterialSystem : public std::enable_shared_from_this<MaterialSystem> {
  public:
    MaterialSystem(const ShaderCompileContextHandle& compile_context,
                   const ContextHandle& context,
                   const ResourceAllocatorHandle& allocator,
                   const TextureManagerHandle& texture_manager);

    // Register a material model type. Returns the type's dispatch ID.
    MaterialModelID register_material_type(const std::string& slang_type_name,
                                           const std::string& slang_module_path);

    // Add a material instance. Returns MaterialID.
    MaterialID add_material(MaterialModelID type_id, const Material& material);

    // Re-pack the payload at slot `id`. Cheap: only widens the dirty range so
    // the next upload() copies that single entry. The new material's payload
    // size must not exceed the current max_payload_size (asserted) — pick a
    // material model whose payload is stable across updates.
    void update_material(MaterialID id, const Material& material);

    // Drop all materials. Caller is responsible for re-adding (e.g. after a
    // map change). The GPU buffer handle is kept; it gets overwritten on the
    // next upload().
    void clear();

    uint32_t get_material_count() const {
        return static_cast<uint32_t>(materials.size());
    }

    // Upload material buffer to GPU and update ShaderObject state.
    void update(const CommandBufferHandle& cmd);

    float get_alpha_test_threshold() const {
        return alpha_test_threshold;
    }
    void set_alpha_test_threshold(float threshold);

    bool get_clamp_normals() const {
        return clamp_normals;
    }
    void set_clamp_normals(bool clamp);

    float get_min_roughness() const {
        return min_roughness;
    }
    void set_min_roughness(float min_roughness);

    void properties(Properties& props);

    const SlangCompositionHandle& get_composition() const {
        return composition;
    }

    // Bumps whenever the composition changes.
    uint64_t version() const {
        return composition->version();
    }

    const TextureManagerHandle& get_texture_manager() const {
        return texture_manager;
    }

    const ShaderObjectHandle& get_shader_object() const {
        return shader_object.get();
    }

    operator const ShaderObjectHandle&() const {
        return shader_object.get();
    }

  public:
    struct MaterialTypeInfo {
        std::string slang_module_path;
        MaterialModelID dispatch_id;
    };

  private:
    void update_composition_constants();
    ShaderObjectHandle build_shader_object() const;
    void repack_host_buffer();
    uint32_t get_entry_size() const;

    ShaderCompileContextHandle compile_context;
    ContextHandle context;
    ResourceAllocatorHandle allocator;
    TextureManagerHandle texture_manager;

    std::unordered_map<std::string, MaterialTypeInfo> material_types;

    struct StoredMaterial {
        MaterialHeader header;
        std::vector<uint8_t> payload;
    };
    std::vector<StoredMaterial> materials;

    // Packed host-side buffer matching GPU MaterialBufferEntry layout
    std::vector<uint8_t> host_buffer;
    BufferHandle material_buffer;
    uint32_t max_payload_size = 0;
    uint32_t dirty_begin = UINT32_MAX;
    uint32_t dirty_end = 0;
    float alpha_test_threshold = 0.5F;
    bool clamp_normals = true;
    float min_roughness = 0.0316F;
    SlangCompositionHandle composition;
    Versioned<SlangProgram> layout_program;
    Versioned<ShaderObject> shader_object;
};

using MaterialSystemHandle = std::shared_ptr<MaterialSystem>;

} // namespace merian
