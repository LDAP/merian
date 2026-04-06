#pragma once

#include "merian-shaders/shading/materials/material-system-data.slangh"
#include "merian-shaders/utils/texture_manager.hpp"
#include "merian/shader/shader_object.hpp"
#include "merian/shader/slang_composition.hpp"
#include "merian/shader/slang_program.hpp"
#include "merian/utils/versionable.hpp"

#include <vector>

namespace merian {

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

class MaterialSystem : public Versionable,
                       public std::enable_shared_from_this<MaterialSystem> {
  public:
    MaterialSystem(const ShaderCompileContextHandle& compile_context,
                   const ContextHandle& context,
                   const ResourceAllocatorHandle& allocator,
                   const ShaderObjectAllocatorHandle& obj_allocator,
                   const TextureManagerHandle& texture_manager);

    // Register a material model type. Returns the type's dispatch ID.
    MaterialModelID register_material_type(const std::string& slang_type_name,
                                           const std::string& slang_module_path);

    // Add a material instance. Returns MaterialID.
    MaterialID add_material(MaterialModelID type_id, const Material& material);

    uint32_t get_material_count() const {
        return static_cast<uint32_t>(materials.size());
    }

    // Upload material buffer to GPU and update ShaderObject state.
    void upload(const CommandBufferHandle& cmd);

    const SlangCompositionHandle& get_composition() const {
        return composition;
    }

    const TextureManagerHandle& get_texture_manager() const {
        return texture_manager;
    }

    const ShaderObjectHandle& get_shader_object() const {
        return shader_object;
    }

    operator const ShaderObjectHandle&() const {
        return shader_object;
    }

  public:
    struct MaterialTypeInfo {
        std::string slang_type_name;
        std::string slang_module_path;
        MaterialModelID dispatch_id;
    };

  private:
    void update_composition_constants();
    void rebuild_shader_object();
    void repack_host_buffer();
    uint32_t get_entry_size() const;

    ShaderCompileContextHandle compile_context;
    ContextHandle context;
    ResourceAllocatorHandle allocator;
    ShaderObjectAllocatorHandle obj_allocator;
    TextureManagerHandle texture_manager;

    std::vector<MaterialTypeInfo> material_types;

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
    SlangCompositionHandle composition;
    SlangProgramHandle layout_program;
    ShaderObjectHandle shader_object;
};

using MaterialSystemHandle = std::shared_ptr<MaterialSystem>;

} // namespace merian
