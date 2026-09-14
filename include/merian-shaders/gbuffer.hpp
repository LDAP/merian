#pragma once

#include "merian/shader/shader_compile_context.hpp"
#include "merian/shader/shader_object.hpp"
#include "merian/shader/slang_composition.hpp"
#include "merian/vk/memory/resource_allocations.hpp"
#include "merian/vk/utils/enums.hpp"

#include <array>
#include <cstdint>
#include <memory>
#include <string>
#include <vector>

namespace merian {

// field, shader name, shader type, float channels, float storage, packed channels
#define MERIAN_GBUFFER_FIELDS(X)                                                                   \
    /* shading normal, world space */                                                              \
    X(Normal, "normal", "float3", 3, Float16, 1)                                                   \
    /* distance from the camera along the primary ray */                                           \
    X(LinearZ, "linear_z", "float", 1, Float16, 1)                                                 \
    /* screen space gradient of LinearZ */                                                         \
    X(GradZ, "grad_z", "half2", 2, Float16, 1)                                                     \
    /* distance to the camera in the previous frame minus LinearZ */                               \
    X(DeltaZ, "delta_z", "float", 1, Float16, 1)                                                   \
    /* pixel offset to the surface point in the previous frame */                                  \
    X(MotionVectors, "motion_vectors", "float2", 2, Float16, 1)                                    \
    X(Hit, "hit", "Hit", 0, Float16, 4)                                                            \
    /* directional albedo summed over the lobes */                                                 \
    X(Albedo, "albedo", "float3", 3, Float16, 0)                                                   \
    X(DiffuseAlbedo, "diffuse_albedo", "float3", 3, Float16, 0)                                    \
    X(SpecularAlbedo, "specular_albedo", "float3", 3, Float16, 0)                                  \
    /* linear roughness of the specular lobe */                                                    \
    X(Roughness, "roughness", "float", 1, Float16, 0)                                              \
    /* distance from the camera along its view axis */                                             \
    X(ViewDepth, "view_depth", "float", 1, Float32, 1)                                             \
    /* perspective depth, 0 at the near and 1 at the far plane */                                  \
    X(ProjectedDepth, "projected_depth", "float", 1, Float32, 1)

#define MERIAN_GBUFFER_FIELD_ENUMERATOR(field, ...) field,
#define MERIAN_GBUFFER_FIELD_ONE(...) +1

enum class GBufferField : uint8_t { MERIAN_GBUFFER_FIELDS(MERIAN_GBUFFER_FIELD_ENUMERATOR) };

inline constexpr uint32_t GBUFFER_FIELD_COUNT = 0 MERIAN_GBUFFER_FIELDS(MERIAN_GBUFFER_FIELD_ONE);

#undef MERIAN_GBUFFER_FIELD_ENUMERATOR
#undef MERIAN_GBUFFER_FIELD_ONE

template <> uint32_t enum_size<GBufferField>();

template <> const GBufferField* enum_values<GBufferField>();

// Fields a consumer reads together.
struct GBufferGroup {
    std::vector<GBufferField> fields;
    // Fills one float texture from its first channel, in order.
    bool texture = false;
};

// Packs the fields requested by a set of groups into textures and generates the Slang types that
// read and write them, linked to merian::GBuffer and merian::WGBuffer.
class GBufferLayout {
  public:
    enum class Storage : uint8_t {
        Uint32,
        Float16,
        Float32,
    };

    struct Placement {
        GBufferField field;
        uint32_t channel;
        // bit packed into a single uint channel
        bool packed;

        bool operator==(const Placement&) const = default;
    };

    struct Texture {
        Storage storage = Storage::Float16;
        uint32_t channel_count = 0;
        // laid out by a texture group
        bool exact = false;
        std::vector<Placement> placements;

        bool operator==(const Texture&) const = default;
    };

    explicit GBufferLayout(const std::vector<GBufferGroup>& groups);

    // Holds every field.
    static const std::shared_ptr<const GBufferLayout>& complete();

    const std::vector<Texture>& get_textures() const {
        return textures;
    }

    bool contains(GBufferField field) const {
        return field_texture[static_cast<uint32_t>(field)] != NO_TEXTURE;
    }

    // The texture a field is read from. Throws where the layout does not hold it.
    uint32_t get_texture_index(GBufferField field) const;

    // The smallest format the device can sample and store the texture in.
    vk::Format get_format(uint32_t texture_index,
                          const PhysicalDeviceHandle& physical_device) const;

    // Links merian::GBuffer to this layout. Add it to every composition that binds one.
    const SlangCompositionHandle& get_composition() const {
        return composition;
    }

    // The type merian::GBuffer, or merian::WGBuffer with write access, is linked to.
    std::string get_type_name(bool write) const;

    const std::string& get_module_name() const {
        return module_name;
    }

    bool operator==(const GBufferLayout& other) const {
        return textures == other.textures;
    }

  private:
    static constexpr uint8_t NO_TEXTURE = 0xFF;

    void add_texture_group(const std::vector<GBufferField>& fields);

    void place(const std::vector<GBufferField>& fields, const std::vector<GBufferField>& group);

    std::string generate_module() const;

    std::vector<Texture> textures;
    std::array<uint8_t, GBUFFER_FIELD_COUNT> field_texture;
    // unique per layout
    std::string suffix;
    std::string module_name;
    SlangCompositionHandle composition;
};

using GBufferLayoutHandle = std::shared_ptr<const GBufferLayout>;

class GBuffer {
  public:
    GBuffer(const ShaderCompileContextHandle& compile_context,
            const ContextHandle& context,
            const ResourceAllocatorHandle& allocator,
            vk::Extent3D extent,
            const GBufferLayoutHandle& layout);

    const ShaderObjectHandle& get_shader_object() const {
        return r_shader_object;
    }

    const ShaderObjectHandle& get_write_shader_object() const {
        return w_shader_object;
    }

    operator const ShaderObjectHandle&() const {
        return r_shader_object;
    }

    // One view per texture of the layout.
    void set_resources(const std::vector<ImageViewHandle>& textures);

    vk::Extent3D get_extent() const {
        return extent;
    }

  private:
    const vk::Extent3D extent;
    const GBufferLayoutHandle layout;

    ShaderObjectHandle r_shader_object;
    ShaderObjectHandle w_shader_object;
};

using GBufferHandle = std::shared_ptr<GBuffer>;

} // namespace merian
