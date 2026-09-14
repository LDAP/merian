#include "merian-shaders/gbuffer.hpp"

#include "merian/shader/slang_program.hpp"
#include "merian/utils/hash.hpp"
#include "merian/vk/physical_device.hpp"

#include <fmt/format.h>
#include <fmt/ranges.h>

#include <algorithm>
#include <array>
#include <cassert>
#include <ranges>
#include <string_view>
#include <utility>

namespace merian {

namespace {

using Storage = GBufferLayout::Storage;

constexpr uint32_t TEXTURE_CHANNELS = 4;
constexpr std::string_view CHANNEL_NAMES = "xyzw";
constexpr const char* GBUFFER_FIELDS_MODULE = "merian_gbuffer_fields";

struct FieldTraits {
    // GBufferSample member and accessor suffix
    const char* name;
    const char* type;
    // 0 where the field has no such form
    uint32_t float_channels;
    Storage float_storage;
    uint32_t packed_channels;
};

#define MERIAN_GBUFFER_FIELD_TRAITS(field, name, type, float_channels, float_storage,              \
                                    packed_channels)                                               \
    FieldTraits{name, type, float_channels, Storage::float_storage, packed_channels},

constexpr std::array FIELD_TRAITS = {MERIAN_GBUFFER_FIELDS(MERIAN_GBUFFER_FIELD_TRAITS)};

#undef MERIAN_GBUFFER_FIELD_TRAITS

#define MERIAN_GBUFFER_FIELD_VALUE(field, ...) GBufferField::field,

constexpr std::array ALL_FIELDS = {MERIAN_GBUFFER_FIELDS(MERIAN_GBUFFER_FIELD_VALUE)};

#undef MERIAN_GBUFFER_FIELD_VALUE

static_assert(FIELD_TRAITS.size() == GBUFFER_FIELD_COUNT);
static_assert(std::ranges::all_of(FIELD_TRAITS, [](const FieldTraits& field_traits) {
    return field_traits.float_channels > 0 || field_traits.packed_channels > 0;
}));

const FieldTraits& traits(const GBufferField field) {
    return FIELD_TRAITS[static_cast<uint32_t>(field)];
}

uint32_t channels(const GBufferField field, const bool packed) {
    return packed ? traits(field).packed_channels : traits(field).float_channels;
}

uint32_t channels(const std::vector<GBufferField>& fields, const bool packed) {
    uint32_t sum = 0;
    for (const GBufferField field : fields) {
        sum += channels(field, packed);
    }
    return sum;
}

bool has_form(const std::vector<GBufferField>& fields, const bool packed) {
    return std::ranges::all_of(
        fields, [&](const GBufferField field) { return channels(field, packed) > 0; });
}

bool holds(const GBufferLayout::Texture& texture, const GBufferField field) {
    return std::ranges::any_of(texture.placements, [&](const GBufferLayout::Placement& placement) {
        return placement.field == field;
    });
}

bool fits(const GBufferLayout::Texture& texture,
          const std::vector<GBufferField>& fields,
          const bool packed) {
    if (texture.exact || (texture.storage == Storage::Uint32) != packed ||
        !has_form(fields, packed)) {
        return false;
    }
    const bool storage_suffices = packed || std::ranges::all_of(fields, [&](const GBufferField f) {
                                      return traits(f).float_storage <= texture.storage;
                                  });
    return storage_suffices && texture.channel_count + channels(fields, packed) <= TEXTURE_CHANNELS;
}

void append(GBufferLayout::Texture& texture, const GBufferField field, const bool packed) {
    texture.placements.push_back({field, texture.channel_count, packed});
    texture.channel_count += channels(field, packed);
}

struct FormatCandidate {
    vk::Format format;
    uint32_t channel_count;
};

const std::array<FormatCandidate, 3>& format_candidates(const Storage storage) {
    static constexpr std::array<FormatCandidate, 3> UINT32 = {{
        {vk::Format::eR32Uint, 1},
        {vk::Format::eR32G32Uint, 2},
        {vk::Format::eR32G32B32A32Uint, 4},
    }};
    static constexpr std::array<FormatCandidate, 3> FLOAT16 = {{
        {vk::Format::eR32Sfloat, 1},
        {vk::Format::eR16G16Sfloat, 2},
        {vk::Format::eR16G16B16A16Sfloat, 4},
    }};
    static constexpr std::array<FormatCandidate, 3> FLOAT32 = {{
        {vk::Format::eR32Sfloat, 1},
        {vk::Format::eR32G32Sfloat, 2},
        {vk::Format::eR32G32B32A32Sfloat, 4},
    }};
    return storage == Storage::Uint32 ? UINT32 : storage == Storage::Float16 ? FLOAT16 : FLOAT32;
}

vk::DeviceSize texel_bytes(const GBufferLayout::Texture& texture) {
    const std::array<FormatCandidate, 3>& candidates = format_candidates(texture.storage);
    const auto it = std::ranges::find_if(candidates, [&](const FormatCandidate& candidate) {
        return candidate.channel_count >= texture.channel_count;
    });
    return Image::format_size(it->format);
}

// Largest fields first, each into the first texture appended here that takes it.
void fill_fresh_textures(std::vector<GBufferLayout::Texture>& textures,
                         const std::vector<GBufferField>& fields,
                         const bool packed) {
    std::vector<GBufferField> by_size = fields;
    std::ranges::stable_sort(by_size, [&](const GBufferField a, const GBufferField b) {
        return channels(a, packed) > channels(b, packed);
    });
    const size_t first_fresh = textures.size();
    for (const GBufferField field : by_size) {
        const auto fresh = textures | std::views::drop(first_fresh);
        const auto it = std::ranges::find_if(fresh, [&](const GBufferLayout::Texture& texture) {
            return fits(texture, {field}, packed);
        });
        GBufferLayout::Texture& texture =
            it != fresh.end() ? *it
                              : textures.emplace_back(GBufferLayout::Texture{
                                    .storage = packed ? Storage::Uint32 : Storage::Float16});
        texture.storage =
            packed ? Storage::Uint32 : std::max(texture.storage, traits(field).float_storage);
        append(texture, field, packed);
    }
}

vk::DeviceSize bytes_per_pixel(const std::vector<GBufferField>& fields, const bool packed) {
    std::vector<GBufferLayout::Texture> textures;
    fill_fresh_textures(textures, fields, packed);
    vk::DeviceSize bytes = 0;
    for (const GBufferLayout::Texture& texture : textures) {
        bytes += texel_bytes(texture);
    }
    return bytes;
}

std::string decode(const GBufferField field, const bool packed, const std::string& channels) {
    switch (field) {
    case GBufferField::Normal:
        return packed ? fmt::format("decode_normal({})", channels) : channels;
    case GBufferField::LinearZ:
    case GBufferField::DeltaZ:
    case GBufferField::ViewDepth:
    case GBufferField::ProjectedDepth:
        return packed ? fmt::format("asfloat({})", channels) : channels;
    case GBufferField::GradZ:
        return packed ? fmt::format("unpackHalf2x16ToHalf({})", channels)
                      : fmt::format("half2({})", channels);
    case GBufferField::MotionVectors:
        return packed ? fmt::format("unpackHalf2x16ToFloat({})", channels) : channels;
    case GBufferField::Hit:
        return fmt::format("reinterpret<PackedHitInfo>({}).unpack()", channels);
    default:
        return channels;
    }
}

std::string encode(const GBufferField field, const bool packed) {
    const std::string value = fmt::format("sample.{}", traits(field).name);
    switch (field) {
    case GBufferField::Normal:
        return packed ? fmt::format("encode_normal({})", value) : value;
    case GBufferField::LinearZ:
    case GBufferField::DeltaZ:
    case GBufferField::ViewDepth:
    case GBufferField::ProjectedDepth:
        return packed ? fmt::format("asuint({})", value) : value;
    case GBufferField::GradZ:
        return packed ? fmt::format("packHalf2x16(float2({}))", value)
                      : fmt::format("float2({})", value);
    case GBufferField::MotionVectors:
        return packed ? fmt::format("packHalf2x16({})", value) : value;
    case GBufferField::Hit:
        return fmt::format("reinterpret<uint4>(PackedHitInfo::pack({}))", value);
    default:
        return value;
    }
}

std::string zero(const GBufferField field) {
    return field == GBufferField::Hit ? "Hit()" : fmt::format("{}(0)", traits(field).type);
}

const std::string& fields_module() {
    static const std::string source = [] {
        std::string members;
        std::string properties;
        std::string accessors;
        for (const GBufferField field : ALL_FIELDS) {
            const FieldTraits& field_traits = traits(field);
            members += fmt::format("    public {} {} = {};\n", field_traits.type, field_traits.name,
                                   zero(field));
            properties += fmt::format("    property {} {} {{ get; }}\n", field_traits.type,
                                      field_traits.name);
            accessors += fmt::format("    {0} get_{1}(uint2 pixel) {{ return load(pixel).{1}; }}\n",
                                     field_traits.type, field_traits.name);
        }

        return fmt::format(R"(module {0};

import merian_shaders.gbuffer_data;

namespace merian {{

public struct GBufferSample {{
{1}}}

public interface IGBufferTexels {{
{2}    // octahedral encoding of the normal
    property uint encoded_normal {{ get; }}
}}

public interface IGBuffer {{
    associatedtype Texels : IGBufferTexels;

    uint2 get_dimensions();

    Texels load(uint2 pixel);

{3}    uint get_encoded_normal(uint2 pixel) {{ return load(pixel).encoded_normal; }}
}}

}}
)",
                           GBUFFER_FIELDS_MODULE, members, properties, accessors);
    }();
    return source;
}

} // namespace

template <> uint32_t enum_size<GBufferField>() {
    return static_cast<uint32_t>(ALL_FIELDS.size());
}

template <> const GBufferField* enum_values<GBufferField>() {
    return ALL_FIELDS.data();
}

GBufferLayout::GBufferLayout(const std::vector<GBufferGroup>& groups) {
    field_texture.fill(NO_TEXTURE);

    std::vector<const GBufferGroup*> sorted;
    for (const GBufferGroup& group : groups) {
        if (!group.fields.empty()) {
            sorted.push_back(&group);
        }
    }
    // texture groups first, then the largest
    std::ranges::stable_sort(sorted, [](const GBufferGroup* a, const GBufferGroup* b) {
        return std::pair(b->texture, b->fields.size()) < std::pair(a->texture, a->fields.size());
    });

    for (const GBufferGroup* group : sorted) {
        if (group->texture) {
            add_texture_group(group->fields);
            continue;
        }
        if (std::ranges::any_of(textures, [&](const Texture& texture) {
                return std::ranges::all_of(
                    group->fields, [&](const GBufferField field) { return holds(texture, field); });
            })) {
            continue;
        }
        std::vector<GBufferField> missing;
        for (const GBufferField field : group->fields) {
            const bool stored = std::ranges::any_of(
                textures, [&](const Texture& texture) { return holds(texture, field); });
            if (!stored && std::ranges::find(missing, field) == missing.end()) {
                missing.push_back(field);
            }
        }
        if (!missing.empty()) {
            place(missing, group->fields);
        }
    }

    // get_dimensions reads the first texture
    if (textures.empty()) {
        place({GBufferField::LinearZ}, {});
    }

    for (uint32_t i = 0; i < textures.size(); i++) {
        for (const Placement& placement : textures[i].placements) {
            field_texture[static_cast<uint32_t>(placement.field)] = static_cast<uint8_t>(i);
        }
    }

    std::size_t seed = 0;
    for (const Texture& texture : textures) {
        hash_combine(seed, texture.storage, texture.channel_count, texture.exact);
        for (const Placement& placement : texture.placements) {
            hash_combine(seed, placement.field, placement.channel, placement.packed);
        }
    }
    suffix = fmt::format("{:016x}", seed);
    module_name = fmt::format("gbuffer_layout_{}", suffix);

    composition = SlangComposition::create();
    composition->add_module_from_string(GBUFFER_FIELDS_MODULE, fields_module());
    composition->add_module_from_string(module_name, generate_module());
}

const GBufferLayoutHandle& GBufferLayout::complete() {
    static const GBufferLayoutHandle layout =
        std::make_shared<const GBufferLayout>(std::vector<GBufferGroup>{
            {{GBufferField::Normal, GBufferField::LinearZ, GBufferField::GradZ,
              GBufferField::DeltaZ}},
            {{GBufferField::MotionVectors}},
            {{GBufferField::Hit}},
            {{GBufferField::Albedo}},
            {{GBufferField::DiffuseAlbedo}},
            {{GBufferField::SpecularAlbedo}},
            {{GBufferField::Roughness}},
            {{GBufferField::ViewDepth}},
            {{GBufferField::ProjectedDepth}},
        });
    return layout;
}

std::string GBufferLayout::get_type_name(const bool write) const {
    return fmt::format("merian::{}GBuffer_{}", write ? "W" : "", suffix);
}

uint32_t GBufferLayout::get_texture_index(const GBufferField field) const {
    const uint8_t index = field_texture[static_cast<uint32_t>(field)];
    if (index == NO_TEXTURE) {
        throw MerianException{
            fmt::format("the gbuffer layout does not hold {}", traits(field).name)};
    }
    return index;
}

vk::Format GBufferLayout::get_format(const uint32_t texture_index,
                                     const PhysicalDeviceHandle& physical_device) const {
    const Texture& texture = textures[texture_index];
    const std::array<FormatCandidate, 3>& candidates = format_candidates(texture.storage);
    constexpr vk::FormatFeatureFlags required =
        vk::FormatFeatureFlagBits::eStorageImage | vk::FormatFeatureFlagBits::eSampledImage;
    for (const FormatCandidate& candidate : candidates) {
        if (candidate.channel_count >= texture.channel_count &&
            (physical_device->get_physical_device()
                 .getFormatProperties(candidate.format)
                 .optimalTilingFeatures &
             required) == required) {
            return candidate.format;
        }
    }
    return candidates.back().format;
}

void GBufferLayout::add_texture_group(const std::vector<GBufferField>& fields) {
    const bool shared = std::ranges::any_of(textures, [&](const Texture& texture) {
        return texture.exact && texture.placements.size() >= fields.size() &&
               std::equal(fields.begin(), fields.end(), texture.placements.begin(),
                          [](const GBufferField field, const Placement& placement) {
                              return field == placement.field;
                          });
    });
    if (shared) {
        return;
    }

    Texture texture{.storage = Storage::Float16, .exact = true};
    for (const GBufferField field : fields) {
        assert(traits(field).float_channels > 0 && "the field cannot be handed out as a texture");
        texture.storage = std::max(texture.storage, traits(field).float_storage);
        append(texture, field, false);
    }
    assert(texture.channel_count <= TEXTURE_CHANNELS && "too many channels for one texture");
    textures.push_back(std::move(texture));
}

void GBufferLayout::place(const std::vector<GBufferField>& fields,
                          const std::vector<GBufferField>& group) {
    if (fields.empty()) {
        return;
    }

    // 1. next to the fields of the group that are already stored
    for (Texture& texture : textures) {
        if (std::ranges::none_of(group,
                                 [&](const GBufferField field) { return holds(texture, field); })) {
            continue;
        }
        for (const bool packed : {false, true}) {
            if (fits(texture, fields, packed)) {
                for (const GBufferField field : fields) {
                    append(texture, field, packed);
                }
                return;
            }
        }
    }

    const bool can_float = has_form(fields, false);
    const bool can_pack = has_form(fields, true);
    if (!can_float && !can_pack) {
        std::vector<GBufferField> float_fields;
        std::vector<GBufferField> packed_fields;
        for (const GBufferField field : fields) {
            (traits(field).float_channels > 0 ? float_fields : packed_fields).push_back(field);
        }
        place(float_fields, group);
        place(packed_fields, group);
        return;
    }
    // fewer bytes per pixel, float where equal: it needs no decoding
    const bool packed =
        !can_float || (can_pack && bytes_per_pixel(fields, true) < bytes_per_pixel(fields, false));

    // 2. together into the fullest texture that takes them all
    Texture* best = nullptr;
    for (Texture& texture : textures) {
        if (fits(texture, fields, packed) &&
            (best == nullptr || texture.channel_count > best->channel_count)) {
            best = &texture;
        }
    }
    if (best != nullptr) {
        for (const GBufferField field : fields) {
            append(*best, field, packed);
        }
        return;
    }

    // 3. into fresh textures
    fill_fresh_textures(textures, fields, packed);
}

std::string GBufferLayout::generate_module() const {
    std::string read_members;
    std::string write_members;
    std::string texel_members;
    std::string load;
    std::string store;
    for (uint32_t i = 0; i < textures.size(); i++) {
        const Texture& texture = textures[i];
        const char* element = texture.storage == Storage::Uint32 ? "uint4" : "float4";
        read_members += fmt::format("    public RRWWTexture2D<{}, 0> tex{};\n", element, i);
        write_members += fmt::format("    public RRWWTexture2D<{}, 2> tex{};\n", element, i);
        texel_members += fmt::format("    public {} tex{};\n", element, i);
        load += fmt::format("        texels.tex{0} = tex{0}[pixel];\n", i);

        std::vector<std::string> components;
        for (const Placement& placement : texture.placements) {
            components.push_back(encode(placement.field, placement.packed));
        }
        components.resize(components.size() + TEXTURE_CHANNELS - texture.channel_count, "0");
        store += fmt::format("        tex{}.Store(pixel, {}({}));\n", i, element,
                             fmt::join(components, ", "));
    }

    std::string properties;
    for (const GBufferField field : ALL_FIELDS) {
        const FieldTraits& field_traits = traits(field);
        std::string value;
        std::string encoded_normal = "encode_normal(normal)";
        if (contains(field)) {
            const uint32_t index = get_texture_index(field);
            const Placement& placement =
                *std::ranges::find(textures[index].placements, field, &Placement::field);
            const std::string texel = fmt::format(
                "tex{}.{}", index,
                CHANNEL_NAMES.substr(placement.channel, channels(field, placement.packed)));
            value = decode(field, placement.packed, texel);
            if (placement.packed) {
                encoded_normal = texel;
            }
        } else {
            value = field == GBufferField::Hit
                        ? "Hit()"
                        : fmt::format("{}(GBUFFER_UNREQUESTED)", field_traits.type);
        }
        properties += fmt::format("    public property {} {} {{\n"
                                  "        get {{ return {}; }}\n"
                                  "    }}\n",
                                  field_traits.type, field_traits.name, value);
        if (field == GBufferField::Normal) {
            properties += fmt::format("    public property uint encoded_normal {{\n"
                                      "        get {{ return {}; }}\n"
                                      "    }}\n",
                                      encoded_normal);
        }
    }

    return fmt::format(R"(module {0};

import merian_shaders.gbuffer;
import merian_shaders.utils.encoding;
import merian_shaders.utils.textures;

namespace merian {{

public struct GBufferTexels_{1} : IGBufferTexels {{
{2}
{3}}}

public struct GBuffer_{1} : IGBuffer {{
    public typealias Texels = GBufferTexels_{1};

{4}
    public uint2 get_dimensions() {{
        return texture_dimensions(tex0);
    }}

    public Texels load(uint2 pixel) {{
        Texels texels;
{5}        return texels;
    }}
}}

public struct WGBuffer_{1} : IWGBuffer {{
{6}
    public uint2 get_dimensions() {{
        return texture_dimensions(tex0);
    }}

    public void store(uint2 pixel, GBufferSample sample) {{
{7}    }}
}}

export struct GBuffer : IGBuffer = GBuffer_{1};
export struct WGBuffer : IWGBuffer = WGBuffer_{1};

}}
)",
                       module_name, suffix, texel_members, properties, read_members, load,
                       write_members, store);
}

GBuffer::GBuffer(const ShaderCompileContextHandle& compile_context,
                 const ContextHandle& context,
                 const ResourceAllocatorHandle& allocator,
                 const vk::Extent3D extent,
                 const GBufferLayoutHandle& layout)
    : extent(extent), layout(layout) {
    const SlangCompositionHandle composition = SlangComposition::create();
    composition->add_composition(layout->get_composition());
    composition->add_module_from_path("merian-shaders/gbuffer.slang");
    const SlangProgramHandle program = SlangProgram::create(compile_context, composition).get();

    r_shader_object =
        program->create_shader_object_for_type(context, layout->get_type_name(false), allocator);
    w_shader_object =
        program->create_shader_object_for_type(context, layout->get_type_name(true), allocator);
}

void GBuffer::set_resources(const std::vector<ImageViewHandle>& textures) {
    assert(textures.size() == layout->get_textures().size());
    for (const auto& object : {r_shader_object, w_shader_object}) {
        auto cursor = object->get_cursor();
        for (uint32_t i = 0; i < textures.size(); i++) {
            cursor[fmt::format("tex{}", i)].write(textures[i], vk::ImageLayout::eGeneral);
        }
    }
}

} // namespace merian
