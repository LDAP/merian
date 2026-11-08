#include "merian-graph/nodes/bloom/bloom.hpp"

#include "merian-graph/graph/errors.hpp"
#include "merian/vk/pipeline/specialization_info_builder.hpp"

#include <cmath>
#include <numbers>

namespace merian {

namespace {
constexpr const char* DOWNSAMPLE_MODULE = "merian-graph/nodes/bloom/bloom_downsample.slang";
constexpr const char* UPSAMPLE_MODULE = "merian-graph/nodes/bloom/bloom_upsample.slang";
constexpr const char* COMPOSITE_MODULE = "merian-graph/nodes/bloom/bloom_composite.slang";
constexpr vk::Format MIP_FORMAT = vk::Format::eR32G32B32A32Sfloat;

vk::Extent3D level_extent(const ImageHandle& image, const uint32_t level) {
    const vk::Extent3D& base = image->get_extent();
    return {std::max(1u, base.width >> level), std::max(1u, base.height >> level), 1};
}

void write_levels(ShaderCursor cursor,
                  const std::string_view name,
                  const std::vector<TextureHandle>& levels,
                  const bool storage) {
    ShaderCursor field = cursor.find(name);
    if (!field.is_valid()) {
        return;
    }
    for (uint32_t i = 0; i < field.get_element_count(); i++) {
        const TextureHandle& level = levels[std::min<size_t>(i, levels.size() - 1)];
        if (storage) {
            field[i].write(level->get_view(), vk::ImageLayout::eGeneral);
        } else {
            field[i].write(level, vk::ImageLayout::eGeneral);
        }
    }
}
} // namespace

Bloom::Bloom() {}

Bloom::~Bloom() {}

DeviceSupportInfo Bloom::query_device_support(const DeviceSupportQueryInfo& query_info) {
    DeviceSupportInfo support{true};
    support.required_features.emplace_back("shaderSampledImageArrayDynamicIndexing");
    support.required_features.emplace_back("shaderStorageImageArrayDynamicIndexing");
    for (const char* module : {DOWNSAMPLE_MODULE, UPSAMPLE_MODULE, COMPOSITE_MODULE}) {
        const auto composition = SlangComposition::create();
        composition->add_module_from_path(module, true);
        support = support & SlangProgram::create(query_info.compile_context, composition)
                                .get()
                                ->query_device_support(query_info);
    }
    return support;
}

void Bloom::initialize(const ContextHandle& context, const ResourceAllocatorHandle& allocator) {
    this->compile_context = context->get_shader_compile_context();
    con_src = VkSampledImageIn::create(allocator->get_sampler_pool()->linear_clamp_to_edge());

    auto spec_builder = SpecializationInfoBuilder();
    spec_builder.add_entry(local_size_x, local_size_y, mode);
    spec_info.set(spec_builder.build());

    downsample_kernel.emplace(context, allocator, compile_context, DOWNSAMPLE_MODULE, spec_info);
    upsample_kernel.emplace(context, allocator, compile_context, UPSAMPLE_MODULE, spec_info);
    composite_kernel.emplace(context, allocator, compile_context, COMPOSITE_MODULE, spec_info);
}

std::vector<InputConnectorDescriptor> Bloom::describe_inputs() {
    return {{.name = "src",
             .connector = con_src,
             .access = ConnectorAccess::compute_read,
             .pass_through = "out"}};
}

std::vector<OutputConnectorDescriptor> Bloom::describe_outputs(const NodeIOLayout& io_layout) {
    const vk::ImageCreateInfo create_info = io_layout[con_src]->get_create_info_or_throw();
    const vk::Format format =
        overwrite_format != vk::Format::eUndefined ? overwrite_format : create_info.format;

    con_out = ManagedVkImageOut::create(format, create_info.extent);

    return {{"out", con_out, ConnectorAccess::compute_write}};
}

Bloom::NodeStatusFlags Bloom::on_connected(const NodeIOLayout& io_layout,
                                           const NodeIO& io,
                                           [[maybe_unused]] const NodeConnectionInfo& info,
                                           [[maybe_unused]] Submission& submission) {
    io_layout.register_event_listener(
        "/graph/reload_shaders", [this](const GraphEvent::Info&, const GraphEvent::Data& force) {
            for (auto* kernel : {&downsample_kernel, &upsample_kernel, &composite_kernel}) {
                (*kernel)->reload(std::any_cast<bool>(force), compile_context);
            }
        });

    const vk::Extent3D extent = io_layout[con_src]->get_create_info_or_throw().extent;
    const vk::Extent3D base{std::max(1u, extent.width / 2), std::max(1u, extent.height / 2), 1};
    const uint32_t max_level_count =
        static_cast<uint32_t>(std::floor(std::log2(std::max(base.width, base.height)))) + 1;
    chain_level_count = std::min(max_levels, max_level_count);
    if (chain_level_count < 2) {
        throw graph_errors::node_error{"bloom needs an input of at least 4 pixels"};
    }
    image_height = extent.height;

    const ResourceAllocatorHandle& allocator = io.get_allocator();
    const vk::ImageCreateInfo image_info{{},
                                         vk::ImageType::e2D,
                                         MIP_FORMAT,
                                         base,
                                         chain_level_count,
                                         1,
                                         vk::SampleCountFlagBits::e1,
                                         vk::ImageTiling::eOptimal,
                                         vk::ImageUsageFlagBits::eStorage |
                                             vk::ImageUsageFlagBits::eSampled};
    const SamplerHandle sampler = allocator->get_sampler_pool()->linear_clamp_to_edge();
    const auto make_levels = [&](const ImageHandle& image, std::vector<TextureHandle>& out) {
        out.clear();
        for (uint32_t level = 0; level < chain_level_count; level++) {
            const vk::ImageViewCreateInfo view_info{
                {},         *image, vk::ImageViewType::e2D,
                MIP_FORMAT, {},     {vk::ImageAspectFlagBits::eColor, level, 1, 0, 1}};
            out.emplace_back(allocator->create_texture(image, view_info, sampler));
        }
    };
    down_image = allocator->create_image(image_info, MemoryMappingType::NONE, "bloom down");
    up_image = allocator->create_image(image_info, MemoryMappingType::NONE, "bloom up");
    make_levels(down_image, down_levels);
    make_levels(up_image, up_levels);

    return {};
}

std::vector<float> Bloom::level_weights(const uint32_t level_count) const {
    const float exponent = 2.0f - falloff;
    const auto energy = [exponent](const float radius) {
        return std::abs(exponent) < 1e-4f ? std::log(radius)
                                          : std::pow(radius, exponent) / exponent;
    };
    std::vector<float> weights(level_count, 0.0f);
    float sum = 0;
    for (uint32_t level = 0; level < level_count; level++) {
        const float center =
            std::exp2(static_cast<float>(level) + 1.0f) / static_cast<float>(image_height);
        const float lo = std::max(center * std::numbers::sqrt2_v<float> * 0.5f, core);
        const float hi = std::min(center * std::numbers::sqrt2_v<float>, size);
        if (hi > lo) {
            weights[level] = energy(hi) - energy(lo);
            sum += weights[level];
        }
    }
    if (sum <= 0.0f) {
        weights.front() = 1.0f;
        return weights;
    }
    for (float& weight : weights) {
        weight /= sum;
    }
    return weights;
}

void Bloom::bind_mip_chain() {
    for (auto* kernel : {&downsample_kernel, &upsample_kernel, &composite_kernel}) {
        const ShaderCursor cursor = (*kernel)->globals_cursor();
        write_levels(cursor, "down_tex", down_levels, false);
        write_levels(cursor, "down_img", down_levels, true);
        write_levels(cursor, "up_tex", up_levels, false);
        write_levels(cursor, "up_img", up_levels, true);
    }
}

void Bloom::barrier_written(const CommandBufferHandle& cmd, const ImageHandle& image) {
    cmd->barrier(vk::PipelineStageFlagBits::eComputeShader,
                 vk::PipelineStageFlagBits::eComputeShader,
                 image->barrier(vk::ImageLayout::eGeneral, vk::AccessFlagBits::eShaderWrite,
                                vk::AccessFlagBits::eShaderRead));
}

[[nodiscard]] Bloom::NodeStatusFlags
Bloom::process(const NodeIO& io, const NodeProcessInfo& info, Submission& submission) {
    const CommandBufferHandle& cmd = submission.get_cmd();
    bind_mip_chain();

    const auto wanted_level_count = static_cast<uint32_t>(
        std::max(0.0f, std::ceil(std::log2(size * static_cast<float>(image_height) *
                                           std::numbers::sqrt2_v<float>)) -
                           1.0f));
    const uint32_t level_count = std::max(2u, std::min(wanted_level_count, chain_level_count));

    const auto discard = [](const ImageHandle& image) {
        return image->barrier(vk::ImageLayout::eGeneral, {}, vk::AccessFlagBits::eShaderWrite,
                              VK_QUEUE_FAMILY_IGNORED, VK_QUEUE_FAMILY_IGNORED,
                              all_levels_and_layers(), true);
    };
    cmd->barrier(vk::PipelineStageFlagBits::eComputeShader,
                 vk::PipelineStageFlagBits::eComputeShader,
                 {discard(down_image), discard(up_image)});

    const auto down_pipe = downsample_kernel->bind(io, info, submission);
    for (uint32_t level = 0; level < level_count; level++) {
        pc.level = static_cast<int32_t>(level);
        pc.karis = static_cast<int32_t>(karis_average && level == 0);
        cmd->push_constant(down_pipe, pc);
        cmd->dispatch(level_extent(down_image, level), local_size_x, local_size_y);
        barrier_written(cmd, down_image);
    }

    const std::vector<float> weights = level_weights(level_count);
    const auto up_pipe = upsample_kernel->bind(io, info, submission);
    for (int32_t level = static_cast<int32_t>(level_count) - 2; level >= 0; level--) {
        pc.level = level;
        pc.coarse_is_down = static_cast<int32_t>(level == static_cast<int32_t>(level_count) - 2);
        pc.coarse_weight = pc.coarse_is_down != 0 ? weights[level_count - 1] : 1.0f;
        pc.fine_weight = weights[level];
        cmd->push_constant(up_pipe, pc);
        cmd->dispatch(level_extent(up_image, level), local_size_x, local_size_y);
        barrier_written(cmd, up_image);
    }

    const auto composite_pipe = composite_kernel->bind(io, info, submission);
    cmd->push_constant(composite_pipe, pc);
    cmd->dispatch(io[con_out]->get_extent(), local_size_x, local_size_y);
    return {};
}

Bloom::NodeStatusFlags Bloom::properties(Properties& config) {
    config.config_float("intensity", pc.intensity,
                        "Fraction of the light scattered into the glare.", 0.001f, 0.0f, 1.0f);
    config.config_float(
        "falloff", falloff,
        "Exponent of the point spread function's tail (r^-falloff): 2 spreads light "
        "evenly over all sizes, larger values keep it tight around the source.",
        0.01f, 0.5f, 6.0f);
    config.config_float("radius", pc.radius, "Tent filter radius when upsampling, in texels.",
                        0.01f, 0.1f, 4.0f);
    config.config_bool("karis average", karis_average,
                       "Weights the first downsample by 1 / (1 + luminance) against fireflies; "
                       "loses energy of bright sources, so use it after exposure.");
    config.config_float("size", size, "Largest blur radius as a fraction of the image height.",
                        0.01f, 0.001f, 1.0f);
    config.config_float("core", core,
                        "Smallest glare radius as a fraction of the image height; finer detail "
                        "stays sharp, so the glare does not change with the resolution.",
                        0.0001f, 0.0f, 0.1f);

    config.st_separate("Debug");
    const bool mode_changed =
        config.config_options("mode", mode, {"combined", "bloom only", "bloom off"});
    if (mode_changed) {
        auto spec_builder = SpecializationInfoBuilder();
        spec_builder.add_entry(local_size_x, local_size_y, mode);
        spec_info.set(spec_builder.build());
    }

    config.st_separate();
    if (config.config_enum("overwrite format", overwrite_format, Properties::OptionsStyle::COMBO,
                           "Undefined keeps the format of the input.")) {
        return NEEDS_RECONNECT;
    }

    return {};
}

} // namespace merian
