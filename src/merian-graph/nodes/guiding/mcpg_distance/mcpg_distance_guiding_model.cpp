#include "merian-graph/nodes/guiding/mcpg_distance/mcpg_distance_guiding_model.hpp"

#include "merian/utils/properties.hpp"

#include <cmath>
#include <fmt/format.h>

namespace merian {

namespace {

constexpr const char* GUIDING_MODULE =
    "merian-graph/nodes/guiding/mcpg_distance/mcpg-distance-guiding.slang";
constexpr const char* CHAIN_MODULE = "merian-graph/nodes/guiding/mcpg_distance/mc_distance.slang";

} // namespace

SlangCompositionHandle MCPGDistanceGuidingModel::query_device_support_composition() {
    const auto composition = SlangComposition::create();
    composition->add_module_from_path(CHAIN_MODULE);
    composition->add_module_from_path(GUIDING_MODULE);
    return composition;
}

void MCPGDistanceGuidingModel::initialize([[maybe_unused]] const ContextHandle& context,
                                          const ResourceAllocatorHandle& allocator) {
    this->allocator = allocator;
}

SlangCompositionHandle MCPGDistanceGuidingModel::get_composition() const {
    const auto composition = SlangComposition::create();
    composition->add_module_from_path(CHAIN_MODULE);
    composition->add_module_from_path(GUIDING_MODULE);
    composition->add_module_from_string(
        "mcpg_distance_guiding_constants",
        fmt::format("namespace merian {{\n"
                    "export static const int merian_distance_guiding_samples = {};\n"
                    "export static const float merian_distance_guiding_probability = {};\n"
                    "}}\n"
                    "export static const float distance_mc_base_width = {};\n"
                    "export static const uint distance_mc_level_count = {}u;\n"
                    "export static const float distance_mc_distribution_dimension = {};",
                    samples, probability, base_width, std::max(level_count, 1u),
                    distribution_dimension));
    return composition;
}

std::vector<std::string> MCPGDistanceGuidingModel::get_slang_imports() const {
    return {GUIDING_MODULE};
}

std::string MCPGDistanceGuidingModel::get_type_name() const {
    return "merian::MCPGDistanceGuiding";
}

void MCPGDistanceGuidingModel::write_to(ShaderCursor cursor) {
    auto levels = cursor["distance_mc"]["levels"];
    for (uint32_t level = 0; level < level_count; level++) {
        levels[level] = level_views[current][level];
    }
}

void MCPGDistanceGuidingModel::reset([[maybe_unused]] const CommandBufferHandle& cmd) {
    // the grid is transient: the projection rebuilds it every frame
}

bool MCPGDistanceGuidingModel::on_extent(const vk::Extent3D& new_extent) {
    const uint32_t cells_x = uint32_t(std::ceil(new_extent.width / base_width)) + 2;
    const uint32_t cells_y = uint32_t(std::ceil(new_extent.height / base_width)) + 2;
    // Coarser than the configured cell width buys nothing, and a level past the image is empty.
    const uint32_t levels_to_max_width =
        uint32_t(std::floor(std::log2(std::max(max_width / base_width, 1.f)))) + 1;
    const uint32_t levels_in_image =
        uint32_t(std::floor(std::log2(float(std::max(cells_x, cells_y))))) + 1;
    const uint32_t levels = std::min({levels_to_max_width, levels_in_image, MAX_LEVELS});

    if (new_extent == extent && levels == level_count && grids[0]) {
        return false;
    }
    extent = new_extent;
    level_count = levels;

    const vk::ImageCreateInfo info{{},
                                   vk::ImageType::e2D,
                                   vk::Format::eR32G32B32A32Sfloat,
                                   {cells_x, cells_y, 1},
                                   level_count,
                                   1,
                                   vk::SampleCountFlagBits::e1,
                                   vk::ImageTiling::eOptimal,
                                   vk::ImageUsageFlagBits::eStorage |
                                       vk::ImageUsageFlagBits::eSampled,
                                   vk::SharingMode::eExclusive};

    for (uint32_t i = 0; i < grids.size(); i++) {
        grids[i] = allocator->create_image(info, MemoryMappingType::NONE);
        grid_textures[i] = allocator->create_texture(grids[i], grids[i]->make_view_create_info());
        level_views[i].clear();
        for (uint32_t level = 0; level < level_count; level++) {
            const vk::ImageViewCreateInfo view{
                {},
                *grids[i],
                vk::ImageViewType::e2D,
                grids[i]->get_format(),
                {},
                vk::ImageSubresourceRange{vk::ImageAspectFlagBits::eColor, level, 1, 0, 1}};
            level_views[i].emplace_back(allocator->create_texture(grids[i], view));
        }
    }
    return true;
}

bool MCPGDistanceGuidingModel::properties(Properties& props) {
    bool constants_changed = false;
    bool recreate = false;

    constants_changed |= props.config_int("distance MC samples", samples,
                                          "Distance chains resampled towards each ray.", 1, 16);
    constants_changed |= props.config_float(
        "distance guiding probability", probability,
        "Probability of drawing the scattering distance from the chains instead of the "
        "transmittance.",
        0.01f, 0.f, 1.f);
    recreate |= props.config_float("distance MC grid width", base_width,
                                   "Cell width in pixels at the finest level.", 0.5f, 1.f, 64.f);
    recreate |=
        props.config_float("distance MC max width", max_width,
                           "Cell width in pixels the coarsest level stops at.", 1.f, 1.f, 1024.f);
    constants_changed |= props.config_float(
        "distance MC states per vertex", distribution_dimension,
        "Effective dimensionality the levels are spread over; smaller widens the spread.", 0.1f,
        0.1f, 8.f);

    if (recreate && !extent.width) {
        return constants_changed;
    }
    if (recreate) {
        const vk::Extent3D previous = extent;
        extent = vk::Extent3D{};
        on_extent(previous);
        constants_changed = true;
    }
    return constants_changed;
}

} // namespace merian
