#pragma once

#include "merian-shaders/sampling/guiding.hpp"

#include "merian/vk/memory/resource_allocator.hpp"

#include <array>
#include <vector>

namespace merian {

// Per-pixel Markov chains over the scattering distance, in a screen-space grid the node carries
// across frames.
class MCPGDistanceGuidingModel : public GuidingModel {
  public:
    static constexpr uint32_t MAX_LEVELS = 16;

    static SlangCompositionHandle query_device_support_composition();

    void initialize(const ContextHandle& context,
                    const ResourceAllocatorHandle& allocator) override;

    SlangCompositionHandle get_composition() const override;

    std::vector<std::string> get_slang_imports() const override;

    std::string get_type_name() const override;

    void write_to(ShaderCursor cursor) override;

    void reset(const CommandBufferHandle& cmd) override;

    bool properties(Properties& props) override;

    // --- node facing ---

    // Cells and levels follow the render target; returns true where the grid was rebuilt.
    bool on_extent(const vk::Extent3D& extent);

    uint32_t get_level_count() const {
        return level_count;
    }

    // The grid the tracer reads and writes this frame, and the one the projection reads.
    const ImageHandle& get_grid() const {
        return grids[current];
    }

    const ImageHandle& get_prev_grid() const {
        return grids[current ^ 1u];
    }

    const std::vector<TextureHandle>& get_levels() const {
        return level_views[current];
    }

    const TextureHandle& get_prev_texture() const {
        return grid_textures[current ^ 1u];
    }

    void swap() {
        current ^= 1u;
    }

  private:
    ResourceAllocatorHandle allocator;

    vk::Extent3D extent{};
    uint32_t level_count = 0;
    uint32_t current = 0;
    std::array<ImageHandle, 2> grids;
    std::array<TextureHandle, 2> grid_textures;
    std::array<std::vector<TextureHandle>, 2> level_views;

    int32_t samples = 3;
    float probability = 0.9f;
    float base_width = 4.f;
    float max_width = 128.f;
    float distribution_dimension = 2.f;
};

} // namespace merian
