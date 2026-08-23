#pragma once

#include "merian-shaders/sampling/guiding.hpp"

#include "merian/vk/memory/resource_allocator.hpp"

#include <array>
#include <utility>

namespace merian {

// Screen-space mixture models: one vMF lobe per pixel, fitted over frames and reused across the
// neighbourhood. Describes the first scattering vertex only.
class SSMMGuidingModel : public GuidingModel {
  public:
    void initialize(const ContextHandle& context,
                    const ResourceAllocatorHandle& allocator) override;

    static SlangCompositionHandle query_device_support_composition();

    SlangCompositionHandle get_composition() const override;

    std::vector<std::string> get_slang_imports() const override;

    std::string get_type_name() const override;

    // One fit per pixel of the render target.
    void on_extent(const vk::Extent3D& extent);

    // The gbuffer the fits are reprojected through, for the frame about to be rendered.
    void set_gbuffer(const ShaderObjectHandle& gbuffer);

    void write_to(ShaderCursor cursor) override;

    void reset(const CommandBufferHandle& cmd) override;

    // Orders the previous frame's writes against this frame's reads.
    std::array<vk::BufferMemoryBarrier2, 2> carry_barriers() const;

    bool properties(Properties& props) override;

  private:
    ResourceAllocatorHandle allocator;
    ShaderObjectHandle gbuffer;

    vk::Extent3D extent{};
    // read and written alternately, so a frame sees the fits of the one before it
    std::array<BufferHandle, 2> states;
    uint32_t frame = 0;

    int32_t group_size = 5;
    float reuse_radius = 15.0f;
    float probability = 0.5f;
    float alpha_threshold = 0.05f;
    uint32_t max_n = 1024;
    float min_alpha = 0.01f;
    float prior_n = 0.2f;
};

} // namespace merian
