#pragma once

#include "merian-graph/objects/gbuffer_object.hpp"
#include "merian-shaders/sampling/guiding.hpp"

#include "merian/vk/memory/resource_allocator.hpp"

#include <array>
#include <utility>

namespace merian {

class SSMMGuidingModel : public GuidingModel {
  public:
    void initialize(const ContextHandle& context,
                    const ResourceAllocatorHandle& allocator) override;

    static SlangCompositionHandle query_device_support_composition();

    SlangCompositionHandle get_composition() const override;

    std::vector<std::string> get_slang_imports() const override;

    std::string get_type_name() const override;

    void on_extent(const vk::Extent3D& extent);

    void set_gbuffer_output(const std::weak_ptr<GBufferOut>& output);

    void set_gbuffer(const ShaderObjectHandle& gbuffer);

    void write_to(ShaderCursor cursor) override;

    void reset(const CommandBufferHandle& cmd) override;

    std::array<vk::BufferMemoryBarrier2, 2> carry_barriers() const;

    bool properties(Properties& props) override;

  private:
    GBufferLayoutHandle get_gbuffer_layout() const;

    ResourceAllocatorHandle allocator;
    std::weak_ptr<GBufferOut> gbuffer_output;
    mutable GBufferLayoutHandle gbuffer_layout;
    ShaderObjectHandle gbuffer;

    vk::Extent3D extent{};
    std::array<BufferHandle, 2> states;
    uint32_t frame = 0;

    int32_t group_size = 5;
    float reuse_radius = 15.0f;
    uint32_t max_n = 1024;
    float min_alpha = 0.01f;
    float prior_n = 0.2f;
};

} // namespace merian
