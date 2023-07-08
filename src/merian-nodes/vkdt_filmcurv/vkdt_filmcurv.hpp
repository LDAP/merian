#pragma once

#include "merian-nodes/compute_node/compute_node.hpp"
#include "merian/vk/memory/resource_allocator.hpp"

namespace merian {

class VKDTFilmcurv : public ComputeNode {

  private:
    static constexpr uint32_t local_size_x = 16;
    static constexpr uint32_t local_size_y = 16;

  public:
    struct Options {
        float brightness{1.};
        float contrast{1.};
        float bias{0.};
        int32_t colourmode{1};
    };

  public:
    VKDTFilmcurv(const SharedContext context,
                 const ResourceAllocatorHandle allocator,
                 const vk::Format output_format = vk::Format::eR16G16B16A16Sfloat,
                 const std::optional<Options> options = std::nullopt);

    std::string name() override {
        return "VKDT Filmcurv";
    }

    std::tuple<std::vector<NodeInputDescriptorImage>, std::vector<NodeInputDescriptorBuffer>>
    describe_inputs() override;

    std::tuple<std::vector<merian::NodeOutputDescriptorImage>,
               std::vector<merian::NodeOutputDescriptorBuffer>>
    describe_outputs(const std::vector<merian::NodeOutputDescriptorImage>&,
                     const std::vector<merian::NodeOutputDescriptorBuffer>&) override;

    SpecializationInfoHandle get_specialization_info() const noexcept override;

    const void* get_push_constant() override;

    std::tuple<uint32_t, uint32_t, uint32_t> get_group_count() const noexcept override;

    ShaderModuleHandle get_shader_module() override;

    void get_configuration(Configuration& config) override;

  private:
    const vk::Format output_format;
    uint32_t width, height;
    Options pc;
};

} // namespace merian
