#pragma once

#include "merian-nodes/connectors/image/vk_image_in_sampled.hpp"
#include "merian-nodes/nodes/compute_node/compute_node.hpp"
#include "config.h"

#include <slang.h>

namespace merian_nodes {

class SlangCompute : public AbstractCompute {

private:
    static constexpr uint32_t local_size_x = 16;
    static constexpr uint32_t local_size_y = 16;

    struct PushConstant {
        float param1 = 1.0;
        float param2 = 1.0;
        float param3 = 1.0;
        float param4 = 1.0;
        float param5 = 1.0;

        float perceptual_exponent = 2.2;
    };

public:
    SlangCompute(const ContextHandle& context,
            const std::optional<vk::Format> output_format = std::nullopt);

    ~SlangCompute();

    std::vector<InputConnectorHandle> describe_inputs() override;

    std::vector<OutputConnectorHandle> describe_outputs(const NodeIOLayout& io_layout) override;

    const void* get_push_constant(GraphRun& run, const NodeIO& io) override;

    std::tuple<uint32_t, uint32_t, uint32_t>
    get_group_count(const NodeIO& io) const noexcept override;

    VulkanEntryPointHandle get_entry_point() override;

    NodeStatusFlags properties(Properties& config) override;

private:
    void make_spec_info();

    std::vector<InputConnectorHandle> reflectInputConnectors(slang::EntryPointReflection* entry_point);
    std::vector<OutputConnectorHandle> reflectOutputConnectors(const NodeIOLayout& io_layout,
                            slang::EntryPointReflection* entry_point);

    std::vector<slang::VariableLayoutReflection*> getVariableLayoutsFromScope(slang::VariableLayoutReflection* scope_var_layout);
    std::vector<slang::VariableLayoutReflection*> reflectFieldsFromStruct(slang::VariableLayoutReflection* struct_layout);

    vk::Extent3D getExtentForImageOutputConnector(const NodeIOLayout& io_layout,
                                             slang::VariableReflection* var) const;
    vk::Format getFormatForImageOutputConnector(slang::TypeReflection* type);

    static slang::Attribute* findAttributeByName(slang::VariableReflection* var, const std::string& name);
    InputConnectorHandle findInputConnectorByName(const std::string& name) const;

    const std::optional<vk::Format> output_format;

    VkSampledImageInHandle con_src;

    std::vector<InputConnectorHandle> input_connectors;
    std::vector<OutputConnectorHandle> output_connectors;

    vk::Extent3D extent;
    PushConstant pc;
    VulkanEntryPointHandle shader;
    SpecializationInfoHandle spec_info;

    slang::ProgramLayout* program_layout;

    int32_t tonemap = 0;
    int32_t alpha_mode = 0;
    int32_t clamp_output = 1;
};

} // namespace merian_nodes
