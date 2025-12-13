#pragma once

#include "merian-nodes/connectors/buffer/vk_buffer_in.hpp"
#include "merian-nodes/connectors/image/vk_image_in_sampled.hpp"
#include "merian-nodes/nodes/compute_node/compute_node.hpp"

#include <slang.h>

namespace merian_nodes {

class SlangCompute : public AbstractCompute {

private:
    static constexpr std::string_view INPUT_STRUCT_PARAMETER_NAME = "node_in";
    static constexpr std::string_view OUTPUT_STRUCT_PARAMETER_NAME = "node_out";
    static constexpr std::string_view PROPERTY_STRUCT_PARAMETER_NAME = "node_props";

    static constexpr std::string_view TARGET_ATTRIBUTE_NAME = "MerianOperateOn";

    static constexpr std::string_view STATIC_EXTENT_ATTRIBUTE_NAME = "MerianExtentStatic";
    static constexpr std::string_view EXTENT_AS_ATTRIBUTE_NAME = "MerianExtentAs";
    static constexpr std::string_view STATIC_SIZE_ATTRIBUTE_NAME = "MerianSizeStatic";
    static constexpr std::string_view SIZE_AS_ATTRIBUTE_NAME = "MerianExtentAs";

    static constexpr std::string_view INT_RANGE_ATTRIBUTE_NAME = "MerianIntRange";
    static constexpr std::string_view FLOAT_RANGE_ATTRIBUTE_NAME = "MerianFloatRange";

public:
    SlangCompute(const ContextHandle& context,
            const std::optional<vk::Format> output_format = std::nullopt);

    ~SlangCompute();

    std::vector<InputConnectorHandle> describe_inputs() override;

    std::vector<OutputConnectorHandle> describe_outputs(const NodeIOLayout& io_layout) override;

    const void* get_push_constant(GraphRun& run, const NodeIO& io) override;

    std::tuple<uint32_t, uint32_t, uint32_t>
    get_group_count(const NodeIO& io) const noexcept override;
    static std::tuple<uint32_t, uint32_t, uint32_t>
    reflectWorkgroupSize(slang::EntryPointReflection* entry_point);

    VulkanEntryPointHandle get_entry_point() override;

    NodeStatusFlags properties(Properties& config) override;

private:
    void make_spec_info();
    void loadShader(const std::string& path);

    void reflectInputConnectors(slang::EntryPointReflection* entry_point);

    void reflectOutputConnectors(const NodeIOLayout& io_layout,
                            slang::EntryPointReflection* entry_point);

    void reflectProperties(Properties& config, slang::EntryPointReflection* entry_point);


    std::vector<slang::VariableLayoutReflection*> getVariableLayoutsFromScope(slang::VariableLayoutReflection* scope_var_layout);

    std::vector<slang::VariableLayoutReflection*>
    reflectFieldsFromEntryPointParameterStruct(slang::EntryPointReflection* entry_point,
                                               const std::string& parameter_name);

    static std::vector<slang::VariableLayoutReflection*> reflectFieldsFromStruct(slang::VariableLayoutReflection* struct_layout);

    size_t getSizeForBufferOutputConnector(const NodeIOLayout& io_layout,
                                           slang::VariableReflection* var) const;


    vk::Extent3D getExtentForImageOutputConnector(const NodeIOLayout& io_layout,
                                             slang::VariableReflection* var) const;

    vk::Format getFormatForImageOutputConnector(slang::TypeReflection* type);


    static slang::Attribute* findVarAttributeByName(slang::VariableReflection* var, const std::string& name);
    static slang::Attribute* findFuncAttributeByName(slang::FunctionReflection* var,
                                              const std::string& name);

    const std::optional<vk::Format> output_format;

    VkSampledImageInHandle con_src;

    std::unordered_map<std::string, VkImageInHandle> image_in_connectors;
    std::unordered_map<std::string, VkBufferInHandle> buffer_in_connectors;

    std::unordered_map<std::string, VkImageOutHandle> image_out_connectors;
    std::unordered_map<std::string, VkBufferOutHandle> buffer_out_connectors;

    std::string shader_path;

    VulkanEntryPointHandle shader;
    SpecializationInfoHandle spec_info;

    slang::ProgramLayout* program_layout;
};

} // namespace merian_nodes
