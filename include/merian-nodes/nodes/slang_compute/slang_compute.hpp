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
    static constexpr std::string_view COLOR_ATTRIBUTE_NAME = "MerianColor";

  public:
    SlangCompute(const ContextHandle& context,
                 const std::optional<vk::Format> output_format = std::nullopt);

    ~SlangCompute() override;

    std::vector<InputConnectorHandle> describe_inputs() override;

    std::vector<OutputConnectorHandle> describe_outputs(const NodeIOLayout& io_layout) override;

    const void* get_push_constant(GraphRun& run, const NodeIO& io) override;

    std::tuple<uint32_t, uint32_t, uint32_t>
    get_group_count(const NodeIO& io) const noexcept override;
    static std::tuple<uint32_t, uint32_t, uint32_t>
    reflect_workgroup_size(slang::EntryPointReflection* entry_point);

    VulkanEntryPointHandle get_entry_point() override;

    NodeStatusFlags properties(Properties& config) override;

  private:
    void make_spec_info();
    void load_shader(const std::string& path);

    void reflect_input_connectors(slang::EntryPointReflection* entry_point);

    void reflect_output_connectors(const NodeIOLayout& io_layout,
                                 slang::EntryPointReflection* entry_point);

    bool reflect_properties(Properties& config, slang::EntryPointReflection* entry_point);

    static std::vector<slang::VariableLayoutReflection*>
    get_variable_layouts_from_scope(slang::VariableLayoutReflection* scope_var_layout);

    static std::vector<slang::VariableLayoutReflection*>
    reflect_fields_from_entry_point_parameter_struct(slang::EntryPointReflection* entry_point,
                                               const std::string& parameter_name);

    static std::vector<slang::VariableLayoutReflection*>
    reflect_fields_from_struct(slang::VariableLayoutReflection* struct_layout);

    size_t get_size_for_buffer_output_connector(const NodeIOLayout& io_layout,
                                           slang::VariableReflection* var) const;

    vk::Extent3D get_extent_for_image_output_connector(const NodeIOLayout& io_layout,
                                                  slang::VariableReflection* var) const;

    static vk::Format get_format_for_image_output_connector(slang::TypeReflection* type);

    static slang::Attribute* find_var_attribute_by_name(slang::VariableReflection* var,
                                                    const std::string& name);
    static slang::Attribute* find_func_attribute_by_name(slang::FunctionReflection* var,
                                                     const std::string& name);

    const std::optional<vk::Format> output_format;

    VkSampledImageInHandle con_src;

    std::unordered_map<std::string, VkImageInHandle> image_in_connectors;
    std::unordered_map<std::string, VkBufferInHandle> buffer_in_connectors;

    std::unordered_map<std::string, VkImageOutHandle> image_out_connectors;
    std::unordered_map<std::string, VkBufferOutHandle> buffer_out_connectors;

    std::unordered_map<std::string, std::unique_ptr<int>> int_properties;
    std::unordered_map<std::string, std::unique_ptr<uint>> uint_properties;
    std::unordered_map<std::string, std::unique_ptr<float>> float_properties;
    std::unordered_map<std::string, std::unique_ptr<bool>> bool_properties;
    std::unordered_map<std::string, std::unique_ptr<std::string>> string_properties;
    std::unordered_map<std::string, std::unique_ptr<glm::vec4>> vector_properties;

    std::string shader_path;

    VulkanEntryPointHandle shader;
    SpecializationInfoHandle spec_info;

    slang::ProgramLayout* program_layout;
};

} // namespace merian_nodes
