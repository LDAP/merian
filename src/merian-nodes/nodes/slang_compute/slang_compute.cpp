#include "merian-nodes/connectors/image/vk_image_in_sampled.hpp"
#include "merian-nodes/connectors/image/vk_image_out_managed.hpp"
#include "merian-nodes/nodes/slang_compute/slang_compute.hpp"
#include "merian/vk/pipeline/specialization_info_builder.hpp"
#include "merian/vk/shader/entry_point.hpp"

#include "merian-nodes/connectors/buffer/vk_buffer_in.hpp"
#include "merian-nodes/connectors/buffer/vk_buffer_out_managed.hpp"
#include "merian-nodes/graph/errors.hpp"
#include "merian/vk/shader/slang_entry_point.hpp"
#include "tonemap.slang.spv.h"

#include <fmt/chrono.h>

namespace merian_nodes {

SlangCompute::SlangCompute(const ContextHandle& context, const std::optional<vk::Format> output_format)
    : AbstractCompute(context, 0), output_format(output_format) {
    make_spec_info();
}

SlangCompute::~SlangCompute() {}

void SlangCompute::make_spec_info() {
    auto spec_builder = SpecializationInfoBuilder();
    spec_info = spec_builder.build();
}

void SlangCompute::loadShader(const std::string& path_string) {
    std::filesystem::path path = path_string;
    if (path.empty()) {
        throw graph_errors::node_error{"no shader set"};
    }
    if (!std::filesystem::exists(path)) {
        throw graph_errors::node_error{fmt::format("shader does not exist: {}", path.string())};
    }
    if (path.extension().string() != ".slang") {
        throw graph_errors::node_error{fmt::format("shader is not a slang shader: {}", path.string())};
    }

    const GLSLShaderCompilerHandle compiler = GLSLShaderCompiler::get();
    ShaderCompileContextHandle compilation_session_desc = ShaderCompileContext::create(context);
    compilation_session_desc->add_search_path(path.parent_path().string());

    SlangProgramEntryPointHandle slang_entry_point = SlangProgramEntryPoint::create(compilation_session_desc, path.filename().string());
    shader = VulkanEntryPoint::create(slang_entry_point, spec_info);
    program_layout = slang_entry_point->get_program()->get_program_reflection();
}

std::vector<slang::VariableLayoutReflection*> SlangCompute::getVariableLayoutsFromScope(slang::VariableLayoutReflection* scope_var_layout) {
    std::vector<slang::VariableLayoutReflection*> result{};

    auto scope_type_layout = scope_var_layout->getTypeLayout();
    switch (scope_type_layout->getKind())
    {
    case slang::TypeReflection::Kind::Struct:
        {
            int paramCount = scope_type_layout->getFieldCount();
            for (int i = 0; i < paramCount; i++)
            {

                slang::VariableLayoutReflection* param = scope_type_layout->getFieldByIndex(i);
                result.push_back(param);
            }
        }
        break;
    }

    return result;
}

std::vector<InputConnectorHandle> SlangCompute::reflectInputConnectors(slang::EntryPointReflection* entry_point) {
    std::vector<InputConnectorHandle> result{};
    std::vector<slang::VariableLayoutReflection*> reflected_inputs{};

    const uint32_t param_count = entry_point->getParameterCount();
    for (uint32_t i = 0; i < param_count; i++) {
        slang::VariableLayoutReflection* var_layout = entry_point->getParameterByIndex(i);
        if (std::string(var_layout->getName()) == INPUT_STRUCT_PARAMETER_NAME) {
            reflected_inputs = reflectFieldsFromStruct(var_layout);
        }
    }

    for (const auto& reflected_input : reflected_inputs) {
        slang::TypeLayoutReflection* type_layout = reflected_input->getTypeLayout();
        slang::TypeReflection* type = type_layout->getType();

        if (type->getKind() != slang::TypeReflection::Kind::Resource)
            continue;

        if (type->getResourceShape() == (SLANG_TEXTURE_COMBINED_FLAG | SLANG_TEXTURE_2D)) {
            slang::TypeReflection* result_type = type->getResourceResultType();
            result.push_back(VkSampledImageIn::compute_read(reflected_input->getName()));
        } else if (type->getResourceShape() == SLANG_TEXTURE_2D) {
            slang::TypeReflection* result_type = type->getResourceResultType();
            // TODO texture connector
        } else if (type->getResourceShape() == SLANG_STRUCTURED_BUFFER) {
            result.push_back(VkBufferIn::compute_read(reflected_input->getName()));
        }
    }

    return result;
}

std::vector<OutputConnectorHandle> SlangCompute::reflectOutputConnectors(const NodeIOLayout& io_layout, slang::EntryPointReflection* entry_point) {
    std::vector<OutputConnectorHandle> result{};
    std::vector<slang::VariableLayoutReflection*> reflected_outputs{};

    const uint32_t param_count = entry_point->getParameterCount();
    for (uint32_t i = 0; i < param_count; i++) {
        slang::VariableLayoutReflection* var_layout = entry_point->getParameterByIndex(i);
        if (std::string(var_layout->getName()) == OUTPUT_STRUCT_PARAMETER_NAME) {
            reflected_outputs = reflectFieldsFromStruct(var_layout);
        }
    }

    for (const auto& reflected_output : reflected_outputs) {
        slang::TypeLayoutReflection* type_layout = reflected_output->getTypeLayout();
        slang::TypeReflection* type = type_layout->getType();
        slang::VariableReflection* var = reflected_output->getVariable();

        if (type->getKind() != slang::TypeReflection::Kind::Resource)
            continue;

        if (type->getResourceShape() == (SLANG_TEXTURE_COMBINED_FLAG | SLANG_TEXTURE_2D)) {
            slang::TypeReflection* result_type = type->getResourceResultType();

            // TODO sampler connector
        } else if (type->getResourceShape() == SLANG_TEXTURE_2D) {
            const vk::Extent3D connector_extent = getExtentForImageOutputConnector(io_layout, var);
            const vk::Format format = getFormatForImageOutputConnector(type);
            extent = connector_extent; // TODO this is only needed for the group count calc, which should be configurable

            result.push_back(ManagedVkImageOut::compute_write(reflected_output->getName(), format, extent));
        } else if (type->getResourceShape() == SLANG_STRUCTURED_BUFFER) {
            const size_t size = getSizeForBufferOutputConnector(io_layout, var);
            result.push_back(ManagedVkBufferOut::compute_write(reflected_output->getName(), vk::BufferCreateInfo{{}, size, vk::BufferUsageFlagBits::eStorageBuffer}));
        }
    }

    return result;
}

std::vector<slang::VariableLayoutReflection*> SlangCompute::reflectFieldsFromStruct(slang::VariableLayoutReflection* struct_layout) {
    std::vector<slang::VariableLayoutReflection*> result{};

    slang::TypeLayoutReflection* type_layout = struct_layout->getTypeLayout();
    slang::TypeReflection* type = type_layout->getType();

    slang::TypeReflection::Kind kind = type->getKind();
    if (kind == slang::TypeReflection::Kind::Struct) {
        uint32_t field_count = type->getFieldCount();
        result.reserve(field_count);
        for (int f = 0; f < field_count; f++)
        {
            slang::VariableLayoutReflection* field = type_layout->getFieldByIndex(f);
            result.push_back(field);
        }
    }

    return result;
}

size_t SlangCompute::getSizeForBufferOutputConnector(const NodeIOLayout& io_layout, slang::VariableReflection* var) const {
    constexpr const char* STATIC_SIZE_ATTRIBUTE_NAME = "MerianSizeStatic";
    constexpr const char* SIZE_AS_ATTRIBUTE_NAME = "MerianExtentAs";

    slang::Attribute* extent_attribute = nullptr;
    if ((extent_attribute = findAttributeByName(var, STATIC_SIZE_ATTRIBUTE_NAME)) != nullptr) {
        int32_t size = 0;
        extent_attribute->getArgumentValueInt(0, &size);

        return  static_cast<size_t>(size);
    }

    if ((extent_attribute = findAttributeByName(var, SIZE_AS_ATTRIBUTE_NAME)) != nullptr) {
        std::string const mirrored_input_name = extent_attribute->getArgumentValueString(0, nullptr);
        InputConnectorHandle mirrored_input = findInputConnectorByName(mirrored_input_name);

        if (typeid(*mirrored_input) == typeid(VkSampledImageIn)) {
            auto mirrored_image = dynamic_pointer_cast<VkSampledImageIn>(mirrored_input);
            const vk::ImageCreateInfo create_info = io_layout[mirrored_image]->get_create_info_or_throw();
            return create_info.extent.width;
        }

        if (typeid(*mirrored_input) == typeid(VkBufferIn)) {
            auto mirrored_buffer = dynamic_pointer_cast<VkBufferIn>(mirrored_input);
            const vk::BufferCreateInfo create_info = io_layout[mirrored_buffer]->get_create_info_or_throw();
            return create_info.size;
        }

        throw graph_errors::node_error("Input connector " + mirrored_input_name + " can not be mirrored by output connector " + std::string(var->getName()));
    }

    throw graph_errors::node_error("No size defined for output connector %s" + std::string(var->getName()));
}

 vk::Extent3D SlangCompute::getExtentForImageOutputConnector(const NodeIOLayout& io_layout, slang::VariableReflection* var) const {
    constexpr const char* STATIC_EXTENT_ATTRIBUTE_NAME = "MerianExtentStatic";
    constexpr const char* EXTENT_AS_ATTRIBUTE_NAME = "MerianExtentAs";

    slang::Attribute* extent_attribute = nullptr;
    if ((extent_attribute = findAttributeByName(var, STATIC_EXTENT_ATTRIBUTE_NAME)) != nullptr) {
        glm::ivec3 dims = glm::ivec3(0);
        extent_attribute->getArgumentValueInt(0, &dims.x);
        extent_attribute->getArgumentValueInt(1, &dims.y);
        extent_attribute->getArgumentValueInt(2, &dims.z);

        return  vk::Extent3D(dims.x, dims.y, dims.z);
    }

    if ((extent_attribute = findAttributeByName(var, EXTENT_AS_ATTRIBUTE_NAME)) != nullptr) {
        std::string const mirrored_input_name = extent_attribute->getArgumentValueString(0, nullptr);
        auto mirrored_input = dynamic_pointer_cast<VkSampledImageIn>(findInputConnectorByName(mirrored_input_name)); // TODO look at this

        if (mirrored_input == nullptr) {
            throw graph_errors::node_error("Input connector " + mirrored_input_name + " can not be mirrored by output connector " + std::string(var->getName()));
        }

        const vk::ImageCreateInfo create_info = io_layout[mirrored_input]->get_create_info_or_throw();
        return create_info.extent;
    }

    throw graph_errors::node_error("No extent defined for output connector %s" + std::string(var->getName()));
}

vk::Format SlangCompute::getFormatForImageOutputConnector(slang::TypeReflection* type) {
    slang::TypeReflection* result_type = type->getResourceResultType();
    // const vk::Format format = output_format.value_or(create_info.format);
    return vk::Format::eR8G8B8A8Unorm; // TODO dont hard code this
}

slang::Attribute* SlangCompute::findAttributeByName(slang::VariableReflection* var, const std::string& name) {
    const uint32_t attribute_count = var->getUserAttributeCount();
    for (uint32_t i = 0; i < attribute_count; i++) {
        slang::Attribute* attribute = var->getUserAttributeByIndex(i);
        if (attribute->getName() == name) {
            return attribute;
        }
    }
    return nullptr;
}

InputConnectorHandle SlangCompute::findInputConnectorByName(const std::string& name) const {
    for (const auto& input : input_connectors) {
        if (input->name == name) {
            return input;
        }
    }
    return nullptr;
}

std::vector<InputConnectorHandle> SlangCompute::describe_inputs() {
    using namespace slang;

    if (!shader) {
        loadShader(shader_path);
    }

    assert(program_layout->getEntryPointCount() == 1);
    EntryPointReflection* entry_point = program_layout->getEntryPointByIndex(0);
    input_connectors = reflectInputConnectors(entry_point);
    con_src = static_pointer_cast<VkSampledImageIn>(input_connectors.at(0)); // TODO remove this
    return input_connectors;
}

std::vector<OutputConnectorHandle>
SlangCompute::describe_outputs([[maybe_unused]] const NodeIOLayout& io_layout) {
    using namespace slang;

    if (!shader) {
        loadShader(shader_path);
    }

    EntryPointReflection* entry_point = program_layout->getEntryPointByIndex(0);
    output_connectors = reflectOutputConnectors(io_layout, entry_point);
    return output_connectors;
}

const void* SlangCompute::get_push_constant([[maybe_unused]] GraphRun& run,
                                       [[maybe_unused]] const NodeIO& io) {
    return nullptr;
}

std::tuple<uint32_t, uint32_t, uint32_t>
SlangCompute::get_group_count([[maybe_unused]] const NodeIO& io) const noexcept {
    slang::EntryPointReflection* entry_point = program_layout->getEntryPointByIndex(0);
    auto [x, y, z] = reflectWorkgroupSize(entry_point);
    return {(extent.width + x - 1) / x,
            (extent.height + y - 1) / y, 1};
};

std::tuple<uint32_t, uint32_t, uint32_t>
SlangCompute::reflectWorkgroupSize(slang::EntryPointReflection* entry_point) {
    assert(entry_point->getStage() == SLANG_STAGE_COMPUTE);
    SlangUInt sizes[3];
    entry_point->getComputeThreadGroupSize(3, sizes);
    return {sizes[0], sizes[1], sizes[2]};
}

VulkanEntryPointHandle SlangCompute::get_entry_point() {
    return shader;
}

AbstractCompute::NodeStatusFlags SlangCompute::properties(Properties& config) {
    bool needs_rebuild = false;

    if (config.config_text("shader path", shader_path, true)) {
        needs_rebuild = true;
    }

    if (needs_rebuild) {
        return NEEDS_RECONNECT;
    }
    return {};
}

} // namespace merian_nodes
