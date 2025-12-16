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

void SlangCompute::reflectInputConnectors(slang::EntryPointReflection* entry_point) {
    std::vector<slang::VariableLayoutReflection*> reflected_inputs = reflectFieldsFromEntryPointParameterStruct(entry_point, INPUT_STRUCT_PARAMETER_NAME.data());

    for (const auto& reflected_input : reflected_inputs) {
        slang::TypeLayoutReflection* type_layout = reflected_input->getTypeLayout();
        slang::TypeReflection* type = type_layout->getType();

        if (type->getKind() != slang::TypeReflection::Kind::Resource)
            continue;

        std::string const reflected_name = reflected_input->getName();
        if (type->getResourceShape() == (SLANG_TEXTURE_COMBINED_FLAG | SLANG_TEXTURE_2D)) {
            slang::TypeReflection* result_type = type->getResourceResultType();
            image_in_connectors.emplace(reflected_name, VkSampledImageIn::compute_read(reflected_name));
        } else if (type->getResourceShape() == SLANG_TEXTURE_2D) {
            slang::TypeReflection* result_type = type->getResourceResultType();
            image_in_connectors.emplace(reflected_name, VkSampledImageIn::compute_read(reflected_name)); // TODO use a not sample compute tex here
        } else if (type->getResourceShape() == SLANG_STRUCTURED_BUFFER) {
            buffer_in_connectors.emplace(reflected_name, VkBufferIn::compute_read(reflected_name));
        }
    }
}

void SlangCompute::reflectOutputConnectors(const NodeIOLayout& io_layout, slang::EntryPointReflection* entry_point) {
    std::vector<slang::VariableLayoutReflection*> reflected_outputs = reflectFieldsFromEntryPointParameterStruct(entry_point, OUTPUT_STRUCT_PARAMETER_NAME.data());

    for (const auto& reflected_output : reflected_outputs) {
        slang::TypeLayoutReflection* type_layout = reflected_output->getTypeLayout();
        slang::TypeReflection* type = type_layout->getType();
        slang::VariableReflection* var = reflected_output->getVariable();

        if (type->getKind() != slang::TypeReflection::Kind::Resource)
            continue;

        std::string const reflected_name = reflected_output->getName();
        if (type->getResourceShape() == (SLANG_TEXTURE_COMBINED_FLAG | SLANG_TEXTURE_2D)) {
            throw graph_errors::node_error{fmt::format(
            "Error for declared output connector {}: Sampled textures are not supported as outputs as they are read_only, use RWTexture2D instead",
                reflected_name)};
        }

        if (type->getResourceShape() == SLANG_TEXTURE_2D) {
            const vk::Extent3D extent = getExtentForImageOutputConnector(io_layout, var);
            const vk::Format format = getFormatForImageOutputConnector(type);

            image_out_connectors.emplace(reflected_name,
                ManagedVkImageOut::compute_write(reflected_name, format, extent));
        } else if (type->getResourceShape() == SLANG_STRUCTURED_BUFFER) {
            const size_t size = getSizeForBufferOutputConnector(io_layout, var);

            buffer_out_connectors.emplace(reflected_name,
                ManagedVkBufferOut::compute_write(reflected_name,
                    vk::BufferCreateInfo{{}, size, vk::BufferUsageFlagBits::eStorageBuffer}));
        }
    }
}
bool SlangCompute::reflectProperties(Properties& config, slang::EntryPointReflection* entry_point) {
    std::vector<slang::VariableLayoutReflection*> reflected_props{};
    try {
        reflected_props = reflectFieldsFromEntryPointParameterStruct(entry_point, PROPERTY_STRUCT_PARAMETER_NAME.data());
    } catch (graph_errors::node_error& e) {
        SPDLOG_WARN(e.what());
    }

    bool needs_rebuild = false;

    for (const auto& reflected_prop : reflected_props) {
        slang::TypeLayoutReflection* type_layout = reflected_prop->getTypeLayout();
        slang::TypeReflection* type = type_layout->getType();
        slang::VariableReflection* var = reflected_prop->getVariable();

        const std::string type_name = type->getName();
        const std::string prop_name = reflected_prop->getName();
        if (type_name == "int") {
            if (!int_properties.contains(prop_name)) {
                int_properties[prop_name] = std::make_unique<int>();
            }

            slang::Attribute* range_attribute = findVarAttributeByName(var, INT_RANGE_ATTRIBUTE_NAME.data());
            if (range_attribute != nullptr) {
                int32_t min, max;
                range_attribute->getArgumentValueInt(0, &min);
                range_attribute->getArgumentValueInt(1, &max);

                needs_rebuild |= config.config_int(prop_name, *int_properties[prop_name], min, max, "");
            } else {
                needs_rebuild |= config.config_int(prop_name, *int_properties[prop_name], "");
            }
        } else if (type_name == "uint") {
            if (!uint_properties.contains(prop_name)) {
                uint_properties[prop_name] = std::make_unique<uint>();
            }

            slang::Attribute* range_attribute = findVarAttributeByName(var, INT_RANGE_ATTRIBUTE_NAME.data());
            if (range_attribute != nullptr) {
                int32_t min, max;
                range_attribute->getArgumentValueInt(0, &min);
                range_attribute->getArgumentValueInt(1, &max);

                if (min < 0 || max < 0) {
                    throw graph_errors::node_error("No negative range values allowed for reflected uint property");
                }

                needs_rebuild |= config.config_uint(prop_name, *uint_properties[prop_name], min, max, "");
            } else {
                needs_rebuild |= config.config_uint(prop_name, *uint_properties[prop_name], "");
            }
        } else if (type_name == "float") {
            if (!float_properties.contains(prop_name)) {
                float_properties[prop_name] = std::make_unique<float>();
            }

            slang::Attribute* range_attribute = findVarAttributeByName(var, FLOAT_RANGE_ATTRIBUTE_NAME.data());
            if (range_attribute != nullptr) {
                float min, max;
                range_attribute->getArgumentValueFloat(0, &min);
                range_attribute->getArgumentValueFloat(1, &max);

                needs_rebuild |= config.config_float(prop_name, *float_properties[prop_name], min, max, "");
            } else {
                needs_rebuild |= config.config_float(prop_name, *float_properties[prop_name], "");
            }
        } else if (type_name == "String") {
            if (!string_properties.contains(prop_name)) {
                string_properties[prop_name] = std::make_unique<std::string>();
            }
            needs_rebuild |= config.config_text(prop_name, *string_properties[prop_name], "");
        } else if (type_name == "vector") {
            size_t const element_count = type->getElementCount();
            slang::TypeReflection* element_type = type->getElementType();
            std::string element_type_name = element_type->getName();

            if (element_count < 3 || element_type_name != "float") {
                throw graph_errors::node_error("Only float3 or float4 vectors are supported as reflected properties!");
            }

            if (!vector_properties.contains(prop_name)) {
                vector_properties[prop_name] = std::make_unique<glm::vec4>();
            }
            config.config_color(prop_name, *vector_properties[prop_name], "");
        } else {
            throw graph_errors::node_error(fmt::format("Type {} is not supported as reflectable property!", type_name));
        }
    }

    return needs_rebuild;
}

std::vector<slang::VariableLayoutReflection*> SlangCompute::reflectFieldsFromEntryPointParameterStruct(slang::EntryPointReflection* entry_point,
                                                         const std::string& parameter_name) {
    const uint32_t param_count = entry_point->getParameterCount();
    for (uint32_t i = 0; i < param_count; i++) {
        slang::VariableLayoutReflection* var_layout = entry_point->getParameterByIndex(i);
        if (std::string(var_layout->getName()) == parameter_name) {
            return reflectFieldsFromStruct(var_layout);
        }
    }

    throw graph_errors::node_error("Parameter '" + parameter_name + "' not found on entry point '" + entry_point->getName() + "'");
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
    slang::Attribute* extent_attribute = nullptr;
    if ((extent_attribute = findVarAttributeByName(var, STATIC_SIZE_ATTRIBUTE_NAME.data())) != nullptr) {
        int32_t size = 0;
        extent_attribute->getArgumentValueInt(0, &size);

        return  static_cast<size_t>(size);
    }

    if ((extent_attribute = findVarAttributeByName(var, SIZE_AS_ATTRIBUTE_NAME.data())) != nullptr) {
        std::string const mirrored_input_name = extent_attribute->getArgumentValueString(0, nullptr);
        if (image_in_connectors.contains(mirrored_input_name)) {
            auto mirrored_image = image_in_connectors.at(mirrored_input_name);
            const vk::ImageCreateInfo create_info = io_layout[mirrored_image]->get_create_info_or_throw();
            return create_info.extent.width * create_info.extent.height;
        } else if (buffer_in_connectors.contains(mirrored_input_name)) {
            auto mirrored_buffer = buffer_in_connectors.at(mirrored_input_name);
            const vk::BufferCreateInfo create_info = io_layout[mirrored_buffer]->get_create_info_or_throw();
            return create_info.size;
        }

        throw graph_errors::node_error("Input connector " + mirrored_input_name + " can not be mirrored by output connector " + std::string(var->getName()));
    }

    throw graph_errors::node_error("No size defined for output connector %s" + std::string(var->getName()));
}

 vk::Extent3D SlangCompute::getExtentForImageOutputConnector(const NodeIOLayout& io_layout, slang::VariableReflection* var) const {
    slang::Attribute* extent_attribute = nullptr;
    if ((extent_attribute = findVarAttributeByName(var, STATIC_EXTENT_ATTRIBUTE_NAME.data())) != nullptr) {
        auto dims = glm::ivec3(0);
        extent_attribute->getArgumentValueInt(0, &dims.x);
        extent_attribute->getArgumentValueInt(1, &dims.y);
        extent_attribute->getArgumentValueInt(2, &dims.z);

        return vk::Extent3D(dims.x, dims.y, dims.z);
    }

    if ((extent_attribute = findVarAttributeByName(var, EXTENT_AS_ATTRIBUTE_NAME.data())) != nullptr) {
        std::string const mirrored_input_name = extent_attribute->getArgumentValueString(0, nullptr);
        if (!image_in_connectors.contains(mirrored_input_name)) {
            throw graph_errors::node_error("Input connector " + mirrored_input_name + " can not be mirrored by output connector " + std::string(var->getName()));
        }

        const vk::ImageCreateInfo create_info = io_layout[image_in_connectors.at(mirrored_input_name)]->get_create_info_or_throw();
        return create_info.extent;
    }

    throw graph_errors::node_error("No extent defined for output connector %s" + std::string(var->getName()));
}

vk::Format SlangCompute::getFormatForImageOutputConnector(slang::TypeReflection* type) {
    slang::TypeReflection* result_type = type->getResourceResultType();
    // const vk::Format format = output_format.value_or(create_info.format);
    return vk::Format::eR8G8B8A8Unorm; // TODO dont hard code this
}

slang::Attribute* SlangCompute::findVarAttributeByName(slang::VariableReflection* var, const std::string& name) {
    const uint32_t attribute_count = var->getUserAttributeCount();
    for (uint32_t i = 0; i < attribute_count; i++) {
        slang::Attribute* attribute = var->getUserAttributeByIndex(i);
        if (attribute->getName() == name) {
            return attribute;
        }
    }
    return nullptr;
}

slang::Attribute* SlangCompute::findFuncAttributeByName(slang::FunctionReflection* var, const std::string& name) {
    const uint32_t attribute_count = var->getUserAttributeCount();
    for (uint32_t i = 0; i < attribute_count; i++) {
        slang::Attribute* attribute = var->getUserAttributeByIndex(i);
        if (attribute->getName() == name) {
            return attribute;
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
    reflectInputConnectors(entry_point);

    std::vector<InputConnectorHandle> in_connectors;
    in_connectors.reserve(image_in_connectors.size() + buffer_in_connectors.size());

    for (const auto& con : image_in_connectors | std::views::values)
        in_connectors.push_back(con);

    for (const auto& con : buffer_in_connectors | std::views::values)
        in_connectors.push_back(con);
    return in_connectors;
}

std::vector<OutputConnectorHandle>
SlangCompute::describe_outputs([[maybe_unused]] const NodeIOLayout& io_layout) {
    using namespace slang;

    if (!shader) {
        loadShader(shader_path);
    }

    EntryPointReflection* entry_point = program_layout->getEntryPointByIndex(0);
    reflectOutputConnectors(io_layout, entry_point);

    std::vector<OutputConnectorHandle> output_connectors;
    output_connectors.reserve(image_out_connectors.size() + buffer_out_connectors.size());

    for (const auto& con : image_out_connectors | std::views::values)
        output_connectors.push_back(con);

    for (const auto& con : buffer_out_connectors | std::views::values)
        output_connectors.push_back(con);
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

    slang::Attribute* target_attribute = findFuncAttributeByName(entry_point->getFunction(), TARGET_ATTRIBUTE_NAME.data());
    if (!target_attribute) {
        SPDLOG_WARN("Entry point '{}' is missing {} attribute", entry_point->getName(), TARGET_ATTRIBUTE_NAME.data());
    }

    std::string target_input_name = target_attribute->getArgumentValueString(0, nullptr);
    if (image_in_connectors.contains(target_input_name)) {
        VkImageInHandle target_img_con = image_in_connectors.at(target_input_name);
        const ImageArrayResource& target_img = io[target_img_con];
        vk::Extent3D extent = target_img->get_extent();
        return {(extent.width + x - 1) / x, (extent.height + y - 1) / y, 1};
    } else if (buffer_in_connectors.contains(target_input_name)) {
        VkBufferInHandle target_buffer_con = buffer_in_connectors.at(target_input_name);
        const BufferArrayResource& target_buffer = io[target_buffer_con];
        std::size_t size = target_buffer->get_size();
        return {(size + x - 1) / x, 1, 1};
    } else {
        throw graph_errors::node_error("Input connector " + target_input_name + " can not be used as target");
    }
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
    using namespace slang;

    bool needs_rebuild = false;

    if (config.config_text("shader path", shader_path, true)) {
        needs_rebuild = true;
    }

    if (!shader) {
        try {
            loadShader(shader_path);
        } catch (graph_errors::node_error& e) {}
    }

    if (shader) {
        EntryPointReflection* entry_point = program_layout->getEntryPointByIndex(0);
        needs_rebuild |= reflectProperties(config, entry_point);
    }

    if (needs_rebuild) {
        return NEEDS_RECONNECT;
    }
    return {};
}

} // namespace merian_nodes
