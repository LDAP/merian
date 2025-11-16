#include "merian-nodes/nodes/slang_compute/slang_compute.hpp"
#include "merian-nodes/connectors/image/vk_image_in_sampled.hpp"
#include "merian-nodes/connectors/image/vk_image_in_sampled.hpp"
#include "merian-nodes/connectors/image/vk_image_out_managed.hpp"
#include "merian/vk/pipeline/specialization_info_builder.hpp"
#include "merian/vk/shader/entry_point.hpp"

#include "merian-nodes/graph/errors.hpp"
#include "merian/vk/shader/slang_entry_point.hpp"
#include "tonemap.slang.spv.h"

namespace merian_nodes {

SlangCompute::SlangCompute(const ContextHandle& context, const std::optional<vk::Format> output_format)
    : AbstractCompute(context, sizeof(PushConstant)), output_format(output_format) {
    make_spec_info();
}

SlangCompute::~SlangCompute() {}

void SlangCompute::make_spec_info() {
    auto spec_builder = SpecializationInfoBuilder();
    spec_builder.add_entry(local_size_x, local_size_y, tonemap, alpha_mode, clamp_output);
    spec_info = spec_builder.build();

    const GLSLShaderCompilerHandle compiler = GLSLShaderCompiler::get();
    ShaderCompileContextHandle compilation_session_desc = ShaderCompileContext::create(context);
    compilation_session_desc->add_search_path("merian-nodes/nodes/slang_compute");

    SlangProgramEntryPointHandle slang_entry_point = SlangProgramEntryPoint::create(compilation_session_desc, "shader.slang");
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
            const vk::Extent3D extent = getExtentForImageOutputConnector(io_layout, var);
            const vk::Format format = getFormatForImageOutputConnector(type);

            result.push_back(ManagedVkImageOut::compute_write(reflected_output->getName(), format, extent));
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

 vk::Extent3D SlangCompute::getExtentForImageOutputConnector(const NodeIOLayout& io_layout, slang::VariableReflection* var) const {
    constexpr const char* STATIC_EXTENT_ATTRIBUTE_NAME = "MerianExtentStatic";
    constexpr const char* EXTENT_AS_ATTRIBUTE_NAME = "MerianExtentAs";

    slang::Attribute* extent_attribute = nullptr;


    if ((extent_attribute = findAttributeByName(var, "MerianExtent")) != nullptr) {
        SPDLOG_INFO("{}", extent_attribute->getArgumentType(0)->getName());
    }

    if ((extent_attribute = findAttributeByName(var, STATIC_EXTENT_ATTRIBUTE_NAME)) != nullptr) {
        glm::ivec3 dims = glm::ivec3(0);
        extent_attribute->getArgumentValueInt(0, &dims.x);
        extent_attribute->getArgumentValueInt(1, &dims.y);
        extent_attribute->getArgumentValueInt(2, &dims.z);

        return  vk::Extent3D(dims.x, dims.y, dims.z);
    }

    if ((extent_attribute = findAttributeByName(var, EXTENT_AS_ATTRIBUTE_NAME)) != nullptr) {
        std::string const mirrored_input_name = extent_attribute->getArgumentValueString(0, nullptr);
        VkSampledImageInHandle mirrored_input = dynamic_pointer_cast<VkSampledImageIn>(findInputConnectorByName(mirrored_input_name)); // TODO look at this

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

    assert(program_layout->getEntryPointCount() == 1);
    EntryPointReflection* entry_point = program_layout->getEntryPointByIndex(0);
    input_connectors = reflectInputConnectors(entry_point);
    con_src = static_pointer_cast<VkSampledImageIn>(input_connectors.at(0)); // TODO remove this
    return input_connectors;
}

std::vector<OutputConnectorHandle>
SlangCompute::describe_outputs([[maybe_unused]] const NodeIOLayout& io_layout) {
    using namespace slang;

    const vk::ImageCreateInfo create_info = io_layout[con_src]->get_create_info_or_throw();

    extent = create_info.extent;
    const vk::Format format = output_format.value_or(create_info.format);

    EntryPointReflection* entry_point = program_layout->getEntryPointByIndex(0);
    output_connectors = reflectOutputConnectors(io_layout, entry_point);
    return output_connectors;
}

const void* SlangCompute::get_push_constant([[maybe_unused]] GraphRun& run,
                                       [[maybe_unused]] const NodeIO& io) {
    return &pc;
}

std::tuple<uint32_t, uint32_t, uint32_t>
SlangCompute::get_group_count([[maybe_unused]] const NodeIO& io) const noexcept {
    return {(extent.width + local_size_x - 1) / local_size_x,
            (extent.height + local_size_y - 1) / local_size_y, 1};
};

VulkanEntryPointHandle SlangCompute::get_entry_point() {
    return shader;
}

AbstractCompute::NodeStatusFlags SlangCompute::properties(Properties& config) {
    bool needs_rebuild = false;

    const int old_tonemap = tonemap;
    config.config_options(
        "tonemap", tonemap,
        {"None", "Clamp", "Uncharted 2", "Reinhard Extended", "Aces", "Aces-Approx", "Lottes"});
    needs_rebuild |= old_tonemap != tonemap;

    if (tonemap == TONEMAP_REINHARD_EXTENDED) {
        if (old_tonemap != TONEMAP_REINHARD_EXTENDED)
            pc.param1 = 1.0;
        config.config_float("max white", pc.param1, "max luminance found in the scene", .05);
    }

    if (tonemap == TONEMAP_UNCHARTED_2) {
        if (old_tonemap != TONEMAP_UNCHARTED_2) {
            pc.param1 = 2.0;
            pc.param2 = 11.2;
        }
        config.config_float("exposure bias", pc.param1, "see UNCHARTED 2", .05);
        config.config_float("W", pc.param2, "see UNCHARTED 2", .1);
    }

    if (tonemap == TONEMAP_LOTTES) {
        if (old_tonemap != TONEMAP_LOTTES) {
            pc.param1 = 1.0;
            pc.param2 = 1.0;
            pc.param3 = 16.0;
            pc.param4 = 0.18;
            pc.param5 = 0.18;
        }
        config.config_float("contrast", pc.param1, "See Lottes talk", 0.01);
        config.config_float("shoulder", pc.param2, "See Lottes talk", 0.01);
        config.config_float("hdrMax", pc.param3, "See Lottes talk", 0.1);
        config.config_float("midIn", pc.param4, "See Lottes talk", 0.001);
        config.config_float("midOut", pc.param5, "See Lottes talk", 0.001);
    }

    if (tonemap == TONEMAP_ACES_APPROX) {
        if (old_tonemap != TONEMAP_ACES_APPROX) {
            pc.param1 = 2.51;
            pc.param2 = 0.03;
            pc.param3 = 2.43;
            pc.param4 = 0.59;
            pc.param5 = 0.14;
        }

        config.config_float("a", pc.param1, "", 0.01);
        config.config_float("b", pc.param2, "", 0.01);
        config.config_float("c", pc.param3, "", 0.01);
        config.config_float("d", pc.param4, "", 0.01);
        config.config_float("e", pc.param5, "", 0.01);
    }

    config.st_separate();
    int32_t old_clamp_output = clamp_output;
    config.config_bool("clamp output", clamp_output,
                       "clamps the output (before computing the alpha channel)");
    needs_rebuild |= old_clamp_output != clamp_output;

    config.st_separate();
    int32_t old_alpha_mode = alpha_mode;
    config.config_options("alpha mode", alpha_mode,
                          {
                              "Passthrough",
                              "Luminance",
                              "Perceptual luminance",
                          },
                          Properties::OptionsStyle::DONT_CARE,
                          "Decides what is written in the alpha channel.");
    if (alpha_mode == ALPHA_MODE_PERCEPTUAL_LUMINANCE) {
        config.config_float(
            "perceptual exponent", pc.perceptual_exponent,
            "Adjust the exponent that is used to convert the luminance to perceptual space.", 0.1);
    }

    needs_rebuild |= old_alpha_mode != alpha_mode;

    if (needs_rebuild) {
        make_spec_info();
    }

    return {};
}

} // namespace merian_nodes
