#include "merian/shader/slang_entry_point.hpp"
#include "merian/shader/shader_object.hpp"
#include "merian/shader/slang_global_session.hpp"
#include "merian/vk/pipeline/pipeline_layout_builder.hpp"

namespace merian {

SlangProgramEntryPoint::SlangProgramEntryPoint(const SlangProgramHandle& program,
                                               const uint64_t entry_point_index)
    : program(program), entry_point_index(entry_point_index) {
    assert(entry_point_index < program->get_program_reflection()->getEntryPointCount());
}

const char* SlangProgramEntryPoint::get_name() const {
    return get_entry_point_reflection()->getNameOverride();
}

vk::ShaderStageFlagBits SlangProgramEntryPoint::get_stage() const {
    return vk_stage_for_slang_stage(get_entry_point_reflection()->getStage());
}

ShaderModuleHandle
SlangProgramEntryPoint::vulkan_shader_module(const ContextHandle& context) const {
    return program->get_shader_module(context);
}

slang::EntryPointReflection* SlangProgramEntryPoint::get_entry_point_reflection() const {
    return program->get_program_reflection()->getEntryPointByIndex(entry_point_index);
}

const SlangProgramHandle& SlangProgramEntryPoint::get_program() const {
    return program;
}

// ---------------------------------------------------------------
// ShaderObject helpers

SlangProgramEntryPoint::ParameterBlockInfo&
SlangProgramEntryPoint::find_or_create_param_info(const ContextHandle& context,
                                                  const std::string& param_name) {
    auto it = param_cache.find(param_name);
    if (it != param_cache.end()) {
        return it->second;
    }

    auto* entry_reflection = get_entry_point_reflection();

    // Walk parameters to find the named ParameterBlock
    for (uint32_t i = 0; i < entry_reflection->getParameterCount(); i++) {
        auto* param = entry_reflection->getParameterByIndex(i);
        auto* type_layout = param->getTypeLayout();

        if (type_layout->getKind() != slang::TypeReflection::Kind::ParameterBlock) {
            continue;
        }

        if (param_name == param->getName()) {
            auto* element_type_layout = type_layout->getElementTypeLayout();
            auto object_layout =
                std::make_shared<SlangObjectLayout>(context, element_type_layout, program);

            // Set index will be assigned by get_pipeline_layout() during DFS walk.
            // Initialize to 0; get_pipeline_layout() updates it before bind() uses it.
            auto [inserted_it, _] =
                param_cache.emplace(param_name, ParameterBlockInfo{object_layout, 0, {}});
            return inserted_it->second;
        }
    }

    SPDLOG_ERROR("ParameterBlock parameter '{}' not found in entry point", param_name);
    assert(false && "ParameterBlock parameter not found");
    // unreachable, but satisfy compiler
    static ParameterBlockInfo dummy{};
    return dummy;
}

SlangObjectLayoutHandle SlangProgramEntryPoint::get_object_layout(const ContextHandle& context,
                                                                  const std::string& param_name) {
    return find_or_create_param_info(context, param_name).object_layout;
}

uint32_t SlangProgramEntryPoint::get_descriptor_set_index(const std::string& param_name) {
    auto it = param_cache.find(param_name);
    assert(it != param_cache.end() && "Call get_object_layout first to populate cache");
    return it->second.descriptor_set_index;
}

ShaderObjectHandle
SlangProgramEntryPoint::create_shader_object(const ContextHandle& context,
                                             const std::string& param_name,
                                             const ShaderObjectAllocatorHandle& allocator) {
    auto layout = get_object_layout(context, param_name);
    return std::make_shared<ShaderObject>(context, layout, allocator);
}

// Recursively collect descriptor set layouts for nested ParameterBlock fields (DFS order).
// Uses the pre-computed sub-object ranges from SlangObjectLayout.
static constexpr uint32_t NO_DESCRIPTOR_SET = UINT32_MAX;

static void
collect_nested_pb_layouts(const SlangObjectLayoutHandle& parent_layout,
                          PipelineLayoutBuilder& builder,
                          uint32_t& next_set,
                          std::vector<SlangProgramEntryPoint::NestedPBInfo>& nested_pb_infos) {
    for (uint32_t sor = 0; sor < parent_layout->get_sub_object_range_count(); sor++) {
        const auto& range = parent_layout->get_sub_object_range(sor);
        if (range.binding_type != slang::BindingType::ParameterBlock || !range.element_layout)
            continue;

        // Only allocate a Vulkan descriptor set if the PB element type has bindings.
        // Slang's SPIR-V output skips descriptor sets for types with no bindings.
        const bool has_bindings =
            !range.element_layout->get_descriptor_set_layout()->get_bindings().empty();
        uint32_t set_idx = NO_DESCRIPTOR_SET;
        if (has_bindings) {
            set_idx = next_set++;
            builder.add_descriptor_set_layout(range.element_layout->get_descriptor_set_layout());
        }

        SlangProgramEntryPoint::NestedPBInfo info{sor, set_idx, {}};

        // Recurse for deeply nested PBs
        collect_nested_pb_layouts(range.element_layout, builder, next_set, info.children);

        nested_pb_infos.push_back(std::move(info));
    }
}

PipelineLayoutHandle SlangProgramEntryPoint::get_pipeline_layout(const ContextHandle& context) {
    if (cached_pipeline_layout) {
        return cached_pipeline_layout;
    }

    auto* entry_reflection = get_entry_point_reflection();
    PipelineLayoutBuilder builder(context);

    // Set indices are assigned sequentially in DFS order of ParameterBlock nesting.
    // This matches Slang's SPIR-V output where PBs get consecutive descriptor set indices.
    uint32_t next_set = 0;

    for (uint32_t i = 0; i < entry_reflection->getParameterCount(); i++) {
        auto* param = entry_reflection->getParameterByIndex(i);
        auto* type_layout = param->getTypeLayout();

        if (type_layout->getKind() != slang::TypeReflection::Kind::ParameterBlock) {
            continue;
        }

        auto& info = find_or_create_param_info(context, param->getName());

        // Only allocate a Vulkan descriptor set if the PB element type has bindings.
        // Slang's SPIR-V output skips descriptor sets for types with no bindings.
        const bool has_bindings =
            !info.object_layout->get_descriptor_set_layout()->get_bindings().empty();
        if (has_bindings) {
            info.descriptor_set_index = next_set++;
            builder.add_descriptor_set_layout(info.object_layout->get_descriptor_set_layout());
        } else {
            info.descriptor_set_index = NO_DESCRIPTOR_SET;
        }

        // Recursively add nested PBs (DFS order) using pre-computed sub-object ranges
        collect_nested_pb_layouts(info.object_layout, builder, next_set, info.nested_pb_infos);
    }

    cached_pipeline_layout = builder.build_pipeline_layout();
    return cached_pipeline_layout;
}

// ---------------------------------------------------------------
// Binding

void SlangProgramEntryPoint::bind(const std::string& param_name,
                                  const ShaderObjectHandle& object,
                                  const ShaderObjectAllocatorHandle& allocator,
                                  const CommandBufferHandle& cmd,
                                  const PipelineHandle& pipeline) {
    // Ensure pipeline layout + param info is populated
    get_pipeline_layout(object->get_context());
    auto& info = find_or_create_param_info(object->get_context(), param_name);

    // Bind this PB at its cached set index (skip if type has no bindings)
    if (info.descriptor_set_index != NO_DESCRIPTOR_SET) {
        object->bind_as_parameter_block(cmd, pipeline, info.descriptor_set_index);
    }

    // Bind nested PB sub-objects using cached set indices from pipeline layout construction
    bind_nested_pbs(object, info.nested_pb_infos, allocator, cmd, pipeline);
}

void SlangProgramEntryPoint::bind_nested_pbs(const ShaderObjectHandle& object,
                                             const std::vector<NestedPBInfo>& nested_infos,
                                             const ShaderObjectAllocatorHandle& allocator,
                                             const CommandBufferHandle& cmd,
                                             const PipelineHandle& pipeline) {
    for (const auto& ni : nested_infos) {
        const auto& sub = object->get_sub_object(ni.sub_object_range_index);
        if (!sub)
            continue;
        if (ni.set_index != NO_DESCRIPTOR_SET) {
            sub->bind_as_parameter_block(cmd, pipeline, ni.set_index);
        }
        bind_nested_pbs(sub, ni.children, allocator, cmd, pipeline);
    }
}

// ---------------------------------------------------------------
// Parameter discovery

bool SlangProgramEntryPoint::has_parameter(const std::string& param_name) const {
    auto* entry_reflection = get_entry_point_reflection();
    for (uint32_t i = 0; i < entry_reflection->getParameterCount(); i++) {
        auto* param = entry_reflection->getParameterByIndex(i);
        if (param->getTypeLayout()->getKind() == slang::TypeReflection::Kind::ParameterBlock &&
            param_name == param->getName()) {
            return true;
        }
    }
    return false;
}

std::vector<std::string> SlangProgramEntryPoint::get_parameter_block_names() const {
    auto* entry_reflection = get_entry_point_reflection();
    std::vector<std::string> names;
    for (uint32_t i = 0; i < entry_reflection->getParameterCount(); i++) {
        auto* param = entry_reflection->getParameterByIndex(i);
        if (param->getTypeLayout()->getKind() == slang::TypeReflection::Kind::ParameterBlock) {
            names.emplace_back(param->getName());
        }
    }
    return names;
}

std::vector<std::string> SlangProgramEntryPoint::get_all_parameter_names() const {
    auto* entry_reflection = get_entry_point_reflection();
    std::vector<std::string> names;
    names.reserve(entry_reflection->getParameterCount());
    for (uint32_t i = 0; i < entry_reflection->getParameterCount(); i++) {
        names.emplace_back(entry_reflection->getParameterByIndex(i)->getName());
    }
    return names;
}

// ---------------------------------------------------------------
// Debug

static void format_nested_pb_tree(const std::vector<SlangProgramEntryPoint::NestedPBInfo>& infos,
                                  std::string& out,
                                  const std::string& indent) {
    for (const auto& ni : infos) {
        out += fmt::format("{}sub_object_range={}, set_index={}\n", indent,
                           ni.sub_object_range_index, ni.set_index);
        if (!ni.children.empty()) {
            format_nested_pb_tree(ni.children, out, indent + "  ");
        }
    }
}

std::string SlangProgramEntryPoint::format_reflection(const ContextHandle& context) {
    // Ensure pipeline layout is built so param_cache is populated
    get_pipeline_layout(context);

    std::string out;
    auto* er = get_entry_point_reflection();
    out += fmt::format("SlangProgramEntryPoint '{}'\n", er->getNameOverride());
    out += fmt::format("  stage: {}\n", vk::to_string(get_stage()));

    out += fmt::format("  parameters ({}):\n", er->getParameterCount());
    for (uint32_t i = 0; i < er->getParameterCount(); i++) {
        auto* param = er->getParameterByIndex(i);
        auto* tl = param->getTypeLayout();
        out += fmt::format("    [{}] '{}': kind={}\n", i, param->getName(),
                           slang_type_kind_to_string(tl->getKind()));

        if (tl->getKind() == slang::TypeReflection::Kind::ParameterBlock) {
            auto* element_tl = tl->getElementTypeLayout();
            out += fmt::format("      element type:\n");
            out += format_type_layout(element_tl, "        ");
        }
    }

    out += fmt::format("  cached parameter blocks ({}):\n", param_cache.size());
    for (const auto& [name, info] : param_cache) {
        out += fmt::format("    '{}': set_index={}\n", name, info.descriptor_set_index);
        if (!info.nested_pb_infos.empty()) {
            out += fmt::format("    nested PB tree:\n");
            format_nested_pb_tree(info.nested_pb_infos, out, "      ");
        }
    }

    return out;
}

// ---------------------------------------------------------------
// Factory methods

SlangProgramEntryPointHandle SlangProgramEntryPoint::create(const SlangProgramHandle& program,
                                                            const uint64_t entry_point_index) {
    return SlangProgramEntryPointHandle(new SlangProgramEntryPoint(program, entry_point_index));
}

SlangProgramEntryPointHandle SlangProgramEntryPoint::create(const SlangProgramHandle& program,
                                                            const std::string& entry_point_name) {
    return create(program, program->get_entry_point_index(entry_point_name));
}

SlangProgramEntryPointHandle
SlangProgramEntryPoint::create(const ShaderCompileContextHandle& compile_context,
                               const std::filesystem::path& module_path,
                               const std::string& entry_point_name) {
    SlangProgramHandle program = SlangProgram::create(compile_context, module_path, true);
    return create(program, entry_point_name);
}

} // namespace merian
