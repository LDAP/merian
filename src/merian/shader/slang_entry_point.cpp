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

        SPDLOG_DEBUG("Parameter element type layout:\n{}",
                     format_type_layout(type_layout->getElementTypeLayout()));
        SPDLOG_DEBUG("Parameter element var (type) layout:\n{}",
                     format_type_layout(type_layout->getElementVarLayout()->getTypeLayout()));
        SPDLOG_DEBUG("Parameter container type layout:\n{}",
                     format_type_layout(type_layout->getContainerVarLayout()->getTypeLayout()));

        if (param_name == param->getName()) {
            auto* element_type_layout = type_layout->getElementTypeLayout();
            auto object_layout =
                std::make_shared<ShaderObjectLayout>(context, element_type_layout, program);

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

ShaderObjectLayoutHandle SlangProgramEntryPoint::get_object_layout(const ContextHandle& context,
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
                                             const ResourceAllocatorHandle& allocator) {
    auto layout = get_object_layout(context, param_name);
    return std::make_shared<ShaderObject>(layout, allocator);
}

// Recursively collect descriptor set layouts for nested ParameterBlock fields (DFS order).
// Uses the pre-computed sub-object ranges from SlangObjectLayout.
static constexpr uint32_t NO_DESCRIPTOR_SET = UINT32_MAX;

static void
collect_nested_pb_layouts(const ShaderObjectLayoutHandle& parent_layout,
                          PipelineLayoutBuilder& builder,
                          uint32_t& next_set,
                          std::vector<SlangProgramEntryPoint::NestedPBInfo>& nested_pb_infos) {
    for (uint32_t sor = 0; sor < parent_layout->get_subobject_range_count(); sor++) {
        const auto& range = parent_layout->get_subobject_range_info(sor);
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

    auto* program_layout = program->get_program_reflection();
    auto* entry_reflection = get_entry_point_reflection();
    PipelineLayoutBuilder builder(context);

    // Set indices are assigned sequentially: global sets first, then PB sets in DFS order.
    // This matches Slang's SPIR-V output.
    uint32_t next_set = 0;

    // ---- Global parameters ----
    auto* global_tl = program_layout->getGlobalParamsTypeLayout();
    if (global_tl != nullptr) {
        // Direct global descriptor sets (resources, CBs at global scope)
        const uint32_t global_ds_count = global_tl->getDescriptorSetCount();
        for (uint32_t ds = 0; ds < global_ds_count; ds++) {
            auto layout =
                create_descriptor_set_layout_from_slang_type_layout(context, global_tl, program_layout, ds);
            if (!layout->get_bindings().empty()) {
                if (!global_object_layout) {
                    global_object_layout =
                        std::make_shared<ShaderObjectLayout>(context, global_tl, program);
                    global_set_index = next_set;
                }
                global_set_layouts.push_back(layout);
                builder.add_descriptor_set_layout(layout);
                next_set++;
            }
        }

        // Create global object layout even if no direct bindings,
        // as long as there are sub-object ranges (PB/CB fields at global scope)
        if (!global_object_layout && global_tl->getSubObjectRangeCount() > 0) {
            global_object_layout =
                std::make_shared<ShaderObjectLayout>(context, global_tl, program);
        }

        // Collect nested PB sub-objects from global scope (global ParameterBlock fields)
        if (global_object_layout) {
            collect_nested_pb_layouts(global_object_layout, builder, next_set,
                                      global_nested_pb_infos);
        }

        // Push constants: check global params first, then entry point
        push_constant_size = global_tl->getSize(SLANG_PARAMETER_CATEGORY_PUSH_CONSTANT_BUFFER);
        if (push_constant_size > 0) {
            builder.add_push_constant(static_cast<uint32_t>(push_constant_size));
        }
    }

    // Also check entry point type layout for push constants
    if (push_constant_size == 0) {
        auto* entry_tl = entry_reflection->getTypeLayout();
        if (entry_tl) {
            push_constant_size =
                entry_tl->getSize(SLANG_PARAMETER_CATEGORY_PUSH_CONSTANT_BUFFER);
            if (push_constant_size > 0) {
                builder.add_push_constant(static_cast<uint32_t>(push_constant_size));
            }
        }
    }

    // Scan global params for push constant bindings if still not found
    if (push_constant_size == 0 && global_tl != nullptr) {
        for (uint32_t br = 0; br < global_tl->getBindingRangeCount(); br++) {
            if (global_tl->getBindingRangeType(br) == slang::BindingType::PushConstant) {
                auto* leaf_tl = global_tl->getBindingRangeLeafTypeLayout(br);
                if (leaf_tl) {
                    // For ConstantBuffer<T>, the element type layout has the actual size
                    auto* elem_tl = leaf_tl->getElementTypeLayout();
                    push_constant_size = elem_tl ? elem_tl->getSize() : leaf_tl->getSize();
                }
                if (push_constant_size > 0) {
                    builder.add_push_constant(static_cast<uint32_t>(push_constant_size));
                }
                break;
            }
        }
    }

    // ---- Entry point ParameterBlock parameters ----
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

void SlangProgramEntryPoint::bind_entry_point_parameter(
    const std::string& param_name,
    const ShaderObjectHandle& object,
    const CommandBufferHandle& cmd,
    const PipelineHandle& pipeline,
    const ShaderObjectAllocatorHandle& obj_allocator) {
    const auto& context = pipeline->get_layout()->get_context();
    get_pipeline_layout(context);
    auto& info = find_or_create_param_info(context, param_name);

    const auto shader_stages = pipeline->get_pipeline_stage_flags2();

    cmd->barrier(vk::MemoryBarrier2{
        shader_stages,
        vk::AccessFlagBits2::eShaderRead | vk::AccessFlagBits2::eUniformRead,
        vk::PipelineStageFlagBits2::eTransfer,
        vk::AccessFlagBits2::eTransferWrite,
    });

    if (info.descriptor_set_index != NO_DESCRIPTOR_SET) {
        object->bind_as_parameter_block(cmd, pipeline, info.descriptor_set_index, obj_allocator);
    }

    bind_nested_pbs(object, info.nested_pb_infos, cmd, pipeline, obj_allocator);

    cmd->barrier(vk::MemoryBarrier2{
        vk::PipelineStageFlagBits2::eTransfer,
        vk::AccessFlagBits2::eTransferWrite,
        shader_stages,
        vk::AccessFlagBits2::eShaderRead | vk::AccessFlagBits2::eUniformRead,
    });
}

void SlangProgramEntryPoint::bind_nested_pbs(const ShaderObjectHandle& object,
                                             const std::vector<NestedPBInfo>& nested_infos,
                                             const CommandBufferHandle& cmd,
                                             const PipelineHandle& pipeline,
                                             const ShaderObjectAllocatorHandle& obj_allocator) {
    for (const auto& ni : nested_infos) {
        const auto& sub = object->get_subobject(ni.subobject_range_index);
        if (!sub)
            continue;
        if (ni.set_index != NO_DESCRIPTOR_SET) {
            sub->bind_as_parameter_block(cmd, pipeline, ni.set_index, obj_allocator);
        }
        bind_nested_pbs(sub, ni.children, cmd, pipeline, obj_allocator);
    }
}

// ---------------------------------------------------------------
// Global parameter support

bool SlangProgramEntryPoint::has_globals(const ContextHandle& context) {
    get_pipeline_layout(context); // ensure layout is built
    return global_set_index != NO_DESCRIPTOR_SET;
}

ShaderObjectHandle
SlangProgramEntryPoint::create_global_shader_object(const ContextHandle& context,
                                                    const ResourceAllocatorHandle& allocator) {
    get_pipeline_layout(context);
    assert(global_object_layout && "No global parameters in this program");
    return std::make_shared<ShaderObject>(global_object_layout, allocator);
}

void SlangProgramEntryPoint::bind_global_parameter(
    const ShaderObjectHandle& globals,
    const CommandBufferHandle& cmd,
    const PipelineHandle& pipeline,
    const ShaderObjectAllocatorHandle& obj_allocator) {
    const auto shader_stages = pipeline->get_pipeline_stage_flags2();

    cmd->barrier(vk::MemoryBarrier2{
        shader_stages,
        vk::AccessFlagBits2::eShaderRead | vk::AccessFlagBits2::eUniformRead,
        vk::PipelineStageFlagBits2::eTransfer,
        vk::AccessFlagBits2::eTransferWrite,
    });

    if (global_set_index != NO_DESCRIPTOR_SET) {
        globals->bind_as_parameter_block(cmd, pipeline, global_set_index, obj_allocator);
    }
    if (!global_nested_pb_infos.empty()) {
        bind_nested_pbs(globals, global_nested_pb_infos, cmd, pipeline, obj_allocator);
    }

    cmd->barrier(vk::MemoryBarrier2{
        vk::PipelineStageFlagBits2::eTransfer,
        vk::AccessFlagBits2::eTransferWrite,
        shader_stages,
        vk::AccessFlagBits2::eShaderRead | vk::AccessFlagBits2::eUniformRead,
    });
}

DescriptorSetLayoutHandle SlangProgramEntryPoint::get_global_set_layout(uint32_t set_index) const {
    if (set_index < global_set_layouts.size()) {
        return global_set_layouts[set_index];
    }
    return nullptr;
}

uint32_t SlangProgramEntryPoint::get_global_set_count() const {
    return static_cast<uint32_t>(global_set_layouts.size());
}

vk::DeviceSize SlangProgramEntryPoint::get_push_constant_size() const {
    return push_constant_size;
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
        out += fmt::format("{}subobject_range={}, set_index={}\n", indent, ni.subobject_range_index,
                           ni.set_index);
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
            out += format_type_layout(element_tl, 2, "        ");
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
// Rebuild

void SlangProgramEntryPoint::rebuild() {
    param_cache.clear();
    cached_pipeline_layout = nullptr;
    global_object_layout = nullptr;
    global_set_index = UINT32_MAX;
    global_set_layouts.clear();
    global_nested_pb_infos.clear();
    push_constant_size = 0;
    increment_version();
}

// ---------------------------------------------------------------
// Factory methods

SlangProgramEntryPointHandle SlangProgramEntryPoint::create(const SlangProgramHandle& program,
                                                            const uint64_t entry_point_index) {
    auto ep = SlangProgramEntryPointHandle(new SlangProgramEntryPoint(program, entry_point_index));
    program->on_changed(ep, [raw = ep.get()]() { raw->rebuild(); });
    return ep;
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
