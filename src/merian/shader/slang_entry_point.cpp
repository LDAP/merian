#include "merian/shader/slang_entry_point.hpp"
#include "merian/shader/shader_object.hpp"
#include "merian/shader/slang_global_session.hpp"
#include "merian/vk/pipeline/pipeline_layout_builder.hpp"

namespace merian {

static constexpr uint32_t NO_DESCRIPTOR_SET = UINT32_MAX;

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
    for (uint32_t i = 0; i < entry_reflection->getParameterCount(); i++) {
        auto* param = entry_reflection->getParameterByIndex(i);
        auto* type_layout = param->getTypeLayout();

        if (type_layout->getKind() != slang::TypeReflection::Kind::ParameterBlock ||
            param_name != param->getName()) {
            continue;
        }

        // Set index will be assigned by get_pipeline_layout() before bind() uses it.
        auto [inserted_it, _] = param_cache.try_emplace(
            param_name,
            ParameterBlockInfo{program->get_or_create_object_layout(context, type_layout), 0, {}});
        return inserted_it->second;
    }

    throw std::invalid_argument{
        fmt::format("ParameterBlock parameter '{}' not found in entry point", param_name)};
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
    return std::make_shared<ShaderObject>(get_object_layout(context, param_name), allocator);
}

// Recursively assign descriptor sets to nested ParameterBlocks (DFS order, matching Slang's
// SPIR-V output). ParameterBlocks without bindings are skipped and never build a layout.
static void collect_nested_parameter_block_layouts(
    const ShaderObjectLayoutHandle& container_layout,
    PipelineLayoutBuilder& builder,
    uint32_t& next_set,
    std::vector<SlangProgramEntryPoint::NestedParameterBlockInfo>& nested_infos) {
    for (uint32_t subobject_range_index = 0;
         subobject_range_index < container_layout->get_subobject_range_count();
         subobject_range_index++) {
        const auto& range = container_layout->get_subobject_range_info(subobject_range_index);
        if (!range.container_layout || !range.container_layout->is_parameter_block()) {
            continue;
        }

        uint32_t set_index = NO_DESCRIPTOR_SET;
        if (range.container_layout->has_bindings()) {
            set_index = next_set++;
            builder.add_descriptor_set_layout(range.container_layout->get_descriptor_set_layout());
        }

        SlangProgramEntryPoint::NestedParameterBlockInfo info{subobject_range_index, set_index, {}};
        collect_nested_parameter_block_layouts(range.container_layout, builder, next_set,
                                               info.children);
        nested_infos.push_back(std::move(info));
    }
}

// Entry-point uniform parameters and [vk::push_constant] globals map to one push constant range.
static vk::DeviceSize compute_push_constant_size(const ShaderObjectLayoutHandle& global_layout,
                                                 slang::EntryPointReflection* entry_reflection) {
    vk::DeviceSize size = 0;

    if (global_layout) {
        auto* element_type_layout = global_layout->get_element_layout()->get_type_layout();
        const uint32_t binding_range_count = element_type_layout->getBindingRangeCount();
        for (uint32_t binding_range_index = 0; binding_range_index < binding_range_count;
             binding_range_index++) {
            if (element_type_layout->getBindingRangeType(binding_range_index) !=
                slang::BindingType::PushConstant) {
                continue;
            }
            auto* leaf_type_layout =
                element_type_layout->getBindingRangeLeafTypeLayout(binding_range_index);
            assert(leaf_type_layout);
            auto* push_constant_type_layout = leaf_type_layout->getElementTypeLayout();
            size += push_constant_type_layout != nullptr ? push_constant_type_layout->getSize()
                                                         : leaf_type_layout->getSize();
        }
    }

    if (auto* entry_type_layout = entry_reflection->getTypeLayout()) {
        size += entry_type_layout->getSize(SLANG_PARAMETER_CATEGORY_UNIFORM);
    }

    return size;
}

PipelineLayoutHandle SlangProgramEntryPoint::get_pipeline_layout(const ContextHandle& context) {
    if (cached_pipeline_layout) {
        return cached_pipeline_layout;
    }

    auto* program_layout = program->get_program_reflection();
    auto* entry_reflection = get_entry_point_reflection();
    PipelineLayoutBuilder builder(context);

    // Set indices are assigned sequentially: global sets first, then ParameterBlock sets in DFS
    // order. This matches Slang's SPIR-V output.
    uint32_t next_set = 0;

    // 1. global scope, treated as one ParameterBlock-like container
    if (auto* global_var_layout = program_layout->getGlobalParamsVarLayout()) {
        if (auto* global_type_layout = global_var_layout->getTypeLayout()) {
            global_object_layout = std::make_shared<ShaderObjectLayout>(
                context, global_type_layout, program, /*as_scope_container=*/true);
        }
    }
    if (global_object_layout) {
        if (global_object_layout->has_bindings()) {
            global_set_index = next_set++;
            builder.add_descriptor_set_layout(global_object_layout->get_descriptor_set_layout());
        }
        collect_nested_parameter_block_layouts(global_object_layout, builder, next_set,
                                               global_nested_parameter_block_infos);
    }

    // 2. push constants
    push_constant_size = compute_push_constant_size(global_object_layout, entry_reflection);
    if (push_constant_size > 0) {
        builder.add_push_constant(static_cast<uint32_t>(push_constant_size));
    }

    // 3. entry point ParameterBlock parameters
    for (uint32_t i = 0; i < entry_reflection->getParameterCount(); i++) {
        auto* param = entry_reflection->getParameterByIndex(i);
        if (param->getTypeLayout()->getKind() != slang::TypeReflection::Kind::ParameterBlock) {
            continue;
        }

        auto& info = find_or_create_param_info(context, param->getName());
        if (info.object_layout->has_bindings()) {
            info.descriptor_set_index = next_set++;
            builder.add_descriptor_set_layout(info.object_layout->get_descriptor_set_layout());
        } else {
            info.descriptor_set_index = NO_DESCRIPTOR_SET;
        }

        collect_nested_parameter_block_layouts(info.object_layout, builder, next_set,
                                               info.nested_parameter_block_infos);
    }

    cached_pipeline_layout = builder.build_pipeline_layout();
    return cached_pipeline_layout;
}

// ---------------------------------------------------------------
// Binding

// Shared objects can be read by pipelines with other stages than the one binding now, so the
// upload barriers cover all commands instead of the current pipeline's stages.
static void barrier_before_uploads(const CommandBufferHandle& cmd) {
    cmd->barrier(vk::MemoryBarrier2{
        vk::PipelineStageFlagBits2::eAllCommands,
        vk::AccessFlagBits2::eShaderRead | vk::AccessFlagBits2::eUniformRead,
        vk::PipelineStageFlagBits2::eTransfer,
        vk::AccessFlagBits2::eTransferWrite,
    });
}

static void barrier_after_uploads(const CommandBufferHandle& cmd) {
    cmd->barrier(vk::MemoryBarrier2{
        vk::PipelineStageFlagBits2::eTransfer,
        vk::AccessFlagBits2::eTransferWrite,
        vk::PipelineStageFlagBits2::eAllCommands,
        vk::AccessFlagBits2::eShaderRead | vk::AccessFlagBits2::eUniformRead,
    });
}

void SlangProgramEntryPoint::bind_entry_point_parameter(
    const std::string& param_name,
    const ShaderObjectHandle& object,
    const CommandBufferHandle& cmd,
    const PipelineHandle& pipeline,
    const ShaderObjectAllocatorHandle& obj_allocator) {
    const auto& context = pipeline->get_layout()->get_context();
    get_pipeline_layout(context);
    auto& info = find_or_create_param_info(context, param_name);

    const bool needs_upload = object->has_pending_uploads();
    if (needs_upload) {
        barrier_before_uploads(cmd);
    }

    if (info.descriptor_set_index != NO_DESCRIPTOR_SET) {
        object->bind_as_parameter_block(cmd, pipeline, info.descriptor_set_index, obj_allocator);
    }

    bind_nested_parameter_blocks(object, info.nested_parameter_block_infos, cmd, pipeline,
                                 obj_allocator);

    if (needs_upload) {
        barrier_after_uploads(cmd);
    }
}

void SlangProgramEntryPoint::bind_nested_parameter_blocks(
    const ShaderObjectHandle& object,
    const std::vector<NestedParameterBlockInfo>& nested_infos,
    const CommandBufferHandle& cmd,
    const PipelineHandle& pipeline,
    const ShaderObjectAllocatorHandle& obj_allocator) {
    for (const auto& nested_info : nested_infos) {
        const auto& subobject = object->get_subobject(nested_info.subobject_range_index);
        if (!subobject) {
            continue;
        }
        if (nested_info.set_index != NO_DESCRIPTOR_SET) {
            subobject->bind_as_parameter_block(cmd, pipeline, nested_info.set_index, obj_allocator);
        }
        bind_nested_parameter_blocks(subobject, nested_info.children, cmd, pipeline, obj_allocator);
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
    const bool needs_upload = globals->has_pending_uploads();
    if (needs_upload) {
        barrier_before_uploads(cmd);
    }

    if (global_set_index != NO_DESCRIPTOR_SET) {
        globals->bind_as_parameter_block(cmd, pipeline, global_set_index, obj_allocator);
    }
    bind_nested_parameter_blocks(globals, global_nested_parameter_block_infos, cmd, pipeline,
                                 obj_allocator);

    if (needs_upload) {
        barrier_after_uploads(cmd);
    }
}

DescriptorSetLayoutHandle
SlangProgramEntryPoint::get_global_set_layout(const uint32_t set_index) const {
    if (set_index == 0 && global_object_layout && global_object_layout->has_bindings()) {
        return global_object_layout->get_descriptor_set_layout();
    }
    return nullptr;
}

uint32_t SlangProgramEntryPoint::get_global_set_count() const {
    return global_object_layout && global_object_layout->has_bindings() ? 1 : 0;
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

static void format_nested_parameter_block_tree(
    const std::vector<SlangProgramEntryPoint::NestedParameterBlockInfo>& infos,
    std::string& out,
    const std::string& indent) {
    for (const auto& nested_info : infos) {
        out += fmt::format("{}subobject_range={}, set_index={}\n", indent,
                           nested_info.subobject_range_index, nested_info.set_index);
        if (!nested_info.children.empty()) {
            format_nested_parameter_block_tree(nested_info.children, out, indent + "  ");
        }
    }
}

std::string SlangProgramEntryPoint::format_reflection(const ContextHandle& context) {
    // Ensure pipeline layout is built so param_cache is populated
    get_pipeline_layout(context);

    std::string out;
    auto* entry_reflection = get_entry_point_reflection();
    out += fmt::format("SlangProgramEntryPoint '{}'\n", entry_reflection->getNameOverride());
    out += fmt::format("  stage: {}\n", vk::to_string(get_stage()));

    out += fmt::format("  parameters ({}):\n", entry_reflection->getParameterCount());
    for (uint32_t i = 0; i < entry_reflection->getParameterCount(); i++) {
        auto* param = entry_reflection->getParameterByIndex(i);
        auto* type_layout = param->getTypeLayout();
        out += fmt::format("    [{}] '{}': kind={}\n", i, param->getName(),
                           slang_type_kind_to_string(type_layout->getKind()));
    }

    out += fmt::format("  cached parameter blocks ({}):\n", param_cache.size());
    for (const auto& [name, info] : param_cache) {
        out += fmt::format("    '{}': set_index={}\n", name, info.descriptor_set_index);
        out += format_as(*info.object_layout, "      ");
        if (!info.nested_parameter_block_infos.empty()) {
            out += fmt::format("    nested ParameterBlock tree:\n");
            format_nested_parameter_block_tree(info.nested_parameter_block_infos, out, "      ");
        }
    }

    return out;
}

// ---------------------------------------------------------------
// Factory methods

Versioned<SlangProgramEntryPoint>
SlangProgramEntryPoint::create(const Versioned<SlangProgram>& program,
                               const uint64_t entry_point_index) {
    auto ep = Versioned<SlangProgramEntryPoint>([program, entry_point_index] {
        return SlangProgramEntryPointHandle(
            new SlangProgramEntryPoint(program.get(), entry_point_index));
    });
    ep.depends_on(program);
    return ep;
}

Versioned<SlangProgramEntryPoint>
SlangProgramEntryPoint::create(const Versioned<SlangProgram>& program,
                               const std::string& entry_point_name) {
    auto ep = Versioned<SlangProgramEntryPoint>([program, entry_point_name] {
        const SlangProgramHandle p = program.get();
        return SlangProgramEntryPointHandle(
            new SlangProgramEntryPoint(p, p->get_entry_point_index(entry_point_name)));
    });
    ep.depends_on(program);
    return ep;
}

Versioned<SlangProgramEntryPoint>
SlangProgramEntryPoint::create(const ShaderCompileContextHandle& compile_context,
                               const std::filesystem::path& module_path,
                               const std::string& entry_point_name) {
    return create(SlangProgram::create(compile_context, module_path, true), entry_point_name);
}

} // namespace merian
