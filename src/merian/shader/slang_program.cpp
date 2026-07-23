#include "merian/shader/slang_program.hpp"
#include "merian/shader/shader_object.hpp"
#include "merian/shader/shader_object_layout.hpp"
#include "merian/shader/slang_utils.hpp"
#include "merian/shader/spriv_reflect.hpp"

namespace merian {

SlangProgram::SlangProgram(const ShaderCompileContextHandle& compile_context,
                           const SlangCompositionHandle& composition)
    : compile_context(compile_context), composition(composition) {
    session = compile_context->current_session();
    program = merian::SlangSession::link(session->compose(composition));
}

ShaderModuleHandle SlangProgram::get_shader_module(const ContextHandle& context) {
    if (!shader_module) {
        Slang::ComPtr<slang::IBlob> binary = get_binary();
        shader_module =
            ShaderModule::create(context, binary->getBufferPointer(), binary->getBufferSize());
    }

    return shader_module;
}

Slang::ComPtr<slang::IBlob> SlangProgram::get_binary() {
    if (binary == nullptr) {
        binary = merian::SlangSession::compile(program);
    }
    return binary;
}

slang::ProgramLayout* SlangProgram::get_program_reflection() const {
    return program->getLayout();
}

const Slang::ComPtr<slang::IComponentType>& SlangProgram::get_program() const {
    return program;
}

uint64_t SlangProgram::get_entry_point_index(const std::string& entry_point_name) const {
    for (uint64_t entry_point_index = 0;
         entry_point_index < get_program_reflection()->getEntryPointCount(); entry_point_index++) {
        if (entry_point_name ==
            get_program_reflection()->getEntryPointByIndex(entry_point_index)->getNameOverride()) {
            return entry_point_index;
        }
    }

    throw std::invalid_argument{
        fmt::format("no entry point with name {} in program", entry_point_name)};
}

const SlangCompositionHandle& SlangProgram::get_composition() const {
    return composition;
}

// ---------------------------------------------------------------
// Type layout

slang::TypeLayoutReflection* SlangProgram::get_type_layout(const std::string& type_name) const {
    auto* type = get_program_reflection()->findTypeByName(type_name.c_str());
    if (type == nullptr) {
        throw ShaderCompiler::compilation_failed(fmt::format("type '{}' not found", type_name));
    }
    auto* layout = get_program_reflection()->getTypeLayout(type, slang::LayoutRules::Default);
    if (layout == nullptr) {
        throw ShaderCompiler::compilation_failed(
            fmt::format("failed to get type layout for '{}'", type_name));
    }
    return layout;
}

ShaderObjectLayoutHandle
SlangProgram::get_or_create_object_layout(const ContextHandle& context,
                                          slang::TypeLayoutReflection* type_layout) {
    auto& entry = object_layout_cache[type_layout];
    if (auto layout = entry.lock()) {
        return layout;
    }
    auto layout = std::make_shared<ShaderObjectLayout>(context, type_layout, shared_from_this());
    entry = layout;
    return layout;
}

ShaderObjectHandle
SlangProgram::create_shader_object_for_type(const ContextHandle& context,
                                            const std::string& type_name,
                                            const ResourceAllocatorHandle& allocator) {
    auto* type_layout = get_type_layout(fmt::format("ParameterBlock<{}>", type_name));
    return std::make_shared<ShaderObject>(get_or_create_object_layout(context, type_layout),
                                          allocator);
}

// ---------------------------------------------------------------
// Parameter reflection (global parameters)

uint32_t SlangProgram::get_parameter_count() const {
    return get_program_reflection()->getParameterCount();
}

slang::VariableLayoutReflection* SlangProgram::get_parameter(uint32_t index) const {
    return get_program_reflection()->getParameterByIndex(index);
}

slang::VariableLayoutReflection* SlangProgram::get_parameter(const std::string& name) const {
    auto* layout = get_program_reflection();
    for (uint32_t i = 0; i < layout->getParameterCount(); i++) {
        auto* param = layout->getParameterByIndex(i);
        if (name == param->getName()) {
            return param;
        }
    }
    return nullptr;
}

std::vector<std::string> SlangProgram::get_parameter_names() const {
    auto* layout = get_program_reflection();
    std::vector<std::string> names;
    names.reserve(layout->getParameterCount());
    for (uint32_t i = 0; i < layout->getParameterCount(); i++) {
        names.emplace_back(layout->getParameterByIndex(i)->getName());
    }
    return names;
}

bool SlangProgram::has_parameter(const std::string& name) const {
    return get_parameter(name) != nullptr;
}

// ---------------------------------------------------------------
// Debug

std::string SlangProgram::format_reflection() const {
    auto* layout = get_program_reflection();
    std::string out;
    out += fmt::format("SlangProgram\n");

    out += fmt::format("  entry_points ({}):\n", layout->getEntryPointCount());
    for (uint64_t i = 0; i < layout->getEntryPointCount(); i++) {
        auto* ep = layout->getEntryPointByIndex(i);
        out += fmt::format("    [{}] '{}'\n", i, ep->getNameOverride());

        out += fmt::format("      parameters ({}):\n", ep->getParameterCount());
        for (uint32_t p = 0; p < ep->getParameterCount(); p++) {
            auto* param = ep->getParameterByIndex(p);
            auto* tl = param->getTypeLayout();
            out += fmt::format("        {:>2}: '{}': kind={}\n", p, param->getName(),
                               slang_type_kind_to_string(tl->getKind()));
            if (tl->getKind() == slang::TypeReflection::Kind::ParameterBlock ||
                tl->getKind() == slang::TypeReflection::Kind::ConstantBuffer) {
                out += format_type_layout(tl->getElementTypeLayout(), 2, "          ");
            }
        }
    }

    uint32_t global_count = layout->getParameterCount();
    if (global_count > 0) {
        out += fmt::format("  global_parameters ({}):\n", global_count);
        for (uint32_t i = 0; i < global_count; i++) {
            auto* param = layout->getParameterByIndex(i);
            auto* tl = param->getTypeLayout();
            out += fmt::format("    {:>2}: '{}': kind={}\n", i, param->getName(),
                               slang_type_kind_to_string(tl->getKind()));
        }
    }

    return out;
}

Versioned<SlangProgram> SlangProgram::create(const ShaderCompileContextHandle& compile_context,
                                             const SlangCompositionHandle& composition) {
    auto program = Versioned<SlangProgram>([compile_context, composition] {
        return SlangProgramHandle(new SlangProgram(compile_context, composition));
    });
    program.depends_on([composition] { return composition->version(); });
    return program;
}

Versioned<SlangProgram> SlangProgram::create(const ShaderCompileContextHandle& compile_context,
                                             const Versioned<SlangComposition>& composition) {
    auto program = Versioned<SlangProgram>([compile_context, composition] {
        return SlangProgramHandle(new SlangProgram(compile_context, composition.get()));
    });
    program.depends_on(composition);
    program.depends_on([composition] { return composition.get()->version(); });
    return program;
}

Versioned<SlangProgram> SlangProgram::create(const ShaderCompileContextHandle& compile_context,
                                             const std::filesystem::path& path,
                                             const bool with_entry_points) {
    auto comp = SlangComposition::create();
    comp->add_module_from_path(path, with_entry_points);
    return create(compile_context, comp);
}

DeviceSupportInfo SlangProgram::query_device_support(const DeviceSupportQueryInfo& query_info) {
    const Slang::ComPtr<slang::IBlob> binary = get_binary();
    const SpirvReflect reflect(static_cast<const uint32_t*>(binary->getBufferPointer()),
                               binary->getBufferSize());
    return reflect.query_device_support(query_info);
}

} // namespace merian
