#include "merian/vk/shader/slang_composition.hpp"

namespace merian {

SlangComposition::SlangComposition(const ShaderCompileContextHandle& compile_context)
    : compile_context(compile_context) {
    session = SlangSession::create(compile_context);
}

void SlangComposition::add_module(const SlangModule& module, const bool with_entry_points) {
    SlangModule& added_module = modules.emplace(module.name, module).first->second;

    load_module(added_module);
    if (with_entry_points) {
        add_entry_points_from_module(added_module);
    }

    composite.setNull();
}

void SlangComposition::add_module(SlangModule&& module, const bool with_entry_points) {
    auto it = modules.emplace(module.name, module);
    SlangModule& added_module = it.first->second;

    load_module(added_module);
    if (with_entry_points) {
        add_entry_points_from_module(added_module);
    }

    composite.setNull();
}

// shortcut for SlangModule::from_path
void SlangComposition::add_module_from_path(const std::filesystem::path& path,
                                            const bool with_entry_points) {
    add_module(SlangModule::from_path(path), with_entry_points);
}

void SlangComposition::add_type_conformance(const std::string& interface_name,
                                            const std::string& type_name,
                                            const int64_t dynamic_dispatch_id) {
    add_type_conformance(TypeConformance(interface_name, type_name), dynamic_dispatch_id);
}

// can also update existing type conformance
void SlangComposition::add_type_conformance(const TypeConformance& type_conformance,
                                            const int64_t dynamic_dispatch_id) {
    type_conformances[type_conformance] = std::make_pair(dynamic_dispatch_id, nullptr);

    composite.setNull();
}

void SlangComposition::add_type_conformance(TypeConformance&& type_conformance,
                                            const int64_t dynamic_dispatch_id) {
    type_conformances.emplace(std::move(type_conformance),
                              std::make_pair(dynamic_dispatch_id, nullptr));

    composite.setNull();
}

void SlangComposition::add_entry_point(const std::string& defined_entry_point_name,
                                       const std::string& from_module) {
    entry_points.emplace_back(EntryPoint(defined_entry_point_name, from_module));

    composite.setNull();
}

Slang::ComPtr<slang::IComponentType> SlangComposition::get_composite() {
    if (composite == nullptr) {
        compose();
    }

    return composite;
}

SlangCompositionHandle SlangComposition::create(const ShaderCompileContextHandle& compile_context) {
    return SlangCompositionHandle(new SlangComposition(compile_context));
}

void SlangComposition::load_module(SlangModule& module) {
    module.module = session->load_module_from_source(
        module.get_name(), module.get_source(compile_context->get_search_path_file_loader()),
        module.get_import_path());
}

void SlangComposition::add_entry_points_from_module(SlangModule& module) {
    assert(module.module);

    for (uint32_t entry_point_index = 0;
         entry_point_index < merian::SlangSession::get_defined_entry_point_count(module.module);
         entry_point_index++) {

        Slang::ComPtr<slang::IEntryPoint> entry_point =
            merian::SlangSession::get_defined_entry_point(module.module, entry_point_index);
        const char* name = entry_point->getFunctionReflection()->getName();
        add_entry_point(name, module.name);
    }
}

void SlangComposition::compose() {
    std::vector<slang::IComponentType*> components;
    components.reserve(
        std::max(modules.size(), 1 + type_conformances.size() + entry_points.size()));

    for (auto& [_, module] : modules) {
        if (module.module == nullptr) {
            load_module(module);
        }
        components.emplace_back(module.module);
    }

    Slang::ComPtr<slang::IComponentType> composed_modules = session->compose(components);
    components.clear();
    components.emplace_back(composed_modules);

    for (auto& [type_conformance, id_comp] : type_conformances) {
        if (id_comp.second == nullptr) {
            id_comp.second =
                session->create_type_conformance(composed_modules, type_conformance.type_name,
                                                 type_conformance.interface_name, id_comp.first);
        }
        components.emplace_back(id_comp.second);
    }

    for (auto& entry_point : entry_points) {
        SlangModule& module = modules.at(entry_point.from_module);
        if (entry_point.entry_point == nullptr) {
            entry_point.entry_point = merian::SlangSession::find_entry_point_or_fail(
                module.module, entry_point.defined_name);
        }
        auto it = module.get_entry_point_map().find(entry_point.defined_name);
        if (it != module.get_entry_point_map().end()) {
            entry_point.entry_point->renameEntryPoint(it->second.c_str(),
                                                      entry_point.renamed_entry_point.writeRef());
            components.push_back(entry_point.renamed_entry_point);
        } else {
            components.push_back(entry_point.entry_point);
        }
    }

    composite = session->compose(components);
}

} // namespace merian
