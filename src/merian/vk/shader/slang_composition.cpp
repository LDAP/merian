#include "merian/vk/shader/slang_composition.hpp"

namespace merian {

SlangComposition::SlangComposition() {}

void SlangComposition::add_module(const SlangModule& module) {
    auto [it, inserted] = modules.emplace(module.name, module);
    if (!inserted) {
        it->second = module;
    }
}

void SlangComposition::add_module(SlangModule&& module) {
    auto [it, inserted] = modules.emplace(module.name, module);
    if (!inserted) {
        it->second = module;
    }
}

// shortcut for SlangModule::from_path
void SlangComposition::add_module_from_path(
    const std::filesystem::path& path,
    const bool with_entry_points,
    const std::map<std::string, std::string>& entry_point_renames) {
    add_module(SlangModule::from_path(path, with_entry_points, entry_point_renames));
}

void SlangComposition::add_type_conformance(const std::string& interface_name,
                                            const std::string& type_name,
                                            const int64_t dynamic_dispatch_id) {
    add_type_conformance(TypeConformance(interface_name, type_name), dynamic_dispatch_id);
}

// can also update existing type conformance
void SlangComposition::add_type_conformance(const TypeConformance& type_conformance,
                                            const int64_t dynamic_dispatch_id) {
    auto [it, inserted] = type_conformances.emplace(type_conformance, dynamic_dispatch_id);
    if (!inserted) {
        it->second = dynamic_dispatch_id;
    }
}

void SlangComposition::add_type_conformance(TypeConformance&& type_conformance,
                                            const int64_t dynamic_dispatch_id) {
    auto [it, inserted] =
        type_conformances.emplace(std::move(type_conformance), dynamic_dispatch_id);
    if (!inserted) {
        it->second = dynamic_dispatch_id;
    }
}

void SlangComposition::add_entry_point(const std::string& defined_entry_point_name,
                                       const std::string& from_module) {
    entry_points.emplace(EntryPoint(defined_entry_point_name, from_module));
}

void SlangComposition::add_composition(const SlangCompositionHandle& composition) {
    compositions.emplace(composition);
}

SlangCompositionHandle SlangComposition::create() {
    return SlangCompositionHandle(new SlangComposition());
}

} // namespace merian
