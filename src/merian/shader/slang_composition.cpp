#include "merian/shader/slang_composition.hpp"

namespace merian {

SlangComposition::SlangComposition() {}

void SlangComposition::add_module(const SlangModule& module) {
    auto [it, inserted] = modules.emplace(module.name, module);
    if (!inserted) {
        it->second = module;
    }
    increment_version();
}

void SlangComposition::add_module(SlangModule&& module) {
    auto [it, inserted] = modules.emplace(module.name, module);
    if (!inserted) {
        it->second = module;
    }
    increment_version();
}

// shortcut for SlangModule::from_path
void SlangComposition::add_module_from_path(
    const std::filesystem::path& path,
    const bool with_entry_points,
    const std::map<std::string, std::string>& entry_point_renames) {
    add_module(SlangModule::from_path(path, with_entry_points, entry_point_renames));
}

void SlangComposition::add_module_from_string(
    const std::string& name,
    const std::string& source,
    const bool with_entry_points,
    const std::map<std::string, std::string>& entry_point_renames) {
    add_module(SlangModule::from_source(name, source, std::nullopt, with_entry_points,
                                        entry_point_renames));
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
    increment_version();
}

void SlangComposition::add_type_conformance(TypeConformance&& type_conformance,
                                            const int64_t dynamic_dispatch_id) {
    auto [it, inserted] =
        type_conformances.emplace(std::move(type_conformance), dynamic_dispatch_id);
    if (!inserted) {
        it->second = dynamic_dispatch_id;
    }
    increment_version();
}

void SlangComposition::add_entry_point(const std::string& defined_entry_point_name,
                                       const std::string& from_module) {
    entry_points.emplace(EntryPoint(defined_entry_point_name, from_module));
    increment_version();
}

void SlangComposition::add_composition(const SlangCompositionHandle& composition) {
    compositions.emplace(composition);
    composition->on_changed(shared_from_this(), [this]() { increment_version(); });
    increment_version();
}

bool SlangComposition::reload(const FileLoader& file_loader) {
    bool changed = false;

    // Check path-based modules for file changes
    for (const auto& [name, module] : modules) {
        const auto& source_path = module.get_source_path();
        if (!source_path.has_value()) {
            continue;
        }

        auto resolved = file_loader.find_file(*source_path);
        if (!resolved) {
            continue;
        }

        std::error_code ec;
        auto mtime = std::filesystem::last_write_time(*resolved, ec);
        if (ec) {
            continue;
        }

        auto it = module_mtimes.find(*resolved);
        if (it == module_mtimes.end()) {
            module_mtimes[*resolved] = mtime;
        } else if (it->second != mtime) {
            it->second = mtime;
            changed = true;
        }
    }

    if (changed) {
        increment_version();
    }
    return changed;
}

void SlangComposition::force_reload() {
    increment_version();
}

SlangCompositionHandle SlangComposition::create() {
    return SlangCompositionHandle(new SlangComposition());
}

} // namespace merian
