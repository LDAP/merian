#include "merian/shader/slang_composition.hpp"
#include "merian/shader/slang_global_session.hpp"

#include <algorithm>
#include <cctype>

namespace merian {

std::string slang_import_spelling(const std::filesystem::path& import_path) {
    const std::string quoted = fmt::format("\"{}\"", import_path.generic_string());
    if (import_path.extension() != ".slang") {
        return quoted;
    }
    const std::string stem =
        std::filesystem::path{import_path}.replace_extension().generic_string();
    std::string name = stem;
    std::ranges::replace(name, '-', '_');
    std::string dashed = name;
    std::ranges::replace(dashed, '_', '-');
    if (name != stem && dashed != stem) {
        return quoted;
    }
    for (const std::filesystem::path& component : std::filesystem::path{name}) {
        const std::string identifier = component.string();
        if (identifier.empty() || std::isdigit(static_cast<unsigned char>(identifier[0])) != 0 ||
            !std::ranges::all_of(identifier, [](const char c) {
                return std::isalnum(static_cast<unsigned char>(c)) != 0 || c == '_';
            })) {
            return quoted;
        }
    }
    std::ranges::replace(name, '/', '.');
    return name;
}

SlangComposition::SlangComposition()
    : creation_time(std::filesystem::file_time_type::clock::now()) {}

void SlangComposition::add_module(const SlangModule& module) {
    auto [it, inserted] = module_index.try_emplace(module.name, modules.size());
    if (inserted) {
        modules.emplace_back(module);
    } else {
        modules[it->second] = module;
    }
    edits++;
}

void SlangComposition::add_module(SlangModule&& module) {
    auto [it, inserted] = module_index.try_emplace(module.name, modules.size());
    if (inserted) {
        modules.emplace_back(std::move(module));
    } else {
        modules[it->second] = std::move(module);
    }
    edits++;
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
    edits++;
}

void SlangComposition::add_type_conformance(TypeConformance&& type_conformance,
                                            const int64_t dynamic_dispatch_id) {
    auto [it, inserted] =
        type_conformances.emplace(std::move(type_conformance), dynamic_dispatch_id);
    if (!inserted) {
        it->second = dynamic_dispatch_id;
    }
    edits++;
}

void SlangComposition::add_entry_point(const std::string& defined_entry_point_name,
                                       const std::string& from_module) {
    entry_points.emplace(EntryPoint(defined_entry_point_name, from_module));
    edits++;
}

void SlangComposition::add_composition(const SlangCompositionHandle& composition) {
    if (std::ranges::find(compositions, composition) == compositions.end()) {
        compositions.emplace_back(composition);
    }
    edits++;
}

bool SlangComposition::reload(const FileLoader& file_loader) {
    // Recurse first: a shared subcomposition dedups via its own mtime state, so only the first
    // caller that sees the change bumps it.
    bool nested_changed = false;
    for (const auto& composition : compositions) {
        nested_changed |= composition->reload(file_loader);
    }

    bool source_changed = false;

    // Check path-based modules for file changes
    for (const auto& module : modules) {
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
            // edited before the first reload
            source_changed |= mtime > creation_time;
        } else if (it->second != mtime) {
            it->second = mtime;
            source_changed = true;
        }
    }

    if (source_changed) {
        bump_slang_source_epoch();
        edits++;
    }
    return source_changed || nested_changed;
}

void SlangComposition::force_reload() {
    bump_slang_source_epoch();
    edits++;
}

SlangCompositionHandle SlangComposition::create() {
    return SlangCompositionHandle(new SlangComposition());
}

} // namespace merian
