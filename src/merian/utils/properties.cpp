#include "merian/utils/properties.hpp"

namespace merian {

std::vector<const Properties::SplitPart*>
Properties::shown_split_parts(const std::vector<SplitPart>& parts, bool& reset) {
    std::vector<const SplitPart*> shown;
    float total = 0.f;
    for (const SplitPart& part : parts) {
        if (part.shown) {
            shown.push_back(&part);
            total += *part.weight;
        }
    }
    if (shown.size() < 2) {
        return {};
    }
    reset = !(total > 0.f);
    if (reset) {
        for (const SplitPart* part : shown) {
            *part->weight = 1.f;
        }
    }
    return shown;
}

bool Properties::config_path(const std::string& id,
                             std::filesystem::path& path,
                             const std::string& desc) {
    std::string text = path.string();
    if (config_text(id, text, true, desc)) {
        path = text;
        return true;
    }
    return false;
}

bool Properties::config_path(const std::string& id, std::string& path, const std::string& desc) {
    std::filesystem::path value = path;
    if (config_path(id, value, desc)) {
        path = value.string();
        return true;
    }
    return false;
}

} // namespace merian
