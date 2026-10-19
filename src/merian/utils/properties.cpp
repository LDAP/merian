#include "merian/utils/properties.hpp"

namespace merian {

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
