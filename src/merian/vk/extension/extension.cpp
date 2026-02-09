#include "merian/vk/extension/extension.hpp"

#include "merian/vk/physical_device.hpp"

#include <cstring>
#include <fmt/format.h>
#include <fmt/ranges.h>
#include <functional>

namespace merian {

ContextExtension::~ContextExtension() {}

DeviceSupportInfo
DeviceSupportInfo::check(const DeviceSupportQueryInfo& query_info,
                         const std::vector<const char*>& required_features,
                         const std::vector<const char*>& optional_features,
                         const std::vector<const char*>& required_extensions,
                         const std::vector<const char*>& optional_extensions,
                         const std::vector<const char*>& required_spirv_capabilities,
                         const std::vector<const char*>& optional_spirv_capabilities,
                         const std::vector<const char*>& required_spirv_extensions,
                         const std::vector<const char*>& optional_spirv_extensions) {
    DeviceSupportInfo info;
    const auto& pd = query_info.physical_device;

    const auto str_in = [](const std::vector<const char*>& haystack, const char* needle) {
        return std::find_if(haystack.begin(), haystack.end(), [needle](const char* s) {
                   return std::strcmp(s, needle) == 0;
               }) != haystack.end();
    };

    struct Category {
        const char* label;
        const std::vector<const char*>& required;
        const std::vector<const char*>& optional;
        std::vector<const char*>& out;
        std::function<bool(const char*)> is_supported;
    };

    Category categories[] = {
        {
            "features",
            required_features,
            optional_features,
            info.required_features,
            [&](const char* n) { return pd->get_supported_features().get_feature(n); },
        },
        {
            "device extensions",
            required_extensions,
            optional_extensions,
            info.required_extensions,
            [&](const char* n) { return pd->extension_supported(n); },
        },
        {
            "SPIR-V capabilities",
            required_spirv_capabilities,
            optional_spirv_capabilities,
            info.required_spirv_capabilities,
            [&, &caps = pd->get_supported_spirv_capabilities()](const char* n) {
                return str_in(caps, n);
            },
        },
        {
            "SPIR-V extensions",
            required_spirv_extensions,
            optional_spirv_extensions,
            info.required_spirv_extensions,
            [&, &exts = pd->get_supported_spirv_extensions()](const char* n) {
                return str_in(exts, n);
            },
        },
    };

    for (auto& [label, required, optional, out, is_supported] : categories) {
        std::vector<const char*> missing;
        for (const char* name : required) {
            if (is_supported(name)) {
                out.emplace_back(name);
            } else {
                missing.emplace_back(name);
            }
        }
        if (!missing.empty()) {
            info.supported = false;
            if (!info.unsupported_reason.empty())
                info.unsupported_reason += "; ";
            info.unsupported_reason +=
                fmt::format("missing {}: {}", label, fmt::join(missing, ", "));
            insert_all(out, missing);
        }
        for (const char* name : optional) {
            if (is_supported(name))
                out.emplace_back(name);
        }
    }

    return info;
}

} // namespace merian
