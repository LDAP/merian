#pragma once

#define MERIAN_EXT_ENABLE_IF_REQUESTED(name, supported, enabled, required_features_set,            \
                                       optional_features_set)                                      \
    do {                                                                                           \
        const bool required = required_features_set.contains(#name);                               \
        const bool optional = optional_features_set.contains(#name);                               \
        if ((supported).name && (required || optional)) {                                          \
            (enabled).name = VK_TRUE;                                                              \
            SPDLOG_DEBUG("enable feature {}", #name);                                              \
        }                                                                                          \
        if (required && !(supported).name) {                                                       \
            all_required_supported = false;                                                        \
            SPDLOG_ERROR("feature {} required but not supported", #name);                          \
        }                                                                                          \
        if (optional && !(supported).name) {                                                       \
            SPDLOG_DEBUG("feature {} optionally requested but not supported", #name);              \
        }                                                                                          \
        required_features_set.erase(#name);                                                        \
        optional_features_set.erase(#name);                                                        \
    } while (0)

#define MERIAN_EXT_ENABLE_IF_REQUESTED_PREFIXED(prefix, name, supported, enabled,                  \
                                                required_features_set, optional_features_set)      \
    do {                                                                                           \
        const bool required = required_features_set.contains(#prefix "/" #name);                   \
        const bool optional = optional_features_set.contains(#prefix "/" #name);                   \
        if ((supported).name && (required || optional)) {                                          \
            (enabled).name = VK_TRUE;                                                              \
            SPDLOG_DEBUG("enable feature {}/{}", #prefix, #name);                                  \
        }                                                                                          \
        if (required && !(supported).name) {                                                       \
            all_required_supported = false;                                                        \
            SPDLOG_ERROR("feature {}/{} required but not supported", #prefix, #name);              \
        }                                                                                          \
        if (optional && !(supported).name) {                                                       \
            SPDLOG_DEBUG("feature {}/{} optionally requested but not supported", #prefix, #name);  \
        }                                                                                          \
        required_features_set.erase(#prefix "/" #name);                                            \
        optional_features_set.erase(#prefix "/" #name);                                            \
    } while (0)
