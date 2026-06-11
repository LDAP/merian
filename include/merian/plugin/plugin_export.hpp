#pragma once

// Shared by both merian plugin ABIs (context extensions and graph nodes). A plugin shared library
// (merian-plugin-*) exports a small set of well-known C entry points; the host resolves them by
// name and calls those that are present.

// Decoration for the exported plugin entry points. A plugin only ever exports these symbols; the
// host resolves them dynamically, so there is no import side.
#if defined(_WIN32)
#define MERIAN_PLUGIN_EXPORT __declspec(dllexport)
#else
#define MERIAN_PLUGIN_EXPORT __attribute__((visibility("default")))
#endif

extern "C" {

// Optional. Human-readable name used in logs. Defaults to the library file name.
MERIAN_PLUGIN_EXPORT const char* merian_plugin_name(void);

using merian_plugin_name_fn = const char* (*)();

} // extern "C"
