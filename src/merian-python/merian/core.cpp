// Python.h must precede the standard headers
#include "json_caster.hpp"

#include "merian/io/image_io.hpp"
#include "merian/plugin/plugins.hpp"
#include "merian/utils/properties.hpp"
#include "merian/vk/context.hpp"
#include "merian/vk/extension/extension_device_fault.hpp"
#include "merian/vk/extension/extension_vk_validation_layers.hpp"
#include "merian/vk/utils/profiler.hpp"

#include <array>
#include <nanobind/ndarray.h>
#include <nanobind/stl/filesystem.h>
#include <nanobind/stl/optional.h>
#include <nanobind/stl/shared_ptr.h>
#include <nanobind/stl/string.h>
#include <nanobind/stl/vector.h>
#include <spdlog/spdlog.h>

namespace nb = nanobind;
using namespace nb::literals;

NB_MODULE(_core, m) {
    nb::set_leak_warnings(false);
    m.attr("DEBUG_BUILD") = merian::Context::IS_DEBUG_BUILD;
    m.attr("DEVICE_FAULT_EXTENSION") = merian::ExtensionDeviceFault::name;

    const nb::object merian_error = nb::exception<merian::MerianException>(m, "MerianError");
    nb::exception<merian::MissingExtension>(m, "MissingExtension", merian_error);
    nb::exception<merian::VulkanException>(m, "VulkanError", merian_error);

    m.def(
        "set_log_level",
        [](const std::string& level) { spdlog::set_level(spdlog::level::from_str(level)); },
        "level"_a, "One of trace, debug, info, warn, error, critical, off.");

    nb::class_<merian::Context>(m, "Context", "Vulkan instance and device.")
        .def(nb::new_([](const std::vector<std::string>& extensions, const bool validation,
                         const std::vector<std::filesystem::path>& search_paths,
                         const std::string& device) {
                 std::vector<std::string> context_extensions = extensions;
                 if (validation) {
                     context_extensions.emplace_back(merian::ExtensionVkValidationLayers::name);
                 }
                 const nb::gil_scoped_release release;
                 return merian::Context::create({
                     .context_extensions = context_extensions,
                     .additional_search_paths = search_paths,
                     .application_name = "merian-python",
                     .filter_device_name = device,
                 });
             }),
             "extensions"_a = std::vector<std::string>{}, "validation"_a = false,
             "search_paths"_a = std::vector<std::filesystem::path>{}, "device"_a = "",
             "search_paths: for shaders and other files, device: only consider the device with "
             "this name.")
        .def_prop_ro(
            "device_name",
            [](const merian::Context& context) -> std::string {
                return context.get_physical_device()->get_properties().get_properties().deviceName;
            })
        .def(
            "dump_device_fault",
            [](merian::Context& context) {
                if (const auto fault =
                        context.get_context_extension<merian::ExtensionDeviceFault>(true)) {
                    fault->dump();
                }
            },
            "Logs the device fault report, if the context loaded DEVICE_FAULT_EXTENSION.");

    nb::class_<merian::ProfilerReport> profiler_report(
        m, "ProfilerReport", "Times of the profiled sections over a period.");
    nb::class_<merian::ProfilerReport::Entry>(profiler_report, "Entry")
        .def_ro("name", &merian::ProfilerReport::Entry::name)
        .def_ro("duration_ms", &merian::ProfilerReport::Entry::duration_ms, "Mean.")
        .def_ro("std_deviation_ms", &merian::ProfilerReport::Entry::std_deviation_ms)
        .def_ro("last_seen_ms_ago", &merian::ProfilerReport::Entry::last_seen_ms_ago,
                "Set if the section did not run in the period.")
        .def_ro("children", &merian::ProfilerReport::Entry::children)
        .def("__repr__", [](const merian::ProfilerReport::Entry& entry) {
            return fmt::format("Entry('{}', {:.4f} ms, std {:.4f} ms, {} children)", entry.name,
                               entry.duration_ms, entry.std_deviation_ms, entry.children.size());
        });
    profiler_report.def_ro("cpu", &merian::ProfilerReport::cpu)
        .def_ro("gpu", &merian::ProfilerReport::gpu)
        .def("cpu_total_ms", &merian::ProfilerReport::cpu_total_ms)
        .def("cpu_total_std_deviation_ms", &merian::ProfilerReport::cpu_total_std_deviation_ms)
        .def("gpu_total_ms", &merian::ProfilerReport::gpu_total_ms)
        .def("gpu_total_std_deviation_ms", &merian::ProfilerReport::gpu_total_std_deviation_ms)
        .def("to_text", &merian::ProfilerReport::to_text)
        .def("to_json", &merian::ProfilerReport::to_json)
        .def("__str__", &merian::ProfilerReport::to_text);

    nb::enum_<merian::Properties::ChildFlagBits>(m, "ChildFlags", nb::is_flag(),
                                                 nb::is_arithmetic())
        .value("DEFAULT_OPEN", merian::Properties::ChildFlagBits::DEFAULT_OPEN)
        .value("FRAMED", merian::Properties::ChildFlagBits::FRAMED)
        .value("TABS", merian::Properties::ChildFlagBits::TABS);
    nb::enum_<merian::Properties::OptionsStyle>(m, "OptionsStyle")
        .value("DONT_CARE", merian::Properties::OptionsStyle::DONT_CARE)
        .value("RADIO_BUTTON", merian::Properties::OptionsStyle::RADIO_BUTTON)
        .value("COMBO", merian::Properties::OptionsStyle::COMBO)
        .value("LIST_BOX", merian::Properties::OptionsStyle::LIST_BOX);

    nb::module_ io = m.def_submodule("io", "Image files");
    io.def(
        "load_image",
        [](const std::filesystem::path& path) {
            merian::ImageInfo info;
            merian::BlobHandle blob;
            {
                const nb::gil_scoped_release release;
                blob = merian::image_load_f32(path, info, 0);
            }
            auto* const owner = new merian::BlobHandle(blob);
            const nb::capsule capsule(owner, [](void* pointer) noexcept {
                delete static_cast<merian::BlobHandle*>(pointer);
            });
            const std::array<size_t, 3> shape = {static_cast<size_t>(info.height),
                                                 static_cast<size_t>(info.width),
                                                 static_cast<size_t>(info.channels)};
            return nb::ndarray<nb::numpy, float>(blob->get_data<float>(), shape.size(),
                                                 shape.data(), capsule);
        },
        "path"_a, "float32 (height, width, channels); LDR files are linearized with gamma 2.2.");
    io.def(
        "save_image",
        [](const std::filesystem::path& path,
           const nb::ndarray<nb::c_contig, nb::device::cpu>& array) {
            if (array.ndim() != 2 && array.ndim() != 3) {
                throw std::invalid_argument("expected (height, width[, channels])");
            }
            const int height = static_cast<int>(array.shape(0));
            const int width = static_cast<int>(array.shape(1));
            const int channels = array.ndim() == 3 ? static_cast<int>(array.shape(2)) : 1;
            if (array.dtype() == nb::dtype<float>()) {
                const nb::gil_scoped_release release;
                merian::image_save_f32(path, static_cast<const float*>(array.data()), width, height,
                                       channels);
            } else if (array.dtype() == nb::dtype<uint8_t>()) {
                const nb::gil_scoped_release release;
                merian::image_save_u8(path, static_cast<const uint8_t*>(array.data()), width,
                                      height, channels);
            } else {
                throw std::invalid_argument("expected float32 (HDR, PFM) or uint8 (PNG, JPG, "
                                            "BMP, TGA) elements");
            }
        },
        "path"_a, "array"_a, "The format follows the extension.");

    nb::module_ plugins = m.def_submodule("plugins", "Discovery of merian-plugin-* libraries");
    plugins.def("add_search_path", &merian::Plugins::add_search_path, "path"_a,
                "Searched first; only effective before the first context is created.");
    plugins.def("discover", &merian::Plugins::discover);
}
