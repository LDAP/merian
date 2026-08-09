#pragma once

#include "pbrt_params.hpp"

#include <filesystem>
#include <optional>
#include <string_view>

namespace merian::pbrt {

// RGB of a spectrum-typed parameter: inline (lambda, value) pairs, .spd file, blackbody
// temperature, or a named standard illuminant. Named material spectra (metal-*, glass-*)
// return nullopt; use the dedicated lookups below.
std::optional<float3> spectrum_param_rgb(const ParsedParameter& param,
                                         const std::filesystem::path& base_dir);

// Normal-incidence reflectance for named conductor spectra ("metal-Au-eta" -> Au).
std::optional<float3> named_metal_f0(std::string_view spectrum_name);

// Scalar IOR for named dielectric spectra ("glass-BK7").
std::optional<float> named_glass_ior(std::string_view spectrum_name);

float3 blackbody_rgb(float temperature_kelvin);

} // namespace merian::pbrt
