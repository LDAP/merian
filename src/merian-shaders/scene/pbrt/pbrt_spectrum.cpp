#include "pbrt_spectrum.hpp"

#include <charconv>
#include <cmath>
#include <fstream>
#include <sstream>
#include <string>
#include <vector>

namespace merian::pbrt {

namespace {

// CIE 1931 standard observer, multi-lobe Gaussian fit (Wyman, Sloan, Shirley, JCGT 2013).
float gaussian_lobe(const float x, const float mu, const float sigma_l, const float sigma_r) {
    const float sigma = x < mu ? sigma_l : sigma_r;
    const float t = (x - mu) / sigma;
    return std::exp(-0.5f * t * t);
}

float3 cie_xyz(const float lambda) {
    const float x = 1.056f * gaussian_lobe(lambda, 599.8f, 37.9f, 31.0f) +
                    0.362f * gaussian_lobe(lambda, 442.0f, 16.0f, 26.7f) -
                    0.065f * gaussian_lobe(lambda, 501.1f, 20.4f, 26.2f);
    const float y = 0.821f * gaussian_lobe(lambda, 568.8f, 46.9f, 40.5f) +
                    0.286f * gaussian_lobe(lambda, 530.9f, 16.3f, 31.1f);
    const float z = 1.217f * gaussian_lobe(lambda, 437.0f, 11.8f, 36.0f) +
                    0.681f * gaussian_lobe(lambda, 459.0f, 26.0f, 13.8f);
    return float3(x, y, z);
}

float3 xyz_to_linear_srgb(const float3& xyz) {
    return float3(3.2404542f * xyz.x - 1.5371385f * xyz.y - 0.4985314f * xyz.z,
                  -0.9692660f * xyz.x + 1.8760108f * xyz.y + 0.0415560f * xyz.z,
                  0.0556434f * xyz.x - 0.2040259f * xyz.y + 1.0572252f * xyz.z);
}

constexpr float LAMBDA_MIN = 360.0f;
constexpr float LAMBDA_MAX = 830.0f;
constexpr float LAMBDA_STEP = 5.0f;

// Integrates against the CIE curves, normalized by the y-curve integral so a constant
// spectrum of 1 maps to luminance 1. Values outside the sample range evaluate to 0.
template <typename SpectrumFn> float3 integrate_to_rgb(const SpectrumFn& value) {
    float3 xyz(0);
    float y_integral = 0;
    for (float lambda = LAMBDA_MIN; lambda <= LAMBDA_MAX; lambda += LAMBDA_STEP) {
        const float3 bar = cie_xyz(lambda);
        xyz += bar * value(lambda);
        y_integral += bar.y;
    }
    const float3 rgb = xyz_to_linear_srgb(xyz / y_integral);
    return max(rgb, float3(0));
}

// (lambda, value) pairs with ascending lambda.
float3 piecewise_to_rgb(const std::vector<float2>& samples) {
    const auto eval = [&](const float lambda) {
        if (lambda < samples.front().x || lambda > samples.back().x) {
            return 0.0f;
        }
        for (size_t i = 1; i < samples.size(); i++) {
            if (lambda <= samples[i].x) {
                const float t = (lambda - samples[i - 1].x) / (samples[i].x - samples[i - 1].x);
                return lerp(samples[i - 1].y, samples[i].y, t);
            }
        }
        return samples.back().y;
    };
    return integrate_to_rgb(eval);
}

std::optional<std::vector<float2>> load_spd(const std::filesystem::path& path) {
    std::ifstream file(path);
    if (!file) {
        return std::nullopt;
    }
    std::vector<float2> samples;
    std::string line;
    while (std::getline(file, line)) {
        const size_t comment = line.find('#');
        if (comment != std::string::npos) {
            line.resize(comment);
        }
        std::istringstream stream(line);
        float lambda;
        float value;
        while (stream >> lambda >> value) {
            samples.emplace_back(lambda, value);
        }
    }
    if (samples.size() < 2) {
        return std::nullopt;
    }
    return samples;
}

struct NamedF0 {
    std::string_view name;
    float3 f0;
};

// Normal-incidence reflectance in linear sRGB, standard published values.
constexpr NamedF0 METAL_F0[] = {
    {"Au", {1.000f, 0.766f, 0.336f}}, {"Ag", {0.972f, 0.960f, 0.915f}},
    {"Al", {0.913f, 0.921f, 0.925f}}, {"Cu", {0.955f, 0.637f, 0.538f}},
    {"Cr", {0.550f, 0.556f, 0.554f}}, {"Ni", {0.660f, 0.609f, 0.526f}},
    {"Ti", {0.542f, 0.497f, 0.449f}}, {"W", {0.510f, 0.500f, 0.480f}},
};

} // namespace

float3 blackbody_rgb(const float temperature_kelvin) {
    constexpr float H = 6.62607015e-34f;
    constexpr float C = 2.99792458e8f;
    constexpr float KB = 1.380649e-23f;
    const auto planck = [&](const float lambda_nm) {
        const float lambda = lambda_nm * 1e-9f;
        const float l5 = lambda * lambda * lambda * lambda * lambda;
        return (2.0f * H * C * C) /
               (l5 * (std::exp((H * C) / (lambda * KB * temperature_kelvin)) - 1.0f));
    };
    // pbrt normalizes to a peak value of 1 over the visible range.
    float peak = 0;
    for (float lambda = LAMBDA_MIN; lambda <= LAMBDA_MAX; lambda += LAMBDA_STEP) {
        peak = std::max(peak, planck(lambda));
    }
    if (peak <= 0) {
        return float3(0);
    }
    return integrate_to_rgb([&](const float lambda) { return planck(lambda) / peak; });
}

std::optional<float3> spectrum_param_rgb(const ParsedParameter& param,
                                         const std::filesystem::path& base_dir) {
    if (param.type == "blackbody") {
        if (param.floats.empty()) {
            return std::nullopt;
        }
        return blackbody_rgb(param.floats[0]);
    }
    if (!param.strings.empty()) {
        const std::string& name = param.strings[0];
        if (name == "stdillum-D65" || name == "stdillum-D50" || name == "stdillum-E") {
            return float3(1);
        }
        if (name == "stdillum-A") {
            return blackbody_rgb(2856.0f);
        }
        const std::filesystem::path file = std::filesystem::path(name).is_absolute()
                                               ? std::filesystem::path(name)
                                               : base_dir / name;
        if (const auto samples = load_spd(file)) {
            return piecewise_to_rgb(*samples);
        }
        return std::nullopt;
    }
    if (param.floats.size() >= 4) {
        std::vector<float2> samples;
        samples.reserve(param.floats.size() / 2);
        for (size_t i = 0; i + 1 < param.floats.size(); i += 2) {
            samples.emplace_back(param.floats[i], param.floats[i + 1]);
        }
        return piecewise_to_rgb(samples);
    }
    if (param.floats.size() == 1) {
        return float3(param.floats[0]);
    }
    return std::nullopt;
}

std::optional<float3> named_metal_f0(const std::string_view spectrum_name) {
    std::string_view name = spectrum_name;
    if (!name.starts_with("metal-")) {
        return std::nullopt;
    }
    name.remove_prefix(6);
    if (name.ends_with("-eta")) {
        name.remove_suffix(4);
    } else if (name.ends_with("-k")) {
        name.remove_suffix(2);
    }
    for (const NamedF0& metal : METAL_F0) {
        if (metal.name == name) {
            return metal.f0;
        }
    }
    return std::nullopt;
}

std::optional<float> named_glass_ior(const std::string_view spectrum_name) {
    if (spectrum_name == "glass-BK7") {
        return 1.5168f;
    }
    if (spectrum_name == "glass-SF11") {
        return 1.7847f;
    }
    return std::nullopt;
}

} // namespace merian::pbrt
