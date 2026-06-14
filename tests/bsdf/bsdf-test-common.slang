// Shared by the generic BSDF check shader and the per-type link-time config
// modules. Each config module reads the fields relevant to its BSDF.

namespace merian_test {

struct BSDFParams {
    float3 albedo;
    float alpha;
    float3 f0;
    float ior_f0;
    float alpha_bitangent; // second GGX axis for anisotropic configs
    float iridescence;     // thin-film strength for iridescent configs
}

} // namespace merian_test
