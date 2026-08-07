__exported import merian_shaders.shading.shading_function;

namespace merian {

interface PhaseFunction : ShadingFunction {
    /** A phase function depends only on the deflection, so it can be evaluated from the cosine
     * between the previous and the new particle direction: dot(prev_dir, new_dir) = dot(-wi, wo).
     */
    float eval(const float cos_theta);
}

}
