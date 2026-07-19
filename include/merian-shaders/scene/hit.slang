__exported import merian_shaders.scene.scene_data;
__exported import merian_shaders.shading.materials.material;
import merian_shaders.utils.encoding;

namespace merian {

static const float SCENE_RAY_TMIN = 0.f;
static const float SCENE_RAY_TMAX = 1e7f;

// Offsets pos along n to avoid self-intersection: a fixed number of ULPs so the offset scales
// with the position's magnitude, a small constant offset near the origin.
// Wächter and Binder, "A Fast and Robust Method for Avoiding Self-Intersection", RTG 2019.
// Assumes pos was computed via barycentric interpolation, not from the ray t.
float3 offset_ray_origin(const float3 pos, const float3 n) {
    static const float origin = 1.f / 16.f;
    static const float float_scale = 3.f / 65536.f;
    static const float int_scale = 3.f * 256.f;

    const int3 int_offset = int3(n * int_scale);
    const float3 int_pos = asfloat(asint(pos) + select(pos < 0.f, -int_offset, int_offset));

    return select(abs(pos) < origin, pos + n * float_scale, int_pos);
}

struct Hit {
    uint instance_id;
    uint instance_index;
    uint primitive_id;
    uint geometry_index;
    float2 barycentrics;

    __init(uint instance_id,
           uint instance_index,
           uint primitive_id,
           uint geometry_index,
           float2 barycentrics) {
        this.instance_id = instance_id;
        this.instance_index = instance_index;
        this.primitive_id = primitive_id;
        this.geometry_index = geometry_index;
        this.barycentrics = barycentrics;
    }

    // invalid hit
    __init() {
        this.instance_id = uint(-1);
        this.instance_index = uint(-1);
        this.primitive_id = uint(-1);
        this.geometry_index = uint(-1);
        this.barycentrics = float2(0);
    }

    bool is_valid() {
        return instance_id != uint(-1);
    }

    // Returns (1 - u - v, u, v) for vertex interpolation
    float3 get_barycentrics3() {
        return float3(1.0 - barycentrics.x - barycentrics.y, barycentrics.x, barycentrics.y);
    }

    GeometryID get_geometry_id() {
        return instance_id + geometry_index;
    }
}

struct TriangleHit {
    GeometryData gd;
    PackedVertexData v0, v1, v2;

    // in object space
    float3 get_position(const Hit hit) {
        const float3 b = hit.get_barycentrics3();
        return b.x * v0.position + b.y * v1.position + b.z * v2.position;
    }
    float2 get_uv(const Hit hit) {
        const float3 b = hit.get_barycentrics3();
        return b.x * float2(v0.uv) + b.y * float2(v1.uv) + b.z * float2(v2.uv);
    }
    // in object space
    float3 get_triangle_edge01() {
        return v1.position - v0.position;
    }
    // in object space
    float3 get_triangle_edge02() {
        return v2.position - v0.position;
    }
}

struct Surface {
    MaterialID material_id;
    float3 pos; // in worldspace
    float2 uv;
    float3 face_normal; // in worldspace
    OrthonormalFrame frame;

    // per-pixel screen-space derivative of linear_z along r_x, r_y
    half2
    compute_grad_z(const RayDesc ray, const float3 r_x, const float3 r_y, const float linear_z) {
        const float n_dot_oo = dot(face_normal, pos - ray.Origin);
        return half2(half(n_dot_oo / dot(face_normal, r_x) - linear_z),
                     half(n_dot_oo / dot(face_normal, r_y) - linear_z));
    }
}

struct ShadingPoint {
    float3 pos;      // in worldspace
    float3 prev_pos; // in worldspace, mcpg only — see Scene::get_prev_world_position
    uint enc_face_normal;
    MaterialSample material;

    // in worldspace
    float3 get_face_normal() {
        return decode_normal(enc_face_normal);
    }

    // Continuation ray for a scatter event; transmission is the shading-frame classification
    // (wi_local.z * wo_local.z < 0). Returns none if the geometric sides disagree, e.g. a
    // normal-mapped "reflection" that would tunnel through the surface.
    Optional<RayDesc> make_ray(const float3 wi, const float3 wo, const bool transmission) {
        const float3 ng = get_face_normal();
        const float wo_ng = dot(wo, ng);
        if (((dot(wi, ng) < 0.f) != (wo_ng < 0.f)) != transmission)
            return none;

        RayDesc ray;
        ray.Origin = offset_ray_origin(pos, wo_ng < 0.f ? -ng : ng);
        ray.Direction = wo;
        ray.TMin = SCENE_RAY_TMIN;
        ray.TMax = SCENE_RAY_TMAX;
        return ray;
    }
}

} // namespace merian
