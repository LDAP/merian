namespace merian {

struct RayDifferential {
    // current differential of origin
    float3 dOdx;
    float3 dOdy;
    // current differential of direction
    float3 dDdx;
    float3 dDdy;

    // Pinhole camera init from neighbour pixel ray directions (finite difference of Igehy Eq. 8).
    static RayDifferential from_pinhole_camera(const float3 ray_dir,
                                               const float3 ray_dir_x,
                                               const float3 ray_dir_y) {
        RayDifferential rd;
        rd.dOdx = float3(0);
        rd.dOdy = float3(0);
        rd.dDdx = ray_dir_x - ray_dir;
        rd.dDdy = ray_dir_y - ray_dir;
        return rd;
    }

    // Igehy Eq. 10/12: propagate to a hit at distance t along D, surface normal N.
    [mutating]
    void propagate(const float3 D, const float t, const float3 N) {
        dOdx += t * dDdx;
        dOdy += t * dDdy;

        const float rcpDN = 1.0f / dot(D, N);
        dOdx += D * -dot(dOdx, N) * rcpDN;
        dOdy += D * -dot(dOdy, N) * rcpDN;
    }

    // Screen-space UV gradients from triangle context.
    // Igehy "Normal-Interpolated Triangles" (SIGGRAPH '99) — barycentric differentials + UV interpolation.
    void compute_uv_gradients(const float3 edge01,
                              const float3 edge02,
                              const float3 face_normal,
                              const float2 uv0,
                              const float2 uv1,
                              const float2 uv2,
                              out float2 ddx,
                              out float2 ddy) {
        const float3 Nu = cross(edge02, face_normal);
        const float3 Nv = cross(edge01, face_normal);
        const float3 Lu = Nu / dot(Nu, edge01);
        const float3 Lv = Nv / dot(Nv, edge02);

        const float2 dBarydx = float2(dot(Lu, dOdx), dot(Lv, dOdx));
        const float2 dBarydy = float2(dot(Lu, dOdy), dot(Lv, dOdy));

        const float2 delta1 = uv1 - uv0;
        const float2 delta2 = uv2 - uv0;
        ddx = dBarydx.x * delta1 + dBarydx.y * delta2;
        ddy = dBarydy.x * delta1 + dBarydy.y * delta2;
    }
};

}
