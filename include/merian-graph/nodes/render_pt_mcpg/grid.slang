static const uint GRID_TYPE_EXPONENTIAL = 0;
static const uint GRID_TYPE_QUADRATIC = 1;

// Backing-array dimension for DistanceMCVertex; the runtime count
// (distance_mc_vertex_state_count) may be smaller.
static const uint MAX_DISTANCE_MC_VERTEX_STATE_COUNT = 10;

struct MCState {
    float3 w_tgt;
    float sum_w;
    float w_cos;

    half3 mv;
    float T;

    uint16_t N;
    uint16_t hash; // grid_idx and level
};

struct LightCacheVertex {
    uint hash; // grid_idx and level
    uint lock;
    half3 irr;
    uint16_t N;
};

struct DistanceMCState {
    float sum_w;
    uint N;
    float2 moments;
};

struct DistanceMCVertex {
    DistanceMCState states[MAX_DISTANCE_MC_VERTEX_STATE_COUNT];
};
