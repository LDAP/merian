import merian_shaders.utils.encoding;
import merian_shaders.utils.hash;
import merian_shaders.colors.colorspaces;

namespace merian {

// Exported true by a renderer when a debugger consumes its record stream; everything in this
// module folds away when false.
extern static const bool merian_path_record = false;

// Record stream, one uint buffer with three regions:
// [0, HEADER_UINTS)  header: counters (renderer) + control words (consumer)
// [paths]            PATH_RECORD_UINTS per path
// [vertices]         PATH_RECORD_VERTEX_UINTS per vertex, one contiguous block per path
// Region capacities derive from the buffer size alone (see path_record.hpp) so producer and
// consumer agree without further coordination.
static const uint PATH_RECORD_HEADER_UINTS = 16;
// paths get 1 of PATH_RECORD_REGION_RATIO payload parts, vertices the rest
static const uint PATH_RECORD_REGION_RATIO = 17;
static const uint PATH_RECORD_UINTS = 4;
static const uint PATH_RECORD_VERTEX_UINTS = 8;

// May exceed the capacity: excess reservations are dropped, the overshoot measures demand.
static const uint PATH_RECORD_HDR_PATH_COUNT = 0;
static const uint PATH_RECORD_HDR_VERTEX_COUNT = 1;
// Control words, written by the consumer for the next iteration.
static const uint PATH_RECORD_HDR_ENABLE = 2;
static const uint PATH_RECORD_HDR_PIXEL_FILTER = 3; // packed pixel or PATH_RECORD_ALL_PIXELS
static const uint PATH_RECORD_HDR_KEEP_PROB = 4;    // float bits, path subsampling probability
// Written by the producer: the frame the records belong to, i.e. the rng seed to replay them.
static const uint PATH_RECORD_HDR_FRAME = 5;

static const uint PATH_RECORD_ALL_PIXELS = 0xFFFFFFFFu;

// vertex meta word: bounce | lobe << 8 | flags | method << 16 | material << 20. The lobe
// carries BSDFLobe bits, zero where the producer cannot classify the scatter event.
static const uint PATH_RECORD_LOBE_SHIFT = 8;
static const uint PATH_RECORD_LOBE_MASK = 0xFu;
static const uint PATH_RECORD_FLAG_TERMINAL = 1u << 12;
// the outgoing direction connects to a light instead of continuing a scatter chain
static const uint PATH_RECORD_FLAG_NEE = 1u << 13;

static const uint PATH_RECORD_MATERIAL_SHIFT = 20;
static const uint PATH_RECORD_MATERIAL_MASK = 0xFFFu;
// not recorded, or the scene has more materials than the field can hold
static const uint PATH_RECORD_MATERIAL_NONE = 0xFFFu;

uint path_record_pack_material(const uint material_id) {
    return min(material_id, PATH_RECORD_MATERIAL_NONE);
}

// How the outgoing direction of a vertex was drawn. Producers pass one of these; the consumer
// reports anything it does not know as unknown, so a renderer can add methods without lockstep.
static const uint PATH_RECORD_METHOD_SHIFT = 16;
static const uint PATH_RECORD_METHOD_MASK = 0xFu;
static const uint PATH_RECORD_METHOD_UNKNOWN = 0;
static const uint PATH_RECORD_METHOD_UNIFORM = 1;
static const uint PATH_RECORD_METHOD_COSINE = 2;
static const uint PATH_RECORD_METHOD_BSDF = 3;
static const uint PATH_RECORD_METHOD_MCPG = 4;
static const uint PATH_RECORD_METHOD_RESTIR = 5;
static const uint PATH_RECORD_METHOD_SSMM = 6;

// shared-exponent RGB for radiometric quantities: [0, ~65k], ~2^-24 smallest step
uint path_record_pack_rgb9e5(const float3 c) {
    const float maxc = max(max(c.r, c.g), max(c.b, 1e-8f));
    const int exp_shared = clamp(int(floor(log2(maxc))) + 1 + 15, 0, 31);
    const float scale = exp2(float(exp_shared - 15 - 9));
    const uint3 m = min(uint3(max(c, 0.f) / scale + 0.5f), 511u);
    return m.r | (m.g << 9) | (m.b << 18) | (uint(exp_shared) << 27);
}

float3 path_record_unpack_rgb9e5(const uint packed) {
    const float scale = exp2(float(int(packed >> 27) - 15 - 9));
    return float3(packed & 511, (packed >> 9) & 511, (packed >> 18) & 511) * scale;
}

typealias PathRecordStream = RWStructuredBuffer<uint, ScalarDataLayout>;

uint path_record_pack_pixel(const uint2 pixel) {
    return (pixel.y << 16) | (pixel.x & 0xFFFFu);
}

uint2 path_record_unpack_pixel(const uint packed) {
    return uint2(packed & 0xFFFFu, packed >> 16);
}

// Region capacities from the buffer size; must match path_record_capacities() on the host.
uint path_record_path_capacity(PathRecordStream buf) {
    uint count;
    uint stride;
    buf.GetDimensions(count, stride);
    const uint payload = count - PATH_RECORD_HEADER_UINTS;
    return payload / PATH_RECORD_REGION_RATIO / PATH_RECORD_UINTS;
}

uint path_record_vertex_capacity(PathRecordStream buf) {
    uint count;
    uint stride;
    buf.GetDimensions(count, stride);
    const uint payload = count - PATH_RECORD_HEADER_UINTS;
    const uint path_uints =
        payload / PATH_RECORD_REGION_RATIO / PATH_RECORD_UINTS * PATH_RECORD_UINTS;
    return (payload - path_uints) / PATH_RECORD_VERTEX_UINTS;
}

uint path_record_vertex_region_offset(PathRecordStream buf) {
    return PATH_RECORD_HEADER_UINTS + path_record_path_capacity(buf) * PATH_RECORD_UINTS;
}

struct PathRecordPath {
    uint2 pixel;
    uint sample_idx;
    uint vertex_offset;
    uint vertex_count;
    float luminance; // total path contribution
}

struct PathRecordVertex {
    float3 pos;
    float3 wo;         // sampled outgoing direction, world space; zero for terminal vertices
    float pdf;         // solid-angle pdf of wo
    float3 contrib;    // throughput * emission gathered at this vertex
    float throughput;  // luminance of the path throughput along wo
    uint bounce;
    uint lobe;
    uint method;   // PATH_RECORD_METHOD_*
    uint material; // material id, PATH_RECORD_MATERIAL_NONE where not recorded
    bool terminal;
    bool nee;
}

PathRecordPath path_record_load_path(PathRecordStream buf, const uint index) {
    const uint base = PATH_RECORD_HEADER_UINTS + index * PATH_RECORD_UINTS;
    PathRecordPath p;
    const uint pixel = buf[base + 0];
    p.pixel = path_record_unpack_pixel(pixel);
    p.vertex_offset = buf[base + 1];
    const uint cs = buf[base + 2];
    p.vertex_count = cs & 0xFFFFu;
    p.sample_idx = cs >> 16;
    p.luminance = asfloat(buf[base + 3]);
    return p;
}

// Just the meta word (bounce | lobe | flags); for scans that only classify vertices.
uint path_record_load_vertex_meta(PathRecordStream buf,
                                  const uint vertex_region_offset,
                                  const uint index) {
    return buf[vertex_region_offset + index * PATH_RECORD_VERTEX_UINTS + 7];
}

PathRecordVertex path_record_load_vertex(PathRecordStream buf,
                                         const uint vertex_region_offset,
                                         const uint index) {
    const uint base = vertex_region_offset + index * PATH_RECORD_VERTEX_UINTS;
    PathRecordVertex v;
    v.pos = float3(asfloat(buf[base + 0]), asfloat(buf[base + 1]), asfloat(buf[base + 2]));
    const uint enc_wo = buf[base + 3];
    v.wo = enc_wo == 0 ? float3(0) : decode_normal(enc_wo);
    v.contrib = path_record_unpack_rgb9e5(buf[base + 4]);
    v.pdf = asfloat(buf[base + 5]);
    v.throughput = asfloat(buf[base + 6]);
    const uint meta = buf[base + 7];
    v.bounce = meta & 0xFFu;
    v.lobe = (meta >> PATH_RECORD_LOBE_SHIFT) & PATH_RECORD_LOBE_MASK;
    v.method = (meta >> PATH_RECORD_METHOD_SHIFT) & PATH_RECORD_METHOD_MASK;
    v.material = (meta >> PATH_RECORD_MATERIAL_SHIFT) & PATH_RECORD_MATERIAL_MASK;
    v.terminal = (meta & PATH_RECORD_FLAG_TERMINAL) != 0;
    v.nee = (meta & PATH_RECORD_FLAG_NEE) != 0;
    return v;
}

// Per-path recorder for the sampling loop of a path tracer. All methods fold away unless
// merian_path_record is exported true. One vertex block of max_vertices is reserved up front so
// records stay contiguous per path without per-vertex atomics or register-array staging.
// The stream is passed per call instead of stored: resource members trip Slang's function-call
// specialization.
struct PathRecorder {
    uint vertex_region_offset;
    uint vertex_base;
    uint max_vertices;
    uint path_slot;
    uint written;
    uint pixel_packed;
    uint sample_idx;
    float3 pending_contrib;
    // prefix weight for renderers that factor a deterministic chain out of the loop (e.g. PSR)
    float3 scale;
    float path_lum;
    bool active;

    static PathRecorder begin(PathRecordStream buf,
                              const uint2 pixel,
                              const uint sample_idx,
                              const uint frame,
                              const uint max_vertices) {
        PathRecorder r;
        r.active = false;
        if (!merian_path_record) {
            return r;
        }

        if (buf[PATH_RECORD_HDR_ENABLE] == 0) {
            return r;
        }

        const uint filter = buf[PATH_RECORD_HDR_PIXEL_FILTER];
        const uint pixel_packed = path_record_pack_pixel(pixel);
        if (filter != PATH_RECORD_ALL_PIXELS && filter != pixel_packed) {
            return r;
        }

        if (filter == PATH_RECORD_ALL_PIXELS) {
            const float keep_prob = asfloat(buf[PATH_RECORD_HDR_KEEP_PROB]);
            const uint h = pcg(pcg(pixel_packed) ^ pcg(frame * 0x9E3779B9u + sample_idx));
            if (float(h) * (1.f / 4294967296.f) >= keep_prob) {
                return r;
            }
        }

        const uint path_capacity = path_record_path_capacity(buf);
        const uint vertex_capacity = path_record_vertex_capacity(buf);

        uint vertex_base;
        InterlockedAdd(buf[PATH_RECORD_HDR_VERTEX_COUNT], max_vertices, vertex_base);
        // the counter keeps counting past the capacity to measure demand, so this must not add
        if (max_vertices > vertex_capacity || vertex_base > vertex_capacity - max_vertices) {
            return r;
        }
        uint path_slot;
        InterlockedAdd(buf[PATH_RECORD_HDR_PATH_COUNT], 1, path_slot);
        if (path_slot >= path_capacity) {
            return r;
        }
        if (path_slot == 0) {
            buf[PATH_RECORD_HDR_FRAME] = frame;
        }

        r.vertex_region_offset = PATH_RECORD_HEADER_UINTS + path_capacity * PATH_RECORD_UINTS;
        r.vertex_base = vertex_base;
        r.max_vertices = max_vertices;
        r.path_slot = path_slot;
        r.written = 0;
        r.pixel_packed = pixel_packed;
        r.sample_idx = sample_idx;
        r.pending_contrib = float3(0.f);
        r.scale = float3(1.f);
        r.path_lum = 0.f;
        r.active = true;
        return r;
    }

    // Radiance gathered at the surface the next recorded vertex sits on (throughput * emission).
    [mutating]
    void emission(const float3 contrib) {
        if (!merian_path_record || !active) {
            return;
        }
        pending_contrib += scale * contrib;
        path_lum += yuv_luminance(scale * contrib);
    }

    // lobe carries BSDFLobe bits (0 where the caller cannot classify), method one of
    // PATH_RECORD_METHOD_*, material a pre-packed PATH_RECORD_MATERIAL_* value.
    [mutating]
    void vertex(PathRecordStream buf,
                const float3 pos,
                const float3 wo,
                const float pdf,
                const float3 throughput,
                const uint bounce,
                const uint lobe,
                const uint method = PATH_RECORD_METHOD_UNKNOWN,
                const uint material = PATH_RECORD_MATERIAL_NONE,
                const uint flags = 0) {
        if (!merian_path_record || !active || written >= max_vertices) {
            return;
        }
        store_vertex(buf, pos, encode_normal(wo), pdf, yuv_luminance(scale * throughput), bounce,
                     lobe,
                     (method << PATH_RECORD_METHOD_SHIFT) |
                         (material << PATH_RECORD_MATERIAL_SHIFT) | flags);
    }

    // Writes the terminal vertex (with any trailing emission) and the path record.
    [mutating]
    void end(PathRecordStream buf, const float3 end_pos) {
        if (!merian_path_record || !active) {
            return;
        }
        if (written < max_vertices) {
            store_vertex(buf, end_pos, 0, 0.f, 0.f, written, 0,
                         PATH_RECORD_FLAG_TERMINAL |
                             (PATH_RECORD_MATERIAL_NONE << PATH_RECORD_MATERIAL_SHIFT));
        }

        const uint base = PATH_RECORD_HEADER_UINTS + path_slot * PATH_RECORD_UINTS;
        buf[base + 0] = pixel_packed;
        buf[base + 1] = vertex_base;
        buf[base + 2] = (written & 0xFFFFu) | (sample_idx << 16);
        buf[base + 3] = asuint(path_lum);
        active = false;
    }

    [mutating]
    void store_vertex(PathRecordStream buf,
                      const float3 pos,
                      const uint enc_wo,
                      const float pdf,
                      const float throughput_lum,
                      const uint bounce,
                      const uint lobe,
                      const uint flags) {
        const uint base = vertex_region_offset + (vertex_base + written) * PATH_RECORD_VERTEX_UINTS;
        buf[base + 0] = asuint(pos.x);
        buf[base + 1] = asuint(pos.y);
        buf[base + 2] = asuint(pos.z);
        buf[base + 3] = enc_wo;
        buf[base + 4] = path_record_pack_rgb9e5(pending_contrib);
        buf[base + 5] = asuint(pdf);
        buf[base + 6] = asuint(throughput_lum);
        buf[base + 7] = (bounce & 0xFFu) |
                        ((lobe & PATH_RECORD_LOBE_MASK) << PATH_RECORD_LOBE_SHIFT) | flags;
        pending_contrib = float3(0.f);
        written++;
    }
}

} // namespace merian
