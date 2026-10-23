import merian_shaders.shading.shading_function;
import merian_shaders.utils.encoding;
import merian_shaders.utils.hash;
import merian_shaders.colors.colorspaces;
import merian_shaders.scene.hit;

#include "merian-shaders/debug/path_record.slangh"

namespace merian {

// true while a debugger consumes the stream
extern static const bool merian_path_record = false;

uint path_record_pack_material(const uint material_id) {
    return min(material_id, PATH_RECORD_MATERIAL_NONE);
}

static const uint2 PATH_RECORD_SURFACE_NONE = uint2(PATH_RECORD_GEOMETRY_NONE, 0u);
static const uint2 PATH_RECORD_SURFACE_ENVIRONMENT = uint2(PATH_RECORD_GEOMETRY_ENVIRONMENT, 0u);

// the surface of a hit, the environment for a distant one
uint2 path_record_surface(const Hit hit, const bool distant) {
    return distant ? PATH_RECORD_SURFACE_ENVIRONMENT
                   : uint2(hit.get_geometry_id(), hit.primitive_id);
}

uint path_record_pack_channel(const float value) {
    if (!(value > exp2(PATH_RECORD_RGB_LOG2_MIN))) {
        return 0u;
    }
    const float stops =
        min(log2(value) - PATH_RECORD_RGB_LOG2_MIN, 1022.f * PATH_RECORD_RGB_LOG2_STEP);
    return uint(round(stops / PATH_RECORD_RGB_LOG2_STEP)) + 1u;
}

// see path_record_unpack_rgb
uint path_record_pack_rgb(const float3 rgb) {
    return path_record_pack_channel(rgb.r) | (path_record_pack_channel(rgb.g) << 10) |
           (path_record_pack_channel(rgb.b) << 20);
}

uint path_record_pack_lobe(const BSDFLobe lobe) {
    const uint lobes = uint(lobe);
    const uint glossy = uint(BSDFLobe::GlossyReflection) | uint(BSDFLobe::GlossyTransmission);
    const uint diffuse = uint(BSDFLobe::DiffuseReflection) | uint(BSDFLobe::DiffuseTransmission);
    return ((lobes & uint(BSDFLobe::Delta)) != 0u ? PATH_RECORD_LOBE_DELTA : 0u) |
           ((lobes & glossy) != 0u ? PATH_RECORD_LOBE_GLOSSY : 0u) |
           ((lobes & diffuse) != 0u ? PATH_RECORD_LOBE_DIFFUSE : 0u) |
           ((lobes & uint(BSDFLobe::Transmission)) != 0u ? PATH_RECORD_LOBE_TRANSMISSION : 0u);
}

typealias PathRecordStream = RWStructuredBuffer<uint, ScalarDataLayout>;

// Region capacities from the buffer size; must match path_record_capacities() on the host.
uint path_record_path_capacity(PathRecordStream buf) {
    uint count;
    uint stride;
    buf.GetDimensions(count, stride);
    const uint payload = count - PATH_RECORD_HEADER_UINTS;
    return payload / PATH_RECORD_REGION_RATIO / PATH_RECORD_UINTS;
}

uint path_record_vertex_region_offset(PathRecordStream buf) {
    return PATH_RECORD_HEADER_UINTS + path_record_path_capacity(buf) * PATH_RECORD_UINTS;
}

uint path_record_vertex_capacity(PathRecordStream buf) {
    uint count;
    uint stride;
    buf.GetDimensions(count, stride);
    return (count - path_record_vertex_region_offset(buf)) / PATH_RECORD_VERTEX_UINTS;
}

uint path_record_path_count(PathRecordStream buf) {
    return min(buf[PATH_RECORD_HDR_PATH_COUNT], path_record_path_capacity(buf));
}

struct PathRecordPath {
    uint2 pixel;
    uint sample_idx;
    uint vertex_offset;
    uint vertex_count;
    float luminance;
}

struct PathRecordVertex {
    float3 pos;
    float3 wo;
    float pdf;
    float3 contrib;
    float3 throughput_rgb;
    float throughput;
    uint bounce;
    uint lobe;
    uint method;
    uint material;
    uint2 surface;
    bool terminal;
    bool nee;
    bool reused;
    bool resampled;
}

PathRecordPath path_record_load_path(PathRecordStream buf, const uint index) {
    const uint base = PATH_RECORD_HEADER_UINTS + index * PATH_RECORD_UINTS;
    PathRecordPath p;
    p.pixel = path_record_unpack_pixel(buf[base + PATH_RECORD_PATH_PIXEL]);
    p.vertex_offset = buf[base + PATH_RECORD_PATH_VERTEX_OFFSET];
    const uint count_sample = buf[base + PATH_RECORD_PATH_COUNT_SAMPLE];
    p.vertex_count = count_sample & 0xFFFFu;
    p.sample_idx = count_sample >> 16;
    p.luminance = asfloat(buf[base + PATH_RECORD_PATH_LUMINANCE]);
    return p;
}

uint path_record_load_vertex_meta(PathRecordStream buf,
                                  const uint vertex_region_offset,
                                  const uint index) {
    return buf[vertex_region_offset + index * PATH_RECORD_VERTEX_UINTS + PATH_RECORD_VERTEX_META];
}

float3 path_record_load_vertex_float3(PathRecordStream buf,
                                      const uint vertex_region_offset,
                                      const uint index,
                                      const uint word) {
    const uint at = vertex_region_offset + index * PATH_RECORD_VERTEX_UINTS + word;
    return float3(asfloat(buf[at]), asfloat(buf[at + 1]), asfloat(buf[at + 2]));
}

PathRecordVertex path_record_load_vertex(PathRecordStream buf,
                                         const uint vertex_region_offset,
                                         const uint index) {
    const uint base = vertex_region_offset + index * PATH_RECORD_VERTEX_UINTS;
    PathRecordVertex v;
    const uint pos = base + PATH_RECORD_VERTEX_POS;
    const uint contrib = base + PATH_RECORD_VERTEX_CONTRIB;
    v.pos = float3(asfloat(buf[pos + 0]), asfloat(buf[pos + 1]), asfloat(buf[pos + 2]));
    v.contrib =
        float3(asfloat(buf[contrib + 0]), asfloat(buf[contrib + 1]), asfloat(buf[contrib + 2]));
    v.pdf = asfloat(buf[base + PATH_RECORD_VERTEX_PDF]);
    v.throughput_rgb = path_record_unpack_rgb(buf[base + PATH_RECORD_VERTEX_THROUGHPUT]);
    v.throughput = yuv_luminance(v.throughput_rgb);
    const uint meta = buf[base + PATH_RECORD_VERTEX_META];
    v.bounce = meta & PATH_RECORD_BOUNCE_MASK;
    v.lobe = (meta >> PATH_RECORD_LOBE_SHIFT) & PATH_RECORD_LOBE_MASK;
    v.method = (meta >> PATH_RECORD_METHOD_SHIFT) & PATH_RECORD_METHOD_MASK;
    v.material = (meta >> PATH_RECORD_MATERIAL_SHIFT) & PATH_RECORD_MATERIAL_MASK;
    v.terminal = (meta & PATH_RECORD_FLAG_TERMINAL) != 0;
    v.nee = (meta & PATH_RECORD_FLAG_NEE) != 0;
    v.reused = (meta & PATH_RECORD_FLAG_REUSED) != 0;
    v.resampled = (meta & PATH_RECORD_FLAG_RESAMPLED) != 0;
    v.surface =
        uint2(buf[base + PATH_RECORD_VERTEX_SURFACE], buf[base + PATH_RECORD_VERTEX_SURFACE + 1]);
    // the terminal flag decides, not the encoding: encode_normal(+z) is 0 and would alias
    v.wo = v.terminal ? float3(0) : decode_normal(buf[base + PATH_RECORD_VERTEX_WO]);
    return v;
}

// Records a path from the camera on; its first vertex is the first surface it scatters at.
interface IPathRecorder {
    static This begin(PathRecordStream buf,
                      const uint2 pixel,
                      const uint sample_idx,
                      const uint frame,
                      const uint max_vertices);

    static This disabled();

    // Radiance gathered at the surface the next recorded vertex sits on (throughput * emission).
    [mutating]
    void emission(const float3 contrib);

    // lobe carries PATH_RECORD_LOBE_* bits, material a packed PATH_RECORD_MATERIAL_* value
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
                const uint flags = 0,
                const uint2 surface = PATH_RECORD_SURFACE_NONE);

    // A next-event connection from the vertex recorded last to a point on a light.
    [mutating]
    void connection(PathRecordStream buf,
                    const float3 from,
                    const float3 light_pos,
                    const float pdf,
                    const float3 throughput,
                    const float3 contrib,
                    const uint bounce,
                    const uint method,
                    const uint material,
                    const uint flags = 0,
                    const uint2 surface = PATH_RECORD_SURFACE_NONE);

    [mutating]
    void terminal(PathRecordStream buf,
                  const float3 end_pos,
                  const uint2 surface = PATH_RECORD_SURFACE_NONE);

    // Writes the path record; every begun path must reach it.
    [mutating]
    void finish(PathRecordStream buf);

    [mutating]
    void end(PathRecordStream buf,
             const float3 end_pos,
             const uint2 surface = PATH_RECORD_SURFACE_NONE);
}

struct NullPathRecorder : IPathRecorder {
    static NullPathRecorder begin(PathRecordStream buf,
                                  const uint2 pixel,
                                  const uint sample_idx,
                                  const uint frame,
                                  const uint max_vertices) {
        return {};
    }
    static NullPathRecorder disabled() {
        return {};
    }
    [mutating]
    void emission(const float3 contrib) {}
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
                const uint flags = 0,
                const uint2 surface = PATH_RECORD_SURFACE_NONE) {}
    [mutating]
    void connection(PathRecordStream buf,
                    const float3 from,
                    const float3 light_pos,
                    const float pdf,
                    const float3 throughput,
                    const float3 contrib,
                    const uint bounce,
                    const uint method,
                    const uint material,
                    const uint flags = 0,
                    const uint2 surface = PATH_RECORD_SURFACE_NONE) {}
    [mutating]
    void terminal(PathRecordStream buf,
                  const float3 end_pos,
                  const uint2 surface = PATH_RECORD_SURFACE_NONE) {}
    [mutating]
    void finish(PathRecordStream buf) {}
    [mutating]
    void end(PathRecordStream buf,
             const float3 end_pos,
             const uint2 surface = PATH_RECORD_SURFACE_NONE) {}
}

struct LivePathRecorder : IPathRecorder {
    uint vertex_region_offset;
    uint vertex_base;
    uint max_vertices;
    uint path_slot;
    uint written;
    uint pixel_packed;
    uint sample_idx;
    float3 pending_contrib;
    float path_lum;
    bool active;

    static LivePathRecorder disabled() {
        LivePathRecorder r;
        r.active = false;
        return r;
    }

    static LivePathRecorder begin(PathRecordStream buf,
                                  const uint2 pixel,
                                  const uint sample_idx,
                                  const uint frame,
                                  const uint max_vertices) {
        LivePathRecorder r;
        r.active = false;
        if (buf[PATH_RECORD_HDR_ENABLE] == 0) {
            return r;
        }

        const uint filter = buf[PATH_RECORD_HDR_PIXEL_FILTER];
        const uint pixel_packed = path_record_pack_pixel(pixel);
        if (filter != PATH_RECORD_ALL_PIXELS && filter != pixel_packed) {
            return r;
        }

        if (filter == PATH_RECORD_ALL_PIXELS && buf[PATH_RECORD_HDR_KEEP_PIXEL] != pixel_packed) {
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
        r.path_lum = 0.f;
        r.active = true;
        return r;
    }

    [mutating]
    void emission(const float3 contrib) {
        if (!active) {
            return;
        }
        pending_contrib += contrib;
        path_lum += yuv_luminance(contrib);
    }

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
                const uint flags = 0,
                const uint2 surface = PATH_RECORD_SURFACE_NONE) {
        if (!active || written >= max_vertices) {
            return;
        }
        store_vertex(buf, pos, encode_normal(wo), pdf, throughput, bounce, lobe, method, material,
                     flags, pending_contrib, surface);
        pending_contrib = float3(0.f);
    }

    [mutating]
    void connection(PathRecordStream buf,
                    const float3 from,
                    const float3 light_pos,
                    const float pdf,
                    const float3 throughput,
                    const float3 contrib,
                    const uint bounce,
                    const uint method,
                    const uint material,
                    const uint flags = 0,
                    const uint2 surface = PATH_RECORD_SURFACE_NONE) {
        if (!active || written >= max_vertices) {
            return;
        }
        path_lum += yuv_luminance(contrib);
        const float3 to_light = light_pos - from;
        store_vertex(buf, light_pos,
                     pdf > 0.f && any(to_light != 0.f) ? encode_normal(normalize(to_light)) : 0,
                     pdf, throughput, bounce, 0, method, material,
                     PATH_RECORD_FLAG_NEE | flags, contrib, surface);
    }

    [mutating]
    void terminal(PathRecordStream buf,
                  const float3 end_pos,
                  const uint2 surface = PATH_RECORD_SURFACE_NONE) {
        if (!active || written >= max_vertices) {
            return;
        }
        store_vertex(buf, end_pos, 0, 0.f, float3(0.f), written, 0, PATH_RECORD_METHOD_UNKNOWN,
                     PATH_RECORD_MATERIAL_NONE, PATH_RECORD_FLAG_TERMINAL, pending_contrib,
                     surface);
        pending_contrib = float3(0.f);
    }

    [mutating]
    void end(PathRecordStream buf,
             const float3 end_pos,
             const uint2 surface = PATH_RECORD_SURFACE_NONE) {
        terminal(buf, end_pos, surface);
        finish(buf);
    }

    [mutating]
    void finish(PathRecordStream buf) {
        if (!active) {
            return;
        }
        const uint base = PATH_RECORD_HEADER_UINTS + path_slot * PATH_RECORD_UINTS;
        buf[base + PATH_RECORD_PATH_PIXEL] = pixel_packed;
        buf[base + PATH_RECORD_PATH_VERTEX_OFFSET] = vertex_base;
        buf[base + PATH_RECORD_PATH_COUNT_SAMPLE] = (written & 0xFFFFu) | (sample_idx << 16);
        buf[base + PATH_RECORD_PATH_LUMINANCE] = asuint(path_lum);
        active = false;
    }

    [mutating]
    void store_vertex(PathRecordStream buf,
                      const float3 pos,
                      const uint enc_wo,
                      const float pdf,
                      const float3 throughput,
                      const uint bounce,
                      const uint lobe,
                      const uint method,
                      const uint material,
                      const uint flags,
                      const float3 contrib,
                      const uint2 surface) {
        const uint base = vertex_region_offset + (vertex_base + written) * PATH_RECORD_VERTEX_UINTS;
        buf[base + PATH_RECORD_VERTEX_POS + 0] = asuint(pos.x);
        buf[base + PATH_RECORD_VERTEX_POS + 1] = asuint(pos.y);
        buf[base + PATH_RECORD_VERTEX_POS + 2] = asuint(pos.z);
        buf[base + PATH_RECORD_VERTEX_WO] = enc_wo;
        buf[base + PATH_RECORD_VERTEX_CONTRIB + 0] = asuint(contrib.r);
        buf[base + PATH_RECORD_VERTEX_CONTRIB + 1] = asuint(contrib.g);
        buf[base + PATH_RECORD_VERTEX_CONTRIB + 2] = asuint(contrib.b);
        buf[base + PATH_RECORD_VERTEX_PDF] = asuint(pdf);
        buf[base + PATH_RECORD_VERTEX_THROUGHPUT] = path_record_pack_rgb(throughput);
        buf[base + PATH_RECORD_VERTEX_META] =
            (bounce & PATH_RECORD_BOUNCE_MASK) |
            ((lobe & PATH_RECORD_LOBE_MASK) << PATH_RECORD_LOBE_SHIFT) |
            (method << PATH_RECORD_METHOD_SHIFT) | (material << PATH_RECORD_MATERIAL_SHIFT) | flags;
        buf[base + PATH_RECORD_VERTEX_SURFACE + 0] = surface.x;
        buf[base + PATH_RECORD_VERTEX_SURFACE + 1] = surface.y;
        written++;
    }
}

extern struct PathRecorder : IPathRecorder = NullPathRecorder;

} // namespace merian
