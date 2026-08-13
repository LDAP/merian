#pragma once

#include "merian-graph/connectors/buffer/vk_buffer_in.hpp"
#include "merian-graph/connectors/image/vk_image_in_sampled.hpp"
#include "merian-graph/connectors/image/vk_image_out_managed.hpp"
#include "merian-graph/connectors/ptr_in.hpp"
#include "merian-graph/connectors/shader_object_in.hpp"
#include "merian-graph/graph/node.hpp"
#include "merian-graph/nodes/compute_node/compute_kernel.hpp"
#include "merian-graph/nodes/path_debug/layout.slangh"
#include "merian-graph/objects/gbuffer_object.hpp"
#include "merian-shaders/scene/scene.hpp"
#include "merian/utils/stopwatch.hpp"
#include "merian/vk/imgui/imgui_context.hpp"
#include "merian/vk/imgui/imgui_merian_backend.hpp"
#include "merian/vk/imgui/imgui_renderer.hpp"
#include "merian/vk/window/viewport_picker.hpp"
#include "merian/vk/window/window.hpp"

#include <array>
#include <deque>
#include <mutex>
#include <optional>

namespace merian {

// Visualizes the path record stream of a renderer: overlays the most contributing paths (or all
// paths of a selected pixel) on the rendered image, accumulates directional sampling maps with a
// BSDF-pdf reference, splats a world-space heat grid, and owns the record stream control words
// (freeze, pixel filter, subsampling) — connecting it is what arms capture in the renderer.
class PathDebugNode : public Node {
  public:
    PathDebugNode() = default;

    ~PathDebugNode() override = default;

    DeviceSupportInfo query_device_support(const DeviceSupportQueryInfo& query_info) override;

    void initialize(const ContextHandle& context,
                    const ResourceAllocatorHandle& allocator) override;

    std::vector<InputConnectorDescriptor> describe_inputs() override;

    std::vector<OutputConnectorDescriptor> describe_outputs(const NodeIOLayout& io_layout) override;

    NodeStatusFlags on_connected(const NodeIOLayout& io_layout,
                                 const NodeIO& io,
                                 const NodeConnectionInfo& info,
                                 Submission& submission) override;

    [[nodiscard]] NodeStatusFlags
    process(const NodeIO& io, const NodeProcessInfo& info, Submission& submission) override;

    NodeStatusFlags properties(Properties& config) override;

  private:
    static constexpr uint32_t MAX_DRAW = PATH_DEBUG_MAX_DRAW;
    static constexpr uint32_t MAX_SEGMENTS = PATH_DEBUG_MAX_SEGMENTS;
    static constexpr uint32_t STATE_UINTS = PATH_DEBUG_STATE_UINTS;
    static constexpr uint32_t STATS_UINTS = PATH_DEBUG_STATS_UINTS;
    static constexpr uint32_t THETA_BANDS = PATH_DEBUG_THETA_BANDS;
    static constexpr uint32_t TOP_COUNT = PATH_DEBUG_TOP_COUNT;
    static constexpr uint32_t VIEW_COUNT = PATH_DEBUG_VIEW_COUNT;
    static constexpr uint32_t MAX_MAP_RES = 1024; // maps_buffer is sized for the largest setting

    // one recorded path, dumped by finalize at PATH_DEBUG_STATE_FOCUS, for the inspector
    struct FocusPath {
        uint32_t valid;
        uint32_t pixel;
        uint32_t sample_and_count;
        uint32_t luminance;
        std::array<uint32_t, MAX_SEGMENTS * PATH_RECORD_VERTEX_UINTS> vertices;
    };

    struct TopPath {
        uint32_t index;
        uint32_t pixel;
        uint32_t luminance;
        uint32_t sample_and_scatter;
        uint32_t classes; // the first 8 scatter events, 4 bits each
    };

    struct Pick {
        uint32_t pixel_plus_one; // packed pixel + 1; 0 while unresolved
        uint32_t x;              // float bits
        uint32_t y;
        uint32_t z;
    };

    struct SelectStats {
        uint32_t draw_count;
        uint32_t threshold_bits;
        uint32_t selected;
        uint32_t heat_max_bits;
        uint32_t paths;
        uint32_t vertices;
        uint32_t records_frame;
        uint32_t keep_prob; // float bits, see PATH_DEBUG_STATE_STAT_KEEP
    };

    struct Probe {
        uint32_t flags; // 1 = pixel values, 2 = panel bin values
        uint32_t a;     // pixel: mean       panel: density
        uint32_t b;     // pixel: rel error  panel: pdf
        uint32_t c;     // pixel: variance   panel: mean contribution
        uint32_t count;
        uint32_t selected_mean;      // float bits, always the selected pixel
        uint32_t selected_rel_error; // float bits
        uint32_t selected_count;
    };

    struct Readback {
        SelectStats select;
        Probe probe;
        FocusPath focus;
        std::array<TopPath, TOP_COUNT> top;
        Pick pick;
        std::array<uint32_t, STATS_UINTS> stats;
    };

    void update_gbuffer_kernels();
    // Compiles a path expression into packed filter tokens; false if it does not parse.
    static bool
    compile_filter(const std::string& text, std::array<uint32_t, 4>& packed, uint32_t& token_count);
    // compile_filter for slot A, applied to the params words.
    bool update_filter();
    void load_reference(Submission& submission);
    void draw_inspector();
    void ensure_bsdf_pipeline(const SceneHandle& scene);
    void bind_globals(const NodeIO& io);
    void handle_pick(const NodeIO& io);
    void draw_window();
    void draw_overlay();

    ContextHandle context;
    ResourceAllocatorHandle allocator;
    ShaderCompileContextHandle compile_context;

    PtrInHandle<Scene> con_scene = PtrIn<Scene>::create();
    VkBufferInHandle con_records = VkBufferIn::create();
    VkSampledImageInHandle con_src = VkSampledImageIn::create();
    ShaderObjectInHandle<GBufferObject> con_gbuffer = ShaderObjectIn<GBufferObject>::create();
    PtrInHandle<InputController> con_controller = PtrIn<InputController>::create();
    PtrInHandle<Window> con_window = PtrIn<Window>::create();
    ManagedVkImageOutHandle con_out;
    ManagedVkImageOutHandle con_filtered;
    ManagedVkImageOutHandle con_filtered_b;

    Versioned<SpecializationInfo> spec_info;
    std::optional<ComputeKernel> select_kernel;
    std::optional<ComputeKernel> threshold_kernel;
    std::optional<ComputeKernel> collect_kernel;
    std::optional<ComputeKernel> draw_kernel;
    std::optional<ComputeKernel> map_reduce_kernel;
    std::optional<ComputeKernel> map_view_kernel;
    std::optional<ComputeKernel> heat_decay_kernel;
    std::optional<ComputeKernel> heat_splat_kernel;
    std::optional<ComputeKernel> filter_render_kernel;
    std::optional<ComputeKernel> finalize_kernel;
    // these read the gbuffer behind the merian_path_debug_gbuffer constant and are recreated on
    // connectivity changes (the constant must be right at the first build)
    std::optional<ComputeKernel> stats_kernel;
    std::optional<ComputeKernel> compose_kernel;
    std::optional<ComputeKernel> sphere_kernel;

    // BSDF pdf reference: manual pipeline, the Scene parameter block needs entry-point binding
    SlangCompositionHandle bsdf_composition;
    Versioned<SlangProgram> bsdf_program;
    Versioned<SlangProgramEntryPoint> bsdf_entry_point;
    Versioned<Pipeline> bsdf_pipeline;
    Versioned<ShaderObject> bsdf_globals;

    vk::Extent3D extent{};
    BufferHandle state_buffer;
    BufferHandle overlay_buffer;
    BufferHandle stats_buffer;
    BufferHandle maps_buffer;
    BufferHandle grid_buffer;
    BufferHandle moments_buffer;
    BufferHandle filtered_buffer;
    BufferHandle matches_buffer;
    TextureHandle reference_texture;
    std::vector<BufferHandle> readback_buffers;

    ImGuiContextHandle imgui_ctx;
    ImGuiRendererHandle imgui_renderer;
    ImGuiMerianBackendHandle imgui_backend;
    Stopwatch frametime;

    std::shared_ptr<ViewportPicker> picker = std::make_shared<ViewportPicker>();
    std::weak_ptr<InputController> registered_controller;

    std::mutex stats_mutex;
    Readback latest_readback{};
    bool readback_valid = false;

    // The shared struct cannot carry C++ member initializers, so the fields the properties
    // bind directly (exposures, radii, opacities) get their defaults here; everything else is
    // derived per frame in process().
    static PathDebugParams default_params() {
        PathDebugParams defaults{};
        defaults.k = 16;
        defaults.firefly_fraction = 1e-4f;
        defaults.overlay_alpha = 0.85f;
        defaults.color_scale = 1.f;
        defaults.keep_prob = 1.f;
        defaults.pixel_filter = PATH_RECORD_ALL_PIXELS;
        defaults.max_draw = PATH_DEBUG_MAX_DRAW;
        defaults.map_res = 32;
        defaults.map_panel_size = 192;
        defaults.map_exposure = 1.f;
        defaults.sphere_radius = 0.05f;
        defaults.heat_bounce_max = 30;
        defaults.heat_decay = 1.f;
        defaults.heat_cell_size = 0.05f;
        defaults.heat_exposure = 1.f;
        defaults.heat_opacity = 0.6f;
        defaults.grid_slots = 1u << 20;
        defaults.accumulate_moments = 1;
        defaults.error_scale = 10.f;
        defaults.cursor_x = PATH_DEBUG_NO_CURSOR;
        defaults.cursor_y = PATH_DEBUG_NO_CURSOR;
        defaults.filter_len_max = 0xFFFFFFFFu;
        defaults.filter_material = PATH_RECORD_MATERIAL_NONE;
        defaults.ab_split = 0.5f;
        defaults.ab_scale = 10.f;
        defaults.focus_path = PATH_DEBUG_NO_FOCUS;
        return defaults;
    }
    PathDebugParams params = default_params();
    bool freeze = false;
    bool auto_keep_prob = true;
    float keep_prob = 1.f;
    int32_t selection_mode = 0;
    int32_t color_mode = 0;
    int32_t top_k = 16;
    int2 selected_pixel = int2(0, 0);

    int32_t map_res_log2 = 5;
    int32_t map_transform = 0;
    int32_t map_frame = 0;
    int32_t map_scope = 0;
    int32_t map_reference = 0;
    int32_t map_bounce = 0;
    bool map_accumulate = true;
    bool maps_dirty = true;
    uint32_t map_frames = 0;
    std::array<bool, VIEW_COUNT> view_enabled{true, false, false, false, true, false, true};
    bool panels_enabled = true;
    int32_t sphere_view = 0;

    // path class filter
    std::string filter_pattern;
    uint32_t filter_token_count = 0;
    bool filter_valid = true;
    int2 filter_length = int2(0, 32);
    bool filter_enabled = false;
    uint32_t filter_method_mask = 0;
    int32_t filter_method_mode = 0;
    int32_t filter_material = -1;
    std::string filter_pattern_b;
    bool filter_valid_b = true;
    uint32_t filter_token_count_b = 0;
    std::array<uint32_t, 4> filter_words_b{};
    int32_t moments_scope = 0;
    float3 query_pos = float3(0.f);
    float query_radius_scale = 1.f;
    int32_t query_anchor = 0;
    bool pick_pending = false;
    int2 pick_pixel = int2(0);

    // A/B
    int32_t render_mode = 0;
    std::string reference_path;
    bool reference_loaded = false;
    bool reference_dirty = false;
    bool ab_flip = false;
    float ab_flip_hz = 2.f;
    float ab_split = 0.5f;
    float ab_scale = 10.f;
    float filtered_exposure = 1.f;

    // inspector
    int32_t focus_path = -1;

    int32_t error_view = 0;
    bool moments_dirty = true;
    uint32_t moments_frames = 0;
    std::deque<float> mean_history;
    std::deque<float> rel_error_history;

    bool export_next = false;
    bool export_metadata = true;
    int32_t export_format = 0;
    std::string export_path = "path_debug";

    bool gbuffer_connected = false;
    int32_t heat_mode = 0;
    int32_t heat_scope = 0;
    int2 heat_bounce_range = int2(0, 30);
    float heat_alpha = 0.05f; // 0 = keep all
    int32_t grid_slots_log2 = 20;
    bool heat_dirty = true;
    uint32_t heat_frames = 0;

    std::string imgui_event_pattern = "//ui";
};

} // namespace merian
