#pragma once

#include "merian-graph/connectors/buffer/vk_buffer_in.hpp"
#include "merian-graph/connectors/image/vk_image_in_sampled.hpp"
#include "merian-graph/connectors/image/vk_image_out_managed.hpp"
#include "merian-graph/connectors/ptr_in.hpp"
#include "merian-graph/connectors/ptr_out.hpp"
#include "merian-graph/graph/node.hpp"
#include "merian-graph/nodes/compute_node/compute_kernel.hpp"
#include "merian-graph/nodes/path_debug/layout.slangh"
#include "merian-graph/nodes/path_debug/path_debug_input.hpp"
#include "merian-graph/nodes/path_debug/path_expression.hpp"
#include "merian-graph/objects/gbuffer_object.hpp"
#include "merian-shaders/scene/scene.hpp"
#include "merian/utils/stopwatch.hpp"
#include "merian/vk/imgui/imgui_context.hpp"
#include "merian/vk/imgui/imgui_merian_backend.hpp"
#include "merian/vk/imgui/imgui_renderer.hpp"
#include "merian/vk/window/swapchain_manager.hpp"
#include "merian/vk/window/window.hpp"

#include <array>
#include <cstddef>
#include <deque>
#include <mutex>
#include <optional>
#include <string>
#include <vector>

namespace merian {

// Visualizes and analyzes the path record stream of a renderer.
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
    static constexpr uint32_t OUT_UINTS = PATH_DEBUG_OUT_UINTS;
    static constexpr uint32_t STATS_UINTS = PATH_DEBUG_STATS_UINTS;
    static constexpr uint32_t THETA_BANDS = PATH_DEBUG_THETA_BANDS;
    static constexpr uint32_t TOP_COUNT = PATH_DEBUG_TOP_COUNT;
    static constexpr uint32_t MAP_CHANNEL_COUNT = PATH_DEBUG_MAP_CHANNEL_COUNT;
    static constexpr float MIN_KEEP_PROB = 1e-4f;
    static constexpr uint32_t UI_IDLE_FRAMES = 120;

    static constexpr std::array<const char*, 3> SOURCE_NAMES = {"selection A", "selection B",
                                                                "all paths"};
    static constexpr std::array<const char*, 5> RESTRICT_NAMES = {
        "nothing", "the picked pixel", "paths through the region", "paths from the region",
        "paths to the region"};
    static constexpr std::array<const char*, 3> FINITE_NAMES = {"any", "finite only",
                                                                "NaN or infinite only"};
    static constexpr std::array<const char*, 2> METHOD_MODE_NAMES = {"any vertex", "every vertex"};
    static constexpr std::array<const char*, 4> RANK_NAMES = {
        "spread over the image", "brightest", "brightest fraction", "resampled by contribution"};
    static constexpr std::array<const char*, 3> DEPTH_NAMES = {
        "off (x-ray)", "hide behind geometry", "dash behind geometry"};
    static constexpr std::array<const char*, 4> COLOR_NAMES = {"luminance", "bounces", "random",
                                                               "technique"};
    static constexpr std::array<const char*, 8> BACKDROP_NAMES = {
        "the render", "selection A",        "selection B", "reference",
        "difference", "render | reference", "A | B",       "the backdrop input"};
    static constexpr std::array<const char*, 3> TRANSFORM_NAMES = {"octahedral", "lat-long",
                                                                   "hemisphere (equal-area)"};
    static constexpr std::array<const char*, 2> FRAME_NAMES = {"world (y up)",
                                                               "local shading frame"};
    static constexpr std::array<const char*, 2> DRAWS_NAMES = {"scatter directions",
                                                               "light connections"};
    static constexpr std::array<const char*, 2> PDF_REF_NAMES = {"BSDF pdf", "recorded pdf"};
    static constexpr std::array<const char*, 2> WI_NAMES = {"camera ray", "manual"};
    static constexpr std::array<const char*, 4> HEAT_NAMES = {"off", "density", "contribution",
                                                              "throughput"};
    static constexpr std::array<const char*, 4> PIXEL_STAT_NAMES = {"off", "relative error",
                                                                    "variance", "mean"};
    static constexpr std::array<const char*, MAP_CHANNEL_COUNT> MAP_CHANNEL_NAMES = {
        "density",       "contribution", "mean color", "mean lum", "BSDF pdf",
        "density / pdf", "z-score",      "BSDF check", "BSDF eval"};

    struct IsolatedPath {
        uint32_t valid;
        uint32_t pixel;
        uint32_t sample_and_count; // sample | vertex count << 16
        uint32_t luminance;
        std::array<uint32_t, std::size_t{MAX_SEGMENTS} * PATH_RECORD_VERTEX_UINTS> vertices;
    };

    struct TopPath {
        uint32_t index;
        uint32_t pixel;
        uint32_t luminance;
        uint32_t sample_and_scatter; // sample | scatter count << 16
        uint32_t classes;            // the first 8 scatter events, 4 bits each
    };

    struct Pick {
        uint32_t seq;
        uint32_t x;
        uint32_t y;
        uint32_t z;
        uint32_t footprint;
    };

    struct FrameStats {
        uint32_t draw_count;
        uint32_t matched;
        uint32_t paths;
        uint32_t vertices;
        uint32_t records_frame;
        uint32_t keep_prob;
        uint32_t grid_cells;
        uint32_t grid_dropped;
    };

    struct Probe {
        uint32_t kind;
        uint32_t a; // pixel: mean       panel: density
        uint32_t b; // pixel: rel error  panel: pdf
        uint32_t c; // pixel: variance   panel: mean contribution
        uint32_t count;
        uint32_t picked_mean;
        uint32_t picked_rel_error;
        uint32_t picked_count;
        uint32_t picked_variance;
    };

    struct Readback {
        FrameStats frame;
        Probe probe;
        IsolatedPath isolated;
        std::array<TopPath, TOP_COUNT> top;
        Pick pick;
        std::array<uint32_t, STATS_UINTS> stats;
    };
    static_assert(offsetof(Readback, probe) == sizeof(uint32_t) * PATH_DEBUG_OUT_PROBE);
    static_assert(offsetof(Readback, isolated) == sizeof(uint32_t) * PATH_DEBUG_OUT_ISOLATE);
    static_assert(offsetof(Readback, top) == sizeof(uint32_t) * PATH_DEBUG_OUT_TOP);
    static_assert(offsetof(Readback, pick) == sizeof(uint32_t) * PATH_DEBUG_OUT_PICK);
    static_assert(offsetof(Readback, stats) == sizeof(uint32_t) * PATH_DEBUG_OUT_UINTS);

    struct ReadbackFeedback {
        uint32_t observed_paths;
        uint32_t matched_paths;
        bool run_check;
    };

    struct FrameActivity {
        bool frozen;
        bool maps_active;
        bool run_check;
    };

    struct ScenePipeline {
        SlangCompositionHandle composition;
        Versioned<SlangProgram> program;
        Versioned<SlangProgramEntryPoint> entry_point;
        Versioned<Pipeline> pipeline;
        Versioned<ShaderObject> globals;
    };

    struct PanelHit {
        uint32_t channel;
        bool on_close;
    };

    void handle_input(const NodeIO& io);
    ReadbackFeedback apply_readback_feedback(const NodeProcessInfo& info,
                                             const PathRecordCapacities& capacities);
    void adapt_keep_prob(const PathRecordCapacities& capacities);
    void resolve_pick();
    void record_clears(const NodeIO& io, const NodeProcessInfo& info, Submission& submission);
    FrameActivity update_params(const NodeIO& io,
                                const NodeProcessInfo& info,
                                bool ready,
                                const ReadbackFeedback& fed);
    void update_selection_params();
    void update_capture_params();
    void update_overlay_params(const ReadbackFeedback& fed, bool frozen);
    void update_map_params(const ReadbackFeedback& fed);
    void update_heat_params(bool frozen);
    void update_pixel_stats_params(const NodeIO& io, bool frozen, bool ready);
    void update_backdrop_params(const NodeProcessInfo& info);
    void bind_globals(const NodeIO& io);
    void bind_node_buffers(ShaderCursor cursor, const NodeIO& io) const;
    void record_analysis(const NodeIO& io,
                         const NodeProcessInfo& info,
                         Submission& submission,
                         const PathRecordCapacities& capacities,
                         const FrameActivity& activity);
    void record_scene_kernel(ScenePipeline& kernel,
                             const NodeIO& io,
                             const NodeProcessInfo& info,
                             Submission& submission,
                             const vk::Extent3D& threads);
    void record_compose(const NodeIO& io, const NodeProcessInfo& info, Submission& submission);
    void
    record_overlay_lines(const NodeIO& io, const NodeProcessInfo& info, Submission& submission);
    void record_map_panels(const NodeIO& io, const NodeProcessInfo& info, Submission& submission);
    void record_labels(const NodeIO& io, const NodeProcessInfo& info, Submission& submission);
    void record_export(const NodeIO& io, const NodeProcessInfo& info, Submission& submission);
    void record_readback(const NodeIO& io, const NodeProcessInfo& info, Submission& submission);
    void record_present(const NodeIO& io, Submission& submission);

    void build_gbuffer_kernels();
    void build_overlay_pipeline(vk::Format color_format);
    void ensure_scene_pipeline(ScenePipeline& kernel, const SceneHandle& scene, const char* module);
    void load_reference(Submission& submission);

    void invalidate_selection();
    void clear_selection();
    void set_expression(bool slot_b, const std::string& text);
    bool shared_support(bool needs_camera_wi) const;

    uint32_t shown_panel_mask() const;
    int32_t panel_top(uint32_t slot) const;
    int32_t panel_left() const;
    std::optional<PanelHit> panel_at(const int2& pixel) const;

    std::optional<int2> to_image(const float2& window_pos) const;
    std::optional<int2> cursor_pixel() const;

    void draw_window();
    void draw_header();
    void draw_look_tab();
    void draw_select_tab();
    void draw_analyze_tab();
    void draw_convergence_tab();
    void draw_distributions_tab() const;
    void draw_bsdf_tab();
    void draw_inspector();
    void draw_maps_tab();
    void draw_capture_tab();
    void draw_sampling_fit() const;
    std::string overlay_summary() const;
    void draw_properties_section(void (PathDebugNode::*section)(Properties&));

    void draw_overlay();
    void draw_panel_labels() const;
    void draw_probe_tooltip() const;

    void properties_selection(Properties& config);
    void properties_overlay(Properties& config);
    void properties_backdrop(Properties& config);
    void properties_maps(Properties& config);
    void properties_heat(Properties& config);
    void properties_pixel_stats(Properties& config);
    void properties_capture(Properties& config);
    void properties_export(Properties& config);

    static PathDebugParams default_params();

    ContextHandle context;
    ResourceAllocatorHandle allocator;
    ShaderCompileContextHandle compile_context;

    PtrInHandle<Scene> con_scene = PtrIn<Scene>::create();
    VkBufferInHandle con_records = VkBufferIn::create();
    VkSampledImageInHandle con_src = VkSampledImageIn::create();
    VkSampledImageInHandle con_backdrop = VkSampledImageIn::create();
    GBufferInHandle con_gbuffer;
    PtrInHandle<InputController> con_controller = PtrIn<InputController>::create();
    PtrInHandle<Window> con_window = PtrIn<Window>::create();
    PtrInHandle<SwapchainAcquireResult> con_acquire = PtrIn<SwapchainAcquireResult>::create();
    ManagedVkImageOutHandle con_out;
    PtrOutHandle<SwapchainAcquireResult> con_acquire_out =
        PtrOut<SwapchainAcquireResult>::create_optional();
    ManagedVkImageOutHandle con_filtered;
    ManagedVkImageOutHandle con_filtered_b;

    Versioned<SpecializationInfo> spec_info;
    std::optional<ComputeKernel> select_kernel;
    std::optional<ComputeKernel> threshold_kernel;
    std::optional<ComputeKernel> collect_kernel;
    std::optional<ComputeKernel> map_reduce_kernel;
    std::optional<ComputeKernel> map_view_kernel;
    std::optional<ComputeKernel> heat_decay_kernel;
    std::optional<ComputeKernel> heat_splat_kernel;
    std::optional<ComputeKernel> filter_render_kernel;
    std::optional<ComputeKernel> finalize_kernel;
    std::optional<ComputeKernel> stats_kernel;
    std::optional<ComputeKernel> compose_kernel;

    SlangCompositionHandle overlay_composition;
    Versioned<SlangProgram> overlay_program;
    Versioned<SlangProgramEntryPoint> overlay_vertex;
    Versioned<SlangProgramEntryPoint> overlay_fragment;
    Versioned<Pipeline> overlay_pipeline;
    Versioned<ShaderObject> overlay_globals;
    vk::Format overlay_format = vk::Format::eUndefined;

    ScenePipeline bsdf_map;
    ScenePipeline bsdf_check;

    vk::Extent3D extent{};
    BufferHandle state_buffer;
    BufferHandle out_state_buffer;
    uint32_t allocated_map_res = 32;
    BufferHandle stats_buffer;
    BufferHandle maps_buffer;
    BufferHandle grid_buffer;
    BufferHandle moments_buffer;
    BufferHandle filtered_buffer;
    BufferHandle matches_buffer;
    BufferHandle resampled_buffer;
    TextureHandle reference_texture;
    std::vector<BufferHandle> readback_buffers;

    ImGuiContextHandle imgui_ctx;
    ImGuiRendererHandle imgui_renderer;
    ImGuiMerianBackendHandle imgui_backend;
    Stopwatch frametime;

    std::shared_ptr<PathDebugInput> input = std::make_shared<PathDebugInput>();
    std::weak_ptr<InputController> registered_controller;
    vk::Offset3D present_lower{};
    vk::Offset3D present_upper{};
    float present_density = 1.f;
    bool presenting = false;

    std::mutex stats_mutex;
    Readback latest_readback{};
    bool readback_valid = false;
    bool window_drawn = false;
    bool inspector_drawn = false;
    bool convergence_drawn = false;
    bool mouse_over_ui = false;
    uint32_t frames_since_window = UI_IDLE_FRAMES;
    uint32_t frames_since_inspector = UI_IDLE_FRAMES;
    uint32_t frames_since_convergence = UI_IDLE_FRAMES;
    bool pending_reconnect = false;

    PathDebugParams params = default_params();

    bool picked = false;
    int2 picked_pixel = int2(0, 0);
    bool picked_pos_valid = false;
    float3 picked_pos = float3(0.f);
    float picked_radius = 0.f;
    int32_t restrict_mode = static_cast<int32_t>(PATH_DEBUG_RESTRICT_PIXEL);
    bool pending_pick = false;
    int2 pending_pick_pixel = int2(0);
    uint32_t pending_pick_seq = 0;
    std::string expr_a;
    std::string expr_b;
    PathExpression compiled_a;
    PathExpression compiled_b;
    int2 scatter_range = int2(0, 32);
    uint32_t method_mask = 0;
    int32_t method_mode = static_cast<int32_t>(PATH_DEBUG_METHOD_ANY_VERTEX);
    int32_t material_id = -1;
    int32_t contribution_filter = static_cast<int32_t>(PATH_DEBUG_FINITE_ANY);

    int32_t overlay_source = PATH_DEBUG_SOURCE_A;
    int32_t overlay_rank = PATH_DEBUG_RANK_ANY;
    int32_t overlay_max_paths = 64;
    float overlay_brightest_fraction = 1e-4f;
    int32_t overlay_depth_mode = static_cast<int32_t>(PATH_DEBUG_DEPTH_HIDE);
    int32_t overlay_color = static_cast<int32_t>(PATH_DEBUG_COLOR_BOUNCES);
    int32_t isolate_path = -1;
    uint32_t record_path_capacity = 0;

    int32_t backdrop = static_cast<int32_t>(PATH_DEBUG_BACKDROP_RENDER);
    std::string reference_path;
    bool reference_loaded = false;
    bool reference_dirty = false;
    bool reference_flip = false;
    float reference_flip_hz = 2.f;

    int32_t map_source = PATH_DEBUG_SOURCE_A;
    int32_t map_res_log2 = 5;
    int32_t map_transform = 0;
    int32_t map_frame = static_cast<int32_t>(PATH_DEBUG_FRAME_WORLD);
    int32_t map_pdf_ref = static_cast<int32_t>(PATH_DEBUG_PDF_REF_BSDF);
    int32_t map_bounce = 0;
    bool map_accumulate = true;
    int32_t map_draws = static_cast<int32_t>(PATH_DEBUG_DRAWS_SCATTER);
    int32_t map_batch_frames = 64;
    bool check_requested = false;
    uint32_t check_seq = 0;
    int32_t wi_mode = static_cast<int32_t>(PATH_DEBUG_WI_CAMERA);
    float wi_theta_deg = 45.f;
    float wi_phi_deg = 0.f;
    bool maps_dirty = true;
    bool pdf_stale = true;
    uint32_t map_frames = 0;
    std::array<bool, MAP_CHANNEL_COUNT> panel_enabled{true,  false, false, false, true,
                                                      false, true,  false, true};
    bool panels_enabled = true;
    int32_t panel_scroll = 0;
    int32_t panel_overflow = 0;
    int32_t analysis_paths = 1 << 16;
    int32_t sphere_channel = -1;

    int32_t heat_source = PATH_DEBUG_SOURCE_A;
    int32_t heat_mode = static_cast<int32_t>(PATH_DEBUG_HEAT_OFF);
    int2 heat_bounce_range = int2(0, 30);
    float heat_alpha = 0.05f;
    int32_t grid_slots_log2 = 20;
    bool heat_dirty = true;
    uint32_t heat_frames = 0;

    int32_t pixel_stats_source = PATH_DEBUG_SOURCE_ALL;
    int32_t pixel_stats_view = static_cast<int32_t>(PATH_DEBUG_PIXEL_STAT_OFF);
    bool pixel_stats_dirty = true;
    uint32_t pixel_stats_frames = 0;
    bool resampled_dirty = true;
    uint32_t resampled_frames = 0;
    uint64_t resampled_from = 0;
    std::deque<float> mean_history;
    std::deque<float> rel_error_history;
    std::deque<float> variance_history;

    bool freeze = false;
    bool auto_keep_prob = true;
    float keep_prob = 1.f;
    bool record_picked_pixel_only = false;
    bool export_next = false;
    bool export_metadata = true;
    bool export_keep_alpha = false;
    int32_t export_format = 0;
    std::string export_path = "path_debug";

    bool gbuffer_connected = false;
    SlangCompositionHandle gbuffer_composition;

    std::string imgui_event_pattern = "//ui";
};

} // namespace merian
