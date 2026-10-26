#pragma once

#include "merian-graph/connectors/buffer/vk_buffer_in.hpp"
#include "merian-graph/connectors/image/vk_image_in.hpp"
#include "merian-graph/connectors/image/vk_image_in_sampled.hpp"
#include "merian-graph/connectors/image/vk_image_out_managed.hpp"
#include "merian-graph/connectors/ptr_in.hpp"
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
#include "merian/vk/imgui/imgui_merian_window_backend.hpp"
#include "merian/vk/imgui/imgui_renderer.hpp"
#include "merian/vk/window/window.hpp"

#include <array>
#include <cstddef>
#include <deque>
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
    static constexpr float MIN_KEEP_PROB = 1e-4f;
    // the record fill the probability aims for, and the fill below which it rises again
    static constexpr float TARGET_FILL = 0.9f;
    static constexpr float MIN_FILL = 0.5f;
    // a map bin count close to overflow starts the maps over
    static constexpr uint32_t MAX_MAP_COUNT = 1u << 30;
    static constexpr float REGION_RADIUS_PIXELS = 48.f;
    static constexpr float MIN_REGION_RADIUS = 1e-6f;
    static constexpr float REGION_SCROLL_FACTOR = 1.15f;
    static constexpr double PANEL_SCROLL_PIXELS = 60.;
    static constexpr int32_t MIN_PANEL_PX = 16;
    static constexpr uint32_t UI_IDLE_FRAMES = 120;
    static constexpr int32_t GRID_SLOTS_LOG2_MIN = 8;
    static constexpr int32_t GRID_SLOTS_LOG2_MAX = 26;
    static constexpr int32_t MAP_RES_LOG2_MIN = 3;
    static constexpr int32_t MAP_RES_LOG2_MAX = 10;
    // a scatter range ending here is open
    static constexpr int32_t SCATTER_UNLIMITED = 32;
    static constexpr int32_t HEAT_BOUNCE_MAX = 30;

    static constexpr int32_t CAPTURE_LIVE = 0;
    static constexpr int32_t CAPTURE_FROZEN = 1;
    static constexpr int32_t CAPTURE_OFF = 2;
    static constexpr std::array<const char*, 3> CAPTURE_STATE_NAMES = {"live", "frozen", "off"};
    static constexpr int32_t SELECT_POINT = 0;
    static constexpr int32_t SELECT_REGION = 1;
    static constexpr std::array<const char*, 3> MATCH_NAMES = {"hits", "starts at", "ends at"};
    static constexpr int32_t MATCH_STARTS = 1;
    static constexpr int32_t TARGET_SPHERE = 0;
    static constexpr int32_t TARGET_TRIANGLE = 1;
    static constexpr int32_t TARGET_INSTANCE = 2;
    static constexpr int32_t TARGET_NODE = 3;
    static constexpr int32_t TARGET_MESH = 4;
    static constexpr int32_t TARGET_MATERIAL = 5;
    static constexpr std::array<const char*, 7> SELECTION_NAMES = {
        "point", "sphere", "triangle", "instance", "node", "mesh", "material"};
    static constexpr std::array<const char*, 4> MODIFIER_NAMES = {"Alt", "Ctrl", "Shift", "Super"};
    static constexpr std::array<int, 4> MODIFIER_KEYS = {
        InputController::ALT, InputController::CONTROL, InputController::SHIFT,
        InputController::SUPER};
    static constexpr std::array<const char*, 8> COLOR_NAMES = {
        "path luminance", "transported radiance", "scatter events",    "random",
        "technique",      "BSDF·cos/pdf",         "transported color", "off"};
    static constexpr std::array<const char*, 9> BACKDROP_NAMES = {"source input",
                                                                  "current frame",
                                                                  "current frame (filtered)",
                                                                  "accumulated",
                                                                  "accumulated (filtered)",
                                                                  "reference",
                                                                  "difference",
                                                                  "split source input | reference",
                                                                  "backdrop input"};
    static constexpr std::array<const char*, 3> BACKGROUND_NAMES = {"off", "wireframe",
                                                                    "shaded albedo"};
    static constexpr std::array<const char*, 2> WI_NAMES = {"camera ray", "manual"};
    static constexpr int32_t FALSE_COLOR_OFF = 0;
    static constexpr int32_t FALSE_COLOR_HEAT = 1;
    static constexpr int32_t FALSE_COLOR_PIXEL_STAT = 2;
    static constexpr std::array<const char*, 3> FALSE_COLOR_NAMES = {"off", "world space",
                                                                     "screen space"};
    static constexpr int32_t STATISTICS_PIXEL = 0;
    static constexpr int32_t STATISTICS_LIGHT = 1;
    static constexpr std::array<const char*, 2> STATISTICS_NAMES = {"pixel", "light paths"};
    static constexpr std::array<const char*, PATH_DEBUG_MAP_CHANNEL_COUNT> MAP_CHANNEL_NAMES = {
        "integrand (normalized)",
        "incident radiance",
        "BSDF·cos (normalized)",
        "recorded directions",
        "recorded NEE directions",
        "BSDF sample()",
        "recorded pdfs",
        "recorded NEE pdfs",
        "BSDF pdf()",
        "z-score",
        "NEE z-score",
        "sample() z-score",
        "recorded / integrand",
        "second moment",
        "recorded guided directions"};
    static constexpr std::array<const char*, PATH_DEBUG_MAP_CHANNEL_COUNT>
        MAP_CHANNEL_DESCRIPTIONS = {
            "BSDF·L·cos of every technique's draws, normalized to integrate to 1: what a "
            "zero-variance continuation draws.",
            "The light arriving at the point: the integrand over BSDF eval().",
            "BSDF eval()·cos, normalized to integrate to 1: what an ideal BSDF sample() draws.",
            "Where the renderer's continuations went, per steradian.",
            "Where the renderer's NEE connections went, per steradian.",
            "Where the material's own sample() draws, per steradian.",
            "The pdfs the renderer recorded with its continuations, per bin.",
            "The pdfs the renderer recorded with its NEE connections, per bin.",
            "The material's pdf().",
            "The continuations against their recorded pdfs, in standard deviations: blue too few, "
            "red too many draws.",
            "The NEE connections against their recorded pdfs, the environment's and the geometry's "
            "apart, in "
            "standard deviations; lights behind each other read red.",
            "BSDF sample()'s draws against their pdfs, in standard deviations.",
            "The continuations over the normalized integrand: blue undersampled, red oversampled.",
            "Squared contributions per steradian: where the variance comes from.",
            "Where the guided continuations went, per steradian."};

    struct IsolatedPath {
        uint32_t valid;
        uint32_t pixel;
        uint32_t sample_and_count; // sample | vertex count << 16
        uint32_t luminance;
        std::array<uint32_t, std::size_t{PATH_DEBUG_MAX_SEGMENTS} * PATH_RECORD_VERTEX_UINTS>
            vertices;
    };

    struct TopPath {
        uint32_t slot;
        uint32_t pixel;
        uint32_t luminance;
        uint32_t sample_and_scatter; // sample | scatter count << 16
        uint32_t classes;            // the first 8 scatter events, 4 bits each
    };

    struct Pick {
        uint32_t record_seq;
        float3 first;
        float3 seen;
        uint2 seen_surface;
        uint32_t surface_seq;
        float3 surface;
        float3 normal;
        uint2 hit;
    };

    struct RegionTarget {
        bool environment = false;
        int32_t mesh = -1;
        int32_t node = -1;
        int32_t primitive = -1;
        int32_t material = -1;

        bool operator==(const RegionTarget&) const = default;
    };

    // GeometryIDs belong to the frame they were looked up in
    using Placements = std::shared_ptr<const std::vector<Scene::GeometryPlacement>>;

    struct TargetBuild {
        Placements placements;
        RegionTarget target;
        int32_t kind = -1;

        bool operator==(const TargetBuild&) const = default;
    };

    struct ImageSignature {
        std::string expression;
        int32_t contribution = 0;
        int32_t scatters_min = 0;
        int32_t scatters_max = 0;
        uint32_t methods = 0;
        int32_t method_mode = 0;
        int32_t match = -1;
        float sphere_x = 0.f;
        float sphere_y = 0.f;
        float sphere_z = 0.f;
        float sphere_radius = 0.f;
        TargetBuild target;

        bool operator==(const ImageSignature&) const = default;
    };

    struct PointPaths {
        uint32_t at_point;
        uint32_t in_front;
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
        // pixel: mean, relative error, variance; panel or sphere: its value, BSDF pdf(), solid
        // angle
        std::array<uint32_t, 3> values;
        uint32_t count;
        uint32_t selected_mean;
        uint32_t selected_rel_error;
        uint32_t selected_count;
        uint32_t selected_variance;
        uint32_t selected_max;
        uint32_t selected_zeros;
    };

    struct Readback {
        FrameStats frame;
        Probe probe;
        IsolatedPath isolated;
        std::array<TopPath, PATH_DEBUG_TOP_COUNT> top;
        Pick pick;
        PointPaths point;
        std::array<uint32_t, PATH_DEBUG_STATS_UINTS> stats;
    };
    static_assert(offsetof(Readback, probe) == sizeof(uint32_t) * PATH_DEBUG_OUT_PROBE);
    static_assert(offsetof(Probe, values) == sizeof(uint32_t) * PATH_DEBUG_PROBE_VALUES);
    static_assert(offsetof(Probe, count) == sizeof(uint32_t) * PATH_DEBUG_PROBE_COUNT);
    static_assert(offsetof(Probe, selected_mean) == sizeof(uint32_t) * PATH_DEBUG_PROBE_SELECTED);
    static_assert(sizeof(Probe) == sizeof(uint32_t) * PATH_DEBUG_PROBE_UINTS);
    static_assert(offsetof(IsolatedPath, vertices) == sizeof(uint32_t) * PATH_DEBUG_ISOLATE_HEADER);
    static_assert(offsetof(TopPath, classes) == sizeof(uint32_t) * PATH_DEBUG_TOP_EVENTS);
    static_assert(sizeof(TopPath) == sizeof(uint32_t) * PATH_DEBUG_TOP_UINTS);
    static_assert(offsetof(Readback, isolated) == sizeof(uint32_t) * PATH_DEBUG_OUT_ISOLATE);
    static_assert(offsetof(Readback, top) == sizeof(uint32_t) * PATH_DEBUG_OUT_TOP);
    static_assert(offsetof(Readback, pick) == sizeof(uint32_t) * PATH_DEBUG_OUT_PICK);
    static_assert(offsetof(Pick, seen_surface) == sizeof(uint32_t) * PATH_DEBUG_PICK_SEEN_SURFACE);
    static_assert(offsetof(Pick, surface_seq) == sizeof(uint32_t) * PATH_DEBUG_PICK_SURFACE_SEQ);
    static_assert(offsetof(Pick, hit) == sizeof(uint32_t) * PATH_DEBUG_PICK_HIT);
    static_assert(sizeof(Pick) == sizeof(uint32_t) * PATH_DEBUG_PICK_UINTS);
    static_assert(offsetof(Readback, point) == sizeof(uint32_t) * PATH_DEBUG_OUT_POINT);
    static_assert(offsetof(Readback, stats) == sizeof(uint32_t) * PATH_DEBUG_OUT_UINTS);

    struct FrameActivity {
        bool active;
        // the frame after a freeze still sees the held records
        bool frozen;
        bool keep_update;
        bool pixel_stats_accumulate;
        bool heat_splat;
        bool one_pixel;
        bool image_accumulate;
        bool image_depth;
        bool image_voxels;
        uint32_t panel_mask;
        uint32_t sphere_channel;
        bool maps_active;
        bool maps_accumulate;
        bool batch_end;
        bool run_check;
        bool check_shown;
    };

    struct OverlayTarget {
        vk::Format format = vk::Format::eUndefined;
        Versioned<Pipeline> pipeline;
        Versioned<ShaderObject> globals;
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
        uint32_t bin;
        bool on_close;
    };

    struct PixelSample {
        float mean;
        float rel_error;
        float variance;
        float max_ratio;
        float zero_fraction;
        uint32_t count;
    };
    static constexpr std::size_t PIXEL_HISTORY = 512;

    void handle_input(const NodeIO& io);
    void select_at(const int2& pixel);
    void set_select_kind(int32_t kind);
    int32_t selection_choice() const;
    void set_selection_choice(int32_t choice);
    void set_region_match(int32_t match);
    void set_region_target(int32_t kind);
    void pick_target(uint2 surface, const Placements& placements);
    void update_target();
    void upload_target(const ResourceAllocatorHandle& allocator, Submission& submission);
    bool targets_ids() const;
    bool target_resolved() const;
    void set_target(const RegionTarget& ids);
    std::string target_label() const;
    // where the point lies on screen, in record pixels
    std::optional<float2> project_point(Camera& camera) const;
    void reproject_pick(const NodeIO& io);
    uint32_t apply_readback_feedback(const NodeProcessInfo& info,
                                     const PathRecordCapacities& capacities);
    void adapt_keep_prob(const PathRecordCapacities& capacities);
    void resolve_pick(Camera& camera);
    float pixel_angle(const Camera& camera) const;
    CameraHandle ready_camera(const NodeIO& io) const;
    void record_clears(const NodeProcessInfo& info, Submission& submission);
    FrameActivity update_frame(const NodeIO& io, bool ready);
    void update_panel_layout(const NodeIO& io, uint32_t panel_mask);
    PathDebugParams build_params(const NodeIO& io,
                                 const NodeProcessInfo& info,
                                 const FrameActivity& activity,
                                 uint32_t observed_paths) const;
    void build_selection_params(PathDebugParams& p, const NodeIO& io) const;
    void build_capture_params(PathDebugParams& p, const FrameActivity& activity) const;
    void build_overlay_params(PathDebugParams& p, const FrameActivity& activity) const;
    void build_map_params(PathDebugParams& p,
                          const FrameActivity& activity,
                          uint32_t observed_paths) const;
    void build_heat_params(PathDebugParams& p, const FrameActivity& activity) const;
    void build_pixel_stats_params(PathDebugParams& p,
                                  const NodeIO& io,
                                  const FrameActivity& activity) const;
    void build_backdrop_params(PathDebugParams& p, const NodeProcessInfo& info) const;
    std::array<std::optional<ComputeKernel>*, 11> kernels();
    void bind_globals(const NodeIO& io);
    void bind_node_buffers(ShaderCursor cursor, const NodeIO& io) const;
    void record_analysis(const NodeIO& io,
                         const NodeProcessInfo& info,
                         Submission& submission,
                         const PathRecordCapacities& capacities,
                         const FrameActivity& activity);
    PipelineHandle bind_scene_pipeline(ScenePipeline& kernel,
                                       const NodeIO& io,
                                       const NodeProcessInfo& info,
                                       Submission& submission);
    void record_compose(const NodeIO& io, const NodeProcessInfo& info, Submission& submission);
    void record_overlay(const NodeIO& io,
                        const NodeProcessInfo& info,
                        Submission& submission,
                        bool onto_canvas);
    void record_voxels(const NodeIO& io, const NodeProcessInfo& info, Submission& submission);
    void record_map_panels(const NodeIO& io, const NodeProcessInfo& info, Submission& submission);
    void record_labels(const NodeIO& io,
                       const NodeProcessInfo& info,
                       Submission& submission,
                       bool onto_canvas);
    void record_export(const NodeIO& io, const NodeProcessInfo& info, Submission& submission);
    void record_readback(const NodeIO& io, const NodeProcessInfo& info, Submission& submission);
    bool record_present(const NodeIO& io, Submission& submission);

    void build_gbuffer_kernels(const ContextHandle& context,
                               const ResourceAllocatorHandle& allocator);
    void ensure_overlay_target(const ContextHandle& context,
                               const ResourceAllocatorHandle& allocator,
                               OverlayTarget& target,
                               vk::Format color_format);
    void ensure_scene_pipeline(const ContextHandle& context,
                               const ResourceAllocatorHandle& allocator,
                               ScenePipeline& kernel,
                               const SceneHandle& scene,
                               const char* module);
    void ensure_reference(const ResourceAllocatorHandle& allocator, Submission& submission);

    void invalidate_selection();
    void invalidate_filter();
    ImageSignature image_signature() const;
    void clear_selection();
    // re-resolves the surface at the selected pixel, keeping the current one until it is read back
    void repick();
    void request_pick(const int2& pixel);
    void reset_selected_point();
    void set_expression(const std::string& text);
    void set_method(uint32_t method, bool set);
    bool maps_at_point() const;
    // the maps bin the draws the BSDF's pdf() is evaluated for
    bool bsdf_comparable() const;
    bool shows_reference() const;
    bool pixel_stats_read() const;
    bool keeps_top() const;
    bool has_reference() const;

    uint32_t selection_mode() const;
    bool selection_visible() const;
    bool selects_point() const;
    bool shows_pixel_image(const NodeIO& io) const;
    const char* missing_image_input() const;
    bool point_hidden() const;
    const char* modifier_name() const;
    std::string material_label(uint32_t id) const;
    std::string mesh_label(int32_t mesh) const;
    std::string node_label(int32_t node) const;
    uint32_t shown_panel_mask() const;
    int32_t label_px(uint32_t size) const;
    int32_t panel_top(uint32_t slot) const;
    int32_t panel_left() const;
    std::optional<PanelHit> panel_at(const float2& at) const;

    std::optional<int2> to_image(const float2& window_pos) const;

    void draw_window();
    void draw_image_choice();
    void draw_header();
    void draw_paths_tab();
    void draw_view_tab();
    void draw_analysis_tab();
    void draw_filter_tab();
    void draw_preferences_tab();
    void draw_selection_choice();
    void draw_panel_choice();
    void draw_sampling_section();
    void draw_bsdf_section();
    void draw_statistics_section();
    void draw_pixel_statistics();
    void draw_light_statistics() const;
    void draw_list();
    void draw_color_legend() const;
    std::string selection_status() const;
    void draw_target_choice(const std::string& status);
    void draw_properties_section(const char* id, void (PathDebugNode::*section)(Properties&));

    void draw_panel_labels() const;
    void draw_probe_tooltip() const;

    void properties_selection(Properties& config);
    void properties_selection_details(Properties& config);
    void properties_color(Properties& config);
    void properties_filter(Properties& config);
    void properties_techniques(Properties& config);
    void properties_technique_match(Properties& config);
    void properties_path_choice(Properties& config);
    void properties_line_style(Properties& config);
    void properties_false_color(Properties& config);
    void properties_sphere(Properties& config);
    void properties_reference(Properties& config);
    void properties_image(Properties& config);
    void properties_panels(Properties& config);
    void properties_bins(Properties& config);
    void properties_sampling(Properties& config);
    void properties_bsdf_check(Properties& config);
    void properties_capture(Properties& config);
    void properties_export(Properties& config);
    void properties_keys(Properties& config);

    ShaderCompileContextHandle compile_context;

    PtrInHandle<Scene> con_scene = PtrIn<Scene>::create();
    VkBufferInHandle con_records = VkBufferIn::create();
    VkSampledImageInHandle con_src = VkSampledImageIn::create();
    VkSampledImageInHandle con_backdrop = VkSampledImageIn::create();
    VkSampledImageInHandle con_reference = VkSampledImageIn::create();
    GBufferInHandle con_gbuffer;
    PtrInHandle<InputController> con_controller = PtrIn<InputController>::create();
    PtrInHandle<Window> con_window = PtrIn<Window>::create();
    VkImageInHandle con_canvas = VkImageIn::create();
    ManagedVkImageOutHandle con_out;
    ManagedVkImageOutHandle con_filtered;

    Versioned<SpecializationInfo> spec_info;
    std::optional<ComputeKernel> select_kernel;
    std::optional<ComputeKernel> threshold_kernel;
    std::optional<ComputeKernel> collect_kernel;
    std::optional<ComputeKernel> keep_kernel;
    std::optional<ComputeKernel> map_reduce_kernel;
    std::optional<ComputeKernel> map_view_kernel;
    std::optional<ComputeKernel> heat_decay_kernel;
    std::optional<ComputeKernel> heat_splat_kernel;
    std::optional<ComputeKernel> finalize_kernel;
    std::optional<ComputeKernel> stats_kernel;
    std::optional<ComputeKernel> compose_kernel;

    SlangCompositionHandle overlay_composition;
    Versioned<SlangProgram> overlay_program;
    Versioned<SlangProgramEntryPoint> overlay_vertex;
    Versioned<SlangProgramEntryPoint> overlay_fragment;
    OverlayTarget overlay_out;
    OverlayTarget overlay_canvas;
    SlangCompositionHandle voxel_composition;
    Versioned<SlangProgram> voxel_program;
    Versioned<SlangProgramEntryPoint> voxel_vertex;
    Versioned<SlangProgramEntryPoint> voxel_fragment;
    Versioned<Pipeline> voxel_pipeline;
    Versioned<ShaderObject> voxel_globals;

    ScenePipeline bsdf_map;
    ScenePipeline bsdf_check;

    // the pixels the records are traced in
    vk::Extent3D extent{};
    // the displayed image
    vk::Extent3D out_extent{};
    uint32_t record_path_capacity = 0;
    BufferHandle state_buffer;
    BufferHandle out_state_buffer;
    uint32_t allocated_map_res = 0;
    BufferHandle stats_buffer;
    BufferHandle maps_buffer;
    BufferHandle grid_buffer;
    BufferHandle moments_buffer;
    BufferHandle filtered_buffer;
    BufferHandle voxel_depth_buffer;
    TextureHandle voxel_texture;
    ImageViewHandle voxel_depth_view;
    BufferHandle matches_buffer;
    BufferHandle events_buffer;
    BufferHandle target_buffer;
    BufferHandle kept_buffer;
    TextureHandle reference_texture;
    TextureHandle panel_texture;
    ImTextureID panel_texture_id = 0;
    std::vector<BufferHandle> readback_buffers;

    ImGuiContextHandle imgui_ctx;
    ImGuiRendererHandle imgui_renderer;
    ImGuiMerianBackendHandle imgui_backend;
    ImGuiMerianWindowBackendHandle imgui_window_backend;
    std::weak_ptr<Window> imgui_window;
    Stopwatch frametime;

    std::shared_ptr<PathDebugInput> input = std::make_shared<PathDebugInput>();
    std::weak_ptr<InputController> registered_controller;
    vk::Offset3D present_lower{};
    vk::Offset3D present_upper{};
    float present_density = 1.f;
    bool presenting = false;
    // the canvas while presenting, else the output
    int2 label_area = int2(0);
    float label_scale = 1.f;
    std::optional<float2> probe_at;
    std::optional<int2> cursor_pixel;
    std::optional<PanelHit> probe_panel;
    int32_t modifier = 0;

    Readback latest_readback{};
    bool readback_valid = false;
    bool window_drawn = false;
    bool list_shown = false;
    bool pixel_section_shown = false;
    bool path_statistics_shown = false;
    uint32_t frames_since_window = UI_IDLE_FRAMES;
    uint32_t frames_since_list = UI_IDLE_FRAMES;
    uint32_t frames_since_pixel_section = UI_IDLE_FRAMES;
    uint32_t frames_since_path_statistics = UI_IDLE_FRAMES;
    bool pending_reconnect = false;

    PathDebugParams params{};
    // the pixel filter the current records were traced with
    uint32_t traced_pixel_filter = PATH_RECORD_ALL_PIXELS;

    int32_t select_kind = SELECT_POINT;
    int32_t region_match = 0;
    int32_t region_target = TARGET_SPHERE;
    RegionTarget target;
    bool target_from_click = false;
    // in the frame of the matched records: the triangle's GeometryID, a bit per GeometryID of the
    // other targets
    uint32_t target_geometry = PATH_RECORD_GEOMETRY_NONE;
    std::vector<uint32_t> target_bits;
    TargetBuild target_built;
    bool target_bits_dirty = true;
    // of this frame, of the matched records and of the latest readback
    Placements placements;
    Placements records_placements;
    Placements readback_placements;
    bool reproject = true;
    bool has_selection = false;
    // in record pixels
    int2 selected_pixel = int2(0, 0);
    // from the projected point to the selected pixel's center, the sample's jitter
    float2 selected_pixel_offset = float2(0.f);
    bool selected_offscreen = false;
    std::optional<int2> selected_in_records;
    bool selected_pos_valid = false;
    float3 selected_pos = float3(0.f);
    // 0 while unknown
    float3 selected_normal = float3(0.f);
    float selected_radius = 0.f;
    bool point_behind_surface = false;
    bool pending_pick = false;
    int2 pending_pick_pixel = int2(0);
    uint32_t pending_pick_seq = 0;
    std::string expression;
    PathExpression compiled_expression;
    int2 scatter_range = int2(0, SCATTER_UNLIMITED);
    uint32_t method_mask = 0;
    int32_t method_mode = static_cast<int32_t>(PATH_DEBUG_METHOD_ANY_VERTEX);
    std::weak_ptr<Scene> current_scene;
    std::weak_ptr<MaterialSystem> material_system;
    int32_t contribution_filter = static_cast<int32_t>(PATH_DEBUG_FINITE_ANY);

    int32_t overlay_rank = PATH_DEBUG_RANK_RESAMPLED;
    bool paths_filtered = true;
    int32_t overlay_max_paths = 16;
    float overlay_brightest_share = 1e-4f;
    float resample_history = 60.f;
    bool keep_on_camera_move = true;
    int32_t overlay_depth_mode = static_cast<int32_t>(PATH_DEBUG_DEPTH_DASH);
    int32_t overlay_color = static_cast<int32_t>(PATH_DEBUG_COLOR_LUMINANCE);
    std::optional<uint32_t> isolate_path;
    float overlay_opacity = 0.85f;
    float overlay_thickness = 2.f;
    float overlay_exposure = 1.f;

    int32_t backdrop = static_cast<int32_t>(PATH_DEBUG_BACKDROP_ACCUMULATED);
    int32_t background = static_cast<int32_t>(PATH_DEBUG_BACKGROUND_WIREFRAME);
    int32_t false_color = FALSE_COLOR_OFF;
    bool false_color_filtered = false;
    std::string reference_path;
    bool reference_loaded = false;
    bool reference_input_connected = false;
    bool backdrop_missing = false;
    bool reference_dirty = false;
    bool reference_flip = false;
    float reference_flip_hz = 2.f;
    float backdrop_split = 0.5f;
    float difference_gain = 10.f;
    float image_exposure = 1.f;
    bool image_dirty = true;
    bool image_filtered_dirty = true;
    bool image_accumulated = false;
    ImageSignature image_held;
    CameraHandle image_camera;
    float false_color_opacity = 0.6f;

    int32_t map_res_log2 = 5;
    int32_t map_transform = 0;
    int32_t map_frame = static_cast<int32_t>(PATH_DEBUG_FRAME_LOCAL);
    bool map_accumulate = true;
    int32_t map_batch_frames = 64;
    bool check_requested = false;
    uint32_t check_seq = 0;
    uint32_t check_draws = 0;
    bool check_topping_up = false;
    bool surface_captured = false;
    int32_t wi_mode = static_cast<int32_t>(PATH_DEBUG_WI_CAMERA);
    float wi_theta_deg = 45.f;
    float wi_phi_deg = 0.f;
    bool maps_dirty = true;
    bool pdf_stale = true;
    uint32_t map_frames = 0;
    std::array<bool, PATH_DEBUG_MAP_CHANNEL_COUNT> panel_enabled{true,  false, false, true,  false,
                                                                 true,  true,  false, true,  true,
                                                                 false, false, false, false, false};
    bool panels_enabled = true;
    int32_t panel_size = 192;
    int32_t panel_px = 192;
    int32_t panel_scroll = 0;
    int32_t panel_overflow = 0;
    int32_t analysis_paths = 1 << 16;
    float map_exposure = 1.f;
    int32_t sphere_channel = -1;
    // of the distance
    float sphere_size = 0.05f;
    bool sphere_scale = false;
    int32_t statistics_kind = STATISTICS_PIXEL;

    // PATH_DEBUG_HEAT_* - PATH_DEBUG_HEAT_VERTICES
    int32_t heat_quantity = 0;
    int2 heat_bounce_range = int2(0, HEAT_BOUNCE_MAX);
    // frames, 0 = keep forever
    float heat_history = 20.f;
    float heat_cell_size = 0.05f;
    float heat_exposure = 1.f;
    bool heat_smooth = false;
    int32_t grid_slots_log2 = 20;
    bool heat_dirty = true;
    bool heat_splatted = false;

    // PATH_DEBUG_PIXEL_STAT_* - PATH_DEBUG_PIXEL_STAT_MEAN
    int32_t pixel_statistic =
        static_cast<int32_t>(PATH_DEBUG_PIXEL_STAT_REL_ERROR - PATH_DEBUG_PIXEL_STAT_MEAN);
    float pixel_stats_exposure = 1.f;
    bool pixel_stats_dirty = true;
    bool pixel_stats_accumulated = false;
    uint32_t pixel_stats_generation = 0;
    bool kept_dirty = true;
    bool kept_updated = false;
    std::deque<PixelSample> pixel_history;

    int32_t capture = CAPTURE_LIVE;
    bool was_frozen = false;
    bool camera_moved = false;
    float keep_prob = 1.f;
    bool export_next = false;
    bool export_metadata = true;
    bool export_keep_alpha = false;
    int32_t export_format = 0;
    std::string export_path = "path_debug";

    bool gbuffer_connected = false;
    SlangCompositionHandle gbuffer_composition;
};

} // namespace merian
