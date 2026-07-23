#include "merian-graph/nodes/error_plot/error_plot.hpp"

#include "merian/vk/pipeline/specialization_info_builder.hpp"
#include "merian/vk/utils/blits.hpp"

#include "merian-graph/graph/errors.hpp"

#include <imgui.h>

#include <algorithm>
#include <cmath>

namespace merian {

namespace {
constexpr const char* ERROR_TO_BUFFER_MODULE =
    "merian-graph/nodes/error_plot/error_to_buffer.slang";
constexpr const char* REDUCE_BUFFER_MODULE =
    "merian-graph/nodes/error_plot/error_reduce_buffer.slang";
} // namespace

ErrorPlot::ErrorPlot() = default;

ErrorPlot::~ErrorPlot() = default;

DeviceSupportInfo ErrorPlot::query_device_support(const DeviceSupportQueryInfo& query_info) {
    DeviceSupportInfo support{true};
    for (const char* module : {ERROR_TO_BUFFER_MODULE, REDUCE_BUFFER_MODULE}) {
        const auto composition = SlangComposition::create();
        composition->add_module_from_path(module, true);
        support = support & SlangProgram::create(query_info.compile_context, composition)
                                .get()
                                ->query_device_support(query_info);
    }
    return support;
}

void ErrorPlot::initialize(const ContextHandle& context, const ResourceAllocatorHandle& allocator) {
    this->context = context;
    this->allocator = allocator;
    this->compile_context = context->get_shader_compile_context();

    auto error_spec = SpecializationInfoBuilder();
    error_spec.add_entry(local_size_x, local_size_y);
    error_to_buffer_spec.set(error_spec.build());

    auto reduce_spec = SpecializationInfoBuilder();
    reduce_spec.add_entry(workgroup_size, 1u);
    reduce_buffer_spec.set(reduce_spec.build());

    error_to_buffer_kernel.emplace(context, allocator, compile_context, ERROR_TO_BUFFER_MODULE,
                                   error_to_buffer_spec);
    reduce_buffer_kernel.emplace(context, allocator, compile_context, REDUCE_BUFFER_MODULE,
                                 reduce_buffer_spec);

    ::ImGuiContext* const prev_imgui_ctx = ImGui::GetCurrentContext();
    imgui_ctx = std::make_shared<ImGuiContext>();
    imgui_renderer = std::make_shared<ImGuiRenderer>(context, allocator, imgui_ctx);
    imgui_backend = std::make_shared<ImGuiMerianBackend>(imgui_ctx);
    ImGui::SetCurrentContext(prev_imgui_ctx);
}

std::vector<InputConnectorDescriptor> ErrorPlot::describe_inputs() {
    return {
        {"reference", con_reference, ConnectorAccess::compute_read | ConnectorAccess::transfer_src},
        {"input", con_input, ConnectorAccess::compute_read | ConnectorAccess::transfer_src}};
}

std::vector<OutputConnectorDescriptor> ErrorPlot::describe_outputs(const NodeIOLayout& io_layout) {
    const vk::ImageCreateInfo reference_info = io_layout[con_reference]->get_create_info_or_throw();
    const vk::ImageCreateInfo input_info = io_layout[con_input]->get_create_info_or_throw();
    if (reference_info.extent != input_info.extent) {
        throw graph_errors::node_error{"reference and input image extents mismatch"};
    }
    const vk::Extent3D extent = reference_info.extent;

    const auto group_count_x = (extent.width + local_size_x - 1) / local_size_x;
    const auto group_count_y = (extent.height + local_size_y - 1) / local_size_y;
    const std::size_t buffer_size = group_count_x * group_count_y;

    con_error = ManagedVkBufferOut::create(vk::BufferCreateInfo(
        {}, buffer_size * sizeof(float4),
        vk::BufferUsageFlagBits::eStorageBuffer | vk::BufferUsageFlagBits::eTransferSrc));

    // blit target for the split view and color attachment for the overlay
    const vk::ImageCreateInfo out_info{
        {},
        vk::ImageType::e2D,
        reference_info.format,
        extent,
        1,
        1,
        vk::SampleCountFlagBits::e1,
        vk::ImageTiling::eOptimal,
        vk::ImageUsageFlagBits::eTransferDst | vk::ImageUsageFlagBits::eColorAttachment,
        vk::SharingMode::eExclusive,
        {},
        {},
        vk::ImageLayout::eUndefined,
    };
    con_out = ManagedVkImageOut::create(out_info);

    return {
        {"out", con_out, ConnectorAccess::transfer_dst},
        {"error", con_error, ConnectorAccess::compute_read_write | ConnectorAccess::transfer_src}};
}

ErrorPlot::NodeStatusFlags ErrorPlot::on_connected(const NodeConnectedInfo& info) {
    const NodeIOLayout& io_layout = info.io_layout;
    io_layout.register_event_listener(
        "/graph/reload_shaders", [this](const GraphEvent::Info&, const GraphEvent::Data& force) {
            for (auto* kernel : {&error_to_buffer_kernel, &reduce_buffer_kernel}) {
                (*kernel)->reload(std::any_cast<bool>(force), compile_context);
            }
            return true;
        });

    return {};
}

void ErrorPlot::process(GraphRun& run, const NodeIO& io) {
    const CommandBufferHandle& cmd = run.get_cmd();

    // 1. pull the latest async readback into the plot history (graph thread only)
    bool new_sample = false;
    {
        const std::scoped_lock lock(result_mutex);
        if (latest_valid) {
            current_sum = latest_sum;
            latest_valid = false;
            new_sample = true;
        }
    }
    if (new_sample) {
        const float4 e = metric_error();
        history.push_back((e.x + e.y + e.z) / 3.0f);
        while (history.size() > history_size) {
            history.pop_front();
        }
    }

    const vk::Extent3D extent = io[con_reference]->get_extent();
    const auto group_count_x = (extent.width + local_size_x - 1) / local_size_x;
    const auto group_count_y = (extent.height + local_size_y - 1) / local_size_y;

    pc.divisor = extent.width * extent.height;
    pc.squared = metric == ErrorMetric::MAE ? 0u : 1u;

    // 2. per-pixel error into the reduction buffer
    {
        MERIAN_PROFILE_SCOPE_GPU(run.get_profiler(), cmd, "error to buffer");
        const auto pipe = error_to_buffer_kernel->bind(run, io);
        cmd->push_constant(pipe, pc);
        cmd->dispatch(group_count_x, group_count_y, 1);
    }

    // 3. reduce to a single element
    pc.size = group_count_x * group_count_y;
    pc.offset = 1;
    pc.count = group_count_x * group_count_y;
    PipelineHandle reduce_pipe;
    while (pc.count > 1) {
        MERIAN_PROFILE_SCOPE_GPU(run.get_profiler(), cmd,
                                 fmt::format("reduce {} elements", pc.count));
        cmd->barrier(vk::PipelineStageFlagBits::eComputeShader,
                     vk::PipelineStageFlagBits::eComputeShader,
                     io[con_error]->buffer_barrier(
                         vk::AccessFlagBits::eShaderRead | vk::AccessFlagBits::eShaderWrite,
                         vk::AccessFlagBits::eShaderRead | vk::AccessFlagBits::eShaderWrite));
        if (!reduce_pipe) {
            reduce_pipe = reduce_buffer_kernel->bind(run, io);
        }
        cmd->push_constant(reduce_pipe, pc);
        cmd->dispatch((pc.count + workgroup_size - 1) / workgroup_size, 1, 1);

        pc.count = (pc.count + workgroup_size - 1) / workgroup_size;
        pc.offset *= workgroup_size;
    }

    // 4. non-blocking readback of the reduced value (must not stall the graph)
    const uint32_t in_flight = run.get_in_flight_index();
    if (readback_buffers.size() != run.get_iterations_in_flight()) {
        readback_buffers.assign(run.get_iterations_in_flight(), nullptr);
    }
    if (!readback_buffers[in_flight]) {
        readback_buffers[in_flight] =
            allocator->create_buffer(sizeof(float4), vk::BufferUsageFlagBits::eTransferDst,
                                     MemoryMappingType::HOST_ACCESS_RANDOM, "error_plot readback");
    }
    const BufferHandle readback = readback_buffers[in_flight];
    cmd->barrier(vk::PipelineStageFlagBits::eComputeShader, vk::PipelineStageFlagBits::eTransfer,
                 io[con_error]->buffer_barrier(vk::AccessFlagBits::eShaderWrite,
                                               vk::AccessFlagBits::eTransferRead));
    cmd->copy(static_cast<const BufferHandle&>(io[con_error]), readback);
    run.sync_to_cpu([this, readback]() {
        const float4 value = *readback->get_memory()->map_as<float4>();
        readback->get_memory()->unmap();
        const std::scoped_lock lock(result_mutex);
        latest_sum = value;
        latest_valid = true;
    });

    // 5. split view: input fills the output, reference overwrites the left half
    const ImageHandle reference_img = io[con_reference];
    const ImageHandle input_img = io[con_input];
    const ImageHandle out_img = io[con_out];
    {
        MERIAN_PROFILE_SCOPE_GPU(run.get_profiler(), cmd, "split view");
        cmd->barrier(reference_img->barrier2(vk::ImageLayout::eGeneral));
        cmd->barrier(input_img->barrier2(vk::ImageLayout::eGeneral));

        cmd_blit_fit(cmd, input_img, vk::ImageLayout::eGeneral, input_img->get_extent(), out_img,
                     vk::ImageLayout::eGeneral, out_img->get_extent());
        cmd->barrier(out_img->barrier2(vk::ImageLayout::eGeneral));

        vk::Extent3D reference_extent = reference_img->get_extent();
        reference_extent.width /= 2;
        vk::Extent3D out_half = out_img->get_extent();
        out_half.width /= 2;
        cmd_blit_fit(cmd, reference_img, vk::ImageLayout::eGeneral, reference_extent, out_img,
                     vk::ImageLayout::eGeneral, out_half, std::nullopt);
    }

    // 6. overlay the plot, labels and values onto the split view
    {
        MERIAN_PROFILE_SCOPE_GPU(run.get_profiler(), cmd, "overlay");
        const vk::Extent3D out_extent = out_img->get_extent();
        imgui_ctx->get_io().DisplaySize =
            ImVec2(static_cast<float>(out_extent.width), static_cast<float>(out_extent.height));
        imgui_backend->new_frame(static_cast<float>(frametime.seconds()));
        frametime.reset();
        imgui_ctx->with_context([&] { draw_overlay(out_extent.width, out_extent.height); });
        imgui_renderer->render(cmd, io[con_out].get_texture(0)->get_view());
    }
}

float4 ErrorPlot::metric_error() const {
    if (metric == ErrorMetric::RMSE) {
        return {std::sqrt(std::max(current_sum.x, 0.0f)), std::sqrt(std::max(current_sum.y, 0.0f)),
                std::sqrt(std::max(current_sum.z, 0.0f)), std::sqrt(std::max(current_sum.w, 0.0f))};
    }
    return current_sum;
}

void ErrorPlot::draw_overlay(const uint32_t width, const uint32_t height) const {
    ImDrawList* dl = ImGui::GetBackgroundDrawList();

    const ImU32 col_text = IM_COL32(255, 255, 255, 255);
    const ImU32 col_shadow = IM_COL32(0, 0, 0, 200);
    const ImU32 col_line = IM_COL32(80, 200, 255, 255);
    const ImU32 col_panel = IM_COL32(0, 0, 0, 110);
    const ImU32 col_border = IM_COL32(255, 255, 255, 60);

    const float margin = 8.0f;
    const float line_height = ImGui::GetTextLineHeightWithSpacing();
    const auto text = [&](const ImVec2& p, const std::string& s) {
        dl->AddText({p.x + 1, p.y + 1}, col_shadow, s.c_str());
        dl->AddText(p, col_text, s.c_str());
    };

    text({margin, margin}, "Reference");
    const char* input_label = "Input";
    const ImVec2 input_size = ImGui::CalcTextSize(input_label);
    text({static_cast<float>(width) - margin - input_size.x, margin}, input_label);

    const float4 e = metric_error();
    text({margin, margin + line_height},
         fmt::format("{}: {:.6f}", enum_to_string(metric), (e.x + e.y + e.z) / 3.0f));
    text({margin, margin + (2 * line_height)},
         fmt::format("R {:.5f}  G {:.5f}  B {:.5f}", e.x, e.y, e.z));

    // plot panel along the bottom
    const ImVec2 p0{margin, static_cast<float>(height) * 0.6f};
    const ImVec2 p1{static_cast<float>(width) - margin, static_cast<float>(height) - margin};
    dl->AddRectFilled(p0, p1, col_panel, 3.0f);
    dl->AddRect(p0, p1, col_border, 3.0f);

    if (history.size() < 2) {
        return;
    }

    const auto y_transform = [&](const float v) {
        return log_y_axis ? std::log10(std::max(v, 1e-9f)) : v;
    };

    float v_min;
    float v_max;
    if (auto_scale) {
        const auto [mn, mx] = std::minmax_element(history.begin(), history.end());
        v_min = *mn;
        v_max = *mx;
    } else {
        v_min = scale_min;
        v_max = scale_max;
    }
    const float t_min = y_transform(v_min);
    const float t_range = std::max(y_transform(v_max) - t_min, 1e-9f);

    const float plot_w = p1.x - p0.x;
    const float plot_h = p1.y - p0.y;
    const std::size_t n = history.size();
    const float log_den = std::max(std::log10(static_cast<float>(n)), 1e-9f);

    std::vector<ImVec2> points;
    points.reserve(n);
    for (std::size_t i = 0; i < n; i++) {
        const float tx = log_x_axis ? std::log10(static_cast<float>(i + 1)) / log_den
                                    : static_cast<float>(i) / static_cast<float>(n - 1);
        const float ty = (y_transform(history[i]) - t_min) / t_range;
        points.push_back({p0.x + (tx * plot_w), p1.y - (ty * plot_h)});
    }
    dl->AddPolyline(points.data(), static_cast<int>(points.size()), col_line, ImDrawFlags_None,
                    1.5f);

    text({p0.x + 2, p0.y + 2}, fmt::format("{:.4g}", v_max));
    text({p0.x + 2, p1.y - ImGui::GetTextLineHeight() - 2}, fmt::format("{:.4g}", v_min));
}

ErrorPlot::NodeStatusFlags ErrorPlot::properties(Properties& config) {
    config.config_enum("Metric", metric, Properties::OptionsStyle::COMBO,
                       "Error metric computed against the reference.");

    if (config.config_uint("History", history_size, "Number of samples shown in the plot.")) {
        history_size = std::max(history_size, 2u);
        while (history.size() > history_size) {
            history.pop_front();
        }
    }
    if (config.config_bool("Reset history")) {
        history.clear();
    }

    config.st_separate("Axes");
    config.config_bool("Logarithmic x-axis", log_x_axis, "Distribute samples on a log10 x-axis.");
    config.config_bool("Logarithmic y-axis", log_y_axis, "Plot values on a log10 y-axis.");

    config.config_bool("Auto scale", auto_scale, "Fit the y-axis range to the data.");
    if (!auto_scale) {
        config.config_float("Min", scale_min, "Lower y-axis bound.", 0.001f);
        config.config_float("Max", scale_max, "Upper y-axis bound.", 0.001f);
    }

    const float4 e = metric_error();
    config.output_text(fmt::format("{}: {:.6f}\nR {:.5f}  G {:.5f}  B {:.5f}",
                                   enum_to_string(metric), (e.x + e.y + e.z) / 3.0f, e.x, e.y,
                                   e.z));

    return {};
}

} // namespace merian
