#include "merian-graph/nodes/video_read/video_read.hpp"

#include "merian-graph/graph/errors.hpp"
#include "merian/vk/extension/extension_registry.hpp"
#include "merian/vk/pipeline/specialization_info_builder.hpp"

#include <algorithm>

namespace merian {

namespace {
constexpr const char* SHADER_MODULE = "merian-graph/nodes/video_read/video_read.slang";
constexpr uint32_t LOCAL_SIZE_X = 16;
constexpr uint32_t LOCAL_SIZE_Y = 16;
// a clock that ran ahead of the stream must not decode the whole file in one iteration
constexpr uint32_t MAX_FRAMES_PER_ITERATION = 8;
} // namespace

VideoRead::VideoRead() {}

VideoRead::~VideoRead() = default;

std::vector<std::string> VideoRead::request_context_extensions() {
    return ExtensionRegistry::get_instance().get_provider_names<VideoDeviceProvider>();
}

DeviceSupportInfo VideoRead::query_device_support(const DeviceSupportQueryInfo& query_info) {
    const auto provider = query_info.extension_container.find_provider<VideoDeviceProvider>(true);
    if (!provider ||
        (provider->get_supported_operations(query_info) & VideoDeviceProvider::DECODE) == 0u) {
        return DeviceSupportInfo{false, "no hardware video decoder"};
    }

    const auto composition = SlangComposition::create();
    composition->add_module_from_path(SHADER_MODULE, true);
    return SlangProgram::create(query_info.compile_context, composition)
        .get()
        ->query_device_support(query_info);
}

void VideoRead::initialize(const ContextHandle& context, const ResourceAllocatorHandle& allocator) {
    this->context = context;
    this->allocator = allocator;
    sampler = allocator->get_sampler_pool()->linear_clamp_to_edge();

    auto spec_builder = SpecializationInfoBuilder();
    spec_builder.add_entry(LOCAL_SIZE_X, LOCAL_SIZE_Y);
    spec_info.set(spec_builder.build());

    kernel.emplace(
        context, allocator, context->get_shader_compile_context(),
        [] {
            const auto composition = SlangComposition::create();
            composition->add_module_from_path(SHADER_MODULE, true);
            return composition;
        },
        spec_info);
}

std::vector<OutputConnectorDescriptor> VideoRead::describe_outputs(const NodeIOLayout& io_layout) {
    events.register_listeners(io_layout);

    if (filename.empty()) {
        throw graph_errors::node_error{"no file set"};
    }

    if (!decoder) {
        try {
            if (!device) {
                device = context->find_provider<VideoDeviceProvider>()->create_video_device(
                    VideoDeviceProvider::DECODE);
            }
            decoder = device->open_decoder(filename);
        } catch (const std::runtime_error& e) {
            throw graph_errors::node_error{e.what()};
        }
        current.reset();
        pending.reset();
        loop_offset = {};
        reached_end = false;
        advanced_iteration.reset();
        pending = decode_next();
    }

    const VideoStreamInfo& stream = decoder->get_stream_info();
    extent = vk::Extent3D{stream.width, stream.height, 1};
    const vk::Format format = overwrite_format != vk::Format::eUndefined
                                  ? overwrite_format
                                  : vk::Format::eR16G16B16A16Sfloat;

    con_out = ManagedVkImageOut::create(format, extent);
    return {{"out", con_out, ConnectorAccess::compute_write}};
}

std::chrono::nanoseconds VideoRead::presentation_time(const VideoFrameHandle& frame) const {
    return frame->get_presentation_time() + loop_offset;
}

bool VideoRead::ensure_pending() {
    if (!pending) {
        pending = decode_next();
    }
    return pending != nullptr;
}

bool VideoRead::advance_one() {
    if (!ensure_pending()) {
        return false;
    }
    current = std::move(pending);
    return true;
}

std::optional<std::chrono::nanoseconds> VideoRead::provide_time() {
    if (!decoder) {
        return std::nullopt;
    }
    if ((playing || !current) && ensure_pending()) {
        return presentation_time(pending);
    }
    // paused or ended: the content time stands still
    return current ? std::optional{presentation_time(current)} : std::nullopt;
}

VideoFrameHandle VideoRead::decode_next() {
    VideoFrameHandle frame = decoder->read_frame();
    if (frame || !loop || !can_loop) {
        return frame;
    }

    // A stream that cannot say how long it is would repeat the same time forever.
    const VideoStreamInfo& stream = decoder->get_stream_info();
    const auto frame_duration = std::chrono::nanoseconds{
        stream.frame_rate > 0 ? static_cast<int64_t>(1e9 / stream.frame_rate) : 0};
    const auto stream_duration =
        (current ? current->get_presentation_time() : stream.duration) + frame_duration;
    if (stream_duration <= std::chrono::nanoseconds{}) {
        SPDLOG_WARN("cannot loop a stream without a duration, holding the last frame");
        can_loop = false;
        return nullptr;
    }

    decoder->seek({});
    frame = decoder->read_frame();
    if (!frame) {
        SPDLOG_WARN("could not restart the stream, holding the last frame");
        can_loop = false;
        return nullptr;
    }

    loop_offset += stream_duration;
    return frame;
}

void VideoRead::seek_to(const std::chrono::nanoseconds& requested) {
    const std::chrono::nanoseconds duration = decoder->get_stream_info().duration;
    const auto time = std::clamp(requested, std::chrono::nanoseconds{},
                                 duration > std::chrono::nanoseconds{} ? duration : requested);
    decoder->seek(time);
    current.reset();
    pending.reset();
    loop_offset = {};
    can_loop = true;
    reached_end = false;
    // the keyframe can be well before the target
    while (advance_one() && presentation_time(current) < time) {
    }
}

void VideoRead::advance_to(const std::chrono::nanoseconds& time) {
    for (uint32_t i = 0; i < MAX_FRAMES_PER_ITERATION; i++) {
        if (!pending) {
            pending = decode_next();
        }
        if (!pending) {
            break;
        }
        if (current && presentation_time(pending) > time) {
            break;
        }
        current = std::move(pending);
    }
}

VideoRead::NodeStatusFlags VideoRead::pre_process(const NodeIO& io, const NodeProcessInfo& info) {
    if (!decoder) {
        return {};
    }

    if (advanced_iteration == info.get_iteration()) {
        return {};
    }
    advanced_iteration = info.get_iteration();

    // a control picks the frame itself, so the clock must not pick another one over it
    bool moved = false;
    if (seek_requested) {
        seek_to(*seek_requested);
        seek_requested.reset();
        moved = true;
    } else if (step_back) {
        step_back = false;
        const double frame_rate = decoder->get_stream_info().frame_rate;
        const auto frame_duration =
            std::chrono::nanoseconds{frame_rate > 0 ? static_cast<int64_t>(1e9 / frame_rate) : 0};
        seek_to(current ? presentation_time(current) - frame_duration : std::chrono::nanoseconds{});
        moved = true;
    } else if (step_forward) {
        step_forward = false;
        moved = advance_one();
    }

    if (playing && !moved) {
        advance_to(info.get_elapsed_duration());
    }
    if (!current) {
        advance_one();
    }
    return {};
}

VideoRead::NodeStatusFlags
VideoRead::process(const NodeIO& io, const NodeProcessInfo& info, Submission& submission) {
    if (!current) {
        return {};
    }
    // hold the frame until this iteration finished on the GPU
    io.frame_data<VideoFrameHandle>() = current;

    submission.add_wait_semaphore(current->get_semaphore(),
                                  vk::PipelineStageFlagBits::eComputeShader,
                                  current->get_wait_value());

    const CommandBufferHandle& cmd = submission.get_cmd();
    const ImageHandle& image = current->get_plane(0)->get_image();
    image->_set_current_layout(current->get_layout());
    cmd->barrier(vk::PipelineStageFlagBits::eTopOfPipe, vk::PipelineStageFlagBits::eComputeShader,
                 image->barrier(vk::ImageLayout::eShaderReadOnlyOptimal, current->get_access(),
                                vk::AccessFlagBits::eShaderRead));

    const ColorConversion& conversion = current->get_color_conversion();
    pc.matrix_0 = glm::vec4{conversion.matrix[0], 0.};
    pc.matrix_1 = glm::vec4{conversion.matrix[1], 0.};
    pc.matrix_2 = glm::vec4{conversion.matrix[2], 0.};
    pc.scale = glm::vec4{conversion.scale, 0.};
    pc.bias = glm::vec4{conversion.bias, decode_transfer ? 1.f : 0.f};

    ShaderCursor cursor = kernel->globals_cursor();
    cursor["luma"] = Texture::create(current->get_plane(0), sampler);
    cursor["chroma"] = Texture::create(current->get_plane(1), sampler);

    const PipelineHandle pipe = kernel->bind(io, info, submission);
    cmd->push_constant(pipe, pc);
    cmd->dispatch((extent.width + LOCAL_SIZE_X - 1) / LOCAL_SIZE_X,
                  (extent.height + LOCAL_SIZE_Y - 1) / LOCAL_SIZE_Y, 1);

    submission.add_signal_semaphore(
        current->get_semaphore(),
        current->take_signal_value(vk::ImageLayout::eShaderReadOnlyOptimal));

    // after the last frame was output, so a listener that stops still gets it
    if (ensure_pending()) {
        reached_end = false;
    } else if (playing && !reached_end) {
        reached_end = true;
        io.send_event("end");
    }
    return {};
}

VideoRead::NodeStatusFlags VideoRead::properties(Properties& config) {
    bool needs_rebuild = false;

    if (config.config_text("path", config_filename, true) && context) {
        filename = context->get_file_loader()->find_file(config_filename).value_or(config_filename);
        decoder.reset();
        can_loop = true;
        needs_rebuild = true;
    }
    config.st_separate("Playback");
    config.config_bool("playing", playing);
    if (config.config_bool("step forward")) {
        step_forward = true;
    }
    if (config.config_bool("step backward")) {
        step_back = true;
    }
    config.config_float("seek (s)", seek_seconds, "", 0.01);
    if (config.config_bool("seek")) {
        seek_requested = std::chrono::duration_cast<std::chrono::nanoseconds>(
            std::chrono::duration<double>(seek_seconds));
    }
    config.config_bool("loop", loop, "Restart the stream at its end.");
    config.config_bool("decode transfer", decode_transfer,
                       "Apply the sRGB inverse transfer curve, so the output is linear.");
    if (config.config_enum("overwrite format", overwrite_format, Properties::OptionsStyle::COMBO,
                           "Undefined outputs 16 bit float.")) {
        needs_rebuild = true;
    }

    needs_rebuild |= events.properties(config);

    if (decoder) {
        const VideoStreamInfo& stream = decoder->get_stream_info();
        config.output_text("codec: {}\nextent: {}x{}\nframe rate: {:.3f}\nduration: {:.3f}s",
                           stream.codec_name, stream.width, stream.height, stream.frame_rate,
                           to_seconds(stream.duration));
        config.output_text("presentation time: {:.3f}s",
                           current ? to_seconds(current->get_presentation_time() + loop_offset)
                                   : 0.);
    }

    if (needs_rebuild) {
        return NEEDS_RECONNECT;
    }
    return {};
}

} // namespace merian
