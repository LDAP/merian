#include "merian-graph/nodes/video_write/video_write.hpp"

#include "merian/vk/extension/extension_registry.hpp"
#include "merian/vk/pipeline/specialization_info_builder.hpp"

#include <fmt/args.h>

namespace merian {

namespace {
constexpr const char* SHADER_MODULE = "merian-graph/nodes/video_write/video_write.slang";
constexpr uint32_t LOCAL_SIZE_X = 16;
constexpr uint32_t LOCAL_SIZE_Y = 16;
} // namespace

VideoWrite::VideoWrite() {}

VideoWrite::~VideoWrite() = default;

std::vector<std::string> VideoWrite::request_context_extensions() {
    return ExtensionRegistry::get_instance().get_provider_names<VideoDeviceProvider>();
}

DeviceSupportInfo VideoWrite::query_device_support(const DeviceSupportQueryInfo& query_info) {
    const auto provider = query_info.extension_container.find_provider<VideoDeviceProvider>(true);
    if (!provider ||
        (provider->get_supported_operations(query_info) & VideoDeviceProvider::ENCODE) == 0u) {
        return DeviceSupportInfo{false, "no hardware video encoder"};
    }

    const auto composition = SlangComposition::create();
    composition->add_module_from_path(SHADER_MODULE, true);
    return SlangProgram::create(query_info.compile_context, composition)
        .get()
        ->query_device_support(query_info);
}

void VideoWrite::initialize(const ContextHandle& context,
                            const ResourceAllocatorHandle& allocator) {
    this->context = context;
    this->allocator = allocator;
    if (const auto provider = context->find_provider<VideoDeviceProvider>(true)) {
        codecs = provider->get_encoder_codecs();
    }

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

std::vector<InputConnectorDescriptor> VideoWrite::describe_inputs() {
    return {{"src", con_src, ConnectorAccess::compute_read}};
}

VideoWrite::NodeStatusFlags
VideoWrite::on_connected(const NodeIOLayout& io_layout,
                         [[maybe_unused]] const NodeIO& io,
                         [[maybe_unused]] const NodeConnectionInfo& info,
                         [[maybe_unused]] Submission& submission) {
    events.register_listeners(io_layout);

    // the topology a pre_process ran on can still be discarded, so the resolution is taken here
    const vk::Extent3D connected = io_layout[con_src]->get_create_info_or_throw().extent;
    const bool changed = record_enable && connected != extent;
    const bool started = changed && encoder->get_frame_count() > 0;
    extent = connected;
    if (started) {
        SPDLOG_WARN("stopping the recording, the input resolution changed");
        stop();
    } else if (changed) {
        record();
    }
    return {};
}

bool VideoWrite::provides_time() const {
    return time_provider != TimeProviderMode::OFF;
}

std::optional<std::chrono::nanoseconds> VideoWrite::provide_time() {
    // a start requested for this run counts as active, else the run that starts the recording
    // would still be encoded at whatever time the previous clock was at
    if (time_provider == TimeProviderMode::WHEN_ACTIVE && !record_enable && !start_stop_record) {
        // the timeline starts at zero once this node starts providing
        provided_frames = 0;
        return std::nullopt;
    }
    if (!record_enable) {
        provided_frametime_millis = frametime_millis;
    }
    return std::chrono::duration_cast<std::chrono::nanoseconds>(std::chrono::duration<double>(
        static_cast<double>(provided_frames++) * provided_frametime_millis / 1000.));
}

void VideoWrite::record() {
    stop();
    if (codecs.empty()) {
        SPDLOG_ERROR("no hardware video encoder available");
        return;
    }

    fmt::dynamic_format_arg_store<fmt::format_context> arg_store;
    arg_store.push_back(fmt::arg("index", num_recordings));
    arg_store.push_back(fmt::arg("width", extent.width));
    arg_store.push_back(fmt::arg("height", extent.height));
    arg_store.push_back(fmt::arg("framerate", framerate));

    const std::string& codec_name = codecs[std::min<std::size_t>(codec_index, codecs.size() - 1)];
    try {
        recording_path =
            std::filesystem::absolute(fmt::vformat(filename_format, arg_store) +
                                      (codec_name.starts_with("av1") ? ".mkv" : ".mp4"));
    } catch (const std::exception& e) {
        SPDLOG_ERROR("{}", e.what());
        return;
    }

    try {
        if (!device) {
            device = context->find_provider<VideoDeviceProvider>()->create_video_device(
                VideoDeviceProvider::ENCODE);
        }
        encoder = device->open_encoder(
            recording_path, VideoEncoderInfo{extent.width, extent.height, framerate, codec_name,
                                             bit_rate_kbit * 1000LL, high_bit_depth});
    } catch (const std::runtime_error& e) {
        SPDLOG_ERROR("could not start recording: {}", e.what());
        return;
    }
    num_recordings++;
    last_encoded_time.reset();
    submitted_frames = 0;
    record_enable = true;
}

void VideoWrite::stop() {
    if (encoder) {
        const bool empty = encoder->get_frame_count() == 0;
        encoder->finish();
        encoder.reset();
        if (empty) {
            // nothing was written, let the next recording reuse the name
            num_recordings--;
        }
    }
    record_enable = false;
}

VideoWrite::NodeStatusFlags VideoWrite::pre_process(const NodeIO& io, const NodeProcessInfo& info) {
    // provide_time runs before this, so a start has to be known one run ahead of it
    if (!record_enable && start_at_run >= 0 &&
        static_cast<int64_t>(info.get_iteration()) + 1 == start_at_run) {
        start_stop_record = true;
    }

    if (!record_enable &&
        (start_stop_record || static_cast<int64_t>(info.get_iteration()) == start_at_run)) {
        record();
        start_stop_record = false;
        if (record_enable) {
            io.send_event("start");
        }
    } else if (record_enable &&
               (start_stop_record ||
                (stop_after_num_frames >= 0 &&
                 submitted_frames >= static_cast<uint64_t>(stop_after_num_frames)))) {
        stop();
        start_stop_record = false;
        io.send_event("stop");
    }
    return {};
}

VideoWrite::NodeStatusFlags
VideoWrite::process(const NodeIO& io, const NodeProcessInfo& info, Submission& submission) {
    if (!record_enable) {
        return {};
    }

    // a frozen clock means the content did not advance: a paused or finished reader, or a still
    if (last_encoded_time == info.get_elapsed_duration()) {
        return {};
    }

    const VideoFrameHandle frame = encoder->acquire_frame();
    if (!frame) {
        SPDLOG_WARN("dropping a frame, the encoder has none free");
        return {};
    }

    submission.add_wait_semaphore(frame->get_semaphore(), vk::PipelineStageFlagBits::eComputeShader,
                                  frame->get_wait_value());

    const CommandBufferHandle& cmd = submission.get_cmd();
    const ImageHandle& image = frame->get_plane(0)->get_image();
    cmd->barrier(vk::PipelineStageFlagBits::eTopOfPipe, vk::PipelineStageFlagBits::eComputeShader,
                 image->barrier(vk::ImageLayout::eGeneral, frame->get_access(),
                                vk::AccessFlagBits::eShaderWrite));

    const ColorConversion& conversion = frame->get_inverse_color_conversion();
    pc.matrix_0 = glm::vec4{conversion.matrix[0], 0.};
    pc.matrix_1 = glm::vec4{conversion.matrix[1], 0.};
    pc.matrix_2 = glm::vec4{conversion.matrix[2], 0.};
    pc.scale = glm::vec4{conversion.scale, 0.};
    pc.bias = glm::vec4{conversion.bias, encode_transfer ? 1.f : 0.f};

    ShaderCursor cursor = kernel->globals_cursor();
    cursor["luma"] = frame->get_plane(0);
    cursor["chroma"] = frame->get_plane(1);

    const PipelineHandle pipe = kernel->bind(io, info, submission);
    cmd->push_constant(pipe, pc);
    cmd->dispatch(((extent.width + 1) / 2 + LOCAL_SIZE_X - 1) / LOCAL_SIZE_X,
                  ((extent.height + 1) / 2 + LOCAL_SIZE_Y - 1) / LOCAL_SIZE_Y, 1);

    submission.add_signal_semaphore(frame->get_semaphore(),
                                    frame->take_signal_value(vk::ImageLayout::eGeneral));

    last_encoded_time = info.get_elapsed_duration();
    submitted_frames++;
    submission.sync_to_cpu([encoder = encoder, frame] { encoder->submit_frame(frame); });

    return {};
}

VideoWrite::NodeStatusFlags VideoWrite::properties(Properties& config) {
    std::ignore = config.config_text("filename", filename_format, false,
                                     "Format string, variables: index, width, height, framerate.");
    config.config_options("codec", codec_index, codecs, Properties::OptionsStyle::COMBO);
    config.config_int("bit rate (kbit/s)", bit_rate_kbit,
                      "0 lets the encoder choose its default rate control.");
    bit_rate_kbit = std::max(bit_rate_kbit, 0);
    config.config_bool("10 bit", high_bit_depth);
    config.config_bool("encode transfer", encode_transfer,
                       "Apply the sRGB transfer curve, so a linear input is encoded correctly.");

    config.st_separate();
    bool prop_record_enable = record_enable;
    start_stop_record |= config.config_bool("enable", prop_record_enable);
    config.config_enum("time provider", time_provider, Properties::OptionsStyle::COMBO,
                       TIME_PROVIDER_DESCRIPTION);
    config.config_float("framerate", framerate, "", 0.01);
    framerate = std::max(framerate, 0.01f);
    frametime_millis = 1000.f / framerate;
    config.config_float("frametime", frametime_millis, "", 0.01);
    frametime_millis = std::max(frametime_millis, 0.01f);
    framerate = 1000.f / frametime_millis;
    config.config_int("start at run", start_at_run,
                      "Starts recording at the given run. Use it to let a graph that changes its "
                      "resolution over the first iterations settle first. -1 to disable.");
    config.config_int("stop after number frames", stop_after_num_frames,
                      "Finishes the file after the given number of frames. -1 to disable.");

    const bool needs_reconnect = events.properties(config);

    if (encoder) {
        config.output_text("recording: {}\nframes: {}", recording_path.string(),
                           encoder->get_frame_count());
    }

    return needs_reconnect ? NEEDS_RECONNECT : NodeStatusFlags{};
}

} // namespace merian
