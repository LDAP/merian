#include "ffmpeg_video.hpp"

#include <spdlog/spdlog.h>

#include <unordered_set>

namespace merian {

namespace {

// FFmpeg never destroys the image views of the encoder pictures that are still referenced when it
// closes: ff_hw_base_encode_close frees them without the codec's own op->free, and its tail
// picture can never retire because the retire loop skips the last entry.
//
// Vulkan entry points carry no user data, so the record has to be static.
struct ImageViewRecord {
    std::mutex mutex;
    std::unordered_set<VkImageView> views;
    // Sweeping is only safe once no FFmpeg device is left that could still own a view.
    uint32_t devices = 0;
};

ImageViewRecord& image_view_record() {
    static ImageViewRecord record;
    return record;
}

VKAPI_ATTR VkResult VKAPI_CALL recorded_create_image_view(VkDevice device,
                                                          const VkImageViewCreateInfo* create_info,
                                                          const VkAllocationCallbacks* allocator,
                                                          VkImageView* view) {
    const VkResult result =
        VULKAN_HPP_DEFAULT_DISPATCHER.vkCreateImageView(device, create_info, allocator, view);
    if (result == VK_SUCCESS) {
        ImageViewRecord& record = image_view_record();
        const std::lock_guard lock{record.mutex};
        record.views.emplace(*view);
    }
    return result;
}

VKAPI_ATTR void VKAPI_CALL recorded_destroy_image_view(VkDevice device,
                                                       VkImageView view,
                                                       const VkAllocationCallbacks* allocator) {
    {
        ImageViewRecord& record = image_view_record();
        const std::lock_guard lock{record.mutex};
        record.views.erase(view);
    }
    VULKAN_HPP_DEFAULT_DISPATCHER.vkDestroyImageView(device, view, allocator);
}

} // namespace

std::string av_error(const int code) {
    char message[AV_ERROR_MAX_STRING_SIZE];
    av_strerror(code, message, sizeof(message));
    return message;
}

std::vector<const char*> ffmpeg_device_extensions(const PhysicalDeviceHandle& physical_device) {
    int count = 0;
    const char** names = av_vk_get_optional_device_extensions(&count);
    std::vector<const char*> extensions;
    for (int i = 0; i < count; i++) {
        // the context decides on its own whether the device is a portability driver
        if (std::string_view{names[i]} != "VK_KHR_portability_subset" &&
            physical_device->extension_supported(names[i])) {
            extensions.emplace_back(names[i]);
        }
    }
    av_free(names);
    return extensions;
}

FFmpegVideoDevice::FFmpegVideoDevice(const ContextHandle& context,
                                     const VideoDeviceProvider::OperationFlags operations)
    : context(context),
      queue_GCT(context->get_queue(vk::QueueFlagBits::eGraphics | vk::QueueFlagBits::eCompute |
                                   vk::QueueFlagBits::eTransfer)) {

    const QueueAssignment& queues = context->get_queue_info();
    const std::optional<QueueInfo> decode = queues.get(vk::QueueFlagBits::eVideoDecodeKHR);
    const std::optional<QueueInfo> encode = queues.get(vk::QueueFlagBits::eVideoEncodeKHR);
    if (((operations & VideoDeviceProvider::DECODE) != 0u && !decode) ||
        ((operations & VideoDeviceProvider::ENCODE) != 0u && !encode)) {
        throw std::runtime_error{"the context has no video queue for the requested operations"};
    }
    if (!queue_GCT) {
        throw std::runtime_error{"the context has no graphics queue"};
    }

    hw_device_context = av_hwdevice_ctx_alloc(AV_HWDEVICE_TYPE_VULKAN);
    if (hw_device_context == nullptr) {
        throw std::runtime_error{"could not allocate a Vulkan hardware device context"};
    }

    auto* device_context = reinterpret_cast<AVHWDeviceContext*>(hw_device_context->data);
    device_context->user_opaque = this;
    auto* vulkan_context = static_cast<AVVulkanDeviceContext*>(device_context->hwctx);

    vulkan_context->get_proc_addr = get_instance_proc_addr;
    vulkan_context->inst = **context->get_instance();
    vulkan_context->phys_dev = context->get_physical_device()->get_physical_device();
    vulkan_context->act_dev = **context->get_device();

    enabled_features = context->get_device()->get_enabled_features();
    vulkan_context->device_features = vk::PhysicalDeviceFeatures2{
        enabled_features.get_features(),
        enabled_features.build_chain_for_device_creation(context->get_physical_device()),
    };

    for (const std::string& extension : context->get_device()->get_enabled_extensions()) {
        enabled_extensions.emplace_back(extension.c_str());
    }
    vulkan_context->enabled_dev_extensions = enabled_extensions.data();
    vulkan_context->nb_enabled_dev_extensions = static_cast<int>(enabled_extensions.size());

    using Chain = vk::StructureChain<vk::QueueFamilyProperties2, vk::QueueFamilyVideoPropertiesKHR>;
    const std::vector<Chain> properties =
        context->get_physical_device()->get_physical_device().getQueueFamilyProperties2<Chain>();

    const auto add_queue_family = [&](const QueueInfo& queue, const vk::QueueFlagBits flags) {
        vulkan_context->qf[vulkan_context->nb_qf++] = AVVulkanDeviceQueueFamily{
            .idx = static_cast<int>(queue.family_index),
            .num = static_cast<int>(queue.queue_indices.size()),
            .flags = static_cast<VkQueueFlagBits>(flags),
            .video_caps = static_cast<VkVideoCodecOperationFlagBitsKHR>(
                static_cast<uint32_t>(properties[queue.family_index]
                                          .get<vk::QueueFamilyVideoPropertiesKHR>()
                                          .videoCodecOperations)),
        };
    };
    if ((operations & VideoDeviceProvider::DECODE) != 0u) {
        add_queue_family(*decode, vk::QueueFlagBits::eVideoDecodeKHR);
    }
    if ((operations & VideoDeviceProvider::ENCODE) != 0u) {
        add_queue_family(*encode, vk::QueueFlagBits::eVideoEncodeKHR);
    }
    // FFmpeg needs a compute and a transfer family and always takes queue index 0, which is the
    // queue merian submits to. Listing it also shares the frames with it concurrently.
    vulkan_context->qf[vulkan_context->nb_qf++] = AVVulkanDeviceQueueFamily{
        .idx = static_cast<int>(queue_GCT->get_queue_family_index()),
        .num = 1,
        .flags = static_cast<VkQueueFlagBits>(VK_QUEUE_GRAPHICS_BIT | VK_QUEUE_COMPUTE_BIT |
                                              VK_QUEUE_TRANSFER_BIT),
        .video_caps = VK_VIDEO_CODEC_OPERATION_NONE_KHR,
    };

#if FF_API_VULKAN_SYNC_QUEUES
    vulkan_context->lock_queue = lock_queue;
    vulkan_context->unlock_queue = unlock_queue;
#else
#error "FFmpeg dropped lock_queue; the shared graphics queue is no longer serialized"
#endif

    const int result = av_hwdevice_ctx_init(hw_device_context);
    if (result < 0) {
        av_buffer_unref(&hw_device_context);
        throw std::runtime_error{fmt::format(
            "could not initialize the Vulkan hardware device context: {}", av_error(result))};
    }

    ImageViewRecord& record = image_view_record();
    const std::lock_guard lock{record.mutex};
    record.devices++;
}

FFmpegVideoDevice::~FFmpegVideoDevice() {
    av_buffer_unref(&hw_device_context);

    ImageViewRecord& record = image_view_record();
    const std::lock_guard lock{record.mutex};
    if (--record.devices > 0 || record.views.empty()) {
        return;
    }
    SPDLOG_DEBUG("destroying {} image views ffmpeg left behind", record.views.size());
    for (const VkImageView view : record.views) {
        VULKAN_HPP_DEFAULT_DISPATCHER.vkDestroyImageView(**context->get_device(), view, nullptr);
    }
    record.views.clear();
}

PFN_vkVoidFunction VKAPI_CALL FFmpegVideoDevice::get_instance_proc_addr(VkInstance instance,
                                                                        const char* name) {
    if (std::string_view{name} == "vkGetDeviceProcAddr") {
        return reinterpret_cast<PFN_vkVoidFunction>(get_device_proc_addr);
    }
    return VULKAN_HPP_DEFAULT_DISPATCHER.vkGetInstanceProcAddr(instance, name);
}

PFN_vkVoidFunction VKAPI_CALL FFmpegVideoDevice::get_device_proc_addr(VkDevice device,
                                                                      const char* name) {
    const std::string_view function{name};
    if (function == "vkCreateImageView") {
        return reinterpret_cast<PFN_vkVoidFunction>(recorded_create_image_view);
    }
    if (function == "vkDestroyImageView") {
        return reinterpret_cast<PFN_vkVoidFunction>(recorded_destroy_image_view);
    }
    return VULKAN_HPP_DEFAULT_DISPATCHER.vkGetDeviceProcAddr(device, name);
}

VideoDecoderHandle FFmpegVideoDevice::open_decoder(const std::filesystem::path& path) const {
    return std::make_shared<FFmpegVideoDecoder>(
        std::const_pointer_cast<FFmpegVideoDevice>(shared_from_this()), path);
}

VideoEncoderHandle FFmpegVideoDevice::open_encoder(const std::filesystem::path& path,
                                                   const VideoEncoderInfo& info) const {
    return std::make_shared<FFmpegVideoEncoder>(
        std::const_pointer_cast<FFmpegVideoDevice>(shared_from_this()), path, info);
}

void FFmpegVideoDevice::lock_queue(AVHWDeviceContext* device_context,
                                   const uint32_t queue_family,
                                   const uint32_t /*index*/) {
    auto* self = static_cast<FFmpegVideoDevice*>(device_context->user_opaque);
    if (queue_family == self->queue_GCT->get_queue_family_index()) {
        self->queue_GCT->lock();
    }
}

void FFmpegVideoDevice::unlock_queue(AVHWDeviceContext* device_context,
                                     const uint32_t queue_family,
                                     const uint32_t /*index*/) {
    auto* self = static_cast<FFmpegVideoDevice*>(device_context->user_opaque);
    if (queue_family == self->queue_GCT->get_queue_family_index()) {
        self->queue_GCT->unlock();
    }
}

} // namespace merian
