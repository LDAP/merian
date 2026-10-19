#include "merian/vk/window/host_swapchain.hpp"

#include "merian/vk/extension/extension_resources.hpp"

namespace merian {

HostSwapchain::HostSwapchain(const ContextHandle& context,
                             const HostWindowHandle& window,
                             const std::vector<vk::SurfaceFormatKHR>& preferred_surface_formats)
    : Swapchain(context,
                {vk::Format::eR8G8B8A8Srgb, vk::Format::eR8G8B8A8Unorm},
                {vk::PresentModeKHR::eMailbox},
                1,
                preferred_surface_formats,
                {vk::PresentModeKHR::eMailbox}),
      window(window),
      setup_queue(context->get_queue(vk::QueueFlagBits::eGraphics | vk::QueueFlagBits::eCompute |
                                     vk::QueueFlagBits::eTransfer)),
      allocator(context->get_context_extension<ExtensionResources>()->resource_allocator()),
      copied(TimelineSemaphore::create(context)) {}

HostSwapchain::HostSwapchain(const std::shared_ptr<HostSwapchain>& swapchain)
    : Swapchain(swapchain), window(swapchain->window), setup_queue(swapchain->setup_queue),
      allocator(swapchain->allocator), copied(TimelineSemaphore::create(context)) {}

HostSwapchain::~HostSwapchain() {
    copied->wait(presents);
}

std::optional<std::pair<uint32_t, Swapchain::SyncGroup>>
HostSwapchain::acquire(const vk::Extent2D extent, const uint64_t /*timeout*/) {
    if (extent.width == 0 || extent.height == 0) {
        return std::nullopt;
    }
    const uint32_t max_extent =
        context->get_physical_device()->get_device_limits().maxImageDimension2D;
    const vk::Extent2D image_extent{std::min(extent.width, max_extent),
                                    std::min(extent.height, max_extent)};

    if (!info) {
        if (image) {
            throw needs_recreate{"previous acquire or present failed."};
        }
        create_image(image_extent);
    } else if (image_extent != info->extent) {
        info.reset();
        throw needs_recreate("changed framebuffer size");
    } else if (settings_changed()) {
        info.reset();
        throw needs_recreate("changed settings");
    }

    return std::make_pair(0u, sync_group);
}

void HostSwapchain::present(const QueueHandle& queue, const uint32_t /*image_idx*/) {
    if (!info) {
        throw needs_recreate{"previous acquire or present failed."};
    }

    const uint64_t copied_value = presents + 1;
    const std::array<vk::Semaphore, 2> signal = {**sync_group.read_semaphore, **copied};
    const std::array<uint64_t, 2> signal_values = {0, copied_value};
    const vk::TimelineSemaphoreSubmitInfo timeline{{}, signal_values};

    CommandPoolHandle pool;
    while (!copies.empty() && copied->get_counter_value() >= copies.front().first) {
        pool = copies.front().second->get_pool();
        copies.pop_front();
    }
    if (pool) {
        pool->reset();
    } else {
        pool = CommandPool::create(queue);
    }
    const CommandBufferHandle cmd = CommandBuffer::create(pool);
    cmd->begin();
    const HostCopy copy = cmd_copy_to_host(cmd, frame_buffer(), image);
    cmd->end();

    queue->submit({cmd}, VK_NULL_HANDLE, signal, {**sync_group.written_semaphore},
                  {vk::PipelineStageFlagBits::eTransfer}, timeline);
    copies.emplace_back(copied_value, cmd);
    presents = copied_value;
    window->add_presented(copy, copied, copied_value);
}

SwapchainHandle HostSwapchain::recreate() {
    return std::make_shared<HostSwapchain>(
        std::static_pointer_cast<HostSwapchain>(shared_from_this()));
}

uint32_t HostSwapchain::get_max_image_count() {
    return UINT32_MAX;
}

vk::ImageLayout HostSwapchain::get_present_layout() const {
    return vk::ImageLayout::eTransferSrcOptimal;
}

BufferHandle HostSwapchain::frame_buffer() {
    for (const BufferHandle& buffer : frame_buffers) {
        if (buffer.use_count() == 1 && buffer->get_memory().use_count() == 1) {
            return buffer;
        }
    }
    const vk::Extent3D extent = image->get_extent();
    const vk::DeviceSize size =
        Image::format_size(image->get_format()) * extent.width * extent.height;
    return frame_buffers.emplace_back(allocator->create_buffer(
        vk::BufferCreateInfo{{}, size, vk::BufferUsageFlagBits::eTransferDst},
        MemoryMappingType::HOST_ACCESS_RANDOM, "host swapchain frame"));
}

void HostSwapchain::create_image(const vk::Extent2D extent) {
    SwapchainInfo image_info;
    image_info.extent = extent;
    image_info.min_images = 1;
    image_info.present_mode = new_present_mode;
    image_info.surface_format = new_surface_format;
    image_info.image_create_info = vk::ImageCreateInfo{
        {},
        vk::ImageType::e2D,
        new_surface_format.format,
        vk::Extent3D(extent, 1),
        1,
        1,
        vk::SampleCountFlagBits::e1,
        vk::ImageTiling::eOptimal,
        vk::ImageUsageFlagBits::eColorAttachment | vk::ImageUsageFlagBits::eTransferDst |
            vk::ImageUsageFlagBits::eTransferSrc,
    };

    image = allocator->create_image(image_info.image_create_info, MemoryMappingType::NONE,
                                    "host swapchain image");
    sync_group.read_semaphore = BinarySemaphore::create(context);
    sync_group.written_semaphore = BinarySemaphore::create(context);
    image_info.images.emplace_back(**image);

    // the present layout and a signaled read semaphore, as after a present
    const CommandPoolHandle setup_pool = CommandPool::create(setup_queue);
    const CommandBufferHandle setup_cmd = CommandBuffer::create(setup_pool);
    setup_cmd->begin();
    setup_cmd->barrier(image->barrier2(get_present_layout(), true));
    setup_cmd->end();
    setup_queue->submit_wait({setup_cmd}, VK_NULL_HANDLE, {**sync_group.read_semaphore});

    info = std::move(image_info);
}

} // namespace merian
