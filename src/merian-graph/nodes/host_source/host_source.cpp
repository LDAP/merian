#include "merian-graph/nodes/host_source/host_source.hpp"

namespace merian {

void HostSource::initialize([[maybe_unused]] const ContextHandle& context,
                            const ResourceAllocatorHandle& allocator) {
    staging = allocator->get_staging();
}

HostSource::NodeStatusFlags HostSource::pre_process([[maybe_unused]] const NodeIO& io,
                                                    [[maybe_unused]] const NodeProcessInfo& info) {
    const std::scoped_lock lock(mutex);
    return layout_changed ? NodeStatusFlags{NEEDS_RECONNECT} : NodeStatusFlags{};
}

HostSource::NodeStatusFlags HostSource::process(const NodeIO& io,
                                                [[maybe_unused]] const NodeProcessInfo& info,
                                                Submission& submission) {
    const std::scoped_lock lock(mutex);
    if (!pending || layout_changed) {
        return {};
    }
    cmd_copy(submission.get_cmd(), io, current.buffer, current.offset);
    pending = false;
    return {};
}

void HostSource::create_output(const std::function<void()>& create) {
    const std::scoped_lock lock(mutex);
    create();
    pending = true;
    layout_changed = false;
}

void HostSource::stage_upload(const vk::DeviceSize size,
                              const std::function<void(void* data)>& fill,
                              const std::function<bool()>& update_layout) {
    Staged staged;
    staged.memory = staging->get_upload_staging_space(size, staged.buffer, staged.offset);
    fill(staged.memory->map());
    staged.memory->flush();
    staged.memory->unmap();

    const std::scoped_lock lock(mutex);
    layout_changed |= update_layout();
    current = std::move(staged);
    pending = true;
}

} // namespace merian
