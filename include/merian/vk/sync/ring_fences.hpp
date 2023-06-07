#pragma once

#include "merian/vk/context.hpp"
#include <vulkan/vulkan.hpp>

namespace merian {

/**
  RingFences recycles a fixed number of fences, provides information in which cycle
  we are currently at, and prevents accidental access to a cycle in-flight.

  A typical frame would start by "wait_and_get_fence()", which waits for the
  requested cycle to be available.
*/
template <uint32_t RING_SIZE = 3> class RingFences : public std::enable_shared_from_this<RingFences<RING_SIZE>> {
  private:
    struct Entry {
        vk::Fence fence;
        bool active;
    };

  public:
    RingFences(RingFences const&) = delete;
    RingFences& operator=(RingFences const&) = delete;
    RingFences() = delete;

    RingFences(const SharedContext& context) : context(context) {
        fences.assign(RING_SIZE, {{}, false});
        for (uint32_t i = 0; i < RING_SIZE; i++) {
            vk::FenceCreateInfo fence_create_info;
            fences[i].fence = context->device.createFence(fence_create_info);
        }
    }

    ~RingFences() {
        for (uint32_t i = 0; i < RING_SIZE; i++) {
            context->device.destroyFence(fences[i].fence);
        }
    }

    void reset() {
        for (uint32_t i = 0; i < RING_SIZE; i++) {
            reset_fence(fences[i]);
        }
    }

    // Like wait_and_get_fence(uint32_t cycle) but advances the cycle internally by one
    vk::Fence& wait_and_get_fence() {
        return wait_and_get_fence(current_index + 1);
    }

    // ensures the availability of the passed cycle
    // cycle can be absolute (e.g. current frame number)
    vk::Fence& wait_and_get_fence(uint32_t cycle) {
        current_index = cycle % RING_SIZE;

        Entry& entry = fences[current_index];

        if (entry.active) {
            // ensure the cycle we will use now has completed
            check_result(context->device.waitForFences(1, &entry.fence, VK_TRUE, ~0ULL), "failed waiting for fence");
            reset_fence(entry);
        }
    }

    // query current cycle index
    uint32_t current_cycle_index() const {
        return current_index;
    }

    uint32_t ring_size() const {
        return RING_SIZE;
    }

  private:
    void reset_fence(Entry& entry) {
        entry.active = false;
        check_result(context->device.resetFences(1, &entry.fence), "could not reset fence");
    }

    uint32_t current_index = 0;
    const SharedContext context;
    std::vector<Entry> fences;
};

} // namespace merian
