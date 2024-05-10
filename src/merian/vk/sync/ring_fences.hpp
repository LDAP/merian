#pragma once

#include "merian/vk/context.hpp"
#include <vulkan/vulkan.hpp>

#include "merian/vk/utils/check_result.hpp"

namespace merian {

/**
 *  RingFences recycles a fixed number of fences, provides information in which cycle
 *  we are currently at, and prevents accidental access to a cycle in-flight.
 *
 *  A typical frame would start by "next_cycle_wait_and_get()", which waits for the
 *  requested cycle to be available (i.e. the GPU has finished executing).
 *
 *  You can store additional data for every frame.
 */
template <uint32_t RING_SIZE = 2, typename UserDataType = void*>
class RingFences : public std::enable_shared_from_this<RingFences<RING_SIZE, UserDataType>> {
  public:
    struct RingData {
        vk::Fence fence{};
        UserDataType user_data{};
    };

  public:
    RingFences(RingFences const&) = delete;
    RingFences& operator=(RingFences const&) = delete;
    RingFences() = delete;

    RingFences(const SharedContext& context) : context(context) {
        for (uint32_t i = 0; i < RING_SIZE; i++) {
            vk::FenceCreateInfo fence_create_info{vk::FenceCreateFlagBits::eSignaled};
            ring_data[i].fence = context->device.createFence(fence_create_info);
        }
    }

    ~RingFences() {
        context->device.waitIdle();
        for (uint32_t i = 0; i < RING_SIZE; i++) {
            context->device.destroyFence(ring_data[i].fence);
        }
    }

    // Resets the fences of the whole ring.
    // Use with caution.
    void reset() {
        for (uint32_t i = 0; i < RING_SIZE; i++) {
            reset_fence(ring_data[i]);
        }
    }

    // Returns the RingData for the current cycle.
    // Use next_cycle_wait_and_get once per frame to
    // advance the data.
    RingData& get() {
        return ring_data[current_index];
    }

    // Should be called once per frame.
    // Like set_cycle_wait_and_get(uint32_t cycle) but advances the cycle internally by one
    RingData& next_cycle_wait_and_get() {
        return set_cycle_wait_and_get(current_index + 1);
    }

    // ensures the availability of the passed cycle
    // cycle can be absolute (e.g. current frame number)
    RingData& set_cycle_wait_and_get(uint32_t cycle) {
        current_index = cycle % RING_SIZE;
        RingData& data = ring_data[current_index];

        check_result(context->device.waitForFences(1, &data.fence, VK_TRUE, ~0ULL),
                     "failed waiting for fence");
        reset_fence(data);

        return data;
    }

    // query current cycle index [0, RING_SIZE)
    uint32_t current_cycle_index() const {
        return current_index;
    }

    uint32_t ring_size() const {
        return RING_SIZE;
    }

    // Allows access to the user data of the whole ring.
    // Use with caution and do not change data of in-flight processing.
    UserDataType& get_ring_data(const uint32_t index) {
        assert(index < RING_SIZE);
        return ring_data[index].user_data;
    }

  private:
    void reset_fence(RingData& data) {
        check_result(context->device.resetFences(1, &data.fence), "could not reset fence");
    }

    uint32_t current_index = 0;
    const SharedContext context;
    std::array<RingData, RING_SIZE> ring_data;
};

} // namespace merian
