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
 *  requested cycle to be available (i.e. the GPU has finished executing) and resets the fence to be
 *  signaled again.
 *
 *  You can store additional data for every frame.
 */
template <typename UserDataType = void*>
class RingFences : public std::enable_shared_from_this<RingFences<UserDataType>> {
  public:
    struct RingData {
        vk::Fence fence{};
        UserDataType user_data{};
    };

  public:
    RingFences(RingFences const&) = delete;
    RingFences& operator=(RingFences const&) = delete;
    RingFences() = delete;

    RingFences(
        const ContextHandle& context,
        uint32_t ring_size = 2,
        const std::function<UserDataType(const uint32_t index)>& user_data_initializer =
            [](const uint32_t /*index*/) -> UserDataType { return UserDataType(); })
        : context(context), user_data_initializer(user_data_initializer) {

        resize(ring_size);
    }

    ~RingFences() {
        resize(0);
    }

    // Elements are erased starting from the current index. All new indices will be initialized
    // using the provided user_data_initializer.
    void resize(const uint32_t ring_size) {
        const vk::FenceCreateInfo fence_create_info{vk::FenceCreateFlagBits::eSignaled};
        while (ring_data.size() < ring_size) {
            ring_data.emplace_back(context->get_device()->get_device().createFence(fence_create_info),
                                   user_data_initializer(ring_data.size()));
        }

        while (ring_size < ring_data.size()) {
            check_result(
                context->get_device()->get_device().waitForFences(1, &ring_data[current_index].fence, VK_TRUE, ~0ULL),
                "failed waiting for fence");
            context->get_device()->get_device().destroyFence(ring_data[current_index].fence);
            ring_data.erase(ring_data.begin() + current_index);

            if (current_index >= ring_data.size()) {
                current_index = 0;
            }
        }

        assert(ring_data.size() == ring_size);
    }

    // Resets the fence of the current iteration ring and returns the fence.
    // For example, use it together with *_cycle_wait_get().
    const vk::Fence& reset() const {
        reset_fence(ring_data[current_index]);
        return ring_data[current_index].fence;
    }

    // Returns the RingData for the current cycle.
    // Use next_cycle_wait_and_get once per frame to
    // advance the data.
    RingData& get() {
        return ring_data[current_index];
    }

    // Allows access to the user data of the whole ring.
    // Use with caution and do not change data of in-flight processing.
    RingData& get(const uint32_t index) {
        assert(index < size());
        return ring_data[index];
    }

    // Should be called once per frame.
    // Like set_cycle_wait_and_get(uint32_t cycle) but advances the cycle internally by one
    RingData& next_cycle_wait_reset_get() {
        return set_cycle_wait_reset_get(current_index + 1);
    }

    // ensures the availability of the passed cycle
    // cycle can be absolute (e.g. current frame number)
    RingData& set_cycle_wait_reset_get(uint32_t cycle) {
        current_index = cycle % size();
        RingData& data = ring_data[current_index];
        check_result(context->get_device()->get_device().waitForFences(1, &data.fence, VK_TRUE, ~0ULL),
                     "failed waiting for fence");
        reset_fence(data);
        return data;
    }

    // Advances the cycle, waits for the cycle to be available and returns the ring data.
    // reset() has to be manually called.
    UserDataType& next_cycle_wait_get() {
        return set_cycle_wait_get(current_index + 1);
    }

    // Advances the cycle, waits for the cycle to be available and returns the ring data.
    // reset() has to be manually called.
    UserDataType& next_cycle_wait_get(bool& did_wait) {
        return set_cycle_wait_get(current_index + 1, did_wait);
    }

    // Sets cycle, waits for the cycle to be available and returns the ring data.
    // reset() has to be manually called.
    UserDataType& set_cycle_wait_get(uint32_t cycle) {
        current_index = cycle % size();
        RingData& data = ring_data[current_index];
        check_result(context->get_device()->get_device().waitForFences(1, &data.fence, VK_TRUE, ~0ULL),
                     "failed waiting for fence");
        return data.user_data;
    }

    // Sets cycle, waits for the cycle to be available and returns the ring data.
    // reset() has to be manually called.
    UserDataType& set_cycle_wait_get(uint32_t cycle, bool& did_wait) {
        current_index = cycle % size();
        RingData& data = ring_data[current_index];

        const vk::Result status = context->get_device()->get_device().getFenceStatus(data.fence);
        if (status == vk::Result::eSuccess) {
            did_wait = false;
            return data.user_data;
        }

        did_wait = true;
        check_result(context->get_device()->get_device().waitForFences(1, &data.fence, VK_TRUE, ~0ULL),
                     "failed waiting for fence");
        return data.user_data;
    }

    // query current cycle index [0, RING_SIZE)
    uint32_t current_cycle_index() const {
        return current_index;
    }

    uint32_t size() const {
        return ring_data.size();
    }

    uint32_t count_waiting() const {
        uint32_t waiting = 0;

        for (uint32_t i = 0; i < size(); i++) {
            waiting += context->get_device()->get_device().getFenceStatus(ring_data[i].fence) == vk::Result::eNotReady;
        }

        return waiting;
    }

    void wait_all() {
        for (uint32_t i = 0; i < size(); i++) {
            check_result(context->get_device()->get_device().waitForFences(1, &ring_data[i].fence, VK_TRUE, ~0ULL),
                         "failed waiting for fence");
        }
    }

  private:
    void reset_fence(const RingData& data) const {
        check_result(context->get_device()->get_device().resetFences(1, &data.fence), "could not reset fence");
    }

    const ContextHandle context;
    const std::function<UserDataType(const uint32_t index)> user_data_initializer;

    uint32_t current_index = 0;
    std::vector<RingData> ring_data;
};

} // namespace merian
