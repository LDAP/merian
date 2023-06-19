#pragma once

#include "merian/vk/context.hpp"
#include "merian/vk/utils/profiler.hpp"
#include <vulkan/vulkan.hpp>

namespace merian {

/**
 * @brief      Provides a profiler for every frame-in-flight.
 *
 * In GPU processing there are often multiple frames-in-flight.
 * Waiting for the frame to be finished to collect the timestamps would flush the pipeline.
 * Use a profiler for every frame-in-flight instead, and collect the results after a few iterations.
 * 
 * Intended use is together with RingFence of equal (or smaller) RING_SIZE:
 * while (True) {
 *     vk::Fence fence = ring_fence.wait_and_get_fence()
 *     // now the timestamps from iteration i - RING_SIZE are ready
 *     
 *     ring_profiler.set_cycle();
 *     //get the profiler for the current iteration
 *     profiler = ring_profiler.get_profiler();
 *     // collects the results of iteration i - RING_SIZE from the gpu
 *     // and resets the query pool
 *     profiler.collect();
 *     // print, display results...
 *      
 *     profiler.reset(cmd);
 *     // use profiler...
 *     
 *     queue.submit(... fence);
 * }
 * 
 */
template<uint32_t RING_SIZE>
class RingProfiler {

public:
    RingProfiler(const SharedContext context, const uint32_t num_gpu_timers = 1028) {
        for (uint32_t i = 0; i < RING_SIZE; i++) {
            profilers.push_back(Profiler::make(context, num_gpu_timers));
        }
    }

    void set_cycle() {
        set_cycle(cycle + 1);
    }

    void set_cycle(uint32_t cycle) {
        this->cycle = cycle;
    }

    ProfilerHandle get_profiler() {
        return profilers[cycle & RING_SIZE];
    }

private:
    std::vector<ProfilerHandle> profilers;
    uint32_t cycle = 0;

};

} // namespace merian
