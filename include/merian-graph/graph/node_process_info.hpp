#pragma once

#include "merian/shader/glsl_shader_compiler.hpp"
#include "merian/shader/shader_object_allocator.hpp"
#include "merian/utils/chrono.hpp"
#include "merian/utils/concurrent/thread_pool.hpp"
#include "merian/vk/memory/resource_allocator.hpp"
#include "merian/vk/sync/semaphore_timeline.hpp"
#include "merian/vk/utils/cpu_queue.hpp"
#include "merian/vk/utils/profiler.hpp"

#include <cstdint>
#include <string>
#include <utility>
#include <vector>

namespace merian {

// Per-iteration context the graph supplies to Node::pre_process and Node::process.
class NodeProcessInfo {
    friend class Graph;

  public:
    NodeProcessInfo(const ResourceAllocatorHandle& allocator) : allocator(allocator) {}

    NodeProcessInfo(const NodeProcessInfo&) = delete;
    NodeProcessInfo& operator=(const NodeProcessInfo&) = delete;

    // Number of iterations since connect.
    // Use get_total_iteration() for iterations since graph initialization.
    //
    // Iterations are 0-indexed.
    const uint64_t& get_iteration() const noexcept {
        return iteration;
    }

    // Number of iterations since graph initialization.
    // Use get_iteration() for iterations since connect.
    //
    // Iterations are 0-indexed.
    const uint64_t& get_total_iteration() const noexcept {
        return total_iteration;
    }

    // returns the current in-flight index i, with 0 <= i < get_iterations_in_flight().
    // It is guaranteed that processing of the last iteration with that index has finished.
    const uint32_t& get_in_flight_index() const noexcept {
        return in_flight_index;
    }

    // returns the number of iterations that might be in flight at a certain time.
    const uint32_t& get_iterations_in_flight() const noexcept {
        return iterations_in_flight;
    }

    // Graph-owned allocator for binding shader objects; cycles with the in-flight index.
    const ShaderObjectAllocatorHandle& get_shader_object_allocator() const noexcept {
        return shader_object_allocator;
    }

    // Returns the time difference to the last run.
    // For the first run of a build the difference to the last run in the previous run is returned.
    const std::chrono::nanoseconds& get_time_delta_duration() const noexcept {
        return time_delta;
    }

    // Returns the time difference to the last run in seconds.
    // For the first run of a build the difference to the last run in the previous run is returned.
    double get_time_delta() const noexcept {
        return to_seconds(time_delta);
    }

    // Return elapsed time since graph initialization
    const std::chrono::nanoseconds& get_elapsed_duration() const noexcept {
        return elapsed;
    }

    // Return elapsed time since graph initialization in seconds.
    double get_elapsed() const noexcept {
        return to_seconds(elapsed);
    }

    // Return elapsed time since the last connect()
    const std::chrono::nanoseconds& get_elapsed_since_connect_duration() const noexcept {
        return elapsed_since_connect;
    }

    // Return elapsed time since the last connect in seconds.
    double get_elapsed_since_connect() const noexcept {
        return to_seconds(elapsed_since_connect);
    }

    // Returns the profiler that is attached to this run.
    //
    // Can be nullptr if profiling is disabled!
    const ProfilerHandle& get_profiler() const noexcept {
        return profiler;
    }

    const ResourceAllocatorHandle& get_allocator() const noexcept {
        return allocator;
    }

    // Returns a semaphore that is signaled at the end of a run with the value of the next run.
    const TimelineSemaphoreHandle& get_iteration_semaphore() const noexcept {
        return iteration_semaphore;
    }

    // Describes this run — merian version and the current graph config — for nodes that embed it
    // into what they write. Built once per connect.
    const std::vector<std::pair<std::string, std::string>>& get_metadata() const noexcept {
        return *metadata;
    }

  private:
    const ResourceAllocatorHandle allocator;

    const std::vector<std::pair<std::string, std::string>>* metadata = nullptr;

    TimelineSemaphoreHandle iteration_semaphore;
    ShaderObjectAllocatorHandle shader_object_allocator;
    ProfilerHandle profiler;

    uint64_t iteration = 0;
    uint64_t total_iteration = 0;
    uint32_t in_flight_index = 0;
    uint32_t iterations_in_flight = 0;
    std::chrono::nanoseconds time_delta{};
    std::chrono::nanoseconds elapsed{};
    std::chrono::nanoseconds elapsed_since_connect{};
};

// Context the graph supplies to Node::on_connected.
struct NodeConnectionInfo {
    uint32_t iterations_in_flight;
};

} // namespace merian
