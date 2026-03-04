#pragma once

#include "merian/vk/pipeline/pipeline_ray_tracing.hpp"

#include <vector>

namespace merian {

/*
 * Builder for ray tracing pipelines.
 *
 * Accumulates shader group descriptors, validates shader stage flags, and
 * forwards them to RayTracingPipeline::create().
 *
 * If no raygen group is added, the pipeline is treated as a library
 * (eLibraryKHR is added to flags automatically by RayTracingPipeline).
 */
class RayTracingPipelineBuilder {
  public:
    RayTracingPipelineBuilder& add_raygen_group(const VulkanEntryPointHandle& raygen);

    RayTracingPipelineBuilder& add_miss_group(const VulkanEntryPointHandle& miss);

    RayTracingPipelineBuilder& add_triangle_hit_group(const VulkanEntryPointHandle& closest_hit,
                                                      const VulkanEntryPointHandle& any_hit = {});

    RayTracingPipelineBuilder&
    add_procedural_hit_group(const VulkanEntryPointHandle& intersection,
                             const VulkanEntryPointHandle& closest_hit = {},
                             const VulkanEntryPointHandle& any_hit = {});

    RayTracingPipelineBuilder& add_callable_group(const VulkanEntryPointHandle& callable);

    RayTracingPipelineBuilder& add_library(const RayTracingPipelineHandle& library);

    RayTracingPipelineBuilder& set_max_recursion_depth(uint32_t depth);

    RayTracingPipelineBuilder& add_dynamic_state(vk::DynamicState state);

    // If raygen_groups is empty, eLibraryKHR is added to flags automatically.
    RayTracingPipelineHandle build(const PipelineLayoutHandle& layout,
                                   vk::PipelineCreateFlags flags = {},
                                   const PipelineHandle& base_pipeline = {},
                                   const void* pNext = nullptr);

  private:
    std::vector<RayTracingRaygenGroup> raygen_groups;
    std::vector<RayTracingMissGroup> miss_groups;
    std::vector<RayTracingTriangleHitGroup> triangle_hit_groups;
    std::vector<RayTracingProceduralHitGroup> procedural_hit_groups;
    std::vector<RayTracingCallableGroup> callable_groups;
    std::vector<RayTracingPipelineHandle> libraries;

    uint32_t max_recursion_depth = 1;
    std::vector<vk::DynamicState> dynamic_states;
};

} // namespace merian
