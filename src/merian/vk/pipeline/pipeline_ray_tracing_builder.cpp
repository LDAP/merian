#include "merian/vk/pipeline/pipeline_ray_tracing_builder.hpp"

namespace merian {

RayTracingPipelineBuilder&
RayTracingPipelineBuilder::add_raygen_group(const VulkanEntryPointHandle& raygen) {
    assert(raygen);
    assert(raygen->get_stage() == vk::ShaderStageFlagBits::eRaygenKHR);
    raygen_groups.push_back({raygen});
    return *this;
}

RayTracingPipelineBuilder&
RayTracingPipelineBuilder::add_miss_group(const VulkanEntryPointHandle& miss) {
    assert(miss);
    assert(miss->get_stage() == vk::ShaderStageFlagBits::eMissKHR);
    miss_groups.push_back({miss});
    return *this;
}

RayTracingPipelineBuilder&
RayTracingPipelineBuilder::add_triangle_hit_group(const VulkanEntryPointHandle& closest_hit,
                                                   const VulkanEntryPointHandle& any_hit) {
    assert(closest_hit);
    assert(closest_hit->get_stage() == vk::ShaderStageFlagBits::eClosestHitKHR);
    if (any_hit) {
        assert(any_hit->get_stage() == vk::ShaderStageFlagBits::eAnyHitKHR);
    }
    RayTracingTriangleHitGroup g{closest_hit, {}};
    if (any_hit)
        g.any_hit = any_hit;
    triangle_hit_groups.push_back(std::move(g));
    return *this;
}

RayTracingPipelineBuilder&
RayTracingPipelineBuilder::add_procedural_hit_group(const VulkanEntryPointHandle& intersection,
                                                     const VulkanEntryPointHandle& closest_hit,
                                                     const VulkanEntryPointHandle& any_hit) {
    assert(intersection);
    assert(intersection->get_stage() == vk::ShaderStageFlagBits::eIntersectionKHR);
    if (closest_hit) {
        assert(closest_hit->get_stage() == vk::ShaderStageFlagBits::eClosestHitKHR);
    }
    if (any_hit) {
        assert(any_hit->get_stage() == vk::ShaderStageFlagBits::eAnyHitKHR);
    }
    RayTracingProceduralHitGroup g{intersection, {}, {}};
    if (closest_hit)
        g.closest_hit = closest_hit;
    if (any_hit)
        g.any_hit = any_hit;
    procedural_hit_groups.push_back(std::move(g));
    return *this;
}

RayTracingPipelineBuilder&
RayTracingPipelineBuilder::add_callable_group(const VulkanEntryPointHandle& callable) {
    assert(callable);
    assert(callable->get_stage() == vk::ShaderStageFlagBits::eCallableKHR);
    callable_groups.push_back({callable});
    return *this;
}

RayTracingPipelineBuilder&
RayTracingPipelineBuilder::add_library(const RayTracingPipelineHandle& library) {
    assert(library);
    assert(library->is_library());
    libraries.push_back(library);
    return *this;
}

RayTracingPipelineBuilder&
RayTracingPipelineBuilder::set_max_recursion_depth(uint32_t depth) {
    max_recursion_depth = depth;
    return *this;
}

RayTracingPipelineBuilder& RayTracingPipelineBuilder::add_dynamic_state(vk::DynamicState state) {
    dynamic_states.push_back(state);
    return *this;
}

RayTracingPipelineHandle RayTracingPipelineBuilder::build(const PipelineLayoutHandle& layout,
                                                          vk::PipelineCreateFlags flags,
                                                          const PipelineHandle& base_pipeline,
                                                          const void* pNext) {
    assert(layout);
    return RayTracingPipeline::create(layout,
                                      std::move(raygen_groups),
                                      std::move(miss_groups),
                                      std::move(triangle_hit_groups),
                                      std::move(procedural_hit_groups),
                                      std::move(callable_groups),
                                      libraries,
                                      max_recursion_depth,
                                      std::move(dynamic_states),
                                      flags,
                                      base_pipeline,
                                      pNext);
}

} // namespace merian
