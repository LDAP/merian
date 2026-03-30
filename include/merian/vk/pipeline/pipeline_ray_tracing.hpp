#pragma once

#include "merian/shader/entry_point.hpp"
#include "merian/vk/pipeline/pipeline.hpp"
#include "merian/vk/utils/check_result.hpp"

#include <map>
#include <optional>
#include <unordered_map>
#include <vector>

namespace merian {

// Forward declaration (needed for library parameter and handle typedef)
class RayTracingPipeline;
using RayTracingPipelineHandle = std::shared_ptr<RayTracingPipeline>;

// ---------------------------------------------------------------------------
// Shader group descriptor structs
// ---------------------------------------------------------------------------

struct RayTracingRaygenGroup {
    VulkanEntryPointHandle raygen;
};

struct RayTracingMissGroup {
    VulkanEntryPointHandle miss;
};

struct RayTracingCallableGroup {
    VulkanEntryPointHandle callable;
};

struct RayTracingTriangleHitGroup {
    VulkanEntryPointHandle closest_hit;            // required
    std::optional<VulkanEntryPointHandle> any_hit; // optional
};

struct RayTracingProceduralHitGroup {
    VulkanEntryPointHandle intersection;               // required
    std::optional<VulkanEntryPointHandle> closest_hit; // optional
    std::optional<VulkanEntryPointHandle> any_hit;     // optional
};

// ---------------------------------------------------------------------------

class RayTracingPipeline : public Pipeline {
  private:
    // Use std::map with raw-pointer comparison from shared_ptr<T>::operator<
    using TriHitKey = std::pair<VulkanEntryPointHandle, VulkanEntryPointHandle>;
    using ProcHitKey =
        std::tuple<VulkanEntryPointHandle, VulkanEntryPointHandle, VulkanEntryPointHandle>;

    RayTracingPipeline(const PipelineLayoutHandle& pipeline_layout,
                       std::vector<RayTracingRaygenGroup> raygen_groups,
                       std::vector<RayTracingMissGroup> miss_groups,
                       std::vector<RayTracingTriangleHitGroup> triangle_hit_groups,
                       std::vector<RayTracingProceduralHitGroup> procedural_hit_groups,
                       std::vector<RayTracingCallableGroup> callable_groups,
                       const std::vector<RayTracingPipelineHandle>& libraries,
                       uint32_t max_recursion_depth,
                       std::vector<vk::DynamicState> dynamic_states,
                       vk::PipelineCreateFlags flags,
                       const PipelineHandle& base_pipeline,
                       const void* pNext)
        : Pipeline(pipeline_layout->get_context(),
                   pipeline_layout,
                   raygen_groups.empty() ? flags | vk::PipelineCreateFlagBits::eLibraryKHR : flags),
          raygen_groups(std::move(raygen_groups)), miss_groups(std::move(miss_groups)),
          triangle_hit_groups(std::move(triangle_hit_groups)),
          procedural_hit_groups(std::move(procedural_hit_groups)),
          callable_groups(std::move(callable_groups)), base_pipeline(base_pipeline) {

        SPDLOG_DEBUG("create RayTracingPipeline ({})", fmt::ptr(this));

        std::vector<vk::PipelineShaderStageCreateInfo> stages;
        std::vector<vk::RayTracingShaderGroupCreateInfoKHR> groups;

        const auto add_stage = [&](const VulkanEntryPointHandle& ep) -> uint32_t {
            if (!ep)
                return VK_SHADER_UNUSED_KHR;
            stages.push_back(ep->get_shader_stage_create_info(context));
            return static_cast<uint32_t>(stages.size() - 1);
        };

        uint32_t abs_idx = 0;

        // Raygen groups
        for (const auto& g : this->raygen_groups) {
            const uint32_t stage = add_stage(g.raygen);
            groups.push_back(vk::RayTracingShaderGroupCreateInfoKHR{
                vk::RayTracingShaderGroupTypeKHR::eGeneral, stage, VK_SHADER_UNUSED_KHR,
                VK_SHADER_UNUSED_KHR, VK_SHADER_UNUSED_KHR});
            raygen_ep_map[g.raygen] = abs_idx++;
        }

        // Miss groups
        for (const auto& g : this->miss_groups) {
            const uint32_t stage = add_stage(g.miss);
            groups.push_back(vk::RayTracingShaderGroupCreateInfoKHR{
                vk::RayTracingShaderGroupTypeKHR::eGeneral, stage, VK_SHADER_UNUSED_KHR,
                VK_SHADER_UNUSED_KHR, VK_SHADER_UNUSED_KHR});
            miss_ep_map[g.miss] = abs_idx++;
        }

        // Triangle hit groups
        for (const auto& g : this->triangle_hit_groups) {
            const uint32_t chit = add_stage(g.closest_hit);
            const uint32_t ahit = g.any_hit ? add_stage(*g.any_hit) : VK_SHADER_UNUSED_KHR;
            groups.push_back(vk::RayTracingShaderGroupCreateInfoKHR{
                vk::RayTracingShaderGroupTypeKHR::eTrianglesHitGroup, VK_SHADER_UNUSED_KHR, chit,
                ahit, VK_SHADER_UNUSED_KHR});
            tri_hit_tuple_map[{g.closest_hit, g.any_hit.value_or(nullptr)}] = abs_idx;
            hit_flat_ep_map[g.closest_hit] = abs_idx;
            if (g.any_hit)
                hit_flat_ep_map[*g.any_hit] = abs_idx;
            abs_idx++;
        }

        // Procedural hit groups
        for (const auto& g : this->procedural_hit_groups) {
            const uint32_t isect = add_stage(g.intersection);
            const uint32_t chit = g.closest_hit ? add_stage(*g.closest_hit) : VK_SHADER_UNUSED_KHR;
            const uint32_t ahit = g.any_hit ? add_stage(*g.any_hit) : VK_SHADER_UNUSED_KHR;
            groups.push_back(vk::RayTracingShaderGroupCreateInfoKHR{
                vk::RayTracingShaderGroupTypeKHR::eProceduralHitGroup, VK_SHADER_UNUSED_KHR, chit,
                ahit, isect});
            proc_hit_tuple_map[{g.intersection, g.closest_hit.value_or(nullptr),
                                g.any_hit.value_or(nullptr)}] = abs_idx;
            hit_flat_ep_map[g.intersection] = abs_idx;
            if (g.closest_hit)
                hit_flat_ep_map[*g.closest_hit] = abs_idx;
            if (g.any_hit)
                hit_flat_ep_map[*g.any_hit] = abs_idx;
            abs_idx++;
        }

        // Callable groups
        for (const auto& g : this->callable_groups) {
            const uint32_t stage = add_stage(g.callable);
            groups.push_back(vk::RayTracingShaderGroupCreateInfoKHR{
                vk::RayTracingShaderGroupTypeKHR::eGeneral, stage, VK_SHADER_UNUSED_KHR,
                VK_SHADER_UNUSED_KHR, VK_SHADER_UNUSED_KHR});
            callable_ep_map[g.callable] = abs_idx++;
        }

        // Collect library pipeline handles
        std::vector<vk::Pipeline> lib_handles;
        lib_handles.reserve(libraries.size());
        for (const auto& lib : libraries) {
            lib_handles.push_back(lib->get_pipeline());
        }

        const vk::PipelineLibraryCreateInfoKHR library_info{lib_handles};
        const vk::PipelineDynamicStateCreateInfo dynamic_state_info{{}, dynamic_states};

        const auto create_info =
            vk::RayTracingPipelineCreateInfoKHR{}
                .setFlags(this->flags)
                .setStages(stages)
                .setGroups(groups)
                .setMaxPipelineRayRecursionDepth(max_recursion_depth)
                .setPLibraryInfo(libraries.empty() ? nullptr : &library_info)
                .setPDynamicState(dynamic_states.empty() ? nullptr : &dynamic_state_info)
                .setLayout(*pipeline_layout)
                .setBasePipelineHandle(base_pipeline ? base_pipeline->get_pipeline()
                                                     : vk::Pipeline{})
                .setBasePipelineIndex(0)
                .setPNext(pNext);

        pipeline = context->get_device()
                       ->get_device()
                       .createRayTracingPipelinesKHR(
                           {}, context->get_device()->get_pipeline_cache(), {create_info})
                       .value[0];
    }

  public:
    ~RayTracingPipeline() override {
        SPDLOG_DEBUG("destroy RayTracingPipeline ({})", fmt::ptr(this));
        context->get_device()->get_device().destroyPipeline(pipeline);
    }

    static RayTracingPipelineHandle
    create(const PipelineLayoutHandle& pipeline_layout,
           std::vector<RayTracingRaygenGroup> raygen_groups,
           std::vector<RayTracingMissGroup> miss_groups,
           std::vector<RayTracingTriangleHitGroup> triangle_hit_groups,
           std::vector<RayTracingProceduralHitGroup> procedural_hit_groups,
           std::vector<RayTracingCallableGroup> callable_groups,
           const std::vector<RayTracingPipelineHandle>& libraries,
           uint32_t max_recursion_depth,
           std::vector<vk::DynamicState> dynamic_states = {},
           vk::PipelineCreateFlags flags = {},
           const PipelineHandle& base_pipeline = {},
           const void* pNext = nullptr) {
        return RayTracingPipelineHandle(new RayTracingPipeline(
            pipeline_layout, std::move(raygen_groups), std::move(miss_groups),
            std::move(triangle_hit_groups), std::move(procedural_hit_groups),
            std::move(callable_groups), libraries, max_recursion_depth, std::move(dynamic_states),
            flags, base_pipeline, pNext));
    }

    // ---------------------------------------------------------------------------

    vk::PipelineBindPoint get_pipeline_bind_point() const override {
        return vk::PipelineBindPoint::eRayTracingKHR;
    }

    vk::PipelineStageFlags get_pipeline_stage_flags() const override {
        return vk::PipelineStageFlagBits::eRayTracingShaderKHR;
    }

    vk::PipelineStageFlags2 get_pipeline_stage_flags2() const override {
        return vk::PipelineStageFlagBits2::eRayTracingShaderKHR;
    }

    // True when created without a raygen group (eLibraryKHR was set automatically).
    bool is_library() const {
        return bool(flags & vk::PipelineCreateFlagBits::eLibraryKHR);
    }

    const ContextHandle& get_context() const {
        return context;
    }

    // ---------------------------------------------------------------------------
    // Group index lookup

    // Flat lookup — works for any entry point that belongs to exactly one group.
    uint32_t group_index(const VulkanEntryPointHandle& ep) const {
        if (auto it = raygen_ep_map.find(ep); it != raygen_ep_map.end())
            return it->second;
        if (auto it = miss_ep_map.find(ep); it != miss_ep_map.end())
            return it->second;
        if (auto it = callable_ep_map.find(ep); it != callable_ep_map.end())
            return it->second;
        if (auto it = hit_flat_ep_map.find(ep); it != hit_flat_ep_map.end())
            return it->second;
        assert(false && "entry point not found in pipeline");
        return VK_SHADER_UNUSED_KHR;
    }

    // Unambiguous triangle hit group lookup (pass both key components explicitly).
    // Pass {} for any_hit if the group has no any-hit shader.
    uint32_t group_index(const VulkanEntryPointHandle& closest_hit,
                         const VulkanEntryPointHandle& any_hit) const {
        const auto it = tri_hit_tuple_map.find({closest_hit, any_hit});
        assert(it != tri_hit_tuple_map.end() && "triangle hit group not found in pipeline");
        return it->second;
    }

    // Unambiguous procedural hit group lookup.
    uint32_t group_index(const VulkanEntryPointHandle& intersection,
                         const VulkanEntryPointHandle& closest_hit,
                         const VulkanEntryPointHandle& any_hit) const {
        const auto it = proc_hit_tuple_map.find({intersection, closest_hit, any_hit});
        assert(it != proc_hit_tuple_map.end() && "procedural hit group not found in pipeline");
        return it->second;
    }

    // ---------------------------------------------------------------------------

    // Opaque handles for all shader groups, in canonical order:
    //   raygen[0..n] | miss[0..n] | triangle_hit[0..n] | procedural_hit[0..n] | callable[0..n]
    // Each handle is shaderGroupHandleSize bytes.
    std::vector<uint8_t> get_shader_group_handles() const {
        const auto& rt_props = context->get_physical_device()
                                   ->get_properties()
                                   .get_ray_tracing_pipeline_properties_khr();
        const uint32_t n = get_total_group_count();
        if (n == 0)
            return {};
        const uint32_t handle_size = rt_props.shaderGroupHandleSize;
        std::vector<uint8_t> handles(static_cast<size_t>(n) * handle_size);
        check_result(context->get_device()->get_device().getRayTracingShaderGroupHandlesKHR(
                         pipeline, 0, n, handles.size(), handles.data()),
                     "failed to get ray tracing shader group handles");
        return handles;
    }

    // ---------------------------------------------------------------------------

    uint32_t get_raygen_count() const {
        return static_cast<uint32_t>(raygen_groups.size());
    }
    uint32_t get_miss_count() const {
        return static_cast<uint32_t>(miss_groups.size());
    }
    uint32_t get_hit_count() const {
        return static_cast<uint32_t>(triangle_hit_groups.size() + procedural_hit_groups.size());
    }
    uint32_t get_callable_count() const {
        return static_cast<uint32_t>(callable_groups.size());
    }
    uint32_t get_total_group_count() const {
        return get_raygen_count() + get_miss_count() + get_hit_count() + get_callable_count();
    }

    const std::vector<RayTracingRaygenGroup>& get_raygen_groups() const {
        return raygen_groups;
    }
    const std::vector<RayTracingMissGroup>& get_miss_groups() const {
        return miss_groups;
    }
    const std::vector<RayTracingTriangleHitGroup>& get_triangle_hit_groups() const {
        return triangle_hit_groups;
    }
    const std::vector<RayTracingProceduralHitGroup>& get_procedural_hit_groups() const {
        return procedural_hit_groups;
    }
    const std::vector<RayTracingCallableGroup>& get_callable_groups() const {
        return callable_groups;
    }

  private:
    std::vector<RayTracingRaygenGroup> raygen_groups;
    std::vector<RayTracingMissGroup> miss_groups;
    std::vector<RayTracingTriangleHitGroup> triangle_hit_groups;
    std::vector<RayTracingProceduralHitGroup> procedural_hit_groups;
    std::vector<RayTracingCallableGroup> callable_groups;

    // Lookup maps (built in constructor body)
    std::unordered_map<VulkanEntryPointHandle, uint32_t> raygen_ep_map;
    std::unordered_map<VulkanEntryPointHandle, uint32_t> miss_ep_map;
    std::unordered_map<VulkanEntryPointHandle, uint32_t> callable_ep_map;
    std::unordered_map<VulkanEntryPointHandle, uint32_t> hit_flat_ep_map;
    std::map<TriHitKey, uint32_t> tri_hit_tuple_map;
    std::map<ProcHitKey, uint32_t> proc_hit_tuple_map;

    const PipelineHandle base_pipeline;
};

} // namespace merian
