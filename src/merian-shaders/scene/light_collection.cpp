#include "merian-shaders/scene/light_collection.hpp"

#include "merian/utils/properties.hpp"
#include "merian/vk/pipeline/pipeline_compute.hpp"
#include "merian/vk/utils/profiler.hpp"

#include <algorithm>
#include <array>
#include <spdlog/spdlog.h>

namespace merian {

namespace {

uint32_t smallest_subgroup_size(const ContextHandle& context) {
    const VulkanProperties& properties = context->get_physical_device()->get_properties();
    if (properties.is_available<vk::PhysicalDeviceSubgroupSizeControlProperties>())
        return properties.get_subgroup_size_control_properties().minSubgroupSize;
    return properties.get_subgroup_properties().subgroupSize;
}

} // namespace

LightCollection::LightCollection(const ShaderCompileContextHandle& compile_context,
                                 const ContextHandle& context,
                                 const ResourceAllocatorHandle& allocator)
    : compile_context(compile_context), context(context), allocator(allocator),
      min_subgroup_size(smallest_subgroup_size(context)) {}

void LightCollection::set_geometries(const std::vector<EmissiveGeometry>& geometries,
                                     const uint32_t geometry_count) {
    std::vector<LightGeometry> new_geometries;
    new_geometries.reserve(geometries.size());
    std::vector<uint32_t> new_offsets(geometry_count, LIGHT_INVALID_INDEX);
    uint32_t first_triangle = 0;
    for (const EmissiveGeometry& g : geometries) {
        if (g.primitive_count == 0)
            continue;
        assert(g.geometry_id < geometry_count);
        new_geometries.push_back({g.geometry_id, g.instance_index, first_triangle,
                                  g.primitive_count, g.emission_offsets, g.emission_records,
                                  g.emission_level, 0});
        new_offsets[g.geometry_id] = first_triangle;
        first_triangle += g.primitive_count;
    }

    if (new_geometries != light_geometries || new_offsets != geometry_light_offsets) {
        light_geometries = std::move(new_geometries);
        geometry_light_offsets = std::move(new_offsets);
        triangle_count = first_triangle;
        tables_dirty = true;
        SPDLOG_DEBUG("light collection: {} emissive geometries, {} triangles",
                     light_geometries.size(), triangle_count);
    }
}

void LightCollection::ensure_buffer(BufferHandle& buffer,
                                    const vk::DeviceSize size,
                                    const std::string& name,
                                    const CommandBufferHandle& cmd) {
    if (buffer && buffer->get_size() >= size)
        return;
    if (buffer)
        cmd->keep_until_pool_reset(std::move(buffer));
    buffer = allocator->create_buffer(std::max<vk::DeviceSize>(size * 3 / 2, 1024),
                                      vk::BufferUsageFlagBits::eStorageBuffer |
                                          vk::BufferUsageFlagBits::eTransferDst,
                                      MemoryMappingType::NONE, name);
}

void LightCollection::retire_buffer(BufferHandle& buffer, const CommandBufferHandle& cmd) {
    if (buffer)
        cmd->keep_until_pool_reset(std::move(buffer));
    buffer.reset();
}

void LightCollection::ensure_pipelines(const SlangCompositionHandle& scene_composition) {
    if (!update_composition) {
        update_composition = SlangComposition::create();
        update_composition->add_composition(scene_composition);
        update_composition->add_module_from_path("merian-shaders/scene/light-update.slang", true);
        update_program = SlangProgram::create(compile_context, update_composition);
        update_entry_point = SlangProgramEntryPoint::create(update_program, "main");
        update_pipeline = Versioned<Pipeline>([this] {
            const auto ep = update_entry_point.get();
            return ComputePipeline::create(ep->get_pipeline_layout(context), ep->specialize());
        });
        update_pipeline.depends_on(update_entry_point);
        update_params = Versioned<ShaderObject>([this] {
            return update_entry_point->create_shader_object_for_parameter(context, "params",
                                                                          allocator);
        });
        update_params.depends_on(update_entry_point);
    }

    if (!preprocess_composition) {
        preprocess_composition = SlangComposition::create();
        preprocess_composition->add_composition(scene_composition);
        preprocess_composition->add_module_from_path("merian-shaders/scene/light-preprocess.slang",
                                                     true);
        preprocess_program = SlangProgram::create(compile_context, preprocess_composition);
        const auto make = [&](const std::string& name, Versioned<SlangProgramEntryPoint>& ep,
                              Versioned<Pipeline>& pipe, Versioned<ShaderObject>& params) {
            ep = SlangProgramEntryPoint::create(preprocess_program, name);
            pipe = Versioned<Pipeline>([this, &ep] {
                const auto e = ep.get();
                return ComputePipeline::create(e->get_pipeline_layout(context), e->specialize());
            });
            pipe.depends_on(ep);
            params = Versioned<ShaderObject>([this, &ep] {
                return ep->create_shader_object_for_parameter(context, "params", allocator);
            });
            params.depends_on(ep);
        };
        make("grid_setup", setup_entry_point, setup_pipeline, setup_params);
        make("grid", grid_entry_point, grid_pipeline, grid_params);
        make("env_cones", env_cones_entry_point, env_cones_pipeline, env_cones_params);
    }

    if (!sort_composition) {
        sort_composition = SlangComposition::create();
        sort_composition->add_module_from_path("merian-shaders/scene/light-sort.slang", true);
        sort_program = SlangProgram::create(compile_context, sort_composition);
        const auto make = [&](const std::string& name, Versioned<SlangProgramEntryPoint>& ep,
                              Versioned<Pipeline>& pipe) {
            ep = SlangProgramEntryPoint::create(sort_program, name);
            pipe = Versioned<Pipeline>([this, &ep] {
                const auto e = ep.get();
                return ComputePipeline::create(e->get_pipeline_layout(context), e->specialize());
            });
            pipe.depends_on(ep);
        };
        make("sort_histogram", sort_histogram_entry_point, sort_histogram_pipeline);
        make("sort_scatter", sort_scatter_entry_point, sort_scatter_pipeline);
        sort_params.clear();
        for (uint32_t i = 0; i <= LIGHT_SORT_PASSES; i++) {
            Versioned<SlangProgramEntryPoint>& ep =
                i == 0 ? sort_histogram_entry_point : sort_scatter_entry_point;
            Versioned<ShaderObject> params([this, &ep] {
                return ep->create_shader_object_for_parameter(context, "params", allocator);
            });
            params.depends_on(ep);
            sort_params.emplace_back(std::move(params));
        }
    }

    if (!env_composition) {
        env_composition = SlangComposition::create();
        env_composition->add_composition(scene_composition);
        env_composition->add_module_from_path("merian-shaders/scene/env-importance-build.slang",
                                              true);
        env_program = SlangProgram::create(compile_context, env_composition);
        env_pool_entry_point = SlangProgramEntryPoint::create(env_program, "env_pool");
        env_pool_pipeline = Versioned<Pipeline>([this] {
            const auto ep = env_pool_entry_point.get();
            return ComputePipeline::create(ep->get_pipeline_layout(context), ep->specialize());
        });
        env_pool_pipeline.depends_on(env_pool_entry_point);
        env_pool_params = Versioned<ShaderObject>([this] {
            return env_pool_entry_point->create_shader_object_for_parameter(context, "params",
                                                                            allocator);
        });
        env_pool_params.depends_on(env_pool_entry_point);
        env_build_entry_point = SlangProgramEntryPoint::create(env_program, "build");
        env_build_pipeline = Versioned<Pipeline>([this] {
            const auto ep = env_build_entry_point.get();
            return ComputePipeline::create(ep->get_pipeline_layout(context), ep->specialize());
        });
        env_build_pipeline.depends_on(env_build_entry_point);
        env_build_params = Versioned<ShaderObject>([this] {
            return env_build_entry_point->create_shader_object_for_parameter(context, "params",
                                                                             allocator);
        });
        env_build_params.depends_on(env_build_entry_point);
    }

    if (!cdf_composition) {
        cdf_composition = SlangComposition::create();
        cdf_composition->add_module_from_path("merian-shaders/scene/light-cdf.slang", true);
        cdf_program = SlangProgram::create(compile_context, cdf_composition);
        cdf_entry_point = SlangProgramEntryPoint::create(cdf_program, "scan");
        cdf_pipeline = Versioned<Pipeline>([this] {
            const auto ep = cdf_entry_point.get();
            return ComputePipeline::create(ep->get_pipeline_layout(context), ep->specialize());
        });
        cdf_pipeline.depends_on(cdf_entry_point);
        cdf_params = Versioned<ShaderObject>([this] {
            return cdf_entry_point->create_shader_object_for_parameter(context, "params",
                                                                       allocator);
        });
        cdf_params.depends_on(cdf_entry_point);
    }
}

void LightCollection::prepare(const CommandBufferHandle& cmd) {
    if (!enabled)
        return;

    if (env_emissive) {
        if (env_importance_resized && env_cdf_buffer) {
            cmd->keep_until_pool_reset(std::move(env_cdf_buffer));
        }
        env_importance_resized = false;
        const uint32_t texels = env_importance_size() * env_importance_size();
        ensure_buffer(env_cdf_buffer, texels * sizeof(float2), "LightCollection::env_cdf", cmd);
        if (env_cdf_buffer != env_importance_built_buffer) {
            env_importance_built = false;
            env_importance_built_buffer = env_cdf_buffer;
        }
        ensure_buffer(env_cdf_state_buffer, cdf_state_size(texels),
                      "LightCollection::env_cdf_state", cmd);
        if (uses_env_pool()) {
            ensure_buffer(env_pool_buffer, static_cast<uint32_t>(env_pool_size) * sizeof(uint32_t),
                          "LightCollection::env_pool", cmd);
        } else {
            retire_buffer(env_pool_buffer, cmd);
        }
        if (lists_env()) {
            const BufferHandle previous = env_cones_buffer;
            ensure_buffer(env_cones_buffer, 2 * texels * sizeof(float4),
                          "LightCollection::env_cones", cmd);
            if (env_cones_buffer != previous)
                env_cones_size = 0;
        }
    }

    grid_slot ^= 1u;
    if (triangle_count > 0) {
        for (uint32_t i = 0; i < 2; i++) {
            ensure_buffer(tree_keys_buffer[i], triangle_count * sizeof(uint32_t),
                          "LightCollection::tree_keys", cmd);
            ensure_buffer(tree_values_buffer[i], triangle_count * sizeof(uint32_t),
                          "LightCollection::tree_values", cmd);
        }
        ensure_buffer(tree_rank_buffer, triangle_count * sizeof(uint32_t),
                      "LightCollection::tree_rank", cmd);
        ensure_buffer(tree_cdf_buffer, triangle_count * sizeof(float2), "LightCollection::tree_cdf",
                      cmd);
        ensure_buffer(tree_cdf_state_buffer, cdf_state_size(triangle_count),
                      "LightCollection::tree_cdf_state", cmd);
        ensure_buffer(setup_state_buffer, LIGHT_GRID_SETUP_STATE * sizeof(uint32_t),
                      "LightCollection::setup_state", cmd);
        ensure_buffer(sort_state_buffer, sort_state_size(), "LightCollection::sort_state", cmd);

        bool info_grew = false;
        for (uint32_t i = 0; i < 2; i++) {
            const vk::DeviceSize info_size =
                static_cast<uint32_t>(grid_cascades) * sizeof(LightGridInfo);
            if (!grid_info_buffer[i] || grid_info_buffer[i]->get_size() < info_size) {
                ensure_buffer(grid_info_buffer[i], info_size, "LightCollection::grid_info", cmd);
                // read as the previous cascades before the first setup
                cmd->fill(grid_info_buffer[i]);
                info_grew = true;
            }
            if (!tree_info_buffer[i]) {
                ensure_buffer(tree_info_buffer[i], sizeof(LightTreeInfo),
                              "LightCollection::tree_info", cmd);
                cmd->fill(tree_info_buffer[i]);
                info_grew = true;
            }
        }
        if (info_grew) {
            cmd->barrier(vk::MemoryBarrier2{
                vk::PipelineStageFlagBits2::eTransfer,
                vk::AccessFlagBits2::eTransferWrite,
                vk::PipelineStageFlagBits2::eComputeShader,
                vk::AccessFlagBits2::eShaderRead | vk::AccessFlagBits2::eShaderWrite,
            });
        }

        // a cell index names another cell once the count changes
        const uint32_t cells = cell_count();
        grid_reset |= cells != allocated_cells;
        allocated_cells = cells;
        const vk::DeviceSize slot_count = static_cast<vk::DeviceSize>(cells) * LIGHT_GRID_SLOTS;
        for (uint32_t i = 0; i < 2; i++) {
            const std::array<BufferHandle, 4> previous = {
                grid_keys_buffer[i], grid_contribution_buffer[i], grid_feedback_buffer[i],
                grid_touched_buffer[i]};
            ensure_buffer(grid_keys_buffer[i], slot_count * sizeof(uint32_t),
                          "LightCollection::grid_keys", cmd);
            ensure_buffer(grid_contribution_buffer[i], slot_count * sizeof(float3),
                          "LightCollection::grid_contribution", cmd);
            ensure_buffer(grid_feedback_buffer[i],
                          LIGHT_GRID_FEEDBACK_STRIDE * slot_count * sizeof(uint32_t),
                          "LightCollection::grid_feedback", cmd);
            ensure_buffer(grid_touched_buffer[i], cells * sizeof(uint32_t),
                          "LightCollection::grid_touched", cmd);
            const bool reallocated =
                previous !=
                std::array<BufferHandle, 4>{grid_keys_buffer[i], grid_contribution_buffer[i],
                                            grid_feedback_buffer[i], grid_touched_buffer[i]};
            grid_reset |= reallocated;
            grid_feedback_fresh |= reallocated;
        }
        ensure_buffer(grid_starts_buffer, slot_count * sizeof(uint32_t),
                      "LightCollection::grid_starts", cmd);
        ensure_buffer(grid_estimate_buffer, slot_count * sizeof(float),
                      "LightCollection::grid_estimate", cmd);
        if (slot_weighing == LightSlotWeighing::LightSlotWeighingShadingPoint) {
            ensure_buffer(grid_slot_bounds_buffer, slot_count * sizeof(LightBound),
                          "LightCollection::grid_slot_bounds", cmd);
            retire_buffer(grid_probability_buffer, cmd);
        } else {
            ensure_buffer(grid_probability_buffer, slot_count * sizeof(float),
                          "LightCollection::grid_probability", cmd);
            retire_buffer(grid_slot_bounds_buffer, cmd);
        }
    }

    if (triangle_count > 0 && tables_dirty) {
        const auto staging = allocator->get_staging();
        const vk::DeviceSize geometries_size = light_geometries.size() * sizeof(LightGeometry);
        const vk::DeviceSize offsets_size = geometry_light_offsets.size() * sizeof(uint32_t);
        ensure_buffer(light_geometries_buffer, geometries_size, "LightCollection::geometries", cmd);
        ensure_buffer(geometry_light_offsets_buffer, offsets_size, "LightCollection::offsets", cmd);
        ensure_buffer(triangles_buffer, triangle_count * sizeof(EmissiveTriangle),
                      "LightCollection::triangles", cmd);
        ensure_buffer(proxies_buffer, triangle_count * sizeof(EmissiveLightProxy),
                      "LightCollection::proxies", cmd);
        ensure_buffer(regions_buffer, triangle_count * sizeof(EmissiveRegion),
                      "LightCollection::regions", cmd);
        staging->cmd_to_device(cmd, light_geometries_buffer, light_geometries.data(), 0,
                               geometries_size);
        staging->cmd_to_device(cmd, geometry_light_offsets_buffer, geometry_light_offsets.data(), 0,
                               offsets_size);
        cmd->barrier(vk::MemoryBarrier2{
            vk::PipelineStageFlagBits2::eTransfer,
            vk::AccessFlagBits2::eTransferWrite,
            vk::PipelineStageFlagBits2::eComputeShader,
            vk::AccessFlagBits2::eShaderRead,
        });
        tables_dirty = false;
    }
}

// Changes the composition, so it has to settle before the scene's shader object is fetched.
void LightCollection::update_constants(const SlangCompositionHandle& scene_composition) {
    if (!constants_dirty) {
        return;
    }
    scene_composition->add_module_from_string(
        "scene_light_constants",
        fmt::format("namespace merian {{\n"
                    "export static const int merian_nee_draws = {};\n"
                    "export static const int merian_nee_source = {};\n"
                    "export static const int merian_nee_slot_weighing = {};\n"
                    "}}",
                    std::clamp(draws, 1, static_cast<int32_t>(LIGHT_GRID_MAX_DRAWS)), source,
                    slot_weighing));
    constants_dirty = false;
}

void LightCollection::update(const CommandBufferHandle& cmd,
                             const SlangCompositionHandle& scene_composition,
                             const ShaderObjectHandle& scene_object,
                             const ShaderObjectAllocatorHandle& obj_allocator_in) {
    if (!enabled || (triangle_count == 0 && !env_emissive))
        return;

    MERIAN_PROFILE_SCOPE_GPU(cmd, "LightCollection::update");
    ensure_pipelines(scene_composition);

    ShaderObjectAllocatorHandle obj_allocator = obj_allocator_in;
    if (!obj_allocator) {
        if (!fallback_obj_allocator)
            fallback_obj_allocator = std::make_shared<SimpleShaderObjectAllocator>(allocator);
        obj_allocator = fallback_obj_allocator;
    }

    // the renderer's reads and feedback, from any stage
    cmd->barrier(vk::MemoryBarrier2{
        vk::PipelineStageFlagBits2::eAllCommands,
        vk::AccessFlagBits2::eShaderWrite,
        vk::PipelineStageFlagBits2::eComputeShader | vk::PipelineStageFlagBits2::eTransfer,
        vk::AccessFlagBits2::eShaderRead | vk::AccessFlagBits2::eShaderWrite |
            vk::AccessFlagBits2::eTransferWrite,
    });
    if (triangle_count > 0) {
        cmd->fill(setup_state_buffer);
        cmd->fill(sort_state_buffer, 0, sort_state_size());
        cmd->fill(tree_cdf_state_buffer, 0, cdf_state_size(triangle_count));
    }
    if (env_emissive && !env_importance_built)
        cmd->fill(env_cdf_state_buffer, 0,
                  cdf_state_size(env_importance_size() * env_importance_size()));
    if (grid_feedback_fresh) {
        for (uint32_t i = 0; i < 2; i++) {
            cmd->fill(grid_feedback_buffer[i]);
            cmd->fill(grid_touched_buffer[i]);
        }
        grid_feedback_fresh = false;
    }
    cmd->barrier(vk::MemoryBarrier2{
        vk::PipelineStageFlagBits2::eTransfer,
        vk::AccessFlagBits2::eTransferWrite,
        vk::PipelineStageFlagBits2::eComputeShader,
        vk::AccessFlagBits2::eShaderRead | vk::AccessFlagBits2::eShaderWrite,
    });

    const auto barrier = [&] {
        cmd->barrier(vk::MemoryBarrier2{
            vk::PipelineStageFlagBits2::eComputeShader,
            vk::AccessFlagBits2::eShaderWrite,
            vk::PipelineStageFlagBits2::eComputeShader,
            vk::AccessFlagBits2::eShaderRead | vk::AccessFlagBits2::eShaderWrite,
        });
    };
    const auto run = [&](Versioned<SlangProgramEntryPoint>& entry_point,
                         Versioned<Pipeline>& pipeline, const ShaderObjectHandle& params,
                         const uint32_t groups_x, const uint32_t groups_y = 1,
                         const ShaderObjectHandle& scene = {}) {
        const auto ep = entry_point.get();
        const auto pipe = pipeline.get();
        cmd->bind(pipe);
        if (scene)
            ep->bind("scene", scene, cmd, pipe, obj_allocator);
        ep->bind("params", params, cmd, pipe, obj_allocator);
        cmd->dispatch(groups_x, groups_y, 1);
    };
    const BufferHandle& dummy = allocator->get_dummy_buffer();
    const auto or_dummy = [&](const BufferHandle& buffer) -> const BufferHandle& {
        return buffer ? buffer : dummy;
    };
    const uint32_t env_size = env_importance_size();
    const bool env_listed = lists_env();
    const auto write_preprocess = [&](const ShaderObjectHandle& params) {
        auto c = params->get_cursor();
        c["proxies"] = proxies_buffer;
        c["tree_keys"] = tree_keys_buffer[0];
        c["tree_cdf"] = tree_cdf_buffer;
        c["env_cdf"] = env_listed ? env_cdf_buffer : dummy;
        c["env_size"] = env_listed ? env_size : 0u;
        c["env_cones"] = env_listed ? env_cones_buffer : dummy;
        c["tree_info"] = tree_info_buffer[grid_slot];
        c["tree_info_prev"] = tree_info_buffer[grid_slot ^ 1];
        c["grid_keys"] = or_dummy(grid_keys_buffer[grid_slot]);
        c["grid_keys_prev"] = or_dummy(grid_keys_buffer[grid_slot ^ 1]);
        c["grid_contribution"] = or_dummy(grid_contribution_buffer[grid_slot]);
        c["grid_contribution_prev"] = or_dummy(grid_contribution_buffer[grid_slot ^ 1]);
        c["grid_feedback"] = or_dummy(grid_feedback_buffer[grid_slot]);
        c["grid_feedback_prev"] = or_dummy(grid_feedback_buffer[grid_slot ^ 1]);
        c["grid_touched"] = or_dummy(grid_touched_buffer[grid_slot]);
        c["grid_touched_prev"] = or_dummy(grid_touched_buffer[grid_slot ^ 1]);
        c["grid_starts"] = or_dummy(grid_starts_buffer);
        c["grid_estimate"] = or_dummy(grid_estimate_buffer);
        c["grid_probability"] = or_dummy(grid_probability_buffer);
        c["grid_slot_bounds"] = or_dummy(grid_slot_bounds_buffer);
        c["grid_info"] = grid_info_buffer[grid_slot];
        c["grid_info_prev"] = grid_info_buffer[grid_slot ^ 1];
        c["grid_coverage"] = grid_coverage;
        c["camera_position"] = camera_position;
        c["triangle_count"] = triangle_count;
        c["grid_dimension"] = static_cast<uint32_t>(grid_dimension);
        c["grid_cascades"] = static_cast<uint32_t>(grid_cascades);
        c["grid_cell_size"] = grid_cell_size;
        c["grid_jitter"] = grid_jitter;
        c["grid_even_share"] = grid_even_share;
        c["grid_refinements"] = static_cast<uint32_t>(grid_refinements);
        c["cell_count"] = cell_count();
        c["single_cut"] = !uses_grid();
        c["triangle_prior"] = triangle_prior;
        c["env_prior"] = env_prior;
        c["setup_state"] = setup_state_buffer;
        c["setup_groups"] = setup_group_count();
        c["slot_weighing"] = static_cast<uint32_t>(slot_weighing);
        c["grid_reset"] = grid_reset;
        return params;
    };

    {
        MERIAN_PROFILE_SCOPE_GPU(cmd, "flux and environment");
        if (env_emissive && !env_importance_built) {
            const auto params = env_build_params.get();
            auto c = params->get_cursor();
            c["cdf"] = env_cdf_buffer;
            c["state"] = env_cdf_state_buffer;
            c["size"] = env_size;
            run(env_build_entry_point, env_build_pipeline, params,
                cdf_block_count(env_size * env_size), 1, scene_object);
            env_importance_built = true;
        }
        if (triangle_count > 0) {
            const auto params = update_params.get();
            auto c = params->get_cursor();
            c["triangles"] = triangles_buffer;
            c["proxies"] = proxies_buffer;
            c["regions"] = regions_buffer;
            c["light_geometries"] = light_geometries_buffer;
            c["light_geometry_count"] = static_cast<uint32_t>(light_geometries.size());
            c["triangle_count"] = triangle_count;
            c["flux_samples"] = static_cast<uint32_t>(flux_samples);
            const uint32_t lanes = update_lanes();
            c["lanes"] = lanes;
            const uint32_t groups =
                (triangle_count * lanes + LIGHT_UPDATE_GROUP - 1) / LIGHT_UPDATE_GROUP;
            run(update_entry_point, update_pipeline, params, std::min(groups, LIGHT_DISPATCH_ROW),
                (groups + LIGHT_DISPATCH_ROW - 1) / LIGHT_DISPATCH_ROW, scene_object);
        }
        barrier();
    }

    {
        MERIAN_PROFILE_SCOPE_GPU(cmd, "setup");
        if (env_emissive && uses_env_pool()) {
            const auto params = env_pool_params.get();
            auto c = params->get_cursor();
            c["importance"]["cdf"] = env_cdf_buffer;
            c["importance"]["size"] = env_size;
            c["pool"] = env_pool_buffer;
            c["pool_size"] = static_cast<uint32_t>(env_pool_size);
            c["frame"] = frame;
            run(env_pool_entry_point, env_pool_pipeline, params,
                (static_cast<uint32_t>(env_pool_size) + 63) / 64);
        }
        if (triangle_count > 0) {
            run(setup_entry_point, setup_pipeline, write_preprocess(setup_params.get()),
                setup_group_count());
        }
        if (env_listed && env_cones_size != env_size) {
            run(env_cones_entry_point, env_cones_pipeline, write_preprocess(env_cones_params.get()),
                (2 * env_size * env_size + 63) / 64);
            env_cones_size = env_size;
        }
        barrier();
    }

    if (triangle_count > 0) {
        MERIAN_PROFILE_SCOPE_GPU(cmd, "sort");
        const uint32_t tiles = sort_tile_count();
        for (uint32_t pass = 0; pass <= LIGHT_SORT_PASSES; pass++) {
            const uint32_t in = pass > 0 ? (pass - 1) % 2 : 0;
            const auto params = sort_params[pass].get();
            auto c = params->get_cursor();
            c["proxies"] = proxies_buffer;
            c["tree_info"] = tree_info_buffer[grid_slot];
            c["keys_in"] = tree_keys_buffer[in];
            c["values_in"] = tree_values_buffer[in];
            c["keys_out"] = tree_keys_buffer[in ^ 1];
            c["values_out"] = tree_values_buffer[in ^ 1];
            c["rank"] = tree_rank_buffer;
            c["state"] = sort_state_buffer;
            c["count"] = triangle_count;
            c["tile_count"] = tiles;
            c["pass"] = pass > 0 ? pass - 1 : 0u;
            if (pass == 0)
                run(sort_histogram_entry_point, sort_histogram_pipeline, params, tiles);
            else
                run(sort_scatter_entry_point, sort_scatter_pipeline, params, tiles);
            barrier();
        }
    }

    if (triangle_count > 0) {
        MERIAN_PROFILE_SCOPE_GPU(cmd, "cdf");
        const auto params = cdf_params.get();
        auto c = params->get_cursor();
        c["proxies"] = proxies_buffer;
        c["order"] = tree_values_buffer[0];
        c["cdf"] = tree_cdf_buffer;
        c["state"] = tree_cdf_state_buffer;
        c["count"] = triangle_count;
        run(cdf_entry_point, cdf_pipeline, params, cdf_block_count(triangle_count));
        barrier();
    }

    {
        MERIAN_PROFILE_SCOPE_GPU(cmd, "grid");
        if (triangle_count > 0) {
            // one group per cell
            const uint32_t cells = cell_count();
            run(grid_entry_point, grid_pipeline, write_preprocess(grid_params.get()),
                std::min(cells, LIGHT_DISPATCH_ROW),
                (cells + LIGHT_DISPATCH_ROW - 1) / LIGHT_DISPATCH_ROW);
        }
    }
    if (triangle_count > 0)
        grid_reset = false;

    cmd->barrier(vk::MemoryBarrier2{
        vk::PipelineStageFlagBits2::eComputeShader,
        vk::AccessFlagBits2::eShaderWrite,
        vk::PipelineStageFlagBits2::eAllCommands,
        vk::AccessFlagBits2::eShaderRead | vk::AccessFlagBits2::eShaderWrite,
    });
}

void LightCollection::write_to(ShaderCursor cursor) const {
    const bool active = enabled && triangle_count > 0 && triangles_buffer && tree_cdf_buffer;
    const BufferHandle& dummy = allocator->get_dummy_buffer();
    auto table = cursor["table"];
    table["triangles"] = active ? triangles_buffer : dummy;
    table["regions"] = active ? regions_buffer : dummy;
    table["geometries"] = active ? light_geometries_buffer : dummy;
    table["geometry_offsets"] = active ? geometry_light_offsets_buffer : dummy;
    table["triangle_count"] = active ? triangle_count : 0u;
    table["geometry_count"] = active ? static_cast<uint32_t>(light_geometries.size()) : 0u;
    table["geometry_offset_count"] =
        active ? static_cast<uint32_t>(geometry_light_offsets.size()) : 0u;

    const bool env_listed = enabled && lists_env();
    const uint32_t env_texels = env_listed ? env_importance_size() * env_importance_size() : 0u;
    auto tree = cursor["tree"];
    tree["cdf"] = active ? tree_cdf_buffer : dummy;
    tree["order"] = active ? tree_values_buffer[0] : dummy;
    tree["rank"] = active ? tree_rank_buffer : dummy;
    tree["env_texels"] = env_texels;

    auto grid = cursor["grid"];
    grid["starts"] = active ? grid_starts_buffer : dummy;
    grid["cascades"] = active ? grid_info_buffer[grid_slot] : dummy;
    grid["in_use"] = active;
    grid["jitter"] = debug_jitter ? grid_jitter : 0.f;
    grid["even_share"] = grid_even_share;
    grid["contribution"] = active ? grid_contribution_buffer[grid_slot] : dummy;
    grid["estimate"] = active ? grid_estimate_buffer : dummy;
    grid["slot_bounds"] = active && grid_slot_bounds_buffer ? grid_slot_bounds_buffer : dummy;
    grid["probability"] = active && grid_probability_buffer ? grid_probability_buffer : dummy;
    grid["feedback"] = active ? grid_feedback_buffer[grid_slot] : dummy;
    grid["touched"] = active ? grid_touched_buffer[grid_slot] : dummy;
    grid["light_count"] = triangle_count + env_texels;

    cursor["has_sky_portals"] = has_sky_portals;
    cursor["debug_view"] = static_cast<uint32_t>(debug_view);

    const bool env_active = enabled && env_emissive && env_cdf_buffer;
    auto env = cursor["env"];
    env["pool"] = env_active && env_pool_buffer ? env_pool_buffer : dummy;
    env["pool_size"] = env_active && uses_env_pool() ? static_cast<uint32_t>(env_pool_size) : 0u;
    auto importance = env["importance"];
    importance["cdf"] = env_active ? env_cdf_buffer : dummy;
    importance["size"] = env_active ? env_importance_size() : 0u;
}

void LightCollection::properties(Properties& props) {
    props.config_bool("enable", enabled,
                      "Keep the emissive triangles as lights a renderer can sample directly (next "
                      "event estimation).");

    constants_dirty |= props.config_options(
        "source", source, {"scene", "grid"}, Properties::OptionsStyle::COMBO,
        "Where a shading point draws its lights from: a cut through a light tree, all "
        "lights grouped finely where they matter and coarsely elsewhere, weighed by what "
        "each group delivers.\n"
        "scene: one cut for every shading point.\n"
        "grid: the cut of the camera-anchored grid cell the point lies in. A point outside "
        "the grid takes its nearest cell.\n"
        "A scene without emissive triangles draws the environment by its importance.");
    constants_dirty |= props.config_int("draws", draws,
                                        "Lights a shading point draws. A resampling renderer "
                                        "weighs all of them and traces one shadow ray; a mixture "
                                        "takes one.",
                                        1, static_cast<int32_t>(LIGHT_GRID_MAX_DRAWS));

    if (props.st_begin_child("cuts", "Cuts")) {
        props.st_separate("Drawing");
        constants_dirty |= props.config_options(
            "weigh at", slot_weighing, {"cell", "shading point"}, Properties::OptionsStyle::COMBO,
            "Where the groups of the cut are weighed by what each delivers.\n"
            "cell: once per frame, at the cell. A draw and its density are a few loads, "
            "which suits renderers that often hit lights with other techniques, such as "
            "guiding.\n"
            "shading point: towards the shading point and its normal, so a draw favours the "
            "closest light. Every draw and density weighs all listed lights.");
        props.config_percent("even share", grid_even_share,
                             "Share of the draws that pick the groups of the cut alike "
                             "instead of by what each delivers.");

        props.st_separate("Cut upkeep");
        if (props.config_bool("clear learned state",
                              "Drop every cell's cut and what it learned about its lights."))
            grid_reset = true;
        if (props.st_begin_child("prior", "Visibility prior")) {
            props.config_float("triangles", triangle_prior,
                               "Share of a group's emission assumed to reach the cell before "
                               "it learned anything about it.",
                               0.01f, 0.f, 1.f);
            props.config_float("environment", env_prior, "The same for the environment.", 0.001f,
                               0.f, 1.f);
            props.st_end_child();
        }
        props.config_int("refinements per frame", grid_refinements,
                         "Groups a cell may split per frame, merging others to make room "
                         "once its slots are full. What a cell learned carries across "
                         "frames.",
                         0, static_cast<int32_t>(LIGHT_GRID_SLOTS));

        if (uses_grid()) {
            props.st_separate("Grid");
            props.config_int("cells per side", grid_dimension, "Cells per side of every cascade.",
                             4, 64);
            props.config_int("cascades", grid_cascades,
                             "Nested grids around the camera, each with twice the cell size of "
                             "the one inside it.",
                             1, 16);
            props.config_float("cell size", grid_cell_size,
                               "World units per cell of the innermost cascade; 0 derives it from "
                               "the distance to the lights.",
                               0.1f, 0.f);
            if (grid_cell_size <= 0.f) {
                props.config_float("reach", grid_coverage,
                                   "Distance the outermost cascade reaches, relative to the "
                                   "farthest light, when the cell size is derived.",
                                   0.05f, 0.01f, 16.f);
            }
            props.config_float("lookup jitter", grid_jitter,
                               "Cells a shading point's lookup is shifted by at random, so the "
                               "cell borders do not show.",
                               0.05f, 0.f, 2.f);
        } else if (grid_cell_size <= 0.f) {
            props.config_float("reach", grid_coverage,
                               "Distance around the camera the cut is shaped for, relative to the "
                               "farthest light.",
                               0.05f, 0.01f, 16.f);
        }
        props.st_end_child();
    }

    if (props.st_begin_child("environment", "Environment")) {
        props.config_options("sampling", env_selection, {"search", "pool"},
                             Properties::OptionsStyle::COMBO,
                             "Without emissive triangles: search the importance map per draw, or "
                             "load from a pool of directions drawn once per frame.");
        if (env_selection == EnvSelection::EnvSelectionPool) {
            props.config_int("pool size", env_pool_size, "Directions in the per-frame pool.", 64,
                             262144);
        }
        int32_t env_size_option = env_importance_log2 - 6;
        if (props.config_options("importance map", env_size_option, {"64", "128", "256", "512"},
                                 Properties::OptionsStyle::COMBO,
                                 "Side length of the importance map over the environment.")) {
            env_importance_log2 = env_size_option + 6;
            env_importance_resized = true;
            grid_reset = true;
        }
        props.st_end_child();
    }

    props.config_int("flux samples", flux_samples,
                     "Emission evaluations per triangle and frame that estimate its power. Where "
                     "the mesh has an emission micromap they land where it emits.",
                     1, 256);

    if (props.st_begin_child("debug", "Debug")) {
        props.config_options(
            "view", debug_view, {"draw outcome", "cell", "slots", "coverage"},
            Properties::OptionsStyle::COMBO,
            fmt::format(
                "What a renderer's NEE debug output shows at the primary hit.\n"
                "draw outcome: of {} light draws, the share blocked before the light (red), "
                "reaching it (green), passing through where it is cut out (blue); dark where "
                "draws produce no ray.\n"
                "cell: the cell of the lookup.\n"
                "slots: occupied slots (red), the contribution the cell learned for what it "
                "picks, relative to its estimate (green), the share of one draw per group that "
                "reaches it (blue).\n"
                "coverage: of the direct light {} cosine-distributed rays find, the share the "
                "draws can reach here (red), the share of lights not offered here (green), the "
                "share no draw can ever reach (blue).",
                LIGHT_DEBUG_DRAWS, LIGHT_DEBUG_COVERAGE_RAYS));
        props.config_bool("jitter the lookup", debug_jitter,
                          "Off shows the cells the lookup sees without its random shift.");
        props.st_end_child();
    }

    props.output_text(fmt::format("emissive geometries: {}\nemissive triangles: {}",
                                  light_geometries.size(), triangle_count));
}

} // namespace merian
