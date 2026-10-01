#include "merian-shaders/scene/light_collection.hpp"

#include "merian/utils/properties.hpp"
#include "merian/vk/pipeline/pipeline_compute.hpp"
#include "merian/vk/utils/profiler.hpp"

#include <spdlog/spdlog.h>

namespace merian {

namespace {
constexpr uint32_t UPDATE_GROUP_SIZE = 64;
constexpr uint32_t ENV_GROUP_SIZE = 8;

// levels from size x size down to 1 x 1; the warp starts at the 2 x 2 level
uint32_t level_count_for(const uint32_t size) {
    uint32_t levels = 1;
    while ((size >> (levels - 1)) > 1) {
        levels++;
    }
    return levels;
}
} // namespace

// quads (float4), one per 2 x 2 block of every level down to 2 x 2
uint32_t LightCollection::env_importance_quad_count() const {
    const uint32_t size = env_importance_size();
    uint32_t quads = 0;
    for (uint32_t level = 0; level + 1 < level_count_for(size); level++) {
        const uint32_t s = size >> level;
        quads += (s * s) / 4;
    }
    return quads;
}

LightCollection::LightCollection(const ShaderCompileContextHandle& compile_context,
                                 const ContextHandle& context,
                                 const ResourceAllocatorHandle& allocator)
    : compile_context(compile_context), context(context), allocator(allocator) {}

void LightCollection::set_geometries(const std::vector<EmissiveGeometry>& geometries,
                                     const uint32_t geometry_count) {
    std::vector<LightGeometry> new_geometries;
    new_geometries.reserve(geometries.size());
    light_geometry_keys.clear();
    std::vector<uint32_t> new_offsets(geometry_count, LIGHT_INVALID_INDEX);
    uint32_t first_triangle = 0;
    for (const EmissiveGeometry& g : geometries) {
        if (g.primitive_count == 0)
            continue;
        assert(g.geometry_id < geometry_count);
        new_geometries.push_back({g.geometry_id, g.instance_index, first_triangle,
                                  g.primitive_count, g.emission_offsets, g.emission_records,
                                  g.emission_level, 0});
        light_geometry_keys.push_back(g.key);
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
        make("pool", pool_entry_point, pool_pipeline, pool_params);
        make("grid", grid_entry_point, grid_pipeline, grid_params);
        make("env_split", env_split_entry_point, env_split_pipeline, env_split_params);
        make("sources", sources_entry_point, sources_pipeline, sources_params);
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
        env_reduce_entry_point = SlangProgramEntryPoint::create(env_program, "reduce");
        env_build_pipeline = Versioned<Pipeline>([this] {
            const auto ep = env_build_entry_point.get();
            return ComputePipeline::create(ep->get_pipeline_layout(context), ep->specialize());
        });
        env_build_pipeline.depends_on(env_build_entry_point);
        env_reduce_pipeline = Versioned<Pipeline>([this] {
            const auto ep = env_reduce_entry_point.get();
            return ComputePipeline::create(ep->get_pipeline_layout(context), ep->specialize());
        });
        env_reduce_pipeline.depends_on(env_reduce_entry_point);
        env_build_params = Versioned<ShaderObject>([this] {
            return env_build_entry_point->create_shader_object_for_parameter(context, "params",
                                                                             allocator);
        });
        env_build_params.depends_on(env_build_entry_point);
        // one object per level: they are bound in the same command buffer
        env_reduce_params.clear();
        for (uint32_t level = 1; level < level_count_for(16384); level++) {
            Versioned<ShaderObject> params([this] {
                return env_reduce_entry_point->create_shader_object_for_parameter(context, "params",
                                                                                  allocator);
            });
            params.depends_on(env_reduce_entry_point);
            env_reduce_params.emplace_back(std::move(params));
        }
    }

    if (!cdf_composition) {
        cdf_composition = SlangComposition::create();
        cdf_composition->add_module_from_path("merian-shaders/scene/light-cdf.slang", true);
        cdf_program = SlangProgram::create(compile_context, cdf_composition);
        const auto make_cdf_pass = [this](const char* name,
                                          Versioned<SlangProgramEntryPoint>& entry_point,
                                          Versioned<Pipeline>& pipeline,
                                          Versioned<ShaderObject>& params) {
            entry_point = SlangProgramEntryPoint::create(cdf_program, name);
            pipeline = Versioned<Pipeline>([&entry_point, this] {
                const auto ep = entry_point.get();
                return ComputePipeline::create(ep->get_pipeline_layout(context), ep->specialize());
            });
            pipeline.depends_on(entry_point);
            params = Versioned<ShaderObject>([&entry_point, this] {
                return entry_point->create_shader_object_for_parameter(context, "params",
                                                                       allocator);
            });
            params.depends_on(entry_point);
        };
        make_cdf_pass("scan_blocks", cdf_blocks_entry_point, cdf_blocks_pipeline,
                      cdf_blocks_params);
        make_cdf_pass("scan_block_sums", cdf_sums_entry_point, cdf_sums_pipeline, cdf_sums_params);
        make_cdf_pass("add_block_offset", cdf_offset_entry_point, cdf_offset_pipeline,
                      cdf_offset_params);
    }
}

void LightCollection::prepare(const CommandBufferHandle& cmd) {
    if (!enabled)
        return;

    if (env_emissive) {
        if (env_importance_resized && env_importance_buffer) {
            cmd->keep_until_pool_reset(std::move(env_importance_buffer));
            env_importance_built = false;
        }
        env_importance_resized = false;
        const bool had_buffer = static_cast<bool>(env_importance_buffer);
        ensure_buffer(env_importance_buffer, env_importance_quad_count() * 4 * sizeof(float),
                      "LightCollection::env_importance", cmd);
        if (!had_buffer || env_importance_buffer != env_importance_built_buffer) {
            env_importance_built = false;
            env_importance_built_buffer = env_importance_buffer;
        }
        ensure_buffer(env_pool_buffer, static_cast<uint32_t>(env_pool_size) * sizeof(uint32_t),
                      "LightCollection::env_pool", cmd);
        ensure_buffer(env_split_buffer, sizeof(float), "LightCollection::env_split", cmd);
    }

    if (triangle_count > 0) {
        ensure_buffer(pool_source_buffer,
                      std::min(static_cast<uint32_t>(pool_size), LIGHT_GRID_POOL) *
                          sizeof(uint32_t),
                      "LightCollection::pool_source", cmd);
        ensure_buffer(sources_buffer, triangle_count * sizeof(LightSource),
                      "LightCollection::sources", cmd);
        ensure_buffer(source_of_buffer, triangle_count * sizeof(uint32_t),
                      "LightCollection::source_of", cmd);
        ensure_buffer(source_cdf_buffer, triangle_count * sizeof(float),
                      "LightCollection::source_cdf", cmd);
        ensure_buffer(pool_buffer, static_cast<uint32_t>(pool_size) * sizeof(uint32_t),
                      "LightCollection::pool", cmd);
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
        }
        if (info_grew) {
            cmd->barrier(vk::MemoryBarrier2{
                vk::PipelineStageFlagBits2::eTransfer,
                vk::AccessFlagBits2::eTransferWrite,
                vk::PipelineStageFlagBits2::eComputeShader,
                vk::AccessFlagBits2::eShaderRead | vk::AccessFlagBits2::eShaderWrite,
            });
        }

        if (selection == LightSelection::LightSelectionGrid) {
            grid_slot ^= 1u;
            const vk::DeviceSize slot_count =
                static_cast<vk::DeviceSize>(grid_cell_count()) * LIGHT_GRID_SLOTS;
            for (uint32_t i = 0; i < 2; i++) {
                const bool grew =
                    !grid_buffer[i] || grid_buffer[i]->get_size() < slot_count * sizeof(uint32_t);
                ensure_buffer(grid_buffer[i], slot_count * sizeof(uint32_t),
                              "LightCollection::grid", cmd);
                ensure_buffer(grid_visibility_buffer[i], slot_count * sizeof(float2),
                              "LightCollection::grid_visibility", cmd);
                ensure_buffer(grid_feedback_buffer[i], 2 * slot_count * sizeof(uint32_t),
                              "LightCollection::grid_feedback", cmd);
                ensure_buffer(grid_discovered_buffer[i],
                              grid_cell_count() * LIGHT_GRID_DISCOVERED * sizeof(uint32_t),
                              "LightCollection::grid_discovered", cmd);
                grid_reset |= grew;
            }
            if (slot_weighing == LightSlotWeighing::LightSlotWeighingShadingPoint) {
                ensure_buffer(grid_slot_sources_buffer, slot_count * sizeof(LightSource),
                              "LightCollection::grid_slot_sources", cmd);
                retire_buffer(grid_probability_buffer, cmd);
            } else {
                ensure_buffer(grid_probability_buffer, slot_count * sizeof(float),
                              "LightCollection::grid_probability", cmd);
                retire_buffer(grid_slot_sources_buffer, cmd);
            }
        } else {
            for (uint32_t i = 0; i < 2; i++) {
                retire_buffer(grid_buffer[i], cmd);
                retire_buffer(grid_visibility_buffer[i], cmd);
                retire_buffer(grid_feedback_buffer[i], cmd);
                retire_buffer(grid_discovered_buffer[i], cmd);
            }
            retire_buffer(grid_slot_sources_buffer, cmd);
            retire_buffer(grid_probability_buffer, cmd);
        }
    }

    if (triangle_count > 0) {
        update_light_remap(cmd);
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
        ensure_buffer(cdf_buffer, triangle_count * sizeof(float), "LightCollection::cdf", cmd);
        ensure_buffer(cdf_block_sums_buffer, cdf_block_count() * sizeof(float),
                      "LightCollection::cdf_block_sums", cmd);
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

void LightCollection::update_light_remap(const CommandBufferHandle& cmd) {
    light_remap.clear();
    if (selection == LightSelection::LightSelectionGrid &&
        grid_table_geometries != light_geometries) {
        light_index_of_key.clear();
        light_index_of_key.reserve(light_geometry_keys.size());
        for (uint32_t i = 0; i < light_geometry_keys.size(); i++) {
            light_index_of_key.try_emplace(light_geometry_keys[i], i);
        }
        light_remap.reserve(grid_table_geometries.size());
        for (uint32_t i = 0; i < grid_table_geometries.size(); i++) {
            const LightGeometry& old_geometry = grid_table_geometries[i];
            LightRemap& remap = light_remap.emplace_back(LightRemap{
                old_geometry.first_triangle, old_geometry.primitive_count, LIGHT_INVALID_INDEX, 0});
            if (const auto it = light_index_of_key.find(grid_table_keys[i]);
                it != light_index_of_key.end()) {
                remap.new_first = light_geometries[it->second].first_triangle;
                remap.new_count = light_geometries[it->second].primitive_count;
            }
        }
    }
    grid_table_keys = light_geometry_keys;
    grid_table_geometries = light_geometries;

    const vk::DeviceSize remap_size = std::max<size_t>(light_remap.size(), 1) * sizeof(LightRemap);
    ensure_buffer(light_remap_buffer, remap_size, "LightCollection::light_remap", cmd);
    if (light_remap.empty()) {
        return;
    }
    allocator->get_staging()->cmd_to_device(cmd, light_remap_buffer, light_remap.data(), 0,
                                            light_remap.size() * sizeof(LightRemap));
    cmd->barrier(vk::MemoryBarrier2{
        vk::PipelineStageFlagBits2::eTransfer,
        vk::AccessFlagBits2::eTransferWrite,
        vk::PipelineStageFlagBits2::eComputeShader,
        vk::AccessFlagBits2::eShaderRead,
    });
}

// Changes the composition, so it has to settle before the scene's shader object is fetched.
void LightCollection::update_constants(const SlangCompositionHandle& scene_composition) {
    if (!constants_dirty) {
        return;
    }
    scene_composition->add_module_from_string(
        "scene_light_constants",
        fmt::format("namespace merian {{\n"
                    "export static const int merian_nee_scene_draws = {};\n"
                    "export static const int merian_nee_cell_draws = {};\n"
                    "export static const int merian_nee_slot_weighing = {};\n"
                    "export static const bool merian_nee_env_regions = {};\n"
                    "}}",
                    scene_draws, cell_draws, slot_weighing, grid_env_regions));
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

    if (env_emissive) {
        MERIAN_PROFILE_SCOPE_GPU(cmd, "env importance");
        const uint32_t size = env_importance_size();
        // the coarsest level is 2 x 2: its single quad holds the mean
        const uint32_t levels = level_count_for(size) - 1;
        if (!env_importance_built) {
            const auto ep = env_build_entry_point.get();
            const auto pipe = env_build_pipeline.get();
            const auto params = env_build_params.get();
            auto c = params->get_cursor();
            c["levels"] = env_importance_buffer;
            c["size"] = size;
            c["level"] = 0u;
            c["level_size"] = size;

            cmd->bind(pipe);
            ep->bind("scene", scene_object, cmd, pipe, obj_allocator);
            ep->bind("params", params, cmd, pipe, obj_allocator);
            cmd->dispatch((size + ENV_GROUP_SIZE - 1) / ENV_GROUP_SIZE,
                          (size + ENV_GROUP_SIZE - 1) / ENV_GROUP_SIZE, 1);
        }

        if (!env_importance_built) {
            const auto ep = env_reduce_entry_point.get();
            const auto pipe = env_reduce_pipeline.get();
            cmd->bind(pipe);
            for (uint32_t level = 1; level < levels; level++) {
                cmd->barrier(vk::MemoryBarrier2{
                    vk::PipelineStageFlagBits2::eComputeShader,
                    vk::AccessFlagBits2::eShaderWrite,
                    vk::PipelineStageFlagBits2::eComputeShader,
                    vk::AccessFlagBits2::eShaderRead,
                });
                const uint32_t level_size = size >> level;
                const auto params = env_reduce_params[level - 1].get();
                auto c = params->get_cursor();
                c["levels"] = env_importance_buffer;
                c["size"] = size;
                c["level"] = level;
                c["level_size"] = level_size;
                ep->bind("params", params, cmd, pipe, obj_allocator);
                cmd->dispatch((level_size + ENV_GROUP_SIZE - 1) / ENV_GROUP_SIZE,
                              (level_size + ENV_GROUP_SIZE - 1) / ENV_GROUP_SIZE, 1);
            }
            env_importance_built = true;
        }

        if (env_selection == EnvSelection::EnvSelectionPool && env_pool_buffer) {
            cmd->barrier(vk::MemoryBarrier2{
                vk::PipelineStageFlagBits2::eComputeShader,
                vk::AccessFlagBits2::eShaderWrite,
                vk::PipelineStageFlagBits2::eComputeShader,
                vk::AccessFlagBits2::eShaderRead,
            });
            const auto pool_ep = env_pool_entry_point.get();
            const auto pool_pipe = env_pool_pipeline.get();
            const auto params = env_pool_params.get();
            auto c = params->get_cursor();
            c["levels"] = env_importance_buffer;
            c["pool"] = env_pool_buffer;
            c["size"] = size;
            c["level_count"] = level_count_for(size);
            c["pool_size"] = static_cast<uint32_t>(env_pool_size);
            c["frame"] = frame;

            cmd->bind(pool_pipe);
            pool_ep->bind("params", params, cmd, pool_pipe, obj_allocator);
            cmd->dispatch((static_cast<uint32_t>(env_pool_size) + 63) / 64, 1, 1);
        }
    }

    if (triangle_count > 0) {
        const auto ep = update_entry_point.get();
        const auto pipe = update_pipeline.get();
        const auto params = update_params.get();
        auto c = params->get_cursor();
        c["triangles"] = triangles_buffer;
        c["proxies"] = proxies_buffer;
        c["regions"] = regions_buffer;
        c["light_geometries"] = light_geometries_buffer;
        c["light_geometry_count"] = static_cast<uint32_t>(light_geometries.size());
        c["triangle_count"] = triangle_count;
        c["flux_samples"] = static_cast<uint32_t>(flux_samples);

        cmd->bind(pipe);
        ep->bind("scene", scene_object, cmd, pipe, obj_allocator);
        ep->bind("params", params, cmd, pipe, obj_allocator);
        cmd->dispatch((triangle_count + UPDATE_GROUP_SIZE - 1) / UPDATE_GROUP_SIZE, 1, 1);
    }

    cmd->barrier(vk::MemoryBarrier2{
        vk::PipelineStageFlagBits2::eComputeShader,
        vk::AccessFlagBits2::eShaderWrite,
        vk::PipelineStageFlagBits2::eComputeShader,
        vk::AccessFlagBits2::eShaderRead,
    });

    const auto split_env = [&] {
        if (!env_emissive || !env_split_buffer) {
            return;
        }
        MERIAN_PROFILE_SCOPE_GPU(cmd, "env split");
        const auto ep = env_split_entry_point.get();
        const auto pipe = env_split_pipeline.get();
        const auto params = env_split_params.get();
        auto c = params->get_cursor();
        c["cdf"] = cdf_buffer ? cdf_buffer : allocator->get_dummy_buffer();
        c["env_split"] = env_split_buffer;
        c["triangle_count"] = cdf_buffer ? triangle_count : 0u;
        c["scene_radius"] = scene_radius;
        c["manual_probability"] = env_share;
        c["from_power"] = static_cast<uint32_t>(env_share_from_power ? 1 : 0);
        c["min_probability"] = env_share_min;
        c["max_probability"] = env_share_max;
        auto env = c["env_importance"];
        env["levels"] = env_importance_buffer;
        env["size"] = env_importance_size();
        env["level_count"] = level_count_for(env_importance_size());

        cmd->bind(pipe);
        ep->bind("params", params, cmd, pipe, obj_allocator);
        cmd->dispatch(1, 1, 1);
    };

    if (triangle_count > 0) {
        MERIAN_PROFILE_SCOPE_GPU(cmd, "light selection");
        const auto barrier = [&] {
            cmd->barrier(vk::MemoryBarrier2{
                vk::PipelineStageFlagBits2::eComputeShader,
                vk::AccessFlagBits2::eShaderWrite,
                vk::PipelineStageFlagBits2::eComputeShader,
                vk::AccessFlagBits2::eShaderRead,
            });
        };
        const auto or_dummy = [&](const BufferHandle& buffer) -> const BufferHandle& {
            return buffer ? buffer : allocator->get_dummy_buffer();
        };
        const auto write_preprocess = [&](const ShaderObjectHandle& params) {
            auto c = params->get_cursor();
            c["triangles"] = triangles_buffer;
            c["proxies"] = proxies_buffer;
            c["cdf"] = cdf_buffer;
            c["pool"] = pool_buffer;
            c["light_geometries"] = light_geometries_buffer;
            c["light_geometry_count"] = static_cast<uint32_t>(light_geometries.size());
            c["light_remap"] = light_remap_buffer;
            c["light_remap_count"] = static_cast<uint32_t>(light_remap.size());
            c["sources_out"] = sources_buffer;
            c["sources"] = sources_buffer;
            c["source_of_out"] = source_of_buffer;
            c["source_of"] = source_of_buffer;
            c["source_cdf_out"] = source_cdf_buffer;
            c["pool_source_out"] = pool_source_buffer;
            c["pool_source"] = pool_source_buffer;
            c["grid"] = or_dummy(grid_buffer[grid_slot]);
            c["grid_visibility"] = or_dummy(grid_visibility_buffer[grid_slot]);
            c["grid_slot_sources"] = or_dummy(grid_slot_sources_buffer);
            c["grid_probability"] = or_dummy(grid_probability_buffer);
            c["grid_even_share"] = grid_even_share;
            c["slot_weighing"] = static_cast<uint32_t>(slot_weighing);
            const bool env_regions =
                grid_env_regions && env_emissive && env_importance_buffer && env_split_buffer;
            c["env_regions"] = env_regions;
            c["env_split"] = env_regions ? env_split_buffer : allocator->get_dummy_buffer();
            auto env = c["env_importance"];
            env["levels"] = env_regions ? env_importance_buffer : allocator->get_dummy_buffer();
            env["size"] = env_regions ? env_importance_size() : 0u;
            env["level_count"] = env_regions ? level_count_for(env_importance_size()) : 0u;
            c["grid_visibility_prev"] = or_dummy(grid_visibility_buffer[grid_slot ^ 1]);
            c["grid_feedback"] = or_dummy(grid_feedback_buffer[grid_slot]);
            c["grid_feedback_prev"] = or_dummy(grid_feedback_buffer[grid_slot ^ 1]);
            c["grid_discovered"] = or_dummy(grid_discovered_buffer[grid_slot]);
            c["grid_discovered_prev"] = or_dummy(grid_discovered_buffer[grid_slot ^ 1]);
            c["grid_prev"] = or_dummy(grid_buffer[grid_slot ^ 1]);
            c["grid_info"] = grid_info_buffer[grid_slot];
            c["grid_info_prev"] = grid_info_buffer[grid_slot ^ 1];
            c["grid_coverage"] = grid_coverage;
            c["camera_position"] = camera_position;
            c["triangle_count"] = triangle_count;
            c["pool_size"] = static_cast<uint32_t>(pool_size);
            c["grid_new_lights"] = static_cast<uint32_t>(grid_new_lights);
            c["grid_dimension"] = static_cast<uint32_t>(grid_dimension);
            c["grid_cascades"] = static_cast<uint32_t>(grid_cascades);
            c["grid_cell_size"] = grid_cell_size;
            c["grid_jitter"] = grid_jitter;
            c["grid_source_extent"] = grid_source_extent;
            c["grid_reset"] = grid_reset;
            c["frame"] = frame;
            return params;
        };

        const bool use_grid = selection == LightSelection::LightSelectionGrid;
        {
            MERIAN_PROFILE_SCOPE_GPU(cmd, "cdf");
            const uint32_t blocks = cdf_block_count();
            const auto write_cdf = [&](const ShaderObjectHandle& params) {
                auto c = params->get_cursor();
                c["proxies"] = proxies_buffer;
                c["cdf"] = cdf_buffer;
                c["block_sums"] = cdf_block_sums_buffer;
                c["triangle_count"] = triangle_count;
                c["block_count"] = blocks;
                return params;
            };
            const auto pass = [&](Versioned<SlangProgramEntryPoint>& entry_point,
                                  Versioned<Pipeline>& pipeline, Versioned<ShaderObject>& params,
                                  const uint32_t groups) {
                const auto ep = entry_point.get();
                const auto pipe = pipeline.get();
                cmd->bind(pipe);
                ep->bind("params", write_cdf(params.get()), cmd, pipe, obj_allocator);
                cmd->dispatch(groups, 1, 1);
            };
            pass(cdf_blocks_entry_point, cdf_blocks_pipeline, cdf_blocks_params, blocks);
            if (blocks > 1) {
                barrier();
                pass(cdf_sums_entry_point, cdf_sums_pipeline, cdf_sums_params, 1);
                barrier();
                pass(cdf_offset_entry_point, cdf_offset_pipeline, cdf_offset_params, blocks);
            }
        }
        barrier();
        split_env();
        if (selection != LightSelection::LightSelectionPower) {
            barrier();
            {
                MERIAN_PROFILE_SCOPE_GPU(cmd, "sources");
                const auto ep = sources_entry_point.get();
                const auto pipe = sources_pipeline.get();
                cmd->bind(pipe);
                ep->bind("params", write_preprocess(sources_params.get()), cmd, pipe,
                         obj_allocator);
                const uint32_t chunks =
                    (triangle_count + LIGHT_SOURCE_CHUNK - 1) / LIGHT_SOURCE_CHUNK;
                cmd->dispatch(chunks, 1, 1);
            }
            barrier();
            {
                MERIAN_PROFILE_SCOPE_GPU(cmd, "pool");
                const auto ep = pool_entry_point.get();
                const auto pipe = pool_pipeline.get();
                cmd->bind(pipe);
                ep->bind("params", write_preprocess(pool_params.get()), cmd, pipe, obj_allocator);
                cmd->dispatch((static_cast<uint32_t>(pool_size) + 63) / 64, 1, 1);
            }
        }
        if (use_grid) {
            barrier();
            {
                MERIAN_PROFILE_SCOPE_GPU(cmd, "grid setup");
                const auto ep = setup_entry_point.get();
                const auto pipe = setup_pipeline.get();
                cmd->bind(pipe);
                ep->bind("params", write_preprocess(setup_params.get()), cmd, pipe, obj_allocator);
                cmd->dispatch(1, 1, 1);
            }
            // the renderer's feedback, from any stage
            cmd->barrier(vk::MemoryBarrier2{
                vk::PipelineStageFlagBits2::eAllCommands,
                vk::AccessFlagBits2::eShaderWrite,
                vk::PipelineStageFlagBits2::eComputeShader,
                vk::AccessFlagBits2::eShaderRead | vk::AccessFlagBits2::eShaderWrite,
            });
            {
                MERIAN_PROFILE_SCOPE_GPU(cmd, "grid fill");
                const auto ep = grid_entry_point.get();
                const auto pipe = grid_pipeline.get();
                cmd->bind(pipe);
                ep->bind("params", write_preprocess(grid_params.get()), cmd, pipe, obj_allocator);
                // one group per cell
                const uint32_t cells = grid_cell_count();
                cmd->dispatch(std::min(cells, LIGHT_GRID_DISPATCH_ROW),
                              (cells + LIGHT_GRID_DISPATCH_ROW - 1) / LIGHT_GRID_DISPATCH_ROW, 1);
            }
            grid_reset = false;
        }
    }

    if (triangle_count == 0) {
        split_env();
    }

    cmd->barrier(vk::MemoryBarrier2{
        vk::PipelineStageFlagBits2::eComputeShader,
        vk::AccessFlagBits2::eShaderWrite,
        vk::PipelineStageFlagBits2::eAllCommands,
        vk::AccessFlagBits2::eShaderRead | vk::AccessFlagBits2::eShaderWrite,
    });
}

void LightCollection::write_to(ShaderCursor cursor) const {
    const bool active = enabled && triangle_count > 0 && triangles_buffer && cdf_buffer;
    const BufferHandle& dummy = allocator->get_dummy_buffer();
    auto table = cursor["table"];
    table["triangles"] = active ? triangles_buffer : dummy;
    table["regions"] = active ? regions_buffer : dummy;
    table["cdf"] = active ? cdf_buffer : dummy;
    table["geometries"] = active ? light_geometries_buffer : dummy;
    table["geometry_offsets"] = active ? geometry_light_offsets_buffer : dummy;
    table["triangle_count"] = active ? triangle_count : 0u;
    table["geometry_count"] = active ? static_cast<uint32_t>(light_geometries.size()) : 0u;
    table["geometry_offset_count"] =
        active ? static_cast<uint32_t>(geometry_light_offsets.size()) : 0u;

    auto pool = cursor["pool"];
    pool["entries"] = active ? pool_buffer : dummy;
    pool["size"] = active && selection != LightSelection::LightSelectionPower
                       ? static_cast<uint32_t>(pool_size)
                       : 0u;

    const bool grid_active =
        active && selection == LightSelection::LightSelectionGrid && grid_buffer[grid_slot];
    auto grid = cursor["grid"];
    grid["slots"] = grid_active ? grid_buffer[grid_slot] : dummy;
    grid["sources"] = active && sources_buffer ? sources_buffer : dummy;
    grid["source_of"] = active && source_of_buffer ? source_of_buffer : dummy;
    grid["source_cdf"] = active && source_cdf_buffer ? source_cdf_buffer : dummy;
    grid["cascades"] = grid_active ? grid_info_buffer[grid_slot] : dummy;
    grid["in_use"] = grid_active;
    grid["jitter"] = debug_jitter ? grid_jitter : 0.f;
    grid["frame"] = frame;
    grid["share"] = grid_share;
    grid["even_share"] = grid_even_share;
    grid["visibility"] = grid_active ? grid_visibility_buffer[grid_slot] : dummy;
    grid["slot_sources"] =
        grid_active && grid_slot_sources_buffer ? grid_slot_sources_buffer : dummy;
    grid["probability"] = grid_active && grid_probability_buffer ? grid_probability_buffer : dummy;
    grid["feedback"] = grid_active ? grid_feedback_buffer[grid_slot] : dummy;
    grid["discovered"] = grid_active ? grid_discovered_buffer[grid_slot] : dummy;

    cursor["selection"] =
        active ? static_cast<uint32_t>(selection) : static_cast<uint32_t>(LightSelectionPower);
    cursor["has_sky_portals"] = has_sky_portals;
    cursor["debug_view"] = static_cast<uint32_t>(debug_view);

    const bool env_active = enabled && env_emissive && env_importance_buffer && env_split_buffer;
    auto env = cursor["env"];
    env["split"] = env_active ? env_split_buffer : dummy;
    env["pool"] = env_active && env_pool_buffer ? env_pool_buffer : dummy;
    env["pool_size"] =
        env_active && env_pool_buffer && env_selection == EnvSelection::EnvSelectionPool
            ? static_cast<uint32_t>(env_pool_size)
            : 0u;
    auto importance = env["importance"];
    importance["levels"] = env_active ? env_importance_buffer : dummy;
    importance["size"] = env_active ? env_importance_size() : 0u;
    importance["level_count"] = env_active ? level_count_for(env_importance_size()) : 0u;
}

void LightCollection::properties(Properties& props) {
    props.config_bool("enable", enabled,
                      "Keep the emissive triangles as lights a renderer can sample directly (next "
                      "event estimation).");

    constants_dirty |= props.config_int(
        "scene draws", scene_draws,
        "Lights a shading point draws from the whole scene, in proportion to their power.", 0, 32);
    if (grid_enabled) {
        constants_dirty |= props.config_int(
            "cell draws", cell_draws,
            "Lights a shading point draws from the list of the cell it lies in, by what each "
            "delivers to it.",
            0, static_cast<int32_t>(LIGHT_GRID_MAX_DRAWS));
    }
    props.output_text(
        "A resampling renderer weighs all draws and traces one shadow ray; a mixture takes one.");

    if (props.st_begin_child("cells", "Cell lists")) {
        props.config_bool("enable", grid_enabled,
                          "Keep a list of the lights that reach every cell of a camera-anchored "
                          "grid, so a shading point draws from the few lights that matter where "
                          "it is.");
        if (grid_enabled) {
            props.st_separate("Drawing");
            props.config_percent("list share", grid_share,
                                 "Share of the cell draws taken from the cell's list. The rest "
                                 "are drawn from the whole scene, so that a light the cell does "
                                 "not list still has a chance.");
            constants_dirty |= props.config_options(
                "weigh at", slot_weighing, {"cell", "shading point"},
                Properties::OptionsStyle::COMBO,
                "Where the listed lights are weighed by what each delivers.\n"
                "cell: once per frame, at the cell. A draw and its density are a few loads, "
                "which suits renderers that often hit lights with other techniques, such as "
                "guiding.\n"
                "shading point: towards the shading point and its normal, so a draw favours the "
                "closest light. Every draw and density weighs all listed lights.");
            if (props.config_bool("sky regions", grid_env_regions,
                                  "List regions of the environment that rays from a cell reached, "
                                  "so the sky the cell sees through an opening is drawn like a "
                                  "light.")) {
                constants_dirty = true;
                grid_reset = true;
            }
            props.config_percent("even share", grid_even_share,
                                 "Share of the list draws that pick the listed lights alike "
                                 "instead of by what each delivers.");

            props.st_separate("List upkeep");
            props.config_int("new lights per frame", grid_new_lights,
                             "Lights drawn from the whole scene that every cell weighs against "
                             "its list each frame. Lights the renderer's rays hit from the cell "
                             "and lights its neighbours list are weighed too, and what a cell "
                             "learned carries across frames.",
                             1, static_cast<int32_t>(LIGHT_GRID_MAX_NEW_LIGHTS));
            props.config_float("source extent", grid_source_extent,
                               "Largest extent of one listed light, in cells of the finest "
                               "cascade. An emissive mesh that spreads further, such as a "
                               "particle system, is listed as several lights.",
                               0.1f, 0.f);

            props.st_separate("Layout");
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
        }
        props.st_end_child();
    }

    if (props.st_begin_child("scene", "Whole-scene draws")) {
        props.config_bool("presampled", pool_presampled,
                          "Draw a pool of lights once per frame, so a draw is one load. Off "
                          "searches all lights per draw.");
        if (pool_presampled) {
            props.config_int("pool size", pool_size, "Lights in the per-frame pool.", 64, 65536);
        }
        props.st_end_child();
    }

    if (props.st_begin_child("environment", "Environment")) {
        props.config_bool("share from power", env_share_from_power,
                          "Derive how often a whole-scene draw takes the environment from the "
                          "power it emits against the triangles. That power is blind to "
                          "occlusion, so the limits keep an enclosed scene from spending its "
                          "draws on a sky it barely sees.");
        if (env_share_from_power) {
            props.config_percent("share min", env_share_min);
            props.config_percent("share max", env_share_max);
        } else {
            props.config_percent("share", env_share,
                                 "How often a whole-scene draw takes the environment instead of "
                                 "a triangle, where both exist.");
        }
        props.config_options("sampling", env_selection, {"warp", "pool"},
                             Properties::OptionsStyle::COMBO,
                             "Descend the importance map per draw, or load from a pool of "
                             "directions drawn once per frame.");
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
                "slots: occupied slots (red), the visibility the cell learned for what it picks "
                "(green), their actual visibility (blue).\n"
                "coverage: of the direct light {} cosine-distributed rays find, the share the "
                "draws can reach here (red), the share of lights not offered here (green), the "
                "share no draw can ever reach (blue).",
                LIGHT_DEBUG_DRAWS, LIGHT_DEBUG_COVERAGE_RAYS));
        props.config_bool("jitter the lookup", debug_jitter,
                          "Off shows the cells the lookup sees without its random shift.");
        props.st_end_child();
    }

    const int32_t previous_selection = selection;
    selection = grid_enabled ? LightSelection::LightSelectionGrid
                             : (pool_presampled ? LightSelection::LightSelectionPool
                                                : LightSelection::LightSelectionPower);
    grid_reset |= selection == LightSelection::LightSelectionGrid &&
                  previous_selection != LightSelection::LightSelectionGrid;

    props.output_text(fmt::format("emissive geometries: {}\nemissive triangles: {}",
                                  light_geometries.size(), triangle_count));
}

} // namespace merian
