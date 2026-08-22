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
    std::vector<uint32_t> new_offsets(geometry_count, LIGHT_INVALID_INDEX);
    uint32_t first_triangle = 0;
    for (const EmissiveGeometry& g : geometries) {
        if (g.primitive_count == 0)
            continue;
        assert(g.geometry_id < geometry_count);
        new_geometries.push_back(
            {g.geometry_id, g.instance_index, first_triangle, g.primitive_count});
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
        cdf_entry_point = SlangProgramEntryPoint::create(cdf_program, "main");
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
        if (env_importance_resized && env_importance_buffer) {
            cmd->keep_until_pool_reset(std::move(env_importance_buffer));
        }
        env_importance_resized = false;
        ensure_buffer(env_importance_buffer, env_importance_quad_count() * 4 * sizeof(float),
                      "LightCollection::env_importance", cmd);
        ensure_buffer(env_pool_buffer, static_cast<uint32_t>(env_pool_size) * sizeof(uint32_t),
                      "LightCollection::env_pool", cmd);
    }

    if (triangle_count > 0) {
        ensure_buffer(pool_buffer, static_cast<uint32_t>(pool_size) * sizeof(uint32_t),
                      "LightCollection::pool", cmd);
        ensure_buffer(grid_buffer, grid_cell_count() * LIGHT_GRID_SLOTS * sizeof(uint32_t),
                      "LightCollection::grid", cmd);
        ensure_buffer(grid_info_buffer, sizeof(LightGridInfo), "LightCollection::grid_info", cmd);
    }

    if (triangle_count > 0 && tables_dirty) {
        const auto staging = allocator->get_staging();
        const vk::DeviceSize geometries_size = light_geometries.size() * sizeof(LightGeometry);
        const vk::DeviceSize offsets_size = geometry_light_offsets.size() * sizeof(uint32_t);
        ensure_buffer(light_geometries_buffer, geometries_size, "LightCollection::geometries", cmd);
        ensure_buffer(geometry_light_offsets_buffer, offsets_size, "LightCollection::offsets", cmd);
        ensure_buffer(triangles_buffer, triangle_count * sizeof(EmissiveTriangle),
                      "LightCollection::triangles", cmd);
        ensure_buffer(cdf_buffer, triangle_count * sizeof(float), "LightCollection::cdf", cmd);
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

void LightCollection::update(const CommandBufferHandle& cmd,
                             const SlangCompositionHandle& scene_composition,
                             const ShaderObjectHandle& scene_object,
                             const ShaderObjectAllocatorHandle& obj_allocator_in,
                             const uint32_t frame) {
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
        {
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
        const auto write_preprocess = [&](const ShaderObjectHandle& params) {
            auto c = params->get_cursor();
            c["triangles"] = triangles_buffer;
            c["cdf"] = cdf_buffer;
            c["pool"] = pool_buffer;
            c["grid"] = grid_buffer;
            c["grid_info"] = grid_info_buffer;
            c["grid_coverage"] = grid_coverage;
            c["camera_position"] = camera_position;
            c["triangle_count"] = triangle_count;
            c["pool_size"] = static_cast<uint32_t>(pool_size);
            c["grid_candidates"] = static_cast<uint32_t>(grid_candidates);
            c["grid_dimension"] = static_cast<uint32_t>(grid_dimension);
            c["grid_cell_size"] = grid_cell_size;
            c["grid_visibility"] =
                static_cast<uint32_t>(grid_visibility && acceleration_structure ? 1 : 0);
            if (acceleration_structure) {
                c["as"] = acceleration_structure;
            }
            c["selection"] = static_cast<uint32_t>(selection);
            c["frame"] = frame;
            return params;
        };

        const bool use_grid = selection == LightSelection::LightSelectionGrid;
        {
            const auto ep = cdf_entry_point.get();
            const auto pipe = cdf_pipeline.get();
            const auto params = cdf_params.get();
            auto c = params->get_cursor();
            c["triangles"] = triangles_buffer;
            c["cdf"] = cdf_buffer;
            c["triangle_count"] = triangle_count;

            cmd->bind(pipe);
            ep->bind("params", params, cmd, pipe, obj_allocator);
            cmd->dispatch(1, 1, 1);
        }
        if (selection != LightSelection::LightSelectionPower) {
            barrier();
            {
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
                const auto ep = setup_entry_point.get();
                const auto pipe = setup_pipeline.get();
                cmd->bind(pipe);
                ep->bind("params", write_preprocess(setup_params.get()), cmd, pipe, obj_allocator);
                cmd->dispatch(1, 1, 1);
            }
            barrier();
            {
                const auto ep = grid_entry_point.get();
                const auto pipe = grid_pipeline.get();
                cmd->bind(pipe);
                ep->bind("params", write_preprocess(grid_params.get()), cmd, pipe, obj_allocator);
                cmd->dispatch((grid_cell_count() + 63) / 64, 1, 1);
            }
        }
    }

    cmd->barrier(vk::MemoryBarrier2{
        vk::PipelineStageFlagBits2::eComputeShader,
        vk::AccessFlagBits2::eShaderWrite,
        vk::PipelineStageFlagBits2::eAllCommands,
        vk::AccessFlagBits2::eShaderRead,
    });
}

void LightCollection::write_to(ShaderCursor cursor) const {
    const bool active = enabled && triangle_count > 0 && triangles_buffer && cdf_buffer;
    const BufferHandle& dummy = allocator->get_dummy_buffer();
    cursor["triangles"] = active ? triangles_buffer : dummy;
    cursor["cdf"] = active ? cdf_buffer : dummy;
    cursor["pool"] = active ? pool_buffer : dummy;
    cursor["grid"] = active ? grid_buffer : dummy;
    cursor["geometry_light_offsets"] = active ? geometry_light_offsets_buffer : dummy;
    cursor["grid_info"] = active && grid_info_buffer ? grid_info_buffer : dummy;
    cursor["pool_size"] = active && selection != LightSelection::LightSelectionPower
                              ? static_cast<uint32_t>(pool_size)
                              : 0u;
    cursor["pool_tile_size"] =
        std::min(static_cast<uint32_t>(pool_tile_size), static_cast<uint32_t>(pool_size));
    cursor["selection"] = static_cast<uint32_t>(selection);
    cursor["grid_probability"] = grid_probability;
    cursor["triangle_count"] = active ? triangle_count : 0u;
    cursor["geometry_count"] = active ? static_cast<uint32_t>(geometry_light_offsets.size()) : 0u;
    cursor["p_env"] = enabled && env_emissive ? env_probability : 0.f;
    cursor["has_sky_portals"] = has_sky_portals;

    const bool env_active = enabled && env_emissive && env_importance_buffer;
    cursor["env_pool"] = env_active && env_pool_buffer ? env_pool_buffer : dummy;
    cursor["env_pool_size"] =
        env_active && env_pool_buffer && env_selection == EnvSelection::EnvSelectionPool
            ? static_cast<uint32_t>(env_pool_size)
            : 0u;
    auto env = cursor["env_importance"];
    env["levels"] = env_active ? env_importance_buffer : dummy;
    env["size"] = env_active ? env_importance_size() : 0u;
    env["level_count"] = env_active ? level_count_for(env_importance_size()) : 0u;
}

void LightCollection::properties(Properties& props) {
    props.config_bool("enable", enabled,
                      "Maintain the emissive triangle list for next event estimation.");
    props.config_percent("env probability", env_probability,
                         "Probability of sampling the environment map instead of an emissive "
                         "triangle when both exist.");
    props.config_options("env selection", env_selection, {"warp", "pool"},
                         Properties::OptionsStyle::COMBO,
                         "How a direction towards the environment is chosen: a pyramid descent per "
                         "sample, or one load from a per-frame pool of texels it drew.");
    if (env_selection == EnvSelection::EnvSelectionPool) {
        props.config_int("env pool size", env_pool_size, "Environment texels pre-drawn per frame.",
                         64, 262144);
    }
    props.config_options("selection", selection, {"power", "pool", "grid"},
                         Properties::OptionsStyle::COMBO,
                         "How a light is chosen: a search over the whole scene, one load from a "
                         "pre-resampled pool, or one load from a per-frame world-space grid.");
    if (selection != LightSelection::LightSelectionPower) {
        props.config_int("pool tile size", pool_tile_size,
                         "Lights a screen tile draws from. Smaller keeps the entries in cache; the "
                         "density does not depend on it.",
                         16, 65536);
        props.config_int("pool size", pool_size,
                         "Lights pre-resampled per frame; a sample is one load from it.", 64,
                         65536);
    }
    if (selection == LightSelection::LightSelectionGrid) {
        props.config_int("grid dimension", grid_dimension, "Cells per side of the light grid.", 4,
                         128);
        props.config_int("grid candidates", grid_candidates,
                         "Pool draws resampled into each of the cell's slots.", 1, 64);
        props.config_percent("grid probability", grid_probability,
                             "How often the cell list is used; the rest falls back to the pool so "
                             "that every light stays reachable.");
        props.config_float("grid cell size", grid_cell_size,
                           "World units per cell; 0 derives it from the light bounds.", 0.1f, 0.f);
        props.config_float("grid coverage", grid_coverage,
                           "Fraction of the light bounds the grid spans, when the cell size is "
                           "derived.",
                           0.05f, 0.01f, 16.f);
        props.config_bool("grid visibility", grid_visibility,
                          "Drop cell entries the cell cannot see. Costs one ray per slot.");
    }
    props.config_int("flux samples", flux_samples,
                     "Emission evaluations per triangle for the flux estimate.", 1, 256);
    int32_t env_size_option = env_importance_log2 - 6;
    if (props.config_options("env importance map", env_size_option, {"64", "128", "256", "512"},
                             Properties::OptionsStyle::COMBO,
                             "Side length of the environment importance map, rebuilt every "
                             "frame.")) {
        env_importance_log2 = env_size_option + 6;
        env_importance_resized = true;
    }
    props.output_text(fmt::format("emissive geometries: {}\nemissive triangles: {}",
                                  light_geometries.size(), triangle_count));
}

} // namespace merian
