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

    if (!env_composition) {
        env_composition = SlangComposition::create();
        env_composition->add_composition(scene_composition);
        env_composition->add_module_from_path("merian-shaders/scene/env-importance-build.slang",
                                              true);
        env_program = SlangProgram::create(compile_context, env_composition);
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
    cursor["geometry_light_offsets"] = active ? geometry_light_offsets_buffer : dummy;
    cursor["triangle_count"] = active ? triangle_count : 0u;
    cursor["geometry_count"] = active ? static_cast<uint32_t>(geometry_light_offsets.size()) : 0u;
    cursor["p_env"] = enabled && env_emissive ? env_probability : 0.f;
    cursor["has_sky_portals"] = has_sky_portals;

    const bool env_active = enabled && env_emissive && env_importance_buffer;
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
