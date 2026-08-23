#include "merian-graph/nodes/guiding/mcpg_distance/mcpg_distance_guiding_node.hpp"

#include "merian/vk/utils/profiler.hpp"

#include <array>

namespace merian {

namespace {

constexpr const char* CLEAR_MODULE =
    "merian-graph/nodes/guiding/mcpg_distance/mcpg-distance-clear.slang";
constexpr const char* PROJECT_MODULE =
    "merian-graph/nodes/guiding/mcpg_distance/mcpg-distance-project.slang";

} // namespace

void MCPGDistanceGuidingNode::initialize(const ContextHandle& context,
                                         const ResourceAllocatorHandle& allocator) {
    this->context = context;
    this->resource_allocator = allocator;
    GuidingNode::initialize(context, allocator);
}

std::vector<InputConnectorDescriptor> MCPGDistanceGuidingNode::describe_inputs() {
    return {{.name = "scene", .connector = con_scene, .optional = true},
            {.name = "gbuffer", .connector = con_gbuffer, .access = ConnectorAccess::compute_read}};
}

void MCPGDistanceGuidingNode::configure(const NodeIOLayout& io_layout) {
    extent = io_layout[con_gbuffer]->get_create_info().extent;
    if (chains().on_extent(extent)) {
        // the level count is a link-time constant of both the slot and the passes
        clear_composition = nullptr;
        project_composition = nullptr;
    }
}

void MCPGDistanceGuidingNode::ensure_pipelines(const SceneHandle& scene) {
    const auto build = [this](Pass& pass, const Versioned<SlangProgram>& prog,
                              const std::string& name) {
        pass.entry_point = SlangProgramEntryPoint::create(prog, name);
        pass.pipeline = Versioned<Pipeline>([&pass, this] {
            const auto ep = pass.entry_point.get();
            return ComputePipeline::create(ep->get_pipeline_layout(context), ep->specialize());
        });
        pass.pipeline.depends_on(pass.entry_point);
        pass.params = Versioned<ShaderObject>([&pass, this] {
            return pass.entry_point->create_shader_object_for_parameter(context, "params",
                                                                        resource_allocator);
        });
        pass.params.depends_on(pass.entry_point);
    };

    if (!clear_composition) {
        clear_composition = SlangComposition::create();
        clear_composition->add_composition(model->get_composition());
        clear_composition->add_module_from_path(CLEAR_MODULE, true);
        clear_program =
            SlangProgram::create(context->get_shader_compile_context(), clear_composition);
        build(clear, clear_program, "main");
    }

    // the projection reprojects through the scene's cameras, so it only exists with one
    if (scene && !project_composition) {
        project_composition = SlangComposition::create();
        project_composition->add_composition(scene->get_composition());
        project_composition->add_composition(model->get_composition());
        project_composition->add_module_from_path(PROJECT_MODULE, true);
        project_program =
            SlangProgram::create(context->get_shader_compile_context(), project_composition);
        build(project, project_program, "main");
        for (auto& level_params : project_params) {
            level_params = Versioned<ShaderObject>([this] {
                return project.entry_point->create_shader_object_for_parameter(context, "params",
                                                                               resource_allocator);
            });
            level_params.depends_on(project.entry_point);
        }
    }
}

MCPGDistanceGuidingNode::NodeStatusFlags MCPGDistanceGuidingNode::process(
    const NodeIO& io, const NodeProcessInfo& info, Submission& submission) {
    const bool has_scene = io.is_connected(con_scene);
    if (has_scene && (!io[con_scene] || !io[con_scene]->is_ready())) {
        return {};
    }
    ensure_pipelines(has_scene ? SceneHandle(io[con_scene]) : SceneHandle{});

    const CommandBufferHandle& cmd = submission.get_cmd();
    const ShaderObjectAllocatorHandle& obj_allocator = info.get_shader_object_allocator();
    const uint32_t levels = chains().get_level_count();

    // the tracer wrote the grid last frame; the projection moves it into this frame's
    chains().swap();

    // the grids are the node's own, so the graph does not order last frame's tracer writes
    // against the passes below
    const std::array<vk::ImageMemoryBarrier2, 2> carry = {
        chains().get_grid()->barrier2(
            vk::ImageLayout::eGeneral, vk::AccessFlagBits2::eMemoryWrite,
            vk::AccessFlagBits2::eMemoryRead | vk::AccessFlagBits2::eMemoryWrite,
            vk::PipelineStageFlagBits2::eAllCommands, vk::PipelineStageFlagBits2::eComputeShader),
        chains().get_prev_grid()->barrier2(
            vk::ImageLayout::eGeneral, vk::AccessFlagBits2::eMemoryWrite,
            vk::AccessFlagBits2::eMemoryRead, vk::PipelineStageFlagBits2::eAllCommands,
            vk::PipelineStageFlagBits2::eComputeShader),
    };
    cmd->barrier({}, {}, carry);

    const auto write_grid = [&](const ShaderObjectHandle& obj) {
        auto grid_levels = obj->get_cursor()["grid"]["levels"];
        for (uint32_t i = 0; i < levels; i++) {
            grid_levels[i] = chains().get_levels()[i];
        }
        return obj;
    };

    const auto write_project = [&](const ShaderObjectHandle& obj, const uint32_t level) {
        auto cursor = write_grid(obj)->get_cursor();
        cursor["prev"].write(chains().get_prev_texture(), vk::ImageLayout::eGeneral);
        cursor["level"] = level;
        cursor["dim"] = uint2{extent.width, extent.height};
        return obj;
    };

    const auto barrier_grid = [&] {
        cmd->barrier(chains().get_grid()->barrier2(
            vk::ImageLayout::eGeneral, vk::AccessFlagBits2::eShaderWrite,
            vk::AccessFlagBits2::eShaderRead | vk::AccessFlagBits2::eShaderWrite,
            vk::PipelineStageFlagBits2::eComputeShader,
            vk::PipelineStageFlagBits2::eComputeShader));
    };

    MERIAN_PROFILE_SCOPE_GPU(info.get_profiler(), cmd, "distance project");
    {
        const auto ep = clear.entry_point.get();
        const auto pipe = clear.pipeline.get();
        cmd->bind(pipe);
        ep->bind("params", write_grid(clear.params.get()), cmd, pipe, obj_allocator);
        cmd->dispatch(chains().get_grid()->get_extent(), 8, 8);
        barrier_grid();
    }

    // the previous grid only holds anything from the second iteration on
    if (has_scene && info.get_iteration() != 0) {
        const auto ep = project.entry_point.get();
        const auto pipe = project.pipeline.get();
        cmd->bind(pipe);
        ep->bind("scene", io[con_scene]->get_shader_object(), cmd, pipe, obj_allocator);
        for (uint32_t level = 0; level < levels; level++) {
            ep->bind("params", write_project(project_params[level].get(), level), cmd, pipe,
                     obj_allocator);
            cmd->dispatch(chains().get_grid()->get_extent(), 8, 8);
            barrier_grid();
        }
    }

    io[con_guiding]->write();
    return {};
}

} // namespace merian
