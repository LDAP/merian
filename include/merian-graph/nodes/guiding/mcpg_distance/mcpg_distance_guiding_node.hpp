#pragma once

#include "merian-graph/connectors/ptr_in.hpp"
#include "merian-graph/connectors/shader_object_in.hpp"
#include "merian-graph/nodes/guiding/guiding_node.hpp"
#include "merian-graph/nodes/guiding/mcpg_distance/mcpg_distance_guiding_model.hpp"
#include "merian-graph/objects/gbuffer_object.hpp"
#include "merian-shaders/scene/scene.hpp"

#include "merian/shader/slang_entry_point.hpp"
#include "merian/vk/pipeline/pipeline_compute.hpp"

namespace merian {

// Distance guiding for a path tracer's volume pass. Owns the screen-space chain grid and carries
// it across frames itself: the projection is this node's work, not the renderer's.
class MCPGDistanceGuidingNode : public GuidingNode {
  public:
    MCPGDistanceGuidingNode() : GuidingNode(std::make_shared<MCPGDistanceGuidingModel>()) {}

    void initialize(const ContextHandle& context,
                    const ResourceAllocatorHandle& allocator) override;

    std::vector<InputConnectorDescriptor> describe_inputs() override;

    NodeStatusFlags
    process(const NodeIO& io, const NodeProcessInfo& info, Submission& submission) override;

  protected:
    void configure(const NodeIOLayout& io_layout) override;

  private:
    void ensure_pipelines(const SceneHandle& scene);

    // Without a scene the chains cannot be moved into this frame, so they start fresh each one.
    MCPGDistanceGuidingModel& chains() const {
        return static_cast<MCPGDistanceGuidingModel&>(*model);
    }

    ContextHandle context;
    ResourceAllocatorHandle resource_allocator;

    PtrInHandle<Scene> con_scene = PtrIn<Scene>::create();
    ShaderObjectInHandle<GBufferObject> con_gbuffer = ShaderObjectIn<GBufferObject>::create();

    vk::Extent3D extent{};

    SlangCompositionHandle clear_composition;
    SlangCompositionHandle project_composition;
    Versioned<SlangProgram> clear_program;
    Versioned<SlangProgram> project_program;
    struct Pass {
        Versioned<SlangProgramEntryPoint> entry_point;
        Versioned<Pipeline> pipeline;
        Versioned<ShaderObject> params;
    };
    Pass clear;
    Pass project;
    // One object per level: a shader object must not be rewritten within a submission.
    std::array<Versioned<ShaderObject>, MCPGDistanceGuidingModel::MAX_LEVELS> project_params;
};

} // namespace merian
