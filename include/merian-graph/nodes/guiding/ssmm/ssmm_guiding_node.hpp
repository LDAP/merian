#pragma once

#include "merian-graph/connectors/shader_object_in.hpp"
#include "merian-graph/nodes/guiding/guiding_node.hpp"
#include "merian-graph/nodes/guiding/ssmm/ssmm_guiding_model.hpp"
#include "merian-graph/objects/gbuffer_object.hpp"

namespace merian {

// Screen-space mixture models: one vMF lobe per pixel, reprojected through the gbuffer.
class SSMMGuidingNode : public GuidingNode {
  public:
    SSMMGuidingNode() : GuidingNode(std::make_shared<SSMMGuidingModel>()) {}

    std::vector<InputConnectorDescriptor> describe_inputs() override {
        return {{"gbuffer", con_gbuffer, ConnectorAccess::ray_tracing_read}};
    }

    NodeStatusFlags
    process(const NodeIO& io, const NodeProcessInfo& info, Submission& submission) override {
        ssmm().set_gbuffer(io[con_gbuffer].r());
        // the fits are the node's own, so the graph does not order last frame's writes against
        // this frame's reads
        submission.get_cmd()->barrier(ssmm().carry_barriers());
        return GuidingNode::process(io, info, submission);
    }

  protected:
    // the fit is one lobe per pixel of whatever the path tracer renders into
    void configure(const NodeIOLayout& io_layout) override {
        ssmm().on_extent(io_layout[con_gbuffer]->get_create_info().extent);
    }

  private:
    SSMMGuidingModel& ssmm() const {
        return static_cast<SSMMGuidingModel&>(*model);
    }

    ShaderObjectInHandle<GBufferObject> con_gbuffer = ShaderObjectIn<GBufferObject>::create();
};

} // namespace merian
