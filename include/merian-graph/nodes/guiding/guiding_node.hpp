#pragma once

#include "merian-graph/connectors/shader_object_out.hpp"
#include "merian-graph/graph/node.hpp"
#include "merian-graph/objects/guiding_object.hpp"

namespace merian {

// A directional guiding method as a graph node: it owns the method and its properties, and hands
// a path tracer the object that plugs into its guiding slot.
class GuidingNode : public Node {
  public:
    ~GuidingNode() override = default;

    void initialize(const ContextHandle& context,
                    const ResourceAllocatorHandle& allocator) override {
        model->initialize(context, allocator);
    }

    std::vector<OutputConnectorDescriptor>
    describe_outputs(const NodeIOLayout& io_layout) override {
        configure(io_layout);
        con_guiding = ShaderObjectOut<GuidingObject>::create({model, version}, true);
        needs_reset = true;
        return {{"guiding", con_guiding}};
    }

    NodeStatusFlags
    process(const NodeIO& io, const NodeProcessInfo& info, Submission& submission) override {
        if (info.get_iteration() == 0 || needs_reset) {
            model->reset(submission.get_cmd());
            needs_reset = false;
        }
        io[con_guiding]->write();
        return {};
    }

    NodeStatusFlags properties(Properties& config) override {
        // the method's type name and constants are baked into the consumer's program
        if (!model->properties(config)) {
            return {};
        }
        version++;
        return NEEDS_RECONNECT;
    }

  protected:
    // Built by the concrete node's constructor: properties are loaded before initialize().
    explicit GuidingNode(const GuidingModelHandle& model) : model(model) {}

    // Whatever the method takes from the graph around it, before the object is created.
    virtual void configure([[maybe_unused]] const NodeIOLayout& io_layout) {}

    const GuidingModelHandle model;
    ShaderObjectOutHandle<GuidingObject> con_guiding;
    bool needs_reset = true;
    uint32_t version = 0;
};

} // namespace merian
