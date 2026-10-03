#pragma once

#include "merian-graph/connectors/shader_object_out.hpp"
#include "merian-graph/graph/node.hpp"
#include "merian-graph/objects/guiding_object.hpp"

namespace merian {

template <typename Model> class GuidingNodeBase : public Node {
  public:
    ~GuidingNodeBase() override = default;

    void initialize(const ContextHandle& context,
                    const ResourceAllocatorHandle& allocator) override {
        model->initialize(context, allocator);
    }

    std::vector<OutputConnectorDescriptor>
    describe_outputs(const NodeIOLayout& io_layout) override {
        configure(io_layout);
        con_guiding =
            ShaderObjectOut<SlotObject<Model>>::create({.model = model, .version = version}, true);
        needs_reset = true;
        return {{.name = "guiding", .connector = con_guiding}};
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
        if (!model->properties(config)) {
            return {};
        }
        version++;
        return NEEDS_RECONNECT;
    }

  protected:
    explicit GuidingNodeBase(const std::shared_ptr<Model>& model) : model(model) {}

    virtual void configure([[maybe_unused]] const NodeIOLayout& io_layout) {}

    const std::shared_ptr<Model> model;
    ShaderObjectOutHandle<SlotObject<Model>> con_guiding;
    bool needs_reset = true;
    uint32_t version = 0;
};

class GuidingNode : public GuidingNodeBase<GuidingModel> {
  public:
    std::vector<OutputConnectorDescriptor>
    describe_outputs(const NodeIOLayout& io_layout) override {
        std::vector<OutputConnectorDescriptor> outputs =
            GuidingNodeBase::describe_outputs(io_layout);
        const IrradianceCacheHandle cache = model->get_irradiance_cache();
        con_irradiance_cache = ShaderObjectOut<IrradianceCacheObject>::create(
            {.model = cache ? cache : std::make_shared<NullIrradianceCache>(), .version = version},
            true);
        outputs.push_back(
            {.name = "irradiance_cache", .connector = con_irradiance_cache, .disabled = !cache});
        return outputs;
    }

    NodeStatusFlags
    process(const NodeIO& io, const NodeProcessInfo& info, Submission& submission) override {
        const NodeStatusFlags flags = GuidingNodeBase::process(io, info, submission);
        if (io.is_connected(con_irradiance_cache)) {
            io[con_irradiance_cache]->write();
        }
        return flags;
    }

  protected:
    explicit GuidingNode(const GuidingModelHandle& model) : GuidingNodeBase(model) {}

  private:
    ShaderObjectOutHandle<IrradianceCacheObject> con_irradiance_cache;
};

using DistanceGuidingNode = GuidingNodeBase<DistanceGuidingModel>;

} // namespace merian
