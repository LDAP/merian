#pragma once

#include "merian-graph/graph/graph_shader_object.hpp"
#include "merian-shaders/sampling/guiding.hpp"

#include "merian/shader/slang_program.hpp"

namespace merian {

// A guiding method transported through the graph: the resources a path tracer binds, together
// with the slang type it aliases into its guiding slot.
class GuidingObject : public GraphShaderObject {
  public:
    struct CreateInfo {
        GuidingModelHandle model;
        // bumped whenever the method's properties changed what a consumer bakes into its program
        uint32_t version;
    };

    GuidingObject(const CreateInfo& create_info) : model(create_info.model) {}

    void allocate(const ShaderObjectAllocateInfo& info) override {
        const SlangProgramHandle program =
            SlangProgram::create(info.compile_context, model->get_composition()).get();
        shader_object = program->create_shader_object_for_type(info.context, model->get_type_name(),
                                                               info.allocator);
    }

    const ShaderObjectHandle& object([[maybe_unused]] const ShaderAccess access) const override {
        return shader_object;
    }

    const GuidingModelHandle& get_model() const {
        return model;
    }

    // Binds the method for the frame about to be rendered.
    void write() const {
        model->write_to(shader_object->get_cursor());
    }

  private:
    const GuidingModelHandle model;

    ShaderObjectHandle shader_object;
};

} // namespace merian
