#pragma once

#include "merian-graph/graph/graph_shader_object.hpp"

#include "merian/shader/slang_program.hpp"

#include <memory>

namespace merian {

template <typename Model> class SlotObject : public GraphShaderObject {
  public:
    struct CreateInfo {
        std::shared_ptr<Model> model;
        // bumped whenever the model's properties changed what a consumer bakes into its program
        uint32_t version;
    };

    SlotObject(const CreateInfo& create_info) : model(create_info.model) {}

    void allocate(const ShaderObjectAllocateInfo& info) override {
        const SlangProgramHandle program =
            SlangProgram::create(info.compile_context, model->get_composition()).get();
        shader_object = program->create_shader_object_for_type(info.context, model->get_type_name(),
                                                               info.allocator);
    }

    const ShaderObjectHandle& object([[maybe_unused]] const ShaderAccess access) const override {
        return shader_object;
    }

    const std::shared_ptr<Model>& get_model() const {
        return model;
    }

    void write() const {
        model->write_to(shader_object->get_cursor());
    }

  private:
    const std::shared_ptr<Model> model;

    ShaderObjectHandle shader_object;
};

} // namespace merian
