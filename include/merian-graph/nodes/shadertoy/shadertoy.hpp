#pragma once

#include "merian-graph/connectors/ptr_in.hpp"
#include "merian-graph/nodes/compute_node/compute_node.hpp"

#include "merian/shader/shader_compiler.hpp"
#include "merian/shader/shader_hotreloader.hpp"
#include "merian/utils/input_listener.hpp"

namespace merian {

// A generator node that pushes the Shadertoy variables as push constant.
class Shadertoy : public AbstractCompute {
  private:
    static constexpr uint32_t local_size_x = 16;
    static constexpr uint32_t local_size_y = 16;

  private:
    struct PushConstant {
        merian::float4 iMouse{};
        merian::float4 iDate{};
        merian::float2 iResolution{};
        float iTime{};
        float iTimeDelta{};
        int32_t iFrame{};
        int32_t pad_1{};
        int32_t pad_2{};
        int32_t pad_3{};
    };

  public:
    Shadertoy();

    DeviceSupportInfo query_device_support(const DeviceSupportQueryInfo& query_info) override;

    void initialize(const ContextHandle& context,
                    const ResourceAllocatorHandle& allocator) override;

    std::vector<InputConnectorDescriptor> describe_inputs() override;

    std::vector<OutputConnectorDescriptor> describe_outputs(const NodeIOLayout& io_layout) override;

    const void* get_push_constant(GraphRun& run, const NodeIO& io) override;

    std::tuple<uint32_t, uint32_t, uint32_t>
    get_group_count(const NodeIO& io) const noexcept override;

    VulkanEntryPointHandle get_entry_point() override;

    NodeStatusFlags properties(Properties& config) override;

  private:
    // none if shader compiler is not available.
    GLSLShaderCompilerHandle compiler = nullptr;
    // none if shader compiler is not available.
    std::unique_ptr<HotReloader> reloader = nullptr;

    int shader_source_selector = 0;
    std::string shader_glsl;
    std::string shader_path = {0};
    std::filesystem::path resolved_shader_path;

    vk::Extent3D extent = {1920, 1080, 1};

    SpecializationInfoHandle spec_info;
    VulkanEntryPointHandle shader;
    std::optional<GLSLShaderCompiler::compilation_failed> error;

    PushConstant constant;

    ShaderCompileContextHandle compile_context;

    // Optional input controller, feeds iMouse.
    struct MouseInput : public InputListener {
        float x = 0.f, y = 0.f, click_x = 0.f, click_y = 0.f;
        bool down = false;

        bool on_cursor(InputController& /*controller*/, double xpos, double ypos) override {
            x = static_cast<float>(xpos);
            y = static_cast<float>(ypos);
            return false;
        }
        bool on_mouse_button(InputController& /*controller*/,
                             const InputController::MouseButton button,
                             const InputController::KeyStatus status) override {
            if (button == InputController::MouseButton::MOUSE1) {
                down = status == InputController::KeyStatus::PRESS;
                if (down) {
                    click_x = x;
                    click_y = y;
                }
            }
            return false;
        }
    };

    PtrInHandle<InputController> con_controller = PtrIn<InputController>::create(0, true);
    std::weak_ptr<InputController> registered_controller;
    std::shared_ptr<MouseInput> mouse_input = std::make_shared<MouseInput>();
};

} // namespace merian
