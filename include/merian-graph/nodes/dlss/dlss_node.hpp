#pragma once

#include "merian-graph/connectors/image/vk_image_in_sampled.hpp"
#include "merian-graph/connectors/image/vk_image_out_managed.hpp"
#include "merian-graph/connectors/ptr_in.hpp"
#include "merian-graph/graph/node.hpp"
#include "merian-graph/objects/gbuffer_object.hpp"
#include "merian-shaders/scene/scene.hpp"

#include "merian/vk/extension/ngx/extension_dlss.hpp"

namespace merian {

// Antialiases with DLSS, optionally denoising with DLSS Ray Reconstruction. Renders at the
// resolution of the connected gbuffer and outputs at the configured one, upscaling if it is
// larger.
class DLSSNode : public Node {

  public:
    enum class Mode : uint8_t {
        Bypass,
        SuperResolution,
        RayReconstruction,
    };

    DLSSNode() = default;

    ~DLSSNode() override = default;

    std::vector<std::string> request_context_extensions() override;

    DeviceSupportInfo query_device_support(const DeviceSupportQueryInfo& query_info) override;

    void initialize(const ContextHandle& context,
                    const ResourceAllocatorHandle& allocator) override;

    std::vector<InputConnectorDescriptor> describe_inputs() override;

    std::vector<OutputConnectorDescriptor> describe_outputs(const NodeIOLayout& io_layout) override;

    NodeStatusFlags on_connected(const NodeIOLayout& io_layout,
                                 const NodeIO& io,
                                 const NodeConnectionInfo& info,
                                 Submission& submission) override;

    [[nodiscard]] NodeStatusFlags pre_process(const NodeIO& io,
                                              const NodeProcessInfo& info) override;

    [[nodiscard]] NodeStatusFlags
    process(const NodeIO& io, const NodeProcessInfo& info, Submission& submission) override;

    NodeStatusFlags properties(Properties& config) override;

  private:
    // null while bypassed or where the device does not run the mode
    const std::shared_ptr<ExtensionDLSS>& get_extension() const;

    std::shared_ptr<ExtensionDLSS> dlss_super_resolution;
    std::shared_ptr<ExtensionDLSS> dlss_ray_reconstruction;

    // Connectors
    PtrInHandle<Scene> con_scene = PtrIn<Scene>::create();
    GBufferInHandle con_gbuffer;
    VkSampledImageInHandle con_src = VkSampledImageIn::create();
    ManagedVkImageOutHandle con_out;

    // 0 follows the active camera's resolution, else the input's
    uint32_t out_width = 0;
    uint32_t out_height = 0;
    Mode mode = Mode::SuperResolution;
    DLSSPreset super_resolution_preset = DLSSPreset::Default;
    DLSSPreset ray_reconstruction_preset = DLSSPreset::Default;
    float responsivity = 0.0;
    vk::Format out_format = vk::Format::eR16G16B16A16Sfloat;
    // applied to the active camera
    Camera::JitterSequence jitter_sequence = Camera::JitterSequence::Halton;
    // 0 takes the count DLSS asks for at this scale
    uint32_t jitter_phases = 0;
    std::string reset_event_pattern = "/user/clear";

    // 0 until the scene is ready or where it names none
    vk::Extent3D camera_extent{};

    vk::Extent3D render_extent{};
    vk::Extent3D target_extent{};
    DLSSQuality quality = DLSSQuality::DLAA;
    bool reset = false;

    DLSSHandle dlss;

    ResourceAllocatorHandle allocator;
    TextureHandle responsivity_mask;
};

} // namespace merian
