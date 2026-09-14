#pragma once

#include "merian-graph/connectors/shader_object_in.hpp"
#include "merian-graph/connectors/shader_object_out.hpp"
#include "merian-graph/graph/graph_shader_object.hpp"
#include "merian-shaders/gbuffer.hpp"

namespace merian {

// GBuffer transported as a graph object: the graph allocates the textures per ring slot and keeps
// them synchronized; consumers bind merian::GBuffer / merian::WGBuffer.
class GBufferObject : public GraphShaderObject {
  public:
    struct CreateInfo {
        vk::Extent3D extent;
        // every field if null
        GBufferLayoutHandle layout;
    };

    GBufferObject(const CreateInfo& create_info);

    void allocate(const ShaderObjectAllocateInfo& info) override;

    const ShaderObjectHandle& object(ShaderAccess access) const override;

    // Cleared so consumers never see garbage (out-of-bounds ids in a gbuffer hang the GPU).
    void on_connected(Submission& submission) override;

    vk::Extent3D get_extent() const {
        return extent;
    }

    const GBufferLayoutHandle& get_layout() const {
        return layout;
    }

    // The texture holding the field, laid out as its texture group asked for. Always in eGeneral.
    const ImageViewHandle& get_view(GBufferField field) const;

  private:
    const vk::Extent3D extent;
    const GBufferLayoutHandle layout;

    GBufferHandle gbuffer;
    std::vector<ImageHandle> images;
    std::vector<ImageViewHandle> views;
};

class GBufferIn;
using GBufferInHandle = std::shared_ptr<GBufferIn>;

// Receives a GBufferObject that holds at least the requested fields. Fields the gbuffer does not
// hold read as NaN (an invalid Hit).
class GBufferIn : public ShaderObjectIn<GBufferObject> {
  public:
    explicit GBufferIn(std::vector<GBufferGroup> groups) : groups(std::move(groups)) {}

    const std::vector<GBufferGroup>& get_groups() const {
        return groups;
    }

    static GBufferInHandle create(std::vector<GBufferGroup> groups) {
        return std::make_shared<GBufferIn>(std::move(groups));
    }

  private:
    const std::vector<GBufferGroup> groups;
};

class GBufferOut;
using GBufferOutHandle = std::shared_ptr<GBufferOut>;

// Outputs a GBufferObject that holds the fields its GBufferIns request.
class GBufferOut : public ShaderObjectOut<GBufferObject> {
  public:
    GBufferOut(const vk::Extent3D& extent, bool persistent);

    // Null until the graph allocates the output.
    const GBufferLayoutHandle& get_layout() const {
        return layout;
    }

    static GBufferOutHandle create(const vk::Extent3D& extent, bool persistent = false);

  protected:
    std::shared_ptr<GBufferObject> create_instance(
        const std::vector<std::tuple<NodeHandle, InputConnectorHandle>>& inputs) override;

  private:
    GBufferLayoutHandle layout;
};

} // namespace merian
