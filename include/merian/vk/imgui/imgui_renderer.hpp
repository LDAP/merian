#pragma once

#include "merian/vk/command/command_buffer.hpp"
#include "merian/vk/imgui/imgui_context.hpp"
#include "merian/vk/memory/resource_allocations.hpp"
#include "merian/vk/memory/resource_allocator.hpp"
#include "merian/vk/pipeline/pipeline_graphics.hpp"
#include "merian/vk/pipeline/pipeline_layout.hpp"
#include "merian/vk/utils/profiler.hpp"

#include "imgui.h"

namespace merian {

/*
 * Custom Vulkan ImGui renderer using merian abstractions.
 *
 * Uses dynamic rendering.
 *
 * Calls ImGui::Render() internally at the start of render(), so callers do not need to.
 *
 * Texture IDs:
 *   ImTextureID is an index into an internal TextureHandle registry.
 *   ID 0 is always the ImGui font atlas (registered automatically on first render).
 *   Use register_texture() / unregister_texture() for additional textures.
 *
 * Usage:
 *   ImGuiRenderer renderer(context, alloc, imgui_ctx);
 *   // in render callback, after ImGui frame is built:
 *   renderer.render(cmd, onto_image_view);
 */
class ImGuiRenderer {
  public:
    ImGuiRenderer(const ContextHandle& context,
                  const ResourceAllocatorHandle& alloc,
                  const ImGuiContextHandle& imgui_ctx);
    ~ImGuiRenderer();

    // Register a texture and return its ImTextureID.
    // The texture remains alive until the ID is unregisted.
    ImTextureID register_texture(const TextureHandle& texture);

    // Release a previously registered texture slot (sets it to null).
    void unregister_texture(ImTextureID id);

    // Calls ImGui::Render() then draws ImGui draw data into the given image view.
    void render(const CommandBufferHandle& cmd,
                const ImageViewHandle& image_view,
                const ProfilerHandle& profiler = nullptr);

  private:
    void init_pipeline(vk::Format color_format);
    void upload_pending_textures(const CommandBufferHandle& cmd,
                                 ImVector<ImTextureData*>* tex_list);
    void ensure_buffers(const ImDrawData* draw_data);
    std::size_t alloc_texture_slot();

  private:
    ContextHandle context;
    ResourceAllocatorHandle alloc;
    ImGuiContextHandle imgui_ctx;

    // Lazy-initialized on first render.
    bool initialized = false;
    vk::Format current_format = vk::Format::eUndefined;

    GraphicsPipelineHandle pipeline;
    PipelineLayoutHandle pipeline_layout;
    DescriptorSetLayoutHandle descriptor_layout;

    std::vector<TextureHandle> textures;

    BufferHandle vertex_buffer;
    BufferHandle index_buffer;
};

using ImGuiRendererHandle = std::shared_ptr<ImGuiRenderer>;

} // namespace merian
